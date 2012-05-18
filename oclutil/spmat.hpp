#ifndef SPMAT_HPP
#define SPMAT_HPP

/**
 * \file   spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL sparse matrix.
 */

#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <CL/cl.hpp>
#include <oclutil/vector.hpp>

/// OpenCL convenience utilities.
namespace clu {

template <typename real>
class SpMat {
    public:
	SpMat(const std::vector<cl::CommandQueue> &queue,
	      uint n, const uint *row, const uint *col, const real *val);

	void mul(const clu::vector<real> &x, clu::vector<real> &y) const;
    private:
	struct ell {
	    uint n, w, pitch;
	    std::unique_ptr<clu::vector<uint>> col;
	    std::unique_ptr<clu::vector<real>> val;
	};

	struct exdata {
	    std::vector<uint> mycols;
	    mutable std::vector<real> myvals;

	    std::unique_ptr<clu::vector<uint>> cols_to_send;
	    std::unique_ptr<clu::vector<real>> vals_to_send;
	    std::unique_ptr<clu::vector<real>> rx;
	};

	cl::Context                   context;
	std::vector<cl::CommandQueue> queue;
	std::vector<size_t>           part;

	std::vector<ell> lm; // Local part of the matrix.
	std::vector<ell> rm; // Remote part of the matrix.

	std::vector<exdata> exc;
	std::vector<uint> cidx;
	mutable std::vector<real> rx;

	static bool compiled;
	static cl::Kernel spmv_set;
	static cl::Kernel spmv_add;
	static cl::Kernel gather_vals_to_send;
	static const char *cl_source;
};

template <typename real>
bool SpMat<real>::compiled = false;

template <typename real>
cl::Kernel SpMat<real>::spmv_set;

template <typename real>
cl::Kernel SpMat<real>::spmv_add;

template <typename real>
cl::Kernel SpMat<real>::gather_vals_to_send;

template <typename real>
const char *SpMat<real>::cl_source = "\
#define NCOL (~0U)\n\
\n\
kernel void spmv_set(\n\
    uint n, uint w, uint pitch,\n\
    global const uint *col,\n\
    global const real *val,\n\
    global const real *x,\n\
    global real *y\n\
    )\n\
{\n\
    uint row = get_global_id(0);\n\
    if (row < n) {\n\
	real sum = 0;\n\
	col += row;\n\
	val += row;\n\
	for(uint j = 0; j < w; j++, col += pitch, val += pitch) {\n\
	    uint c = *col;\n\
	    if (c != NCOL) sum += (*val) * x[c];\n\
	}\n\
	y[row] = sum;\n\
    }\n\
}\n\
\n\
kernel void spmv_add(\n\
    uint n, uint w, uint pitch,\n\
    global const uint *col,\n\
    global const real *val,\n\
    global const real *x,\n\
    global real *y\n\
    )\n\
{\n\
    uint row = get_global_id(0);\n\
    if (row < n) {\n\
	real sum = y[row];\n\
	col += row;\n\
	val += row;\n\
	for(uint j = 0; j < w; j++, col += pitch, val += pitch) {\n\
	    uint c = *col;\n\
	    if (c != NCOL) sum += (*val) * x[c];\n\
	}\n\
	y[row] = sum;\n\
    }\n\
}\n\
\n\
kernel void gather_vals_to_send(\n\
    uint n,\n\
    global const real *vals,\n\
    global const uint *cols_to_send,\n\
    global real *vals_to_send\n\
    )\n\
{\n\
    uint i = get_global_id(0);\n\
    if (i < n) vals_to_send[i] = vals[cols_to_send[i]];\n\
}\n\
";

template <typename real>
struct SpMV {
    SpMV(const SpMat<real> &A, const clu::vector<real> &x) : A(A), x(x) {}

    const SpMat<real>       &A;
    const clu::vector<real> &x;
};

template <typename real>
SpMV<real> operator*(const SpMat<real> &A, const clu::vector<real> &x) {
    return SpMV<real>(A, x);
}

template <typename real>
void vector<real>::operator=(const SpMV<real> &spmv) {
    spmv.A.mul(spmv.x, *this);
}

#define NCOL (~0U)
#define BLOCK_SIZE 256

template <typename real>
SpMat<real>::SpMat(const std::vector<cl::CommandQueue> &queue,
	uint n, const uint *row, const uint *col, const real *val
	)
    : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()),
      queue(queue), part(partition(n, queue.size())),
      lm(queue.size()), rm(queue.size()), exc(queue.size())
{
    // Compile kernels.
    if (!compiled) {
	std::ostringstream source;

	source << "#if defined(cl_khr_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
		  "#elif defined(cl_amd_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
		  "#endif\n"
	       << "typedef " << type_name<real>() << " real;\n"
	       << cl_source;

	std::vector<cl::Device> device;
	device.reserve(queue.size());

	for(auto q = queue.begin(); q != queue.end(); q++)
	    device.push_back(q->getInfo<CL_QUEUE_DEVICE>());

	cl::Program program(context, cl::Program::Sources(
		    1, std::make_pair(source.str().c_str(), source.str().size())
		    ));

	try {
	    program.build(device);
	} catch(const cl::Error&) {
	    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
		<< std::endl;
	    throw;
	}

	spmv_set            = cl::Kernel(program, "spmv_set");
	spmv_add            = cl::Kernel(program, "spmv_add");
	gather_vals_to_send = cl::Kernel(program, "gather_vals_to_send");

	compiled = true;
    }

    std::vector<std::set<uint>> remote_cols(queue.size());

    // Each device get it's own strip of the matrix.
    for(uint d = 0; d < queue.size(); d++) {
	// Each strip is divided into local and remote parts.
	// Local part of the strip is its square diagonal subblock.
	// Remote part is everything else.

	// Convert CSR representation of the matrix to ELL format.
	lm[d].n     = part[d + 1] - part[d];
	lm[d].w     = 0;
	lm[d].pitch = alignup(lm[d].n, 16);

	rm[d].n     = lm[d].n;
	rm[d].w     = 0;
	rm[d].pitch = lm[d].pitch;

	// Get widths of local and remote parts.
	for(uint i = part[d]; i < part[d + 1]; i++) {
	    uint w = 0;
	    for(uint j = row[i]; j < row[i + 1]; j++)
		if (col[j] >= part[d] && col[j] < part[d + 1]) w++;

	    lm[d].w = std::max(lm[d].w, w);
	    rm[d].w = std::max(rm[d].w, row[i + 1] - row[i] - w);
	}

	// Rearrange column numbers and matrix values.
	std::vector<uint> lcol(lm[d].pitch * lm[d].w, NCOL);
	std::vector<real> lval(lm[d].pitch * lm[d].w, 0);

	std::vector<uint> rcol(rm[d].pitch * rm[d].w, NCOL);
	std::vector<real> rval(rm[d].pitch * rm[d].w, 0);


	for(uint i = part[d], k = 0; i < part[d + 1]; i++, k++) {
	    for(uint j = row[i], lc = 0, rc = 0; j < row[i + 1]; j++) {
		if (col[j] >= part[d] && col[j] < part[d + 1]) {
		    lcol[k + lm[d].pitch * lc] = col[j] - part[d];
		    lval[k + lm[d].pitch * lc] = val[j];
		    lc++;
		} else {
		    remote_cols[d].insert(col[j]);

		    rcol[k + rm[d].pitch * rc] = col[j];
		    rval[k + rm[d].pitch * rc] = val[j];
		    rc++;
		}
	    }
	}

	// Copy local part to the device.
	std::vector<cl::CommandQueue> myq(1, queue[d]);

	lm[d].col.reset(new vector<uint>(myq, CL_MEM_READ_ONLY, lcol));
	lm[d].val.reset(new vector<real>(myq, CL_MEM_READ_ONLY, lval));

	// Copy remote part to the device.
	if (rm[d].w) {
	    // Renumber columns.
	    std::unordered_map<uint,uint> r2l(2 * remote_cols[d].size());
	    uint k = 0;
	    for(auto c = remote_cols[d].begin(); c != remote_cols[d].end(); c++)
		r2l[*c] = k++;

	    for(auto c = rcol.begin(); c != rcol.end(); c++)
		if (*c != NCOL) *c = r2l[*c];

	    // Copy data to device.
	    rm[d].col.reset(new vector<uint>(myq, CL_MEM_READ_ONLY, rcol));
	    rm[d].val.reset(new vector<real>(myq, CL_MEM_READ_ONLY, rval));
	}
    }

    std::vector<uint> cols_to_send;
    {
	std::set<uint> cols_to_send_s;
	for(uint d = 0; d < queue.size(); d++)
	    cols_to_send_s.insert(remote_cols[d].begin(), remote_cols[d].end());

	cols_to_send.insert(cols_to_send.begin(), cols_to_send_s.begin(), cols_to_send_s.end());
    }

    if (cols_to_send.size()) {
	for(uint d = 0; d < queue.size(); d++) {
	    if (uint rcols = remote_cols[d].size()) {
		exc[d].mycols.resize(rcols);
		exc[d].myvals.resize(rcols);

		std::vector<cl::CommandQueue> myq(1, queue[d]);
		exc[d].rx.reset(new clu::vector<real>(myq, CL_MEM_READ_ONLY, rcols));

		for(uint i = 0, j = 0; i < cols_to_send.size(); i++)
		    if (remote_cols[d].count(cols_to_send[i])) exc[d].mycols[j++] = i;
	    }
	}

	rx.resize(cols_to_send.size());
	cidx.resize(queue.size() + 1);

	cidx[0] = 0;
	for(uint i = 0, d = 0; i < cols_to_send.size(); i++)
	    while(d < queue.size() && cols_to_send[i] >= part[d + 1])
		cidx[++d] = i;
	cidx.back() = cols_to_send.size();

	for(uint d = 0; d < queue.size(); d++) {
	    std::vector<cl::CommandQueue> myq(1, queue[d]);

	    if (uint ncols = cidx[d + 1] - cidx[d]) {
		exc[d].cols_to_send.reset(new clu::vector<uint>(myq, CL_MEM_READ_ONLY, ncols));
		exc[d].vals_to_send.reset(new clu::vector<real>(myq, CL_MEM_READ_WRITE, ncols));

		for(uint i = cidx[d]; i < cidx[d + 1]; i++)
		    cols_to_send[i] -= part[d];

		copy(&cols_to_send[cidx[d]], *exc[d].cols_to_send);
	    }
	}
    }
}

template <typename real>
void SpMat<real>::mul(const clu::vector<real> &x, clu::vector<real> &y) const {
    // Compute contribution from local part of the matrix.
    for(uint d = 0; d < queue.size(); d++) {
	size_t g_size = alignup(lm[d].n, BLOCK_SIZE);
	size_t l_size = BLOCK_SIZE;

	cl::KernelFunctor kset = spmv_set.bind(queue[d], g_size, l_size);

	kset(lm[d].n, lm[d].w, lm[d].pitch, (*lm[d].col)(), (*lm[d].val)(),
		x(d), y(d)
		);
    }

    if (rx.size()) {
	// Transfer remote parts of the input vector.
	for(uint d = 0; d < queue.size(); d++) {
	    if (uint ncols = cidx[d + 1] - cidx[d]) {
		size_t g_size = alignup(ncols, BLOCK_SIZE);
		size_t l_size = BLOCK_SIZE;

		cl::KernelFunctor gather = gather_vals_to_send.bind(
			queue[d], g_size, l_size);

		gather(ncols, x(d), (*exc[d].cols_to_send)(), (*exc[d].vals_to_send)());

		copy(*exc[d].vals_to_send, &rx[cidx[d]]);
	    }
	}

	// Compute contribution from remote part of the matrix.
	for(uint d = 0; d < queue.size(); d++) {
	    if (exc[d].mycols.size()) {
		size_t g_size = alignup(rm[d].n, BLOCK_SIZE);
		size_t l_size = BLOCK_SIZE;

		for(uint i = 0; i < exc[d].mycols.size(); i++)
		    exc[d].myvals[i] = rx[exc[d].mycols[i]];

		copy(exc[d].myvals, *exc[d].rx);

		cl::KernelFunctor kadd = spmv_add.bind(queue[d], g_size, l_size);

		kadd(rm[d].n, rm[d].w, rm[d].pitch, (*rm[d].col)(), (*rm[d].val)(),
			(*exc[d].rx)(), y(d));
	    }
	}
    }
}

} // namespace clu

#endif
