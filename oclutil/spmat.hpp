#ifndef OCLUTIL_SPMAT_HPP
#define OCLUTIL_SPMAT_HPP

/**
 * \file   spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL sparse matrix.
 */

#ifdef WIN32
#  pragma warning(disable : 4290)
#  define NOMINMAX
#endif

#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <CL/cl.hpp>
#include <oclutil/util.hpp>
#include <oclutil/vector.hpp>

/// OpenCL convenience utilities.
namespace clu {

/// Sparse matrix.
template <typename real>
class SpMat {
    public:
	/// Constructor
	/**
	 * Constructs GPU representation of the matrix. Input matrix is in CSR
	 * format. GPU matrix utilizes ELL format and is split equally across
	 * all compute devices. When there are more than one device, secondary
	 * queue can be used to perform transfer of ghost values across GPU
	 * boundaries in parallel with computation kernel.
	 * \param queue vector of primary queues. Each queue represents one
	 *            compute device.
	 * \param squeue vector of secondary queues. Secondary queues are used
	 *            to transfer ghost values across GPU boundaries in
	 *            parallel with computation kernel. It is possible to put
	 *            primary queues here as well.
	 * \param n   number of rows in the matrix.
	 * \param row row index into col and val vectors.
	 * \param col column numbers of nonzero elements of the matrix.
	 * \param val values of nonzero elements of the matrix.
	 */
	SpMat(const std::vector<cl::CommandQueue> &queue,
	      const std::vector<cl::CommandQueue> &squeue,
	      uint n, const uint *row, const uint *col, const real *val
	      );

	/// Matrix-vector multiplication.
	/**
	 * Matrix vector multiplication (y = Ax) is performed in parallel on
	 * all registered compute devices. Ghost values of x are transfered
	 * across GPU boundaries as needed.
	 * \param x input vector.
	 * \param y output vector.
	 */
	void mul(const clu::vector<real> &x, clu::vector<real> &y) const;
    private:
	struct ell {
	    uint n, w, pitch;
	    clu::vector<uint> col;
	    clu::vector<real> val;
	};

	struct exdata {
	    std::vector<uint> mycols;
	    mutable std::vector<real> myvals;

	    clu::vector<uint> cols_to_send;
	    clu::vector<real> vals_to_send;
	    mutable clu::vector<real> rx;
	};

	cl::Context                   context;
	std::vector<cl::CommandQueue> queue;
	std::vector<cl::CommandQueue> squeue;
	std::vector<uint>             part;
	mutable std::vector<cl::Event>        event;

	std::vector<ell> lm; // Local part of the matrix.
	std::vector<ell> rm; // Remote part of the matrix.

	std::vector<exdata> exc;
	std::vector<uint> cidx;
	mutable std::vector<real> rx;

	static bool compiled;
	static cl::Kernel spmv_set;
	static cl::Kernel spmv_add;
	static cl::Kernel gather_vals_to_send;
	static uint wgsize;
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
uint SpMat<real>::wgsize;

/// \internal Sparse matrix-vector product.
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
const vector<real>& vector<real>::operator=(const SpMV<real> &spmv) {
    spmv.A.mul(spmv.x, *this);
    return *this;
}

#define NCOL (~0U)

template <typename real>
SpMat<real>::SpMat(
	const std::vector<cl::CommandQueue> &queue,
	const std::vector<cl::CommandQueue> &squeue,
	uint n, const uint *row, const uint *col, const real *val
	)
    : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()),
      queue(queue), squeue(squeue), part(partition(n, queue.size())),
      event(queue.size()), lm(queue.size()), rm(queue.size()), exc(queue.size())
{
    // Compile kernels.
    if (!compiled) {
	std::ostringstream source;

	source << "#if defined(cl_khr_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
		  "#elif defined(cl_amd_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
		  "#endif\n"
		  "typedef " << type_name<real>() << " real;\n"
		  "#define NCOL (~0U)\n"
		  "kernel void spmv_set(\n"
		  "    uint n, uint w, uint pitch,\n"
		  "    global const uint *col,\n"
		  "    global const real *val,\n"
		  "    global const real *x,\n"
		  "    global real *y\n"
		  "    )\n"
		  "{\n"
		  "    uint row = get_global_id(0);\n"
		  "    if (row < n) {\n"
		  "	real sum = 0;\n"
		  "	col += row;\n"
		  "	val += row;\n"
		  "	for(uint j = 0; j < w; j++, col += pitch, val += pitch) {\n"
		  "	    uint c = *col;\n"
		  "	    if (c != NCOL) sum += (*val) * x[c];\n"
		  "	}\n"
		  "	y[row] = sum;\n"
		  "    }\n"
		  "}\n"
		  "kernel void spmv_add(\n"
		  "    uint n, uint w, uint pitch,\n"
		  "    global const uint *col,\n"
		  "    global const real *val,\n"
		  "    global const real *x,\n"
		  "    global real *y\n"
		  "    )\n"
		  "{\n"
		  "    uint row = get_global_id(0);\n"
		  "    if (row < n) {\n"
		  "	real sum = y[row];\n"
		  "	col += row;\n"
		  "	val += row;\n"
		  "	for(uint j = 0; j < w; j++, col += pitch, val += pitch) {\n"
		  "	    uint c = *col;\n"
		  "	    if (c != NCOL) sum += (*val) * x[c];\n"
		  "	}\n"
		  "	y[row] = sum;\n"
		  "    }\n"
		  "}\n"
		  "kernel void gather_vals_to_send(\n"
		  "    uint n,\n"
		  "    global const real *vals,\n"
		  "    global const uint *cols_to_send,\n"
		  "    global real *vals_to_send\n"
		  "    )\n"
		  "{\n"
		  "    uint i = get_global_id(0);\n"
		  "    if (i < n) vals_to_send[i] = vals[cols_to_send[i]];\n"
		  "}\n";

	auto program = build_sources(context, source.str());

	spmv_set            = cl::Kernel(program, "spmv_set");
	spmv_add            = cl::Kernel(program, "spmv_add");
	gather_vals_to_send = cl::Kernel(program, "gather_vals_to_send");

	std::vector<cl::Device> device;
	device.reserve(queue.size());
	for(auto q = queue.begin(); q != queue.end(); q++)
	    device.push_back(q->getInfo<CL_QUEUE_DEVICE>());

	wgsize = kernel_workgroup_size(spmv_set, device);
	wgsize = std::min(wgsize, kernel_workgroup_size(spmv_add, device));
	wgsize = std::min(wgsize, kernel_workgroup_size(gather_vals_to_send, device));

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
	lm[d].pitch = alignup(lm[d].n, 16U);

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

	lm[d].col = clu::vector<uint>(myq, CL_MEM_READ_ONLY, lcol);
	lm[d].val = clu::vector<real>(myq, CL_MEM_READ_ONLY, lval);

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
	    rm[d].col = clu::vector<uint>(myq, CL_MEM_READ_ONLY, rcol);
	    rm[d].val = clu::vector<real>(myq, CL_MEM_READ_ONLY, rval);
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
		exc[d].rx = clu::vector<real>(myq, CL_MEM_READ_ONLY, rcols);

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
		exc[d].cols_to_send = clu::vector<uint>(myq, CL_MEM_READ_ONLY, ncols);
		exc[d].vals_to_send = clu::vector<real>(myq, CL_MEM_READ_WRITE, ncols);

		for(uint i = cidx[d]; i < cidx[d + 1]; i++)
		    cols_to_send[i] -= part[d];

		copy(&cols_to_send[cidx[d]], exc[d].cols_to_send);
	    }
	}
    }
}

template <typename real>
void SpMat<real>::mul(const clu::vector<real> &x, clu::vector<real> &y) const {
    if (rx.size()) {
	// Transfer remote parts of the input vector.
	for(uint d = 0; d < queue.size(); d++) {
	    if (uint ncols = cidx[d + 1] - cidx[d]) {
		uint g_size = alignup(ncols, wgsize);

		gather_vals_to_send.setArg(0, ncols);
		gather_vals_to_send.setArg(1, x(d));
		gather_vals_to_send.setArg(2, exc[d].cols_to_send());
		gather_vals_to_send.setArg(3, exc[d].vals_to_send());

		squeue[d].enqueueNDRangeKernel(
			gather_vals_to_send, cl::NullRange, g_size, wgsize);

		squeue[d].enqueueReadBuffer(exc[d].vals_to_send(), CL_FALSE,
			0, ncols * sizeof(real), &rx[cidx[d]], 0, &event[d]
			);
	    }
	}
    }

    // Compute contribution from local part of the matrix.
    for(uint d = 0; d < queue.size(); d++) {
	uint g_size = alignup(lm[d].n, wgsize);

	spmv_set.setArg(0, lm[d].n);
	spmv_set.setArg(1, lm[d].w);
	spmv_set.setArg(2, lm[d].pitch);
	spmv_set.setArg(3, lm[d].col());
	spmv_set.setArg(4, lm[d].val());
	spmv_set.setArg(5, x(d));
	spmv_set.setArg(6, y(d));

	queue[d].enqueueNDRangeKernel(spmv_set, cl::NullRange, g_size, wgsize);
    }

    if (rx.size()) {
	// Compute contribution from remote part of the matrix.
	cl::Event::waitForEvents(event);

	for(uint d = 0; d < queue.size(); d++) {
	    if (exc[d].mycols.size()) {
		uint g_size = alignup(rm[d].n, wgsize);

		for(uint i = 0; i < exc[d].mycols.size(); i++)
		    exc[d].myvals[i] = rx[exc[d].mycols[i]];

		squeue[d].enqueueWriteBuffer(
			exc[d].rx(), CL_FALSE,
			0, exc[d].mycols.size() * sizeof(real),
			exc[d].myvals.data(), 0, &event[d]
			);

		std::vector<cl::Event> myevent(1, event[d]);

		spmv_add.setArg(0, rm[d].n);
		spmv_add.setArg(1, rm[d].w);
		spmv_add.setArg(2, rm[d].pitch);
		spmv_add.setArg(3, rm[d].col());
		spmv_add.setArg(4, rm[d].val());
		spmv_add.setArg(5, exc[d].rx());
		spmv_add.setArg(6, y(d));

		queue[d].enqueueNDRangeKernel(
			spmv_add, cl::NullRange, g_size, wgsize, &myevent
			);
	    }
	}
    }
}

} // namespace clu

#endif
