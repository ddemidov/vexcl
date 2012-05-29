#ifndef VEXCL_SPMAT_HPP
#define VEXCL_SPMAT_HPP

/**
 * \file   spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL sparse matrix.
 */

#ifdef WIN32
#  pragma warning(disable : 4290)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/vector.hpp>

#include <cstdlib>

/// OpenCL convenience utilities.
namespace vex {

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
	 * \param queue vector of queues. Each queue represents one
	 *            compute device.
	 * \param n   number of rows in the matrix.
	 * \param row row index into col and val vectors.
	 * \param col column numbers of nonzero elements of the matrix.
	 * \param val values of nonzero elements of the matrix.
	 */
	SpMat(const std::vector<cl::CommandQueue> &queue,
	      uint n, const uint *row, const uint *col, const real *val
	      );

	/// Matrix-vector multiplication.
	/**
	 * Matrix vector multiplication (y = alpha Ax or y += alpha Ax) is
	 * performed in parallel on all registered compute devices. Ghost
	 * values of x are transfered across GPU boundaries as needed.
	 * \param x      input vector.
	 * \param y      output vector.
	 * \param alpha  coefficient in front of matrix-vector product
	 * \param append if set, matrix-vector product is appended to y.
	 *               Otherwise, y is replaced with matrix-vector product.
	 */
	void mul(const vex::vector<real> &x, vex::vector<real> &y,
		 real alpha = 1, bool append = false) const;
    private:
	struct ell {
	    uint n, w, pitch;
	    vex::vector<uint> col;
	    vex::vector<real> val;
	};

	struct exdata {
	    std::vector<uint> mycols;
	    mutable std::vector<real> myvals;

	    vex::vector<uint> cols_to_send;
	    vex::vector<real> vals_to_send;
	    mutable vex::vector<real> rx;
	};

	std::vector<cl::CommandQueue> queue;
	std::vector<uint>             part;
	mutable std::vector<cl::Event>        event;

	std::vector<ell> lm; // Local part of the matrix.
	std::vector<ell> rm; // Remote part of the matrix.

	std::vector<exdata> exc;
	std::vector<uint> cidx;
	mutable std::vector<real> rx;

	static std::map<cl_context, bool>       compiled;
	static std::map<cl_context, cl::Kernel> spmv_set;
	static std::map<cl_context, cl::Kernel> spmv_add;
	static std::map<cl_context, cl::Kernel> gather_vals_to_send;
	static std::map<cl_context, uint>       wgsize;
};

template <typename real>
std::map<cl_context, bool> SpMat<real>::compiled;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::spmv_set;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::spmv_add;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::gather_vals_to_send;

template <typename real>
std::map<cl_context, uint> SpMat<real>::wgsize;

/// \internal Sparse matrix-vector product.
template <typename real>
struct SpMV {
    SpMV(const SpMat<real> &A, const vex::vector<real> &x) : A(A), x(x) {}

    const SpMat<real>       &A;
    const vex::vector<real> &x;
};

/// Multiply sparse matrix and a vector.
template <typename real>
SpMV<real> operator*(const SpMat<real> &A, const vex::vector<real> &x) {
    return SpMV<real>(A, x);
}

/// \internal Expression with matrix-vector product.
template <class Expr, typename real>
struct ExSpMV {
    ExSpMV(const Expr &expr, const real alpha, const SpMV<real> &spmv)
	: expr(expr), alpha(alpha), spmv(spmv) {}

    const Expr &expr;
    const real alpha;
    const SpMV<real> &spmv;
};

/// Add an expression and sparse matrix - vector product.
template <class Expr, typename real>
typename std::enable_if<Expr::is_expression, ExSpMV<Expr,real>>::type
operator+(const Expr &expr, const SpMV<real> &spmv) {
    return ExSpMV<Expr,real>(expr, 1, spmv);
}

/// Subtruct sparse matrix - vector product from an expression.
template <class Expr, typename real>
typename std::enable_if<Expr::is_expression, ExSpMV<Expr,real>>::type
operator-(const Expr &expr, const SpMV<real> &spmv) {
    return ExSpMV<Expr,real>(expr, -1, spmv);
}

template <typename real>
const vector<real>& vector<real>::operator=(const SpMV<real> &spmv) {
    spmv.A.mul(spmv.x, *this);
    return *this;
}

template <typename real>
const vector<real>& vector<real>::operator+=(const SpMV<real> &spmv) {
    spmv.A.mul(spmv.x, *this, 1, true);
    return *this;
}

template <typename real>
const vector<real>& vector<real>::operator-=(const SpMV<real> &spmv) {
    spmv.A.mul(spmv.x, *this, -1, true);
    return *this;
}

template <typename real> template<class Expr>
const vector<real>& vector<real>::operator=(const ExSpMV<Expr,real> &xmv) {
    *this = xmv.expr;
    xmv.spmv.A.mul(xmv.spmv.x, *this, xmv.alpha, true);
    return *this;
}

#define NCOL (~0U)

template <typename real>
SpMat<real>::SpMat(
	const std::vector<cl::CommandQueue> &queue,
	uint n, const uint *row, const uint *col, const real *val
	)
    : queue(queue), part(partition(n, queue.size())), event(queue.size()),
      lm(queue.size()), rm(queue.size()), exc(queue.size())
{
    // Compile kernels.
    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();

	if (!compiled[context()]) {
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
		      "    global real *y,\n"
		      "    real alpha\n"
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
		      "	y[row] = alpha * sum;\n"
		      "    }\n"
		      "}\n"
		      "kernel void spmv_add(\n"
		      "    uint n, uint w, uint pitch,\n"
		      "    global const uint *col,\n"
		      "    global const real *val,\n"
		      "    global const real *x,\n"
		      "    global real *y,\n"
		      "    real alpha\n"
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
		      "	y[row] += alpha * sum;\n"
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

	    spmv_set[context()]            = cl::Kernel(program, "spmv_set");
	    spmv_add[context()]            = cl::Kernel(program, "spmv_add");
	    gather_vals_to_send[context()] = cl::Kernel(program, "gather_vals_to_send");

	    std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

	    wgsize[context()] = kernel_workgroup_size(spmv_set[context()], device);

	    wgsize[context()] = std::min(wgsize[context()],
		    kernel_workgroup_size(spmv_add[context()], device));

	    wgsize[context()] = std::min(wgsize[context()],
		    kernel_workgroup_size(gather_vals_to_send[context()], device));

	    compiled[context()] = true;
	}
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

	lm[d].col = vex::vector<uint>(myq, CL_MEM_READ_ONLY, lcol);
	lm[d].val = vex::vector<real>(myq, CL_MEM_READ_ONLY, lval);

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
	    rm[d].col = vex::vector<uint>(myq, CL_MEM_READ_ONLY, rcol);
	    rm[d].val = vex::vector<real>(myq, CL_MEM_READ_ONLY, rval);
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
		exc[d].rx = vex::vector<real>(myq, CL_MEM_READ_ONLY, rcols);

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
		exc[d].cols_to_send = vex::vector<uint>(myq, CL_MEM_READ_ONLY, ncols);
		exc[d].vals_to_send = vex::vector<real>(myq, CL_MEM_READ_WRITE, ncols);

		for(uint i = cidx[d]; i < cidx[d + 1]; i++)
		    cols_to_send[i] -= part[d];

		vex::copy(&cols_to_send[cidx[d]], &cols_to_send[cidx[d + 1]],
			exc[d].cols_to_send.begin());
	    }
	}
    }
}

template <typename real>
void SpMat<real>::mul(const vex::vector<real> &x, vex::vector<real> &y,
	real alpha, bool append) const
{
    if (rx.size()) {
	// Transfer remote parts of the input vector.
	for(uint d = 0; d < queue.size(); d++) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	    if (uint ncols = cidx[d + 1] - cidx[d]) {
		uint g_size = alignup(ncols, wgsize[context()]);

		gather_vals_to_send[context()].setArg(0, ncols);
		gather_vals_to_send[context()].setArg(1, x(d));
		gather_vals_to_send[context()].setArg(2, exc[d].cols_to_send());
		gather_vals_to_send[context()].setArg(3, exc[d].vals_to_send());

		queue[d].enqueueNDRangeKernel(gather_vals_to_send[context()],
			cl::NullRange, g_size, wgsize[context()]);

		queue[d].enqueueReadBuffer(exc[d].vals_to_send(), CL_FALSE,
			0, ncols * sizeof(real), &rx[cidx[d]], 0, &event[d]
			);
	    }
	}
    }

    // Compute contribution from local part of the matrix.
    for(uint d = 0; d < queue.size(); d++) {
	cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();
	uint g_size = alignup(lm[d].n, wgsize[context()]);

	if (append) {
	    spmv_add[context()].setArg(0, lm[d].n);
	    spmv_add[context()].setArg(1, lm[d].w);
	    spmv_add[context()].setArg(2, lm[d].pitch);
	    spmv_add[context()].setArg(3, lm[d].col());
	    spmv_add[context()].setArg(4, lm[d].val());
	    spmv_add[context()].setArg(5, x(d));
	    spmv_add[context()].setArg(6, y(d));
	    spmv_add[context()].setArg(7, alpha);

	    queue[d].enqueueNDRangeKernel(spmv_add[context()],
		    cl::NullRange, g_size, wgsize[context()]);
	} else {
	    spmv_set[context()].setArg(0, lm[d].n);
	    spmv_set[context()].setArg(1, lm[d].w);
	    spmv_set[context()].setArg(2, lm[d].pitch);
	    spmv_set[context()].setArg(3, lm[d].col());
	    spmv_set[context()].setArg(4, lm[d].val());
	    spmv_set[context()].setArg(5, x(d));
	    spmv_set[context()].setArg(6, y(d));
	    spmv_set[context()].setArg(7, alpha);

	    queue[d].enqueueNDRangeKernel(spmv_set[context()],
		    cl::NullRange, g_size, wgsize[context()]);
	}
    }

    if (rx.size()) {
	// Compute contribution from remote part of the matrix.
	for(uint d = 0; d < queue.size(); d++)
	    if (cidx[d + 1] > cidx[d]) event[d].wait();

	for(uint d = 0; d < queue.size(); d++) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	    if (exc[d].mycols.size()) {
		uint g_size = alignup(rm[d].n, wgsize[context()]);

		for(uint i = 0; i < exc[d].mycols.size(); i++)
		    exc[d].myvals[i] = rx[exc[d].mycols[i]];

		queue[d].enqueueWriteBuffer(
			exc[d].rx(), CL_FALSE,
			0, exc[d].mycols.size() * sizeof(real),
			exc[d].myvals.data()
			);

		spmv_add[context()].setArg(0, rm[d].n);
		spmv_add[context()].setArg(1, rm[d].w);
		spmv_add[context()].setArg(2, rm[d].pitch);
		spmv_add[context()].setArg(3, rm[d].col());
		spmv_add[context()].setArg(4, rm[d].val());
		spmv_add[context()].setArg(5, exc[d].rx());
		spmv_add[context()].setArg(6, y(d));
		spmv_add[context()].setArg(7, alpha);

		queue[d].enqueueNDRangeKernel(spmv_add[context()],
			cl::NullRange, g_size, wgsize[context()]
			);
	    }
	}
    }
}

} // namespace vex

#endif
