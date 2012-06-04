#ifndef VEXCL_SPMAT_HPP
#define VEXCL_SPMAT_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

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
	struct sparse_matrix {
	    virtual void mul_local(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, bool append
		    ) const = 0;

	    virtual void mul_remote(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, const std::vector<cl::Event> &event
		    ) const = 0;

	    virtual ~sparse_matrix() {}
	};

	struct SpMatELL : sparse_matrix {
	    SpMatELL(
		    const cl::CommandQueue &queue, uint beg, uint end,
		    const uint *row, const uint *col, const real *val,
		    const std::set<uint> &remote_cols
		    );

	    void prepare_kernels(const cl::Context &context) const;

	    void mul_local(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, bool append
		    ) const;

	    void mul_remote(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, const std::vector<cl::Event> &event
		    ) const;

	    const cl::CommandQueue &queue;

	    uint n, pitch;

	    struct {
		uint w;
		cl::Buffer col;
		cl::Buffer val;
	    } loc, rem;

	    static std::map<cl_context, bool>       compiled;
	    static std::map<cl_context, cl::Kernel> spmv_set;
	    static std::map<cl_context, cl::Kernel> spmv_add;
	    static std::map<cl_context, uint>       wgsize;
	};

	struct SpMatCSR : public sparse_matrix {
	    SpMatCSR(
		    const cl::CommandQueue &queue, uint beg, uint end,
		    const uint *row, const uint *col, const real *val,
		    const std::set<uint> &remote_cols
		    );

	    void prepare_kernels(const cl::Context &context) const;

	    void mul_local(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, bool append
		    ) const;

	    void mul_remote(
		    const cl::Buffer &x, const cl::Buffer &y,
		    real alpha, const std::vector<cl::Event> &event
		    ) const;

	    const cl::CommandQueue &queue;

	    uint n;

	    struct {
		cl::Buffer row;
		cl::Buffer col;
		cl::Buffer val;
	    } loc, rem;

	    static std::map<cl_context, bool>       compiled;
	    static std::map<cl_context, cl::Kernel> spmv_set;
	    static std::map<cl_context, cl::Kernel> spmv_add;
	    static std::map<cl_context, uint>       wgsize;
	};

	struct exdata {
	    std::vector<uint> cols_to_recv;
	    mutable std::vector<real> vals_to_recv;

	    cl::Buffer cols_to_send;
	    cl::Buffer vals_to_send;
	    mutable cl::Buffer rx;
	};

	const std::vector<cl::CommandQueue> &queue;
	std::vector<cl::CommandQueue>       squeue;
	const std::vector<uint>             part;

	mutable std::vector<std::vector<cl::Event>> event1;
	mutable std::vector<std::vector<cl::Event>> event2;

	std::vector<std::unique_ptr<sparse_matrix>> mtx;

	std::vector<exdata> exc;
	std::vector<uint> cidx;
	mutable std::vector<real> rx;

	static std::map<cl_context, bool>       compiled;
	static std::map<cl_context, cl::Kernel> gather_vals_to_send;
	static std::map<cl_context, uint>       wgsize;

	std::vector<std::set<uint>> setup_exchange(
		uint n, const uint *row, const uint *col, const real *val);
};

template <typename real>
std::map<cl_context, bool> SpMat<real>::compiled;

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
    : queue(queue), part(partition(n, queue)),
      event1(queue.size(), std::vector<cl::Event>(1)),
      event2(queue.size(), std::vector<cl::Event>(1)),
      mtx(queue.size()), exc(queue.size())
{
    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();

	// Compile kernels.
	if (!compiled[context()]) {
	    std::ostringstream source;

	    source << standard_kernel_header <<
		"typedef " << type_name<real>() << " real;\n"
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

#ifdef VEX_SHOW_KERNELS
	    std::cout << source.str() << std::endl;
#endif

	    auto program = build_sources(context, source.str());

	    gather_vals_to_send[context()] = cl::Kernel(program, "gather_vals_to_send");

	    wgsize[context()] = kernel_workgroup_size(
		    gather_vals_to_send[context()],
		    context.getInfo<CL_CONTEXT_DEVICES>()
		    );

	    compiled[context()] = true;
	}

	// Create secondary queues.
	cl::Device device = q->getInfo<CL_QUEUE_DEVICE>();

	squeue.push_back(cl::CommandQueue(context, device));
    }

    std::vector<std::set<uint>> remote_cols = setup_exchange(n, row, col, val);

    // Each device get it's own strip of the matrix.
    for(uint d = 0; d < queue.size(); d++) {
	if (part[d + 1] > part[d]) {
	    cl::Device device = queue[d].getInfo<CL_QUEUE_DEVICE>();

	    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
		mtx[d].reset(
			new SpMatCSR(queue[d], part[d], part[d + 1],
			    row, col, val, remote_cols[d])
			);
	    else
		mtx[d].reset(
			new SpMatELL(queue[d], part[d], part[d + 1],
			    row, col, val, remote_cols[d])
			);
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

		uint pos = 0;
		gather_vals_to_send[context()].setArg(pos++, ncols);
		gather_vals_to_send[context()].setArg(pos++, x(d));
		gather_vals_to_send[context()].setArg(pos++, exc[d].cols_to_send);
		gather_vals_to_send[context()].setArg(pos++, exc[d].vals_to_send);

		queue[d].enqueueNDRangeKernel(gather_vals_to_send[context()],
			cl::NullRange, g_size, wgsize[context()], 0, &event1[d][0]);

		squeue[d].enqueueReadBuffer(exc[d].vals_to_send, CL_FALSE,
			0, ncols * sizeof(real), &rx[cidx[d]], &event1[d], &event2[d][0]
			);
	    }
	}
    }

    // Compute contribution from local part of the matrix.
    for(uint d = 0; d < queue.size(); d++)
	if (mtx[d]) mtx[d]->mul_local(x(d), y(d), alpha, append);

    // Compute contribution from remote part of the matrix.
    if (rx.size()) {
	for(uint d = 0; d < queue.size(); d++)
	    if (cidx[d + 1] > cidx[d]) event2[d][0].wait();

	for(uint d = 0; d < queue.size(); d++) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	    if (exc[d].cols_to_recv.size()) {
		cl::Device  device  = queue[d].getInfo<CL_QUEUE_DEVICE>();
		uint g_size = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
		    * wgsize[context()] * 4;

		for(uint i = 0; i < exc[d].cols_to_recv.size(); i++)
		    exc[d].vals_to_recv[i] = rx[exc[d].cols_to_recv[i]];

		squeue[d].enqueueWriteBuffer(
			exc[d].rx, CL_FALSE,
			0, exc[d].cols_to_recv.size() * sizeof(real),
			exc[d].vals_to_recv.data(), 0, &event2[d][0]
			);

		mtx[d]->mul_remote(exc[d].rx, y(d), alpha, event2[d]);
	    }
	}
    }
}

template <typename real>
std::vector<std::set<uint>> SpMat<real>::setup_exchange(
	uint n, const uint *row, const uint *col, const real *val
	)
{
    std::vector<std::set<uint>> remote_cols(queue.size());

    // Build sets of ghost points.
    for(uint d = 0; d < queue.size(); d++) {
	for(uint i = part[d]; i < part[d + 1]; i++) {
	    for(uint j = row[i]; j < row[i + 1]; j++) {
		if (col[j] < part[d] || col[j] >= part[d + 1]) {
		    remote_cols[d].insert(col[j]);
		}
	    }
	}
    }

    // Complete set of points to be exchanged between devices.
    std::vector<uint> cols_to_send;
    {
	std::set<uint> cols_to_send_s;
	for(uint d = 0; d < queue.size(); d++)
	    cols_to_send_s.insert(remote_cols[d].begin(), remote_cols[d].end());

	cols_to_send.insert(cols_to_send.begin(), cols_to_send_s.begin(), cols_to_send_s.end());
    }

    // Build local structures to facilitate exchange.
    if (cols_to_send.size()) {
	for(uint d = 0; d < queue.size(); d++) {
	    if (uint rcols = remote_cols[d].size()) {
		exc[d].cols_to_recv.resize(rcols);
		exc[d].vals_to_recv.resize(rcols);

		exc[d].rx = cl::Buffer(queue[d].getInfo<CL_QUEUE_CONTEXT>(),
			CL_MEM_READ_ONLY, rcols * sizeof(real));

		for(uint i = 0, j = 0; i < cols_to_send.size(); i++)
		    if (remote_cols[d].count(cols_to_send[i])) exc[d].cols_to_recv[j++] = i;
	    }
	}

	rx.resize(cols_to_send.size());
	cidx.resize(queue.size() + 1);

	{
	    auto beg = cols_to_send.begin();
	    auto end = cols_to_send.end();
	    for(uint d = 0; d <= queue.size(); d++) {
		cidx[d] = std::lower_bound(beg, end, part[d]) - cols_to_send.begin();
		beg = cols_to_send.begin() + cidx[d];
	    }
	}

	for(uint d = 0; d < queue.size(); d++) {
	    if (uint ncols = cidx[d + 1] - cidx[d]) {
		cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

		exc[d].cols_to_send = cl::Buffer(
			context, CL_MEM_READ_ONLY, ncols * sizeof(uint));

		exc[d].vals_to_send = cl::Buffer(
			context, CL_MEM_READ_WRITE, ncols * sizeof(real));

		for(uint i = cidx[d]; i < cidx[d + 1]; i++)
		    cols_to_send[i] -= part[d];

		queue[d].enqueueWriteBuffer(
			exc[d].cols_to_send, CL_TRUE, 0, ncols * sizeof(uint),
			&cols_to_send[cidx[d]]);
	    }
	}
    }

    return remote_cols;
}

//---------------------------------------------------------------------------
// SpMat::SpMatELL
//---------------------------------------------------------------------------
template <typename real>
std::map<cl_context, bool> SpMat<real>::SpMatELL::compiled;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::SpMatELL::spmv_set;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::SpMatELL::spmv_add;

template <typename real>
std::map<cl_context, uint> SpMat<real>::SpMatELL::wgsize;

template <typename real>
SpMat<real>::SpMatELL::SpMatELL(
	const cl::CommandQueue &queue, uint beg, uint end,
	const uint *row, const uint *col, const real *val,
	const std::set<uint> &remote_cols
	)
    : queue(queue), n(end - beg), pitch(alignup(n, 16U))
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    prepare_kernels(context);

    loc.w = rem.w = 0;

    // Get widths of local and remote parts.
    for(uint i = beg; i < end; i++) {
	uint w = 0;
	for(uint j = row[i]; j < row[i + 1]; j++)
	    if (col[j] >= beg && col[j] < end) w++;

	loc.w = std::max(loc.w, w);
	rem.w = std::max(rem.w, row[i + 1] - row[i] - w);
    }

    // Rearrange column numbers and matrix values to ELL format.
    std::vector<uint> lcol(pitch * loc.w, NCOL);
    std::vector<real> lval(pitch * loc.w, 0);

    std::vector<uint> rcol(pitch * rem.w, NCOL);
    std::vector<real> rval(pitch * rem.w, 0);

    {
	// Renumber columns.
	std::unordered_map<uint,uint> r2l(2 * remote_cols.size());
	for(auto c = remote_cols.begin(); c != remote_cols.end(); c++) {
	    uint idx = r2l.size();
	    r2l[*c] = idx;
	}

	for(uint i = beg, k = 0; i < end; i++, k++) {
	    for(uint j = row[i], lc = 0, rc = 0; j < row[i + 1]; j++) {
		if (col[j] >= beg && col[j] < end) {
		    lcol[k + pitch * lc] = col[j] - beg;
		    lval[k + pitch * lc] = val[j];
		    lc++;
		} else {
		    assert(r2l.count(col[j]));
		    rcol[k + pitch * rc] = r2l[col[j]];
		    rval[k + pitch * rc] = val[j];
		    rc++;
		}
	    }
	}
    }

    cl::Event event;

    // Copy local part to the device.
    loc.col = cl::Buffer(
	    context, CL_MEM_READ_ONLY, lcol.size() * sizeof(uint));

    loc.val = cl::Buffer(
	    context, CL_MEM_READ_ONLY, lval.size() * sizeof(real));

    queue.enqueueWriteBuffer(
	    loc.col, CL_FALSE, 0, lcol.size() * sizeof(uint), lcol.data());

    queue.enqueueWriteBuffer(
	    loc.val, CL_FALSE, 0, lval.size() * sizeof(real), lval.data(),
	    0, &event);

    // Copy remote part to the device.
    if (rem.w) {
	rem.col = cl::Buffer(
		context, CL_MEM_READ_ONLY, rcol.size() * sizeof(uint));

	rem.val = cl::Buffer(
		context, CL_MEM_READ_ONLY, rval.size() * sizeof(real));

	queue.enqueueWriteBuffer(
		rem.col, CL_FALSE, 0, rcol.size() * sizeof(uint), rcol.data());

	queue.enqueueWriteBuffer(
		rem.val, CL_FALSE, 0, rval.size() * sizeof(real), rval.data(),
		0, &event);
    }

    // Wait for data to be copied before it gets deallocated.
    event.wait();
}

template <typename real>
void SpMat<real>::SpMatELL::prepare_kernels(const cl::Context &context) const {
    if (!compiled[context()]) {
	std::ostringstream source;

	source << standard_kernel_header <<
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
	    "    uint grid_size = get_num_groups(0) * get_local_size(0);\n"
	    "    for (uint row = get_global_id(0); row < n; row += grid_size) {\n"
	    "        real sum = 0;\n"
	    "        for(uint j = 0; j < w; j++) {\n"
	    "            uint c = col[row + j * pitch];\n"
	    "            if (c != NCOL) sum += val[row + j * pitch] * x[c];\n"
	    "        }\n"
	    "        y[row] = alpha * sum;\n"
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
	    "    uint grid_size = get_num_groups(0) * get_local_size(0);\n"
	    "    for(uint row = get_global_id(0); row < n; row += grid_size) {\n"
	    "        real sum = 0;\n"
	    "        for(uint j = 0; j < w; j++) {\n"
	    "            uint c = col[row + j * pitch];\n"
	    "            if (c != NCOL) sum += val[row + j * pitch] * x[c];\n"
	    "        }\n"
	    "        y[row] += alpha * sum;\n"
	    "    }\n"
	    "}\n";

#ifdef VEX_SHOW_KERNELS
	std::cout << source.str() << std::endl;
#endif

	auto program = build_sources(context, source.str());

	spmv_set[context()] = cl::Kernel(program, "spmv_set");
	spmv_add[context()] = cl::Kernel(program, "spmv_add");

	std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

	wgsize[context()] = std::min(
		kernel_workgroup_size(spmv_set[context()], device),
		kernel_workgroup_size(spmv_add[context()], device)
		);

	compiled[context()] = true;
    }
}

template <typename real>
void SpMat<real>::SpMatELL::mul_local(
	const cl::Buffer &x, const cl::Buffer &y,
	real alpha, bool append
	) const
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device  device  = queue.getInfo<CL_QUEUE_DEVICE>();

    uint g_size = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
	* wgsize[context()] * 4;

    if (append) {
	uint pos = 0;
	spmv_add[context()].setArg(pos++, n);
	spmv_add[context()].setArg(pos++, loc.w);
	spmv_add[context()].setArg(pos++, pitch);
	spmv_add[context()].setArg(pos++, loc.col);
	spmv_add[context()].setArg(pos++, loc.val);
	spmv_add[context()].setArg(pos++, x);
	spmv_add[context()].setArg(pos++, y);
	spmv_add[context()].setArg(pos++, alpha);

	queue.enqueueNDRangeKernel(spmv_add[context()],
		cl::NullRange, g_size, wgsize[context()]);
    } else {
	uint pos = 0;
	spmv_set[context()].setArg(pos++, n);
	spmv_set[context()].setArg(pos++, loc.w);
	spmv_set[context()].setArg(pos++, pitch);
	spmv_set[context()].setArg(pos++, loc.col);
	spmv_set[context()].setArg(pos++, loc.val);
	spmv_set[context()].setArg(pos++, x);
	spmv_set[context()].setArg(pos++, y);
	spmv_set[context()].setArg(pos++, alpha);

	queue.enqueueNDRangeKernel(spmv_set[context()],
		cl::NullRange, g_size, wgsize[context()]);
    }
}
template <typename real>
void SpMat<real>::SpMatELL::mul_remote(
	const cl::Buffer &x, const cl::Buffer &y,
	real alpha, const std::vector<cl::Event> &event
	) const
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device  device  = queue.getInfo<CL_QUEUE_DEVICE>();

    uint g_size = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
	* wgsize[context()] * 4;

    uint pos = 0;
    spmv_add[context()].setArg(pos++, n);
    spmv_add[context()].setArg(pos++, rem.w);
    spmv_add[context()].setArg(pos++, pitch);
    spmv_add[context()].setArg(pos++, rem.col);
    spmv_add[context()].setArg(pos++, rem.val);
    spmv_add[context()].setArg(pos++, x);
    spmv_add[context()].setArg(pos++, y);
    spmv_add[context()].setArg(pos++, alpha);

    queue.enqueueNDRangeKernel(spmv_add[context()],
	    cl::NullRange, g_size, wgsize[context()], &event
	    );
}

//---------------------------------------------------------------------------
// SpMat::SpMatCSR
//---------------------------------------------------------------------------
template <typename real>
std::map<cl_context, bool> SpMat<real>::SpMatCSR::compiled;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::SpMatCSR::spmv_set;

template <typename real>
std::map<cl_context, cl::Kernel> SpMat<real>::SpMatCSR::spmv_add;

template <typename real>
std::map<cl_context, uint> SpMat<real>::SpMatCSR::wgsize;

template <typename real>
SpMat<real>::SpMatCSR::SpMatCSR(
	const cl::CommandQueue &queue, uint beg, uint end,
	const uint *row, const uint *col, const real *val,
	const std::set<uint> &remote_cols
	)
    : queue(queue), n(end - beg)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    prepare_kernels(context);

    if (beg == 0 && remote_cols.empty()) {
	loc.row = cl::Buffer(
		context, CL_MEM_READ_ONLY, (n + 1) * sizeof(uint));

	loc.col = cl::Buffer(
		context, CL_MEM_READ_ONLY, row[n] * sizeof(uint));

	loc.val = cl::Buffer(
		context, CL_MEM_READ_ONLY, row[n] * sizeof(real));

	queue.enqueueWriteBuffer(
		loc.row, CL_FALSE, 0, (n + 1) * sizeof(uint), row);

	queue.enqueueWriteBuffer(
		loc.col, CL_FALSE, 0, row[n] * sizeof(uint), col);

	queue.enqueueWriteBuffer(
		loc.val, CL_TRUE, 0, row[n] * sizeof(real), val);
    } else {
	std::vector<uint> lrow;
	std::vector<uint> lcol;
	std::vector<real> lval;

	std::vector<uint> rrow;
	std::vector<uint> rcol;
	std::vector<real> rval;

	lrow.reserve(end - beg + 1);
	lrow.push_back(0);

	lcol.reserve(row[end] - row[beg]);
	lval.reserve(row[end] - row[beg]);

	if (!remote_cols.empty()) {
	    rrow.reserve(end - beg + 1);
	    rrow.push_back(0);

	    rcol.reserve(row[end] - row[beg]);
	    rval.reserve(row[end] - row[beg]);
	}

	// Renumber columns.
	std::unordered_map<uint,uint> r2l(2 * remote_cols.size());
	for(auto c = remote_cols.begin(); c != remote_cols.end(); c++) {
	    uint idx = r2l.size();
	    r2l[*c] = idx;
	}

	for(uint i = beg; i < end; i++) {
	    for(uint j = row[i]; j < row[i + 1]; j++) {
		if (col[j] >= beg && col[j] < end) {
		    lcol.push_back(col[j] - beg);
		    lval.push_back(val[j]);
		} else {
		    assert(r2l.count(col[j]));
		    rcol.push_back(r2l[col[j]]);
		    rval.push_back(val[j]);
		}
	    }

	    lrow.push_back(lcol.size());
	    rrow.push_back(rcol.size());
	}

	cl::Event event;

	// Copy local part to the device.
	loc.row = cl::Buffer(
		context, CL_MEM_READ_ONLY, lrow.size() * sizeof(uint));

	loc.col = cl::Buffer(
		context, CL_MEM_READ_ONLY, lcol.size() * sizeof(uint));

	loc.val = cl::Buffer(
		context, CL_MEM_READ_ONLY, lval.size() * sizeof(real));

	queue.enqueueWriteBuffer(
		loc.row, CL_FALSE, 0, lrow.size() * sizeof(uint), lrow.data());

	queue.enqueueWriteBuffer(
		loc.col, CL_FALSE, 0, lcol.size() * sizeof(uint), lcol.data());

	queue.enqueueWriteBuffer(
		loc.val, CL_FALSE, 0, lval.size() * sizeof(real), lval.data(),
		0, &event);

	// Copy remote part to the device.
	if (!remote_cols.empty()) {
	    rem.row = cl::Buffer(
		    context, CL_MEM_READ_ONLY, rrow.size() * sizeof(uint));

	    rem.col = cl::Buffer(
		    context, CL_MEM_READ_ONLY, rcol.size() * sizeof(uint));

	    rem.val = cl::Buffer(
		    context, CL_MEM_READ_ONLY, rval.size() * sizeof(real));

	    queue.enqueueWriteBuffer(
		    rem.row, CL_FALSE, 0, rrow.size() * sizeof(uint), rrow.data());

	    queue.enqueueWriteBuffer(
		    rem.col, CL_FALSE, 0, rcol.size() * sizeof(uint), rcol.data());

	    queue.enqueueWriteBuffer(
		    rem.val, CL_FALSE, 0, rval.size() * sizeof(real), rval.data(),
		    0, &event);
	}

	event.wait();
    }
}

template <typename real>
void SpMat<real>::SpMatCSR::prepare_kernels(const cl::Context &context) const {
    if (!compiled[context()]) {
	std::ostringstream source;

	source << standard_kernel_header <<
	    "typedef " << type_name<real>() << " real;\n"
	    "#define NCOL (~0U)\n"
	    "kernel void spmv_set(\n"
	    "    uint n,\n"
	    "    global const uint *row,\n"
	    "    global const uint *col,\n"
	    "    global const real *val,\n"
	    "    global const real *x,\n"
	    "    global real *y,\n"
	    "    real alpha\n"
	    "    )\n"
	    "{\n"
	    "    uint i = get_global_id(0);\n"
	    "    if (i < n) {\n"
	    "        real sum = 0;\n"
	    "        uint beg = row[i];\n"
	    "        uint end = row[i + 1];\n"
	    "        for(uint j = beg; j < end; j++)\n"
	    "            sum += val[j] * x[col[j]];\n"
	    "        y[i] = alpha * sum;\n"
	    "    }\n"
	    "}\n"
	    "kernel void spmv_add(\n"
	    "    uint n,\n"
	    "    global const uint *row,\n"
	    "    global const uint *col,\n"
	    "    global const real *val,\n"
	    "    global const real *x,\n"
	    "    global real *y,\n"
	    "    real alpha\n"
	    "    )\n"
	    "{\n"
	    "    uint i = get_global_id(0);\n"
	    "    if (i < n) {\n"
	    "        real sum = 0;\n"
	    "        uint beg = row[i];\n"
	    "        uint end = row[i + 1];\n"
	    "        for(uint j = beg; j < end; j++)\n"
	    "            sum += val[j] * x[col[j]];\n"
	    "        y[i] += alpha * sum;\n"
	    "    }\n"
	    "}\n";

#ifdef VEX_SHOW_KERNELS
	std::cout << source.str() << std::endl;
#endif

	auto program = build_sources(context, source.str());

	spmv_set[context()] = cl::Kernel(program, "spmv_set");
	spmv_add[context()] = cl::Kernel(program, "spmv_add");

	std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

	wgsize[context()] = std::min(
		kernel_workgroup_size(spmv_set[context()], device),
		kernel_workgroup_size(spmv_add[context()], device)
		);

	compiled[context()] = true;
    }
}

template <typename real>
void SpMat<real>::SpMatCSR::mul_local(
	const cl::Buffer &x, const cl::Buffer &y,
	real alpha, bool append
	) const
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint g_size = alignup(n, wgsize[context()]);

    if (append) {
	uint pos = 0;
	spmv_add[context()].setArg(pos++, n);
	spmv_add[context()].setArg(pos++, loc.row);
	spmv_add[context()].setArg(pos++, loc.col);
	spmv_add[context()].setArg(pos++, loc.val);
	spmv_add[context()].setArg(pos++, x);
	spmv_add[context()].setArg(pos++, y);
	spmv_add[context()].setArg(pos++, alpha);

	queue.enqueueNDRangeKernel(spmv_add[context()],
		cl::NullRange, n, cl::NullRange);
    } else {
	uint pos = 0;
	spmv_set[context()].setArg(pos++, n);
	spmv_set[context()].setArg(pos++, loc.row);
	spmv_set[context()].setArg(pos++, loc.col);
	spmv_set[context()].setArg(pos++, loc.val);
	spmv_set[context()].setArg(pos++, x);
	spmv_set[context()].setArg(pos++, y);
	spmv_set[context()].setArg(pos++, alpha);

	queue.enqueueNDRangeKernel(spmv_set[context()],
		cl::NullRange, n, cl::NullRange);
    }
}

template <typename real>
void SpMat<real>::SpMatCSR::mul_remote(
	const cl::Buffer &x, const cl::Buffer &y,
	real alpha, const std::vector<cl::Event> &event
	) const
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint pos = 0;
    spmv_add[context()].setArg(pos++, n);
    spmv_add[context()].setArg(pos++, rem.row);
    spmv_add[context()].setArg(pos++, rem.col);
    spmv_add[context()].setArg(pos++, rem.val);
    spmv_add[context()].setArg(pos++, x);
    spmv_add[context()].setArg(pos++, y);
    spmv_add[context()].setArg(pos++, alpha);

    queue.enqueueNDRangeKernel(spmv_add[context()],
	    cl::NullRange, n, cl::NullRange, &event
	    );
}

/// Returns device weight after spmv test
double device_spmv_perf(
	const cl::Context &context, const cl::Device &device,
	uint test_size = 64U
	)
{
    static std::map<cl_device_id, double> dev_weights;

    auto dw = dev_weights.find(device());

    if (dw == dev_weights.end()) {
	std::vector<cl::CommandQueue> queue(1,
		cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE)
		);

	// Construct matrix for 3D Poisson problem in cubic domain.
	const uint  n   = test_size;
	const float h   = 1.0 / (n - 1);
	const float h2i = (n - 1) * (n - 1);

	std::vector<uint>  row;
	std::vector<uint>  col;
	std::vector<float> val;

	row.reserve(n * n * n + 1);
	col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
	val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

	row.push_back(0);
	for(uint k = 0, idx = 0; k < n; k++) {
	    float z = k * h;
	    for(uint j = 0; j < n; j++) {
		float y = j * h;
		for(uint i = 0; i < n; i++, idx++) {
		    float x = i * h;
		    if (
			    i == 0 || i == (n - 1) ||
			    j == 0 || j == (n - 1) ||
			    k == 0 || k == (n - 1)
		       )
		    {
			col.push_back(idx);
			val.push_back(1);
			row.push_back(row.back() + 1);
		    } else {
			col.push_back(idx - n * n);
			val.push_back(-h2i);

			col.push_back(idx - n);
			val.push_back(-h2i);

			col.push_back(idx - 1);
			val.push_back(-h2i);

			col.push_back(idx);
			val.push_back(6 * h2i);

			col.push_back(idx + 1);
			val.push_back(-h2i);

			col.push_back(idx + n);
			val.push_back(-h2i);

			col.push_back(idx + n * n);
			val.push_back(-h2i);

			row.push_back(row.back() + 7);
		    }
		}
	    }
	}

	// Create device vectors and copy of the matrix.
	vex::SpMat<float>  A(queue, n * n * n, row.data(), col.data(), val.data());
	vex::vector<float> x(queue, n * n * n);
	vex::vector<float> y(queue, n * n * n);

	// Warming run.
	x = 1;
	y = A * x;

	// Measure performance.
	profiler prof(queue);
	prof.tic_cl("");
	y = A * x;
	double time = prof.toc("");
	std::cout << device.getInfo<CL_DEVICE_NAME>() << " - " << time << std::endl;
	return dev_weights[device()] = 1 / time;
    } else {
	return dw->second;
    }
}

/// Partitions vector wrt to vector performance of devices.
/**
 * Launches the following kernel on each device:
 * \code
 * a = b + c;
 * \endcode
 * where a, b and c are device vectors. Each device gets portion of the vector
 * proportional to the performance of this operation.
 */
std::vector<uint> partition_by_spmv_perf(
	uint n, const std::vector<cl::CommandQueue> &queue)
{

    std::vector<uint> part(queue.size() + 1, 0);

    if (queue.size() > 1) {
	std::vector<double> cumsum;
	cumsum.reserve(queue.size() + 1);
	cumsum.push_back(0);

	for(auto q = queue.begin(); q != queue.end(); q++)
	    cumsum.push_back(cumsum.back() + device_spmv_perf(
			q->getInfo<CL_QUEUE_CONTEXT>(),
			q->getInfo<CL_QUEUE_DEVICE>()
			));

	for(uint d = 0; d < queue.size(); d++)
	    part[d + 1] = std::min(n, alignup(n * cumsum[d + 1] / cumsum.back()));
    }

    part.back() = n;

    return part;
}

} // namespace vex

#endif
