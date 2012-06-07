#ifndef VEXCL_REDUCE_HPP
#define VEXCL_REDUCE_HPP

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
 * \file   reduce.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL vector reduction.
 */

#ifdef WIN32
#  pragma warning(disable : 4290 4715)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <sstream>
#include <numeric>
#include <limits>
#include <CL/cl.hpp>
#include <vexcl/vector.hpp>

namespace vex {

/// Possible kinds of reduction.
enum ReductionKind {
    SUM = 0,
    MAX = 1,
    MIN = 2
};

/// Parallel reduction of arbitrary expression.
/**
 * Reduction uses small temporary buffer on each device present in the queue
 * parameter. One Reductor class for each reduction kind is enough per thread
 * of execution.
 */
template <typename real, ReductionKind RDC>
class Reductor {
    public:
	/// Constructor.
	Reductor(const std::vector<cl::CommandQueue> &queue);

	/// Compute reduction of the input expression.
	/**
	 * The input expression can be as simple as a single vector, although
	 * expressions of arbitrary complexity may be reduced.
	 */
	template <class Expr>
	typename std::enable_if<Expr::is_expr, real>::type
	operator()(const Expr &expr) const;

	template <class Expr>
	typename std::enable_if<Expr::is_multiex, std::array<real,Expr::dim>>::type
	operator()(const Expr &expr) const;
    private:
	const std::vector<cl::CommandQueue> &queue;
	std::vector<size_t> idx;
	std::vector<cl::Buffer> dbuf;

	mutable std::vector<real> hbuf;
	mutable std::vector<cl::Event> event;

	template <class Expr>
	struct exdata {
	    static std::map<cl_context, bool>       compiled;
	    static std::map<cl_context, cl::Kernel> kernel;
	    static std::map<cl_context, size_t>     wgsize;
	};

	static real initial_value() {
	    switch (RDC) {
		case SUM:
		    return 0;
		case MAX:
		    return -std::numeric_limits<real>::max();
		case MIN:
		    return std::numeric_limits<real>::max();
	    }
	}

	template <class Expr>
	std::string cpu_kernel_source(
		const cl::Context &context, const Expr &expr,
		const std::string &kernel_name
		) const;

	template <class Expr>
	std::string gpu_kernel_source(
		const cl::Context &context, const Expr &expr,
		const std::string &kernel_name
		) const;
};

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, bool> Reductor<real,RDC>::exdata<Expr>::compiled;

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, cl::Kernel> Reductor<real,RDC>::exdata<Expr>::kernel;

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, size_t> Reductor<real,RDC>::exdata<Expr>::wgsize;

template <typename real, ReductionKind RDC>
Reductor<real,RDC>::Reductor(const std::vector<cl::CommandQueue> &queue)
    : queue(queue), event(queue.size())
{
    idx.reserve(queue.size() + 1);
    idx.push_back(0);

    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();
	cl::Device d = q->getInfo<CL_QUEUE_DEVICE>();

	size_t bufsize = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 2U;
	idx.push_back(idx.back() + bufsize);

	dbuf.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, bufsize * sizeof(real)));
    }

    hbuf.resize(idx.back());
}

template <typename real, ReductionKind RDC> template <class Expr>
typename std::enable_if<Expr::is_expr, real>::type
Reductor<real,RDC>::operator()(const Expr &expr) const {
    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();

	if (!exdata<Expr>::compiled[context()]) {
	    std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

	    bool device_is_cpu = device[0].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;

	    std::string kernel_name = std::string("reduce_") + expr.kernel_name();

	    std::string source = device_is_cpu ?
		cpu_kernel_source(context, expr, kernel_name) :
		gpu_kernel_source(context, expr, kernel_name) ;

#ifdef VEX_SHOW_KERNELS
	    std::cout << source << std::endl;
#endif

	    auto program = build_sources(context, source);

	    exdata<Expr>::kernel[context()]   = cl::Kernel(program, kernel_name.c_str());
	    exdata<Expr>::compiled[context()] = true;

	    if (device_is_cpu) {
		exdata<Expr>::wgsize[context()] = 1;
	    } else {
		exdata<Expr>::wgsize[context()] = kernel_workgroup_size(
			exdata<Expr>::kernel[context()], device);

		// Strange bug(?) in g++: cannot call getWorkGroupInfo directly on
		// exdata<Expr>::kernel[context()], but it works like this:
		cl::Kernel &krn = exdata<Expr>::kernel[context()];

		for(auto d = device.begin(); d != device.end(); d++) {
		    size_t smem = d->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() -
			krn.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(*d);
		    while(exdata<Expr>::wgsize[context()] * sizeof(real) > smem)
			exdata<Expr>::wgsize[context()] /= 2;
		}
	    }
	}
    }


    for(uint d = 0; d < queue.size(); d++) {
	if (size_t psize = expr.part_size(d)) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	    size_t g_size = (idx[d + 1] - idx[d]) * exdata<Expr>::wgsize[context()];
	    auto lmem = cl::__local(exdata<Expr>::wgsize[context()] * sizeof(real));

	    uint pos = 0;
	    exdata<Expr>::kernel[context()].setArg(pos++, psize);
	    expr.kernel_args(exdata<Expr>::kernel[context()], d, pos);
	    exdata<Expr>::kernel[context()].setArg(pos++, dbuf[d]);
	    exdata<Expr>::kernel[context()].setArg(pos++, lmem);

	    queue[d].enqueueNDRangeKernel(exdata<Expr>::kernel[context()],
		    cl::NullRange, g_size, exdata<Expr>::wgsize[context()]);
	}
    }

    std::fill(hbuf.begin(), hbuf.end(), initial_value());

    for(uint d = 0; d < queue.size(); d++) {
	if (expr.part_size(d))
	    queue[d].enqueueReadBuffer(dbuf[d], CL_FALSE,
		    0, sizeof(real) * (idx[d + 1] - idx[d]), &hbuf[idx[d]], 0, &event[d]);
    }

    for(uint d = 0; d < queue.size(); d++)
	if (expr.part_size(d)) event[d].wait();

    switch(RDC) {
	case SUM:
	    return std::accumulate(
		    hbuf.begin(), hbuf.end(), static_cast<real>(0));
	case MAX:
	    return *std::max_element(hbuf.begin(), hbuf.end());
	case MIN:
	    return *std::min_element(hbuf.begin(), hbuf.end());
    }
}

template <typename real, ReductionKind RDC> template <class Expr>
typename std::enable_if<Expr::is_multiex, std::array<real,Expr::dim>>::type
Reductor<real,RDC>::operator()(const Expr &expr) const {
    std::array<real, Expr::dim> result;
    for (uint i = 0; i < Expr::dim; i++) result[i] = this->operator()(expr(i));
    return result;
}

template <typename real, ReductionKind RDC> template <class Expr>
std::string Reductor<real,RDC>::gpu_kernel_source(
	const cl::Context &context, const Expr &expr,
	const std::string &kernel_name) const
{
    std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

    std::ostringstream source;

    std::ostringstream increment_line;
    switch (RDC) {
	case SUM:
	    increment_line << "mySum += ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ";\n";
	    break;
	case MAX:
	    increment_line << "mySum = max(mySum, ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ");\n";
	    break;
	case MIN:
	    increment_line << "mySum = min(mySum, ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ");\n";
	    break;
    }

    source << standard_kernel_header;
    expr.preamble(source, "prm");
    source << "kernel void " << kernel_name << "(" << type_name<size_t>() << " n";

    expr.kernel_prm(source, "prm");

    source << ",\n\tglobal " << type_name<real>() << " *g_odata,\n"
	"\tlocal  " << type_name<real>() << " *sdata\n"
	"\t)\n"
	"{\n"
	"    size_t tid        = get_local_id(0);\n"
	"    size_t block_size = get_local_size(0);\n"
	"    size_t p          = get_group_id(0) * block_size * 2 + tid;\n"
	"    size_t gridSize   = get_num_groups(0) * block_size * 2;\n"
	"    size_t i;\n"
	"    " << type_name<real>() << " mySum = " << initial_value() << ";\n"
	"    while (p < n) {\n"
	"        i = p;\n"
	"        " << increment_line.str() <<
	"        i = p + block_size;\n"
	"        if (i < n)\n"
	"            " << increment_line.str() <<
	"        p += gridSize;\n"
	"    }\n"
	"    sdata[tid] = mySum;\n"
	"\n";

    switch (RDC) {
	case SUM:
	    source <<
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    if (block_size >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"\n"
		"    if (tid < 32) {\n"
		"        local volatile " << type_name<real>() << "* smem = sdata;\n"
		"        if (block_size >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }\n"
		"        if (block_size >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }\n"
		"        if (block_size >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }\n"
		"        if (block_size >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }\n"
		"        if (block_size >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }\n"
		"        if (block_size >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }\n"
		"    }\n";
	    break;
	case MAX:
	    source <<
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    if (block_size >= 1024) { if (tid < 512) { sdata[tid] = mySum = max(mySum, sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  512) { if (tid < 256) { sdata[tid] = mySum = max(mySum, sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  256) { if (tid < 128) { sdata[tid] = mySum = max(mySum, sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  128) { if (tid <  64) { sdata[tid] = mySum = max(mySum, sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"\n"
		"    if (tid < 32) {\n"
		"        local volatile " << type_name<real>() << "* smem = sdata;\n"
		"        if (block_size >=  64) { smem[tid] = mySum = max(mySum, smem[tid + 32]); }\n"
		"        if (block_size >=  32) { smem[tid] = mySum = max(mySum, smem[tid + 16]); }\n"
		"        if (block_size >=  16) { smem[tid] = mySum = max(mySum, smem[tid +  8]); }\n"
		"        if (block_size >=   8) { smem[tid] = mySum = max(mySum, smem[tid +  4]); }\n"
		"        if (block_size >=   4) { smem[tid] = mySum = max(mySum, smem[tid +  2]); }\n"
		"        if (block_size >=   2) { smem[tid] = mySum = max(mySum, smem[tid +  1]); }\n"
		"    }\n";
	    break;
	case MIN:
	    source <<
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    if (block_size >= 1024) { if (tid < 512) { sdata[tid] = mySum = min(mySum, sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  512) { if (tid < 256) { sdata[tid] = mySum = min(mySum, sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  256) { if (tid < 128) { sdata[tid] = mySum = min(mySum, sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"    if (block_size >=  128) { if (tid <  64) { sdata[tid] = mySum = min(mySum, sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		"\n"
		"    if (tid < 32) {\n"
		"        local volatile " << type_name<real>() << "* smem = sdata;\n"
		"        if (block_size >=  64) { smem[tid] = mySum = min(mySum, smem[tid + 32]); }\n"
		"        if (block_size >=  32) { smem[tid] = mySum = min(mySum, smem[tid + 16]); }\n"
		"        if (block_size >=  16) { smem[tid] = mySum = min(mySum, smem[tid +  8]); }\n"
		"        if (block_size >=   8) { smem[tid] = mySum = min(mySum, smem[tid +  4]); }\n"
		"        if (block_size >=   4) { smem[tid] = mySum = min(mySum, smem[tid +  2]); }\n"
		"        if (block_size >=   2) { smem[tid] = mySum = min(mySum, smem[tid +  1]); }\n"
		"    }\n";
	    break;
    }

    source <<
	"    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
	"}\n";

    return source.str();
}

template <typename real, ReductionKind RDC> template <class Expr>
std::string Reductor<real,RDC>::cpu_kernel_source(
	const cl::Context &context, const Expr &expr,
	const std::string &kernel_name) const
{
    std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

    std::ostringstream source;

    std::ostringstream increment_line;
    switch (RDC) {
	case SUM:
	    increment_line << "mySum += ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ";\n";
	    break;
	case MAX:
	    increment_line << "mySum = max(mySum, ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ");\n";
	    break;
	case MIN:
	    increment_line << "mySum = min(mySum, ";
	    expr.kernel_expr(increment_line, "prm");
	    increment_line << ");\n";
	    break;
    }

    source << standard_kernel_header;
    expr.preamble(source, "prm");
    source << "kernel void " << kernel_name << "(" << type_name<size_t>() << " n";

    expr.kernel_prm(source, "prm");

    source << ",\n\tglobal " << type_name<real>() << " *g_odata,\n"
	"\tlocal  " << type_name<real>() << " *sdata\n"
	"\t)\n"
	"{\n"
	"    size_t grid_size  = get_num_groups(0) * get_local_size(0);\n"
	"    size_t chunk_size = (n + grid_size - 1) / grid_size;\n"
	"    size_t chunk_id   = get_global_id(0);\n"
	"    size_t start      = min(n, chunk_size * chunk_id);\n"
	"    size_t stop       = min(n, chunk_size * (chunk_id + 1));\n"
	"    " << type_name<real>() << " mySum = " << initial_value() << ";\n"
	"    for (size_t i = start; i < stop; i++) {\n"
	"        " << increment_line.str() <<
	"    }\n"
	"\n"
	"    g_odata[get_group_id(0)] = mySum;\n"
	"}\n";

    return source.str();
}

} // namespace vex

#endif
