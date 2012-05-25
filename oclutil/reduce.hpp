#ifndef OCLUTIL_REDUCE_HPP
#define OCLUTIL_REDUCE_HPP

/**
 * \file   reduce.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL vector reduction.
 */

#ifdef WIN32
#  pragma warning(disable : 4290 4715)
#  define NOMINMAX
#endif

#include <vector>
#include <sstream>
#include <numeric>
#include <limits>
#include <CL/cl.hpp>
#include <oclutil/vector.hpp>

namespace clu {

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
	real operator()(const Expr &expr) const;
    private:
	cl::Context context;
	std::vector<cl::CommandQueue> queue;
	std::vector<uint> idx;
	std::vector<clu::vector<real>> dbuf;

	mutable std::vector<real> hbuf;
	mutable std::vector<cl::Event> event;

	template <class Expr>
	struct exdata {
	    static std::map<cl_context, bool>       compiled;
	    static std::map<cl_context, cl::Kernel> kernel;
	    static std::map<cl_context, uint>       wgsize;
	};
};

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, bool> Reductor<real,RDC>::exdata<Expr>::compiled;

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, cl::Kernel> Reductor<real,RDC>::exdata<Expr>::kernel;

template <typename real, ReductionKind RDC> template <class Expr>
std::map<cl_context, uint> Reductor<real,RDC>::exdata<Expr>::wgsize;

template <typename real, ReductionKind RDC>
Reductor<real,RDC>::Reductor(const std::vector<cl::CommandQueue> &queue)
    : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()), queue(queue),
      event(queue.size())
{
    idx.reserve(queue.size() + 1);
    idx.push_back(0);

    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Device d = q->getInfo<CL_QUEUE_DEVICE>();
	uint bufsize = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 2U;
	idx.push_back(idx.back() + bufsize);

	std::vector<cl::CommandQueue> lq(1, *q);
	dbuf.push_back(clu::vector<real>(lq, CL_MEM_READ_WRITE, bufsize));
    }

    hbuf.resize(idx.back());
}

template <typename real, ReductionKind RDC> template <class Expr>
real Reductor<real,RDC>::operator()(const Expr &expr) const {
    if (!exdata<Expr>::compiled[context()]) {
	std::vector<cl::Device> device;
	device.reserve(queue.size());

	for(auto q = queue.begin(); q != queue.end(); q++)
	    device.push_back(q->getInfo<CL_QUEUE_DEVICE>());

	std::ostringstream source;

	std::string kernel_name = std::string("reduce_") + expr.kernel_name();

	std::ostringstream increment_line;
	switch (RDC) {
	    case SUM:
		increment_line << "mySum += ";
		expr.kernel_expr(increment_line);
		increment_line << ";\n";
		break;
	    case MAX:
		increment_line << "mySum = max(mySum, ";
		expr.kernel_expr(increment_line);
		increment_line << ");\n";
		break;
	    case MIN:
		increment_line << "mySum = min(mySum, ";
		expr.kernel_expr(increment_line);
		increment_line << ");\n";
		break;
	}

	source << "#if defined(cl_khr_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
		  "#elif defined(cl_amd_fp64)\n"
		  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
		  "#endif\n"
	       << "typedef " << type_name<real>() << " real;\n"
	       << "kernel void " << kernel_name << "(uint n";

	expr.kernel_prm(source);

	source << ",\n\tglobal real *g_odata,\n"
		  "\tlocal  real *sdata\n"
		  "\t)\n"
		  "{\n"
		  "    uint tid        = get_local_id(0);\n"
		  "    uint block_size = get_local_size(0);\n"
		  "    uint p          = get_group_id(0) * block_size * 2 + tid;\n"
		  "    uint gridSize   = get_num_groups(0) * block_size * 2;\n"
		  "    uint i;\n"
		  "\n"
		  "    real mySum = ";
	switch(RDC) {
	    case SUM:
		source << 0;
		break;
	    case MAX:
		source << -std::numeric_limits<real>::max();
		break;
	    case MIN:
		source << std::numeric_limits<real>::max();
		break;
	}
	source << ";\n"
		  "\n"
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
	if (device[0].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
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
		  "        local volatile real* smem = sdata;\n"
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
		  "        local volatile real* smem = sdata;\n"
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
		  "        local volatile real* smem = sdata;\n"
		  "        if (block_size >=  64) { smem[tid] = mySum = min(mySum, smem[tid + 32]); }\n"
		  "        if (block_size >=  32) { smem[tid] = mySum = min(mySum, smem[tid + 16]); }\n"
		  "        if (block_size >=  16) { smem[tid] = mySum = min(mySum, smem[tid +  8]); }\n"
		  "        if (block_size >=   8) { smem[tid] = mySum = min(mySum, smem[tid +  4]); }\n"
		  "        if (block_size >=   4) { smem[tid] = mySum = min(mySum, smem[tid +  2]); }\n"
		  "        if (block_size >=   2) { smem[tid] = mySum = min(mySum, smem[tid +  1]); }\n"
		  "    }\n";
		break;
	    }
	}

	source << "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
		  "}\n";

	auto program = build_sources(context, source.str());

	exdata<Expr>::kernel[context()]   = cl::Kernel(program, kernel_name.c_str());
	exdata<Expr>::compiled[context()] = true;

	if (device[0].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) {
	    exdata<Expr>::wgsize[context()] = 1;
	} else {
	    exdata<Expr>::wgsize[context()] = kernel_workgroup_size(
		    exdata<Expr>::kernel[context()], device);

	    // Strange bug(?) in g++: cannot call getWorkGroupInfo directly on
	    // exdata<Expr>::kernel[context()], but it works like this:
	    cl::Kernel &krn = exdata<Expr>::kernel[context()];

	    for(auto d = device.begin(); d != device.end(); d++) {
		uint smem = d->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() -
		    krn.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(*d);
		while(exdata<Expr>::wgsize[context()] * sizeof(real) > smem)
		    exdata<Expr>::wgsize[context()] /= 2;
	    }
	}
    }


    for(uint d = 0; d < queue.size(); d++) {
	uint psize = expr.part_size(d);
	uint g_size = (idx[d + 1] - idx[d]) * exdata<Expr>::wgsize[context()];
	auto lmem = cl::__local(exdata<Expr>::wgsize[context()] * sizeof(real));

	uint pos = 0;
	exdata<Expr>::kernel[context()].setArg(pos++, psize);
	expr.kernel_args(exdata<Expr>::kernel[context()], d, pos);
	exdata<Expr>::kernel[context()].setArg(pos++, dbuf[d]());
	exdata<Expr>::kernel[context()].setArg(pos++, lmem);

	queue[d].enqueueNDRangeKernel(exdata<Expr>::kernel[context()], cl::NullRange,
		g_size, exdata<Expr>::wgsize[context()]);
    }

    for(uint d = 0; d < queue.size(); d++) {
	queue[d].enqueueReadBuffer(dbuf[d](), CL_FALSE,
		0, sizeof(real) * dbuf[d].size(), &hbuf[idx[d]], 0, &event[d]);
    }

    cl::Event::waitForEvents(event);

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

/// Sum of vector elements.
template <typename real>
real sum(const clu::vector<real> &x) {
    static Reductor<real,SUM> rdc(x.queue);

    return rdc(x);
}

/// Inner product of two vectors.
template <typename real>
real inner_product(const clu::vector<real> &x, const clu::vector<real> &y) {
    static Reductor<real,SUM> rdc(x.queue);

    return rdc(x * y);
}

} // namespace clu

#endif
