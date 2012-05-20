#ifndef OCLUTIL_REDUCE_HPP
#define OCLUTIL_REDUCE_HPP

/**
 * \file   reduce.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL vector reduction.
 */

#include <vector>
#include <sstream>
#include <numeric>
#include <CL/cl.hpp>
#include <oclutil/vector.hpp>

namespace clu {

enum ReductionKind {
    SUM = 0,
    MAX = 1,
    MIN = 2
};

/// Parallel reduction of arbitrary expression.
template <typename real, ReductionKind RDC>
class Reductor {
    public:
	Reductor(const std::vector<cl::CommandQueue> &queue);

	template <class Expr>
	real operator()(const Expr &expr) const;
    private:
	cl::Context context;
	std::vector<cl::CommandQueue> queue;
	std::vector<uint> idx;
	mutable std::vector<real> hbuf;
	std::vector<clu::vector<real>> dbuf;

	template <class Expr>
	struct exdata {
	    static bool       compiled[3];
	    static cl::Kernel kernel[3];
	};
};

template <typename real, ReductionKind RDC> template <class Expr>
bool Reductor<real,RDC>::exdata<Expr>::compiled[3] = {false, false, false};

template <typename real, ReductionKind RDC> template <class Expr>
cl::Kernel Reductor<real,RDC>::exdata<Expr>::kernel[3];

template <typename real, ReductionKind RDC>
Reductor<real,RDC>::Reductor(const std::vector<cl::CommandQueue> &queue)
    : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()), queue(queue)
{
    idx.reserve(queue.size() + 1);
    idx.push_back(0);

    for(auto q = queue.begin(); q != queue.end(); q++) {
	cl::Device d = q->getInfo<CL_QUEUE_DEVICE>();
	uint bufsize = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 2U;
	idx.push_back(idx.back() + bufsize);

	std::vector<cl::CommandQueue> lq(1, *q);
	dbuf.emplace_back(lq, CL_MEM_READ_WRITE, bufsize);
    }

    hbuf.resize(idx.back());
}

template <typename real, ReductionKind RDC> template <class Expr>
real Reductor<real,RDC>::operator()(const Expr &expr) const {
    if (!exdata<Expr>::compiled[RDC]) {
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
		  "\n"
		  "    sdata[tid] = mySum;\n"
		  "    barrier(CLK_LOCAL_MEM_FENCE);\n"
		  "\n";
	switch (RDC) {
	    case SUM:
		source <<
		  "    if (block_size >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "\n"
		  "    if (tid < 32) {\n"
		  "        local volatile real* smem = sdata;\n"
		  "        if (block_size >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }\n"
		  "        if (block_size >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }\n"
		  "        if (block_size >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }\n"
		  "        if (block_size >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }\n"
		  "        if (block_size >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }\n"
		  "        if (block_size >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }\n"
		  "    }\n"
		  "\n"
		  "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
		  "}\n";
		break;
	    case MAX:
		source <<
		  "    if (block_size >= 512) { if (tid < 256) { sdata[tid] = mySum = max(mySum, sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 256) { if (tid < 128) { sdata[tid] = mySum = max(mySum, sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 128) { if (tid <  64) { sdata[tid] = mySum = max(mySum, sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "\n"
		  "    if (tid < 32) {\n"
		  "        local volatile real* smem = sdata;\n"
		  "        if (block_size >=  64) { smem[tid] = mySum = max(mySum, smem[tid + 32]); }\n"
		  "        if (block_size >=  32) { smem[tid] = mySum = max(mySum, smem[tid + 16]); }\n"
		  "        if (block_size >=  16) { smem[tid] = mySum = max(mySum, smem[tid +  8]); }\n"
		  "        if (block_size >=   8) { smem[tid] = mySum = max(mySum, smem[tid +  4]); }\n"
		  "        if (block_size >=   4) { smem[tid] = mySum = max(mySum, smem[tid +  2]); }\n"
		  "        if (block_size >=   2) { smem[tid] = mySum = max(mySum, smem[tid +  1]); }\n"
		  "    }\n"
		  "\n"
		  "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
		  "}\n";
		break;
	    case MIN:
		source <<
		  "    if (block_size >= 512) { if (tid < 256) { sdata[tid] = mySum = min(mySum, sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 256) { if (tid < 128) { sdata[tid] = mySum = min(mySum, sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "    if (block_size >= 128) { if (tid <  64) { sdata[tid] = mySum = min(mySum, sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
		  "\n"
		  "    if (tid < 32) {\n"
		  "        local volatile real* smem = sdata;\n"
		  "        if (block_size >=  64) { smem[tid] = mySum = min(mySum, smem[tid + 32]); }\n"
		  "        if (block_size >=  32) { smem[tid] = mySum = min(mySum, smem[tid + 16]); }\n"
		  "        if (block_size >=  16) { smem[tid] = mySum = min(mySum, smem[tid +  8]); }\n"
		  "        if (block_size >=   8) { smem[tid] = mySum = min(mySum, smem[tid +  4]); }\n"
		  "        if (block_size >=   4) { smem[tid] = mySum = min(mySum, smem[tid +  2]); }\n"
		  "        if (block_size >=   2) { smem[tid] = mySum = min(mySum, smem[tid +  1]); }\n"
		  "    }\n"
		  "\n"
		  "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
		  "}\n";
		break;
	}

	std::vector<cl::Device> device;
	device.reserve(queue.size());

	for(auto q = queue.begin(); q != queue.end(); q++)
	    device.push_back(q->getInfo<CL_QUEUE_DEVICE>());

	auto program = build_sources(context, source.str());

	exdata<Expr>::kernel[RDC]   = cl::Kernel(program, kernel_name.c_str());
	exdata<Expr>::compiled[RDC] = true;
    }


    for(uint d = 0; d < queue.size(); d++) {
	uint psize = expr.part_size(d);
	uint l_size = 256;
	uint g_size = (idx[d + 1] - idx[d]) * l_size;
	auto lmem = cl::__local(l_size * sizeof(real));

	uint pos = 0;
	exdata<Expr>::kernel[RDC].setArg(pos++, psize);
	expr.kernel_args(exdata<Expr>::kernel[RDC], d, pos);
	exdata<Expr>::kernel[RDC].setArg(pos++, dbuf[d]());
	exdata<Expr>::kernel[RDC].setArg(pos++, lmem);

	queue[d].enqueueNDRangeKernel(exdata<Expr>::kernel[RDC], cl::NullRange,
		g_size, l_size);
    }

    for(uint d = 0; d < queue.size(); d++) {
	copy(dbuf[d], &hbuf[idx[d]]);
    }

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
