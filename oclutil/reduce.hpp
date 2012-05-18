#ifndef REDUCE_HPP
#define REDUCE_HPP

#include <vector>
#include <sstream>
#include <numeric>
#include <CL/cl.hpp>
#include <oclutil/vector.hpp>

namespace clu {
template <typename real>
class Reductor {
    public:
	Reductor(const std::vector<cl::CommandQueue> &queue);

	real sum(const clu::vector<real> &x) const;
	real inner(const clu::vector<real> &x, const clu::vector<real> &y) const;
    private:
	cl::Context context;
	std::vector<cl::CommandQueue> queue;
	std::vector<uint> idx;
	mutable std::vector<real> hbuf;
	std::vector<clu::vector<real>> dbuf;

	static bool sum_compiled;
	static bool inner_compiled;

	static cl::Kernel sum_krn;
	static cl::Kernel inner_krn;

	void compile_sum() const;
	void compile_inner() const;
};

template <typename real> bool Reductor<real>::sum_compiled = false;
template <typename real> bool Reductor<real>::inner_compiled = false;

template <typename real> cl::Kernel Reductor<real>::sum_krn;
template <typename real> cl::Kernel Reductor<real>::inner_krn;

template <typename real>
Reductor<real>::Reductor(const std::vector<cl::CommandQueue> &queue)
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

template <typename real>
real Reductor<real>::sum(const clu::vector<real> &x) const {
    compile_sum();

    for(uint d = 0; d < queue.size(); d++) {
	uint psize = x.part_size(d);

	size_t l_size = 256;
	size_t g_size = (idx[d + 1] - idx[d]) * l_size;
	auto   lmem = cl::__local(l_size * sizeof(real));

	cl::KernelFunctor sum = sum_krn.bind(queue[d], g_size, l_size);

	sum(psize, x(d), dbuf[d](), lmem);
    }

    for(uint d = 0; d < queue.size(); d++) {
	copy(dbuf[d], &hbuf[idx[d]]);
    }

    return std::accumulate(hbuf.begin(), hbuf.end(), static_cast<real>(0));
}

template <typename real>
real Reductor<real>::inner(const clu::vector<real> &x, const clu::vector<real> &y) const {
    compile_inner();

    for(uint d = 0; d < queue.size(); d++) {
	uint psize = x.part_size(d);

	size_t l_size = 256;
	size_t g_size = (idx[d + 1] - idx[d]) * l_size;
	auto   lmem = cl::__local(l_size * sizeof(real));

	cl::KernelFunctor inner = inner_krn.bind(queue[d], g_size, l_size);

	inner(psize, x(d), y(d), dbuf[d](), lmem);
    }

    for(uint d = 0; d < queue.size(); d++) {
	copy(dbuf[d], &hbuf[idx[d]]);
    }

    return std::accumulate(hbuf.begin(), hbuf.end(), static_cast<real>(0));
}

template <typename real>
void Reductor<real>::compile_sum() const {
    if (sum_compiled) return;

    std::ostringstream source;

    source << "#if defined(cl_khr_fp64)\n"
	      "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	      "#elif defined(cl_amd_fp64)\n"
	      "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	      "#endif\n"
	   << "typedef " << type_name<real>() << " real;\n"
	   << "kernel void sum(uint n,\n"
	      "    global const real *g_idata,\n"
	      "    global real *g_odata,\n"
	      "    local  real *sdata\n"
	      "    )\n"
	      "{\n"
	      "    uint tid        = get_local_id(0);\n"
	      "    uint block_size = get_local_size(0);\n"
	      "    uint i          = get_group_id(0) * block_size * 2 + tid;\n"
	      "    uint gridSize   = get_num_groups(0) * block_size * 2;\n"
	      "\n"
	      "    real mySum = 0;\n"
	      "\n"
	      "    while (i < n) {\n"
	      "        mySum += g_idata[i];\n"
	      "        if (i + block_size < n) mySum += g_idata[i + block_size];\n"
	      "        i += gridSize;\n"
	      "    }\n"
	      "\n"
	      "    sdata[tid] = mySum;\n"
	      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
	      "\n"
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

    sum_krn      = cl::Kernel(program, "sum");
    sum_compiled = true;
}

template <typename real>
void Reductor<real>::compile_inner() const {
    if (inner_compiled) return;

    std::ostringstream source;

    source << "#if defined(cl_khr_fp64)\n"
	      "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	      "#elif defined(cl_amd_fp64)\n"
	      "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	      "#endif\n"
	   << "typedef " << type_name<real>() << " real;\n"
	   << "kernel void inner(uint n,\n"
	      "    global const real *g_xdata,\n"
	      "    global const real *g_ydata,\n"
	      "    global real *g_odata,\n"
	      "    local  real *sdata\n"
	      "    )\n"
	      "{\n"
	      "    uint tid        = get_local_id(0);\n"
	      "    uint block_size = get_local_size(0);\n"
	      "    uint i          = get_group_id(0) * block_size * 2 + tid;\n"
	      "    uint gridSize   = get_num_groups(0) * block_size * 2;\n"
	      "\n"
	      "    real mySum = 0;\n"
	      "\n"
	      "    while (i < n) {\n"
	      "        mySum += g_xdata[i] * g_ydata[i];\n"
	      "        if (i + block_size < n)\n"
	      "            mySum += g_xdata[i + block_size] * g_ydata[i + block_size];\n"
	      "        i += gridSize;\n"
	      "    }\n"
	      "\n"
	      "    sdata[tid] = mySum;\n"
	      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
	      "\n"
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

    inner_krn      = cl::Kernel(program, "inner");
    inner_compiled = true;
}

template <typename real>
real sum(const clu::vector<real> &x) {
    static Reductor<real> rdc(x.queue);

    return rdc.sum(x);
}

template <typename real>
real inner_product(const clu::vector<real> &x, const clu::vector<real> &y) {
    static Reductor<real> rdc(x.queue);

    return rdc.inner(x, y);
}

} // namespace clu

#endif
