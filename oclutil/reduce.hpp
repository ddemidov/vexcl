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

	template <class Expr>
	real sum(const Expr &expr) const;
    private:
	cl::Context context;
	std::vector<cl::CommandQueue> queue;
	std::vector<uint> idx;
	mutable std::vector<real> hbuf;
	std::vector<clu::vector<real>> dbuf;

	template <class Expr>
	struct exdata {
	    static bool       compiled;
	    static cl::Kernel kernel;
	};

	void compile_sum() const;
	void compile_inner() const;
};

template <typename real> template <class Expr>
bool Reductor<real>::exdata<Expr>::compiled = false;

template <typename real> template <class Expr>
cl::Kernel Reductor<real>::exdata<Expr>::kernel;

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

template <typename real> template <class Expr>
real Reductor<real>::sum(const Expr &expr) const {
    if (!exdata<Expr>::compiled) {
	std::ostringstream source;

	std::string kernel_name = std::string("reduce_") + expr.kernel_name();

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
		  "    real mySum = 0;\n"
		  "\n"
		  "    while (p < n) {\n"
		  "        i = p;\n"
		  "        mySum += ";
	expr.kernel_expr(source);
	source << ";\n"
		  "        i = p + block_size;\n"
		  "        if (i < n) mySum += ";
	expr.kernel_expr(source);
	source << ";\n"
		  "        p += gridSize;\n"
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

	auto program = build_sources(context, source.str());

	exdata<Expr>::kernel   = cl::Kernel(program, kernel_name.c_str());
	exdata<Expr>::compiled = true;
    }


    for(uint d = 0; d < queue.size(); d++) {
	uint psize = expr.part_size(d);
	uint l_size = 256;
	uint g_size = (idx[d + 1] - idx[d]) * l_size;
	auto lmem = cl::__local(l_size * sizeof(real));

	uint pos = 0;
	exdata<Expr>::kernel.setArg(pos++, psize);
	expr.kernel_args(exdata<Expr>::kernel, d, pos);
	exdata<Expr>::kernel.setArg(pos++, dbuf[d]());
	exdata<Expr>::kernel.setArg(pos++, lmem);

	queue[d].enqueueNDRangeKernel(exdata<Expr>::kernel, cl::NullRange,
		g_size, l_size);
    }

    for(uint d = 0; d < queue.size(); d++) {
	copy(dbuf[d], &hbuf[idx[d]]);
    }

    return std::accumulate(hbuf.begin(), hbuf.end(), static_cast<real>(0));
}

template <typename real>
real sum(const clu::vector<real> &x) {
    static Reductor<real> rdc(x.queue);

    return rdc.sum(x);
}

template <typename real>
real inner_product(const clu::vector<real> &x, const clu::vector<real> &y) {
    static Reductor<real> rdc(x.queue);

    return rdc.sum(x * y);
}

} // namespace clu

#endif
