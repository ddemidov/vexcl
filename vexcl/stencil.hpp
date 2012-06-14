#ifndef VEXCL_STENCIL_HPP
#define VEXCL_STENCIL_HPP

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
 * \file   stencil.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Stencil convolution.
 */

#ifdef WIN32
#  pragma warning(push)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <map>
#include <sstream>
#include <cassert>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/vector.hpp>

namespace vex {

/// Stencil.
/**
 * Should be used for stencil convolutions with vex::vectors as in
 * \code
 * void convolve(
 *	    const vex::stencil<double> &s,
 *	    const vex::vector<double>  &x,
 *	    vex::vector<double> &y)
 * {
 *     y = x * s;
 * }
 * \endcode
 * Stencil should be small enough to fit into local memory of all compute
 * devices it resides on.
 */
template <typename T>
class stencil {
    public:
	/// Costructor.
	/**
	 * \param queue  vector of queues. Each queue represents one
	 *               compute device.
	 * \param st     vector holding stencil values.
	 * \param center center of the stencil.
	 */
	stencil(const std::vector<cl::CommandQueue> &queue,
		const std::vector<T> &st, uint center
		) : queue(queue), s(queue.size())
	{
	    init(st, center);
	}

	/// Costructor.
	/**
	 * \param queue  vector of queues. Each queue represents one
	 *               compute device.
	 * \param begin  iterator to begin of sequence holding stencil data.
	 * \param end    iterator to end of sequence holding stencil data.
	 * \param center center of the stencil.
	 */
	template <class Iterator>
	stencil(const std::vector<cl::CommandQueue> &queue,
		Iterator begin, Iterator end, uint center
		) : queue(queue), s(queue.size())
	{
	    std::vector<T> st(begin, end);
	    init(st, center);
	}

	/// Convolve stencil with a vector.
	/**
	 * \param x input vector.
	 * \param y output vector.
	 */
	void convolve(const vex::vector<T> &x, vex::vector<T> &y) const;
    private:
	const std::vector<cl::CommandQueue> &queue;
	std::vector<cl::CommandQueue> squeue;

	mutable std::vector<T>	hbuf;
	std::vector<cl::Buffer> dbuf;
	std::vector<cl::Buffer> s;
	mutable std::vector<cl::Event>  event;

	int lhalo;
	int rhalo;

	void init(const std::vector<T> &data, uint center);

	static std::map<cl_context, bool>	compiled;
	static std::map<cl_context, cl::Kernel> conv_local;
	static std::map<cl_context, cl::Kernel> conv_remote;
	static std::map<cl_context, uint>	wgsize;
};

template <typename T>
std::map<cl_context, bool> stencil<T>::compiled;

template <typename T>
std::map<cl_context, cl::Kernel> stencil<T>::conv_local;

template <typename T>
std::map<cl_context, cl::Kernel> stencil<T>::conv_remote;

template <typename T>
std::map<cl_context, uint> stencil<T>::wgsize;

template <typename T>
struct Conv {
    Conv(const vex::vector<T> &x, const stencil<T> &s) : x(x), s(s) {}

    const vex::vector<T> &x;
    const stencil<T> &s;
};

template <typename T>
Conv<T> operator*(const vex::vector<T> &x, const stencil<T> &s) {
    return Conv<T>(x, s);
}

template <typename T>
Conv<T> operator*(const stencil<T> &s, const vex::vector<T> &x) {
    return x * s;
}

template <typename T>
const vex::vector<T>& vex::vector<T>::operator=(const Conv<T> &cnv) {
    cnv.s.convolve(cnv.x, *this);
    return *this;
}

template <typename T>
void stencil<T>::init(const std::vector<T> &data, uint center) {
    assert(queue.size());
    assert(data.size());
    assert(center < data.size());

    lhalo = center;
    rhalo = data.size() - center - 1;

    for (uint d = 0; d < queue.size(); d++) {
	cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();
	cl::Device  device  = queue[d].getInfo<CL_QUEUE_DEVICE>();

	if (!compiled[context()]) {
	    std::ostringstream source;

	    source << standard_kernel_header <<
		"typedef " << type_name<T>() << " real;\n"
		"kernel void conv_local(\n"
		"    " << type_name<size_t>() << " n,\n"
		"    int lhalo,\n"
		"    int rhalo,\n"
		"    global const real *x,\n"
		"    global const real *s,\n"
		"    global real *y,\n"
		"    local real *loc_buf\n"
		"    )\n"
		"{\n"
		"    local real *stencil = loc_buf;\n"
		"    local real *xbuf    = loc_buf + lhalo + rhalo + 1 + lhalo;\n"
		"    size_t block_size = get_local_size(0);\n"
		"    long   g_id = get_global_id(0);\n"
		"    int    l_id = get_local_id(0);\n"
		"    if (l_id < lhalo + rhalo + 1)\n"
		"        stencil[l_id] = s[l_id];\n"
		"    stencil += lhalo;\n"
		"    if (g_id < n) {\n"
		"        xbuf[l_id] = x[g_id];\n"
		"        if (l_id < lhalo && g_id >= lhalo)\n"
		"            xbuf[l_id - lhalo] = x[g_id - lhalo];\n"
		"        if (l_id + rhalo < block_size || g_id + rhalo < n)\n"
		"            xbuf[l_id + rhalo] = x[g_id + rhalo];\n"
		"    }\n"
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    if (g_id < n) {\n"
		"        real sum = 0;\n"
		"        for(int k = -lhalo; k <= rhalo; k++)\n"
		"            if (g_id + k >= 0 && g_id + k < n)\n"
		"                sum += stencil[k] * xbuf[l_id + k];\n"
		"        y[g_id] = sum;\n"
		"    }\n"
		"}\n"
		"kernel void conv_remote(\n"
		"    " << type_name<size_t>() << " n,\n"
		"    char has_left,\n"
		"    char has_right,\n"
		"    int lhalo,\n"
		"    int rhalo,\n"
		"    global const real *x,\n"
		"    global const real *h,\n"
		"    global const real *s,\n"
		"    global real *y\n"
		"    )\n"
		"{\n"
		"    long g_id = get_global_id(0);\n"
		"    global const real *xl = h + lhalo;\n"
		"    global const real *xr = h + lhalo;\n"
		"    s += lhalo;\n"
		"    if (g_id < lhalo) {\n"
		"        real sum = 0;\n"
		"        for(int k = -lhalo; k < 0; k++)\n"
		"            if (g_id + k < 0)\n"
		"                sum += s[k] * (has_left ? xl[g_id + k] : x[0]);\n"
		"        y[g_id] += sum;\n"
		"    }\n"
		"    if (g_id < rhalo) {\n"
		"        real sum = 0;\n"
		"        for(int k = 1; k <= rhalo; k++)\n"
		"            if (g_id + k - rhalo >= 0)\n"
		"                sum += s[k] * (has_right ? xr[g_id + k - rhalo] : x[n - 1]);\n"
		"        y[n - rhalo + g_id] += sum;\n"
		"    }\n"
		"}\n";

#ifdef VEXCL_SHOW_KERNELS
	    std::cout << source.str() << std::endl;
#endif

	    auto program = build_sources(context, source.str());

	    conv_local [context()] = cl::Kernel(program, "conv_local");
	    conv_remote[context()] = cl::Kernel(program, "conv_remote");

	    wgsize[context()] = std::min(
		    kernel_workgroup_size(conv_local [context()], device),
		    kernel_workgroup_size(conv_remote[context()], device)
		    );

	    size_t smem = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() -
		std::max(
			conv_local [context()].getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device),
			conv_remote[context()].getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device)
			);

	    while ((wgsize[context()] + lhalo + rhalo) * sizeof(T) > smem)
		wgsize[context()] /= 2;

	    compiled[context()] = true;
	}

	s[d] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		data.size() * sizeof(T), const_cast<T*>(data.data()));
    }

    hbuf.resize(queue.size() * (lhalo + rhalo));
    dbuf.resize(queue.size());
    event.resize(queue.size());
    squeue.resize(queue.size());

    if (lhalo + rhalo > 0) {
	for(uint d = 0; d < queue.size(); d++) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();
	    cl::Device  device  = queue[d].getInfo<CL_QUEUE_DEVICE>();

	    squeue[d] = cl::CommandQueue(context, device);

	    dbuf[d] = cl::Buffer(context, CL_MEM_READ_WRITE, (lhalo + rhalo) * sizeof(T));
	}
    }
}

template <typename T>
void stencil<T>::convolve(const vex::vector<T> &x, vex::vector<T> &y) const {
    if ((queue.size() > 1) && (lhalo + rhalo)) {
	for(uint d = 0; d < queue.size(); d++) {
	    if (d > 0 && rhalo) {
		squeue[d].enqueueReadBuffer(
			x(d), CL_FALSE, 0, rhalo * sizeof(T),
			&hbuf[d * (rhalo + lhalo)], 0, &event[d]);
	    }

	    if (d + 1 < queue.size() && lhalo) {
		squeue[d].enqueueReadBuffer(
			x(d), CL_FALSE, (x.part_size(d) - lhalo) * sizeof(T),
			lhalo * sizeof(T), &hbuf[d * (rhalo + lhalo) + rhalo],
			0, &event[d]);
	    }
	}
    }

    for(uint d = 0; d < queue.size(); d++) {
	cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	size_t g_size = alignup(x.part_size(d), wgsize[context()]);

	auto lmem = cl::__local(
		(wgsize[context()] + 2 * lhalo + 2 * rhalo + 1) * sizeof(T)
		);

	uint pos = 0;
	conv_local[context()].setArg(pos++, x.part_size(d));
	conv_local[context()].setArg(pos++, lhalo);
	conv_local[context()].setArg(pos++, rhalo);
	conv_local[context()].setArg(pos++, x(d));
	conv_local[context()].setArg(pos++, s[d]);
	conv_local[context()].setArg(pos++, y(d));
	conv_local[context()].setArg(pos++, lmem);

	queue[d].enqueueNDRangeKernel(conv_local[context()],
		cl::NullRange, g_size, wgsize[context()]);

    }

    if (lhalo + rhalo) {
	if (queue.size() > 1)
	    for(uint d = 0; d < queue.size(); d++)
		if ((d > 0 && rhalo) || (d + 1 < queue.size() && lhalo))
		    event[d].wait();

	for(uint d = 0; d < queue.size(); d++) {
	    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

	    if (d > 0 && lhalo) {
		queue[d].enqueueWriteBuffer(dbuf[d], CL_FALSE, 0, lhalo * sizeof(T),
			&hbuf[(d - 1) * (rhalo + lhalo) + rhalo]);
	    }

	    if (d + 1 < queue.size() && rhalo) {
		queue[d].enqueueWriteBuffer(dbuf[d], CL_FALSE,
			lhalo * sizeof(T), rhalo * sizeof(T),
			&hbuf[(d + 1) * (rhalo + lhalo)]);
	    }

	    size_t g_size = std::max(lhalo, rhalo);

	    uint prm = 0;
	    conv_remote[context()].setArg(prm++, x.part_size(d));
	    conv_remote[context()].setArg(prm++, d > 0);
	    conv_remote[context()].setArg(prm++, d + 1 < queue.size());
	    conv_remote[context()].setArg(prm++, lhalo);
	    conv_remote[context()].setArg(prm++, rhalo);
	    conv_remote[context()].setArg(prm++, x(d));
	    conv_remote[context()].setArg(prm++, dbuf[d]);
	    conv_remote[context()].setArg(prm++, s[d]);
	    conv_remote[context()].setArg(prm++, y(d));

	    queue[d].enqueueNDRangeKernel(conv_remote[context()],
		    cl::NullRange, g_size, cl::NullRange);
	}
    }
}

} // namespace vex

#ifdef WIN32
#  pragma warning(pop)
#endif
#endif
