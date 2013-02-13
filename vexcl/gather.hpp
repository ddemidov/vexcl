#ifndef VEXCL_GATHER_HPP
#define VEXCL_GATHER_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   vexcl/gather.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Gather scattered points from OpenCL device vector.
 */

#include <CL/cl.hpp>

namespace vex {

template <typename T>
class gather {
    public:
        gather(
                const std::vector<cl::CommandQueue> &queue,
                size_t src_size, std::vector<size_t> indices
              )
            : queue(queue), ptr(queue.size() + 1, 0),
              idx(queue.size()), val(queue.size()), ev(queue.size())
        {
            assert(std::is_sorted(indices.begin(), indices.end()));

            std::vector<size_t> part = partition(src_size, queue);
            column_owner owner(part);

            for(auto i = indices.begin(); i != indices.end(); ++i) {
                uint d = owner(*i);
                *i -= part[d];
                ++ptr[d + 1];
            }

            std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

            for(uint d = 0; d < queue.size(); d++) {
                cl::Context context = qctx(queue[d]);
                cl::Device  device  = qdev(queue[d]);

                if (!compiled[context()]) {
                    std::ostringstream source;

                    source << standard_kernel_header <<
                        "typedef " << type_name<T>() << " real;\n"
                        "kernel void gather(\n"
                        "    " << type_name<size_t>() << " n,\n"
                        "    global const real *src,\n"
                        "    global const " << type_name<size_t>() << " *col,\n"
                        "    global real *dst\n"
                        "    )\n"
                        "{\n"
                        "    size_t i = get_global_id(0);\n"
                        "    if (i < n) dst[i] = src[col[i]];\n"
                        "}\n";

                    auto program = build_sources(context, source.str());

                    krn[context()] = cl::Kernel(program, "gather");

                    wgsize[context()] = kernel_workgroup_size(
                            krn[context()], device);

                    compiled[context()] = true;
                }

                if (size_t n = ptr[d + 1] - ptr[d]) {
                    idx[d] = cl::Buffer(context, CL_MEM_READ_ONLY,  n * sizeof(size_t));
                    val[d] = cl::Buffer(context, CL_MEM_WRITE_ONLY, n * sizeof(T));

                    queue[d].enqueueWriteBuffer(idx[d], CL_FALSE,
                            0, n * sizeof(size_t), &indices[ptr[d]], 0, &ev[d]);
                }
            }

            for(uint d = 0; d < queue.size(); d++)
                if (ptr[d + 1] - ptr[d]) ev[d].wait();
        }

        template <class HostVector>
        void operator()(const vex::vector<T> &src, HostVector &dst) {
            for(uint d = 0; d < queue.size(); d++) {
                if (size_t n = ptr[d + 1] - ptr[d]) {
                    cl::Context context = qctx(queue[d]);

                    size_t g_size = alignup(n, wgsize[context()]);

                    uint pos = 0;
                    krn[context()].setArg(pos++, n);
                    krn[context()].setArg(pos++, src(d));
                    krn[context()].setArg(pos++, idx[d]);
                    krn[context()].setArg(pos++, val[d]);

                    queue[d].enqueueNDRangeKernel(krn[context()],
                            cl::NullRange, g_size, wgsize[context()]);

                    queue[d].enqueueReadBuffer(
                            val[d], CL_FALSE, 0, n * sizeof(T), &dst[ptr[d]],
                            0, &ev[d]);
                }
            }

            for(uint d = 0; d < queue.size(); d++)
                if (ptr[d + 1] - ptr[d]) ev[d].wait();
        }
    private:
        std::vector<cl::CommandQueue> queue;
        std::vector<size_t>     ptr;
        std::vector<cl::Buffer> idx;
        std::vector<cl::Buffer> val;
        std::vector<cl::Event>  ev;

        static std::map<cl_context, bool>       compiled;
        static std::map<cl_context, cl::Kernel> krn;
        static std::map<cl_context, uint>       wgsize;
};

template<typename T> std::map<cl_context, bool>       gather<T>::compiled;
template<typename T> std::map<cl_context, cl::Kernel> gather<T>::krn;
template<typename T> std::map<cl_context, uint>       gather<T>::wgsize;
} // namespace vex

#endif
