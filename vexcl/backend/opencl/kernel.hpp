#ifndef VEXCL_BACKEND_OPENCL_KERNEL_HPP
#define VEXCL_BACKEND_OPENCL_KERNEL_HPP

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
 * \file   vexcl/backend/opencl/kernel.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  An abstraction over OpenCL compute kernel.
 */

#include <functional>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

#include <vexcl/backend/opencl/compiler.hpp>

namespace vex {
namespace backend {

typedef cl::LocalSpaceArg local_mem_arg;

/// Helper function for generating LocalSpaceArg objects.
/**
 * This is a copy of cl::Local that is absent in some of cl.hpp versions.
 */
inline local_mem_arg local_mem(size_t size) {
    cl::LocalSpaceArg ret = { size };
    return ret;
}

struct fixed_workgroup_size_impl {
    size_t size;
};

inline fixed_workgroup_size_impl fixed_workgroup_size(size_t n) {
    fixed_workgroup_size_impl s = {n};
    return s;
}

/// An abstraction over OpenCL compute kernel.
class kernel {
    public:
        kernel() : argpos(0), w_size(0), g_size(0) {}

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const cl::CommandQueue &queue,
               const std::string &src,
               const std::string &name,
               size_t smem_per_thread = 0
               )
            : argpos(0), K(build_sources(queue, src), name.c_str())
        {
            get_launch_cfg(queue,
                    [smem_per_thread](size_t wgs){ return wgs * smem_per_thread; });
        }

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const cl::CommandQueue &queue,
               const std::string &src, const std::string &name,
               std::function<size_t(size_t)> smem
               )
            : argpos(0), K(build_sources(queue, src), name.c_str())
        {
            get_launch_cfg(queue, smem);
        }

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const cl::CommandQueue &queue,
               const std::string &src, const std::string &name,
               fixed_workgroup_size_impl wgs
               )
            : argpos(0),
              K(build_sources(queue, src), name.c_str()),
              w_size(wgs.size),
              g_size(w_size * num_workgroups(queue))
        { }

        /// Adds an argument to the kernel.
        template <class Arg>
        void push_arg(Arg &&arg) {
            K.setArg(argpos++, arg);
        }

        /// Adds an argument to the kernel.
        template <typename T>
        void push_arg(device_vector<T> &&arg) {
            K.setArg(argpos++, arg.raw());
        }

        /// Adds local memory to the kernel.
        void set_smem(size_t smem_per_thread) {
            cl::LocalSpaceArg smem = { smem_per_thread * w_size };
            K.setArg(argpos++, smem);
        }

        /// Adds local memory to the kernel.
        template <class F>
        void set_smem(F &&f) {
            cl::LocalSpaceArg smem = { f(w_size) };
            K.setArg(argpos++, smem);
        }

        void operator()(const cl::CommandQueue &q) {
            q.enqueueNDRangeKernel(K, cl::NullRange, g_size, w_size);
            argpos = 0;
        }

        /// Workgroup size.
        size_t workgroup_size() const {
            return w_size;
        }

        /// Standard number of workgroups to launch on a device.
        static inline size_t num_workgroups(const cl::CommandQueue &q) {
            // This is a simple heuristic-based estimate. More advanced technique may
            // be employed later.
            cl::Device d = q.getInfo<CL_QUEUE_DEVICE>();
            return 4 * d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        }

        /*
        operator const cl::Kernel&() const {
            return K;
        }
        */

    private:
        unsigned argpos;

        cl::Kernel K;

        size_t   w_size;
        size_t   g_size;

        void get_launch_cfg(const cl::CommandQueue &queue, std::function<size_t(size_t)> smem) {
            cl::Device dev = queue.getInfo<CL_QUEUE_DEVICE>();

            if ( is_cpu(queue) ) {
                w_size = 1;
            } else {
                // Select workgroup size that would fit into the device.
                w_size = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];

                size_t max_ws   = K.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev);
                size_t max_smem = static_cast<size_t>(dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>())
                                - static_cast<size_t>(K.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(dev));

                // Reduce workgroup size until it satisfies resource requirements:
                while( (w_size > max_ws) || (smem(w_size) > max_smem) )
                    w_size /= 2;
            }

            g_size = w_size * num_workgroups(queue);
        }
};

} // namespace backend
} // namespace vex

#endif
