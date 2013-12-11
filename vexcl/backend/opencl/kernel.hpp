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
namespace opencl {

/// \cond INTERNAL

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
            config(queue,
                    [smem_per_thread](size_t wgs){ return wgs * smem_per_thread; });
        }

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const cl::CommandQueue &queue,
               const std::string &src, const std::string &name,
               std::function<size_t(size_t)> smem
               )
            : argpos(0), K(build_sources(queue, src), name.c_str())
        {
            config(queue, smem);
        }

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

        /// Enqueue the kernel to the specified command queue.
        void operator()(const cl::CommandQueue &q) {
            q.enqueueNDRangeKernel(K, cl::NullRange, g_size, w_size);
            argpos = 0;
        }

#ifndef BOOST_NO_VARIADIC_TEMPLATES
        /// Enqueue the kernel to the specified command queue with the given arguments
        template <class Arg1, class... OtherArgs>
        void operator()(const cl::CommandQueue &q, Arg1 &&arg1, OtherArgs&&... other_args) {
            push_arg(std::forward<Arg1>(arg1));

            (*this)(q, std::forward<OtherArgs>(other_args)...);
        }
#endif

        /// Workgroup size.
        size_t workgroup_size() const {
            return w_size;
        }

        /// Standard number of workgroups to launch on a device.
        static inline size_t num_workgroups(const cl::CommandQueue &q) {
            // This is a simple heuristic-based estimate. More advanced technique may
            // be employed later.
            cl::Device d = q.getInfo<CL_QUEUE_DEVICE>();
            return 8 * d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        }

        /// The maximum number of threads per block, beyond which a launch of the kernel would fail.
        size_t max_threads_per_block(const cl::CommandQueue &q) const {
            cl::Device d = q.getInfo<CL_QUEUE_DEVICE>();
            return K.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(d);
        }

        /// The size in bytes of shared memory per block available for this kernel.
        size_t max_shared_memory_per_block(const cl::CommandQueue &q) const {
            cl::Device d = q.getInfo<CL_QUEUE_DEVICE>();

            return static_cast<size_t>(d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>())
                 - static_cast<size_t>(K.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(d));
        }

        /// Select best launch configuration for the given shared memory requirements.
        void config(const cl::CommandQueue &queue, std::function<size_t(size_t)> smem) {
            cl::Device dev = queue.getInfo<CL_QUEUE_DEVICE>();

            if ( is_cpu(queue) ) {
                w_size = 1;
            } else {
                // Select workgroup size that would fit into the device.
                w_size = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] / 2;

                size_t max_ws   = max_threads_per_block(queue);
                size_t max_smem = max_shared_memory_per_block(queue);

                // Reduce workgroup size until it satisfies resource requirements:
                while( (w_size > max_ws) || (smem(w_size) > max_smem) )
                    w_size /= 2;
            }

            g_size = w_size * num_workgroups(queue);
        }

        /// Set launch configuration.
        void config(size_t blocks, size_t threads) {
            g_size = blocks * threads;
            w_size = threads;
        }
    private:
        unsigned argpos;

        cl::Kernel K;

        size_t   w_size;
        size_t   g_size;
};

/// \endcond

} // namespace opencl
} // namespace backend
} // namespace vex

#endif
