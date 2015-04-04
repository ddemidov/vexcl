#ifndef VEXCL_BACKEND_OPENCL_KERNEL_HPP
#define VEXCL_BACKEND_OPENCL_KERNEL_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  An abstraction over OpenCL compute kernel.
 */

#include <functional>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
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
               size_t smem_per_thread = 0,
               const std::string &options = ""
               )
            : argpos(0), K(build_sources(queue, src, options), name.c_str())
        {
            config(queue,
                    [smem_per_thread](size_t wgs){ return wgs * smem_per_thread; });
        }

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const cl::CommandQueue &queue,
               const std::string &src, const std::string &name,
               std::function<size_t(size_t)> smem,
               const std::string &options = ""
               )
            : argpos(0), K(build_sources(queue, src, options), name.c_str())
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
            cl::LocalSpaceArg smem = { smem_per_thread * workgroup_size() };
            K.setArg(argpos++, smem);
        }

        /// Adds local memory to the kernel.
        template <class F>
        void set_smem(F &&f) {
            cl::LocalSpaceArg smem = { f(workgroup_size()) };
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
            size_t threads = 1;
            for(size_t i = 0; i < w_size.dimensions(); ++i)
                threads *= static_cast<const size_t*>(w_size)[i];
            return threads;
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

            size_t ws;

            if ( is_cpu(queue) ) {
                ws = 1;
            } else {
                // Select workgroup size that would fit into the device.
                ws = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] / 2;

                size_t max_ws   = max_threads_per_block(queue);
                size_t max_smem = max_shared_memory_per_block(queue);

                // Reduce workgroup size until it satisfies resource requirements:
                while( (ws > max_ws) || (smem(ws) > max_smem) )
                    ws /= 2;
            }

            config(num_workgroups(queue), ws);
        }

        /// Set launch configuration.
        void config(ndrange blocks, ndrange threads) {
            size_t dim = std::max(blocks.dimensions(), threads.dimensions());

            const size_t *b = blocks;
            const size_t *t = threads;

            switch(dim) {
                case 3:
                    g_size = ndrange(b[0] * t[0], b[1] * t[1], b[2] * t[2]);
                    break;
                case 2:
                    g_size = ndrange(b[0] * t[0], b[1] * t[1]);
                    break;
                case 1:
                default:
                    g_size = ndrange(b[0] * t[0]);
                    break;
            }

            w_size = threads;
        }

        /// Set launch configuration.
        void config(size_t blocks, size_t threads) {
            config(ndrange(blocks), ndrange(threads));
        }

        size_t preferred_work_group_size_multiple(const backend::command_queue &q) const {
            return K.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                    q.getInfo<CL_QUEUE_DEVICE>()
                    );
        }
    private:
        unsigned argpos;

        cl::Kernel K;

        backend::ndrange w_size;
        backend::ndrange g_size;
};

/// \endcond

} // namespace opencl
} // namespace backend
} // namespace vex

#endif
