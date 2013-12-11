#ifndef VEXCL_BACKEND_CUDA_KERNEL_HPP
#define VEXCL_BACKEND_CUDA_KERNEL_HPP

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
 * \file   vexcl/backend/cuda/kernel.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  An abstraction over CUDA compute kernel.
 */

#include <functional>

#include <cuda.h>

#include <vexcl/backend/cuda/compiler.hpp>

namespace vex {
namespace backend {
namespace cuda {

/// \cond INTERNAL

/// An abstraction over CUDA compute kernel.
class kernel {
    public:
        kernel() : w_size(0), g_size(0), smem(0) {}

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const command_queue &queue,
               const std::string &src,
               const std::string &name,
               size_t smem_per_thread = 0
               )
            : ctx(queue.context()),
              module(build_sources(queue, src), detail::deleter()),
              smem(0)
        {
            cuda_check( cuModuleGetFunction(&K, module.get(), name.c_str()) );

            config(queue,
                    [smem_per_thread](size_t wgs){ return wgs * smem_per_thread; });
        }

        /// Constructor. Creates a cl::Kernel instance from source.
        kernel(const command_queue &queue,
               const std::string &src, const std::string &name,
               std::function<size_t(size_t)> smem
               )
            : ctx(queue.context()),
              module(build_sources(queue, src), detail::deleter()),
              smem(0)
        {
            cuda_check( cuModuleGetFunction(&K, module.get(), name.c_str()) );
            config(queue, smem);
        }

        /// Adds an argument to the kernel.
        template <class Arg>
        void push_arg(const Arg &arg) {
            char *c = (char*)&arg;
            prm_pos.push_back(stack.size());
            stack.insert(stack.end(), c, c + sizeof(arg));
        }

        /// Adds an argument to the kernel.
        template <typename T>
        void push_arg(const device_vector<T> &arg) {
            push_arg(arg.raw());
        }

        /// Adds local memory to the kernel.
        template <class F>
        void set_smem(F &&f) {
            smem = f(w_size);
        }

        /// Enqueue the kernel to the specified command queue.
        void operator()(const command_queue &q) {
            prm_addr.clear();
            for(auto p = prm_pos.begin(); p != prm_pos.end(); ++p)
                prm_addr.push_back(stack.data() + *p);

            cuda_check(
                    cuLaunchKernel(
                        K,
                        static_cast<unsigned>(g_size), 1, 1,
                        static_cast<unsigned>(w_size), 1, 1,
                        static_cast<unsigned>(smem),
                        q.raw(),
                        prm_addr.data(),
                        0
                        )
                    );

            stack.clear();
            prm_pos.clear();
        }

#ifndef BOOST_NO_VARIADIC_TEMPLATES
        /// Enqueue the kernel to the specified command queue with the given arguments
        template <class Arg1, class... OtherArgs>
        void operator()(const command_queue &q, Arg1 &&arg1, OtherArgs&&... other_args) {
            push_arg(std::forward<Arg1>(arg1));

            (*this)(q, std::forward<OtherArgs>(other_args)...);
        }
#endif

        /// Workgroup size.
        size_t workgroup_size() const {
            return w_size;
        }

        /// Standard number of workgroups to launch on a device.
        static inline size_t num_workgroups(const command_queue &q) {
            return 8 * q.device().multiprocessor_count();
        }

        /// The maximum number of threads per block, beyond which a launch of the kernel would fail.
        size_t max_threads_per_block(const command_queue&) const {
            int n;
            cuda_check( cuFuncGetAttribute(&n, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, K) );
            return n;
        }

        /// The size in bytes of shared memory per block available for this kernel.
        size_t max_shared_memory_per_block(const command_queue &q) const {
            return q.device().max_shared_memory_per_block() -
                shared_size_bytes();
        }

        /// Select best launch configuration for the given shared memory requirements.
        void config(const command_queue &q, std::function<size_t(size_t)> smem) {
            // Select workgroup size that would fit into the device.
            w_size = q.device().max_threads_per_block() / 2;

            size_t max_ws   = max_threads_per_block(q);
            size_t max_smem = max_shared_memory_per_block(q);

            // Reduce workgroup size until it satisfies resource requirements:
            while( (w_size > max_ws) || (smem(w_size) > max_smem) )
                w_size /= 2;

            g_size = num_workgroups(q);
        }

        /// Set launch configuration.
        void config(size_t blocks, size_t threads) {
            g_size = blocks;
            w_size = threads;
        }
    private:
        context ctx;
        std::shared_ptr< std::remove_pointer<CUmodule>::type > module;
        CUfunction K;

        size_t   w_size;
        size_t   g_size;
        size_t   smem;

        std::vector<char>   stack;
        std::vector<size_t> prm_pos;
        std::vector<void*>  prm_addr;

        size_t shared_size_bytes() const {
            int n;
            cuda_check( cuFuncGetAttribute(&n, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, K) );
            return n;
        }

};

/// \endcond

} // namespace cuda
} // namespace backend
} // namespace vex

#endif
