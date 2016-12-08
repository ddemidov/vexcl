#ifndef VEXCL_BACKEND_MAXELER_KERNEL_HPP
#define VEXCL_BACKEND_MAXELER_KERNEL_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/backend/maxeler/kernel.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Compute kernel implementation for the Maxeler backend.
 */

#include <string>
#include <boost/dll/import.hpp>

#include <vexcl/util.hpp>
#include <vexcl/backend/maxeler/compiler.hpp>

namespace vex {
namespace backend {
namespace maxeler {

namespace detail {

struct kernel_api {
    virtual void execute(char *prm) const = 0;
};

} // namespace detail

class kernel {
    public:
        kernel() {}

        kernel(
                const command_queue &q,
                const std::tuple<std::string, std::string> &src,
                const std::string &name,
                size_t smem_per_thread = 0,
                const std::string &options = ""
              )
            : K(boost::dll::import<detail::kernel_api>(build_sources(q, src, options), name))
        {
            stack.reserve(256);
        }

        kernel(
                const command_queue &q,
                const std::tuple<std::string, std::string> &src,
                const std::string &name,
                std::function<size_t(size_t)> smem,
                const std::string &options = ""
              )
            : K(boost::dll::import<detail::kernel_api>(build_sources(q, src, options), name))
        {
            stack.reserve(256);
        }

        kernel(
                const command_queue &q,
                const program &P,
                const std::string &name,
                size_t smem_per_thread = 0
              )
            : K(boost::dll::import<detail::kernel_api>(P, name))
        {
            stack.reserve(256);
        }

        /// Constructor. Extracts a backend::kernel instance from backend::program.
        kernel(
                const command_queue &q, const program &P,
                const std::string &name,
                std::function<size_t(size_t)> smem
              )
            : K(boost::dll::import<detail::kernel_api>(P, name))
        {
            stack.reserve(256);
        }

        template <class Arg>
        void push_arg(const Arg &arg) {
            char *c = (char*)&arg;
            stack.insert(stack.end(), c, c + sizeof(arg));
        }

        template <typename T>
        void push_arg(const device_vector<T> &arg) {
            push_arg(arg.raw());
        }

        void set_smem(size_t smem_per_thread) { }

        template <class F>
        void set_smem(F &&f) { }

        void operator()(const command_queue&) {
            // All parameters have been pushed; time to call the kernel:
            K->execute(stack.data());

            // Reset parameter stack:
            stack.clear();
        }

#ifndef BOOST_NO_VARIADIC_TEMPLATES
        template <class Head, class... Tail>
        void operator()(const command_queue &q, const Head &head, const Tail&... tail) {
            push_arg(head);
            (*this)(q, tail...);
        }
#endif
        size_t workgroup_size() const {
            return 1UL;
        }

        static inline size_t num_workgroups(const command_queue&) {
            return 1UL;
        }

        size_t max_threads_per_block(const command_queue&) const {
            return 1UL;
        }

        size_t max_shared_memory_per_block(const command_queue&) const {
            return 0UL;
        }

        size_t preferred_work_group_size_multiple(const backend::command_queue &q) const {
            return 1;
        }

        void config(const command_queue &q, std::function<size_t(size_t)> smem) { }

        void config(ndrange blocks, ndrange threads) { }

        void config(size_t blocks, size_t threads) { }

        void reset() {
            stack.clear();
        }
    private:
        boost::shared_ptr<detail::kernel_api> K;
        std::vector<char> stack;
};

} // namespace maxeler
} // namespace backend
} // namespace vex

#endif
