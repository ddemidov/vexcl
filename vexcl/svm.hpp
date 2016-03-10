#ifndef VEXCL_SVM_HPP
#define VEXCL_SVM_HPP

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
 * \file   vexcl/svm.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Shared virtual memory support.
 */

#include <vexcl/operations.hpp>

#if defined(VEXCL_BACKEND_OPENCL)
#  include <vexcl/backend/opencl/svm.hpp>
#elif defined(VEXCL_BACKEND_COMPUTE)
#  include <vexcl/backend/compute/svm.hpp>
#elif defined(VEXCL_BACKEND_CUDA)
#  include <vexcl/backend/cuda/svm.hpp>
#endif

namespace vex {
namespace traits {

template <typename T>
struct is_vector_expr_terminal< svm<T> > : std::true_type {};

template <typename T>
struct kernel_param_declaration< svm<T> >
{
    static void get(backend::source_generator &src,
            const svm<T>&,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        src.parameter< global_ptr<T> >(prm_name);
    }
};

template <typename T>
struct partial_vector_expr< svm<T> > {
    static void get(backend::source_generator &src,
            const svm<T>&,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        src << prm_name << "[idx]";
    }
};

template <typename T>
struct kernel_arg_setter< svm<T> >
{
    static void set(const svm<T> &term,
            backend::kernel &kernel, unsigned/*part*/, size_t/*index_offset*/,
            detail::kernel_generator_state_ptr)
    {
        kernel.push_arg(term.get());
    }
};

template <typename T>
struct expression_properties< svm<T> >
{
    static void get(const svm<T> &term,
            std::vector<backend::command_queue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list.clear();
        queue_list.push_back(term.queue());

        size = term.size();

        partition.clear();
        partition.push_back(0);
        partition.push_back(size);
    }
};

} // namespace traits
} // namespace vex



#endif
