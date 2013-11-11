#ifndef VEXCL_VECTOR_POINTER_HPP
#define VEXCL_VECTOR_POINTER_HPP

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
 * \file   vexcl/vector_pointer.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Cast vex::vector to a raw pointer.
 */

#include <vexcl/vector.hpp>
#include <vexcl/operations.hpp>

namespace vex {

template <typename T>
struct vector_pointer {
    typedef T value_type;

    const vector<T> &v;

    vector_pointer(const vector<T> &v) : v(v) {}
};

/// Cast vex::vector to a raw pointer.
/**
 * Useful when user wants to get a pointer to a vector instead of its current
 * element inside a vector expression. Could be combined with calls to
 * address_of/dereference operators or with user-defined functions iterating
 * through the vector. See examples in tests/vector_pointer.cpp.
 */
template <typename T>
#ifdef DOXYGEN
vector_pointer<T>
#else
inline typename boost::proto::result_of::as_expr<vector_pointer<T>, vector_domain>::type
#endif
raw_pointer(const vector<T> &v) {
    precondition(
            v.nparts() == 1,
            "raw_pointer is not supported for multi-device contexts"
            );

    return boost::proto::as_expr<vector_domain>(vector_pointer<T>(v));
}

namespace traits {

template <typename T>
struct is_vector_expr_terminal< vector_pointer<T> > : std::true_type {};

template <typename T>
struct kernel_param_declaration< vector_pointer<T> >
{
    static void get(backend::source_generator &src,
            const vector_pointer<T>&,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        src.parameter< global_ptr<T> >(prm_name);
    }
};

template <typename T>
struct kernel_arg_setter< vector_pointer<T> >
{
    static void set(const vector_pointer<T> &term,
            backend::kernel &kernel, unsigned/*part*/, size_t/*index_offset*/,
            detail::kernel_generator_state_ptr)
    {
        kernel.push_arg(term.v(0));
    }
};

template <typename T>
struct expression_properties< vector_pointer<T> >
{
    static void get(const vector_pointer<T> &term,
            std::vector<backend::command_queue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.v.queue_list();
        partition  = term.v.partition();
        size       = term.v.size();
    }
};

} // namespace traits
}

#endif
