#ifndef VEXCL_CONSTANTS_HPP
#define VEXCL_CONSTANTS_HPP

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
 * \file   vexcl/constants.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Constants for use in vector expressions.
 */

#include <string>
#include <type_traits>

#include <vexcl/types.hpp>
#include <vexcl/operations.hpp>

namespace vex {

/// \cond INTERNAL

template <class T, T v>
struct is_cl_native< std::integral_constant<T, v> > : std::true_type {};

namespace traits {

template <class T, T v>
struct kernel_param_declaration< std::integral_constant<T, v> >
{
    static std::string get(const std::integral_constant<T, v>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

template <class T, T v>
struct partial_vector_expr< std::integral_constant<T, v> >
{
    static std::string get(const std::integral_constant<T, v>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return std::to_string(v);
    }
};

template <class T, T v>
struct kernel_arg_setter< std::integral_constant<T, v> >
{
    static void set(const std::integral_constant<T, v>&,
            cl::Kernel&, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &/*position*/, detail::kernel_generator_state_ptr)
    {
    }
};

} // namespace traits

/// \endcond

} // namespace vex

#endif
