#ifndef VEXCL_ELEMENT_INDEX_HPP
#define VEXCL_ELEMENT_INDEX_HPP

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
 * \file   vexcl/element_index.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Element index for use in vector expressions.
 */

#include <vexcl/operations.hpp>

namespace vex {

/// \cond INTERNAL
struct elem_index {
    typedef size_t value_type;

    size_t offset, length;

    elem_index(size_t offset = 0, size_t length = 0)
        : offset(offset), length(length) {}
};
/// \endcond

/// When used in vector expression, returns current element index plus offset.
/**
 * \param offset Element indices will start from this value.
 * \param length Specify length of vector expression. This is only relevant
 * when parent expression does not contain any vectors. See monte_carlo_pi test
 * in tests/random.cpp for an example.
 */
#ifdef DOXYGEN
elem_index
#else
inline boost::proto::result_of::as_expr<elem_index, vector_domain>::type
#endif
element_index(size_t offset = 0, size_t length = 0) {
    return boost::proto::as_expr<vector_domain>(elem_index(offset, length));
}

namespace traits {

template <>
struct is_vector_expr_terminal< elem_index > : std::true_type {};

template <>
struct is_multivector_expr_terminal< elem_index > : std::true_type {};

template <>
struct kernel_param_declaration< elem_index >
{
    static std::string get(const elem_index&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        std::ostringstream s;
        s << ",\n\t" << type_name<size_t>() << " " << prm_name;
        return s.str();
    }
};

template <>
struct partial_vector_expr< elem_index >
{
    static std::string get(const elem_index&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        std::ostringstream s;
        s << "(" << prm_name << " + idx)";
        return s.str();
    }
};

template <>
struct kernel_arg_setter< elem_index >
{
    static void set(const elem_index &term,
            cl::Kernel &kernel, unsigned/*device*/, size_t index_offset,
            unsigned &position, detail::kernel_generator_state_ptr)
    {
        kernel.setArg(position++, term.offset + index_offset);
    }
};

template <>
struct expression_properties< elem_index >
{
    static void get(const elem_index &term,
            std::vector<cl::CommandQueue> &/*queue_list*/,
            std::vector<size_t> &/*partition*/,
            size_t &size
            )
    {
        size = term.length;
    }
};

} // namespace traits

} // namespace vex;

#endif
