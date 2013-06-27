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
    size_t offset;

    elem_index(size_t offset = 0) : offset(offset) {}
};
/// \endcond

/// When used in vector expression, returns current element index plus offset.
inline boost::proto::result_of::as_expr<elem_index, vector_domain>::type
element_index(size_t offset = 0) {
    return boost::proto::as_expr<vector_domain>(elem_index(offset));
}

namespace traits {

template <>
struct is_vector_expr_terminal< elem_index > : std::true_type {};

template <>
struct is_multivector_expr_terminal< elem_index > : std::true_type {};

template <class T>
struct kernel_name< T, typename std::enable_if<
        boost::proto::matches<
            T,
            boost::proto::terminal<elem_index>
        >::value
    >::type>
{
    static std::string get() {
        return "index_";
    }
};

template <class T>
struct partial_vector_expr< T, typename std::enable_if<
        boost::proto::matches<
            T,
            boost::proto::terminal<elem_index>
        >::value
    >::type >
{
    static std::string get(const cl::Device&, int component, int position,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;
        s << "(prm_" << component << "_" << position << " + idx)";
        return s.str();
    }
};

template <class T>
struct kernel_param_declaration< T, typename std::enable_if<
        boost::proto::matches<
            T,
            boost::proto::terminal<elem_index>
        >::value
    >::type>
{
    static std::string get(const cl::Device&, int component, int position,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;
        s << ",\n\t" << type_name<size_t>() << " prm_"
          << component << "_" << position;
        return s.str();
    }
};

template <class T>
struct kernel_arg_setter< T, typename std::enable_if<
        boost::proto::matches<
            T,
            boost::proto::terminal<elem_index>
        >::value
    >::type>
{
    static void set(cl::Kernel &kernel, unsigned/*device*/, size_t index_offset,
            unsigned &position, const T &term, detail::kernel_generator_state&)
    {
        kernel.setArg(position++, boost::proto::value(term).offset + index_offset);
    }
};

} // namespace traits

} // namespace vex;

#endif
