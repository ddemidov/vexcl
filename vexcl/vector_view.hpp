#ifndef VEXCL_VECTOR_VIEW_HPP
#define VEXCL_VECTOR_VIEW_HPP

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
 * \file   vexcl/vector_view.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Provides sub-view for an existing vex::vector.
 */

#include <array>
#include <algorithm>
#include <vexcl/vector.hpp>

namespace vex {

/// Generalized slice selector.
/**
 * This is very similar to std::gslice. The only difference is the ordering of
 * components inside size and stride arrays. std::gslice specify slower
 * dimensions first, and vex::gslice -- faster dimensions first.
 *
 * Index to base vector is obtained as start + sum(i_k * stride[k]), where i_k
 * is coordinate along each dimension of gslice.
 */
template <size_t NDIM>
struct gslice {
    size_t start;
    std::array<size_t,    NDIM> size;
    std::array<ptrdiff_t, NDIM> stride; // Signed type allows reverse slicing.

#ifndef BOOST_NO_INITIALIZER_LISTS
    gslice(size_t start,
           const std::initializer_list<size_t>    &p_size,
           const std::initializer_list<ptrdiff_t> &p_stride
          ) : start(start)
    {
        assert(p_size.size()   == NDIM);
        assert(p_stride.size() == NDIM);

        std::copy(p_size.begin(),   p_size.end(),   size.begin());
        std::copy(p_stride.begin(), p_stride.end(), stride.begin());
    }
#endif

    gslice(size_t start,
           const std::array<size_t,    NDIM> &size,
           const std::array<ptrdiff_t, NDIM> &stride
          ) : start(start), size(size), stride(stride)
    { }

    gslice(size_t start,
           const size_t    *p_size,
           const ptrdiff_t *p_stride
          ) : start(start)
    {
        std::copy(p_size,   p_size   + NDIM, size.begin());
        std::copy(p_stride, p_stride + NDIM, stride.begin());
    }
};

struct vector_view_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< vector_view_terminal >::type
    > vector_view_terminal_expression;

template <typename T, class Slice>
struct vector_view : public vector_view_terminal_expression
{
    const vector<T> &base;
    const Slice     &slice;

    vector_view(const vector<T> &base, const Slice &slice)
        : base(base), slice(slice)
    { }
};

// Allow vector_view to participate in vector expressions:
template <>
struct is_vector_expr_terminal< vector_view_terminal >
    : std::true_type
{ };

template <typename T, class Slice>
struct kernel_name< vector_view<T, Slice> > {
    static std::string get() {
        return "view_";
    }
};

// TODO
template <typename T, class Slice>
struct partial_vector_expr< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        std::ostringstream s;
        s << "prm_" << component << "_" << position
          << "[slice_" << component << "_" << position << "(idx)]";
        return s.str();
    }
};

// TODO
template <typename T, class Slice>
struct terminal_preamble< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        std::ostringstream s;
        s <<
            "ulong slice_" << component << "_" << position << "(ulong idx) {\n"
            "  return idx;\n"
            "}\n";
        return s.str();
    }
};

// TODO
template <typename T, class Slice>
struct kernel_param_declaration< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        std::ostringstream s;
        s << "global " << type_name<T>() << " * prm_" << component << "_" << position;
        return s.str();
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(cl::Kernel &kernel, uint device, size_t/*index_offset*/, uint &position, const vector_view<T, Slice> &term) {
        assert(device == 0);
        kernel.setArg(position++, term.base(device));
    }
};


} // namespace vex

#endif
