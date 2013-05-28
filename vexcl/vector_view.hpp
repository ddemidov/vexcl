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

template <typename T, class Slice>
struct partial_vector_expr< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        return Slice::partial_expression(component, position);
    }
};

template <typename T, class Slice>
struct terminal_preamble< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        return Slice::indexing_function(component, position);
    }
};

template <typename T, class Slice>
struct kernel_param_declaration< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        return Slice::template parameter_declaration<T>(component, position);
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(cl::Kernel &kernel, uint device, size_t/*index_offset*/, uint &position, const vector_view<T, Slice> &term) {
        assert(device == 0);

        kernel.setArg(position++, term.base(device));
        kernel.setArg(position++, term.slice);
    }
};

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
    static_assert(NDIM > 0, "Incorrect dimension for gslice");

    cl_ulong start;
    cl_ulong size[NDIM];
    cl_long  stride[NDIM]; // Signed type allows reverse slicing.

#ifndef BOOST_NO_INITIALIZER_LISTS
    gslice(cl_ulong start,
           const std::initializer_list<cl_ulong> &p_size,
           const std::initializer_list<cl_long>  &p_stride
          ) : start(start)
    {
        assert(p_size.size()   == NDIM);
        assert(p_stride.size() == NDIM);

        std::copy(p_size.begin(),   p_size.end(),   size);
        std::copy(p_stride.begin(), p_stride.end(), stride);
    }
#endif

    gslice(cl_ulong start,
           const std::array<cl_ulong, NDIM> &p_size,
           const std::array<cl_long,  NDIM> &p_stride
          ) : start(start)
    {
        std::copy(p_size.begin(),   p_size.end(),   size);
        std::copy(p_stride.begin(), p_stride.end(), stride);
    }

    gslice(cl_ulong start,
           const cl_ulong *p_size,
           const cl_long  *p_stride
          ) : start(start)
    {
        std::copy(p_size,   p_size   + NDIM, size);
        std::copy(p_stride, p_stride + NDIM, stride);
    }

    static std::string indexing_function(int component, int position) {
        std::ostringstream s;

        s << "typedef struct {\n"
             "    ulong start;\n"
             "    ulong size[" << NDIM << "];\n"
             "    long  stride[" << NDIM << "];\n"
             "} slice_data_" << component << "_" << position << ";\n\n"
             "ulong slice_" << component << "_" << position << "("
             "slice_data_" << component << "_" << position << " s, "
             "ulong idx) {\n";

        if (NDIM == 1) {
            s << "    return s.start + idx * s.stride[0];\n";
        } else {
            s << "    size_t ptr = s.start + (idx % s.size[0]) * s.stride[0];\n";
            for(size_t k = 1; k < NDIM; ++k) {
                s << "    idx /= size[" << k - 1 << "];\n"
                     "    ptr += (idx % s.size[" << k <<  "]) * s.stride[" << k <<  "];\n";
            }
        }
        s << "}\n\n";

        return s.str();
    }

    static std::string partial_expression(int component, int position) {
        std::ostringstream s;

        s << "prm_" << component << "_" << position << "_base["
          << "slice_" << component << "_" << position << "("
          << "prm_" << component << "_" << position << "_slice, idx)]";

        return s.str();
    }

    template <typename T>
    static std::string parameter_declaration(int component, int position) {
        std::ostringstream s;

        s << "global " << type_name<T>() << " * prm_" << component << "_" << position << "_base, "
          << "slice_data_" << component << "_" << position
          << " prm_" << component << "_" << position << "_slice";

        return s.str();
    }

    template <typename T>
    vector_view<T, gslice> operator()(const vector<T> &base) const {
        return vector_view<T, gslice>(base, *this);
    }
};


} // namespace vex

#endif
