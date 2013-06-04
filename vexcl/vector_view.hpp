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
#include <numeric>
#include <algorithm>
#include <vexcl/vector.hpp>

namespace vex {

/// \cond INTERNAL

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
    static std::string get(int component, int position, kernel_generator_state&) {
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
    static std::string get(int component, int position, kernel_generator_state&) {
        return Slice::template parameter_declaration<T>(component, position);
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(cl::Kernel &kernel, uint device, size_t index_offset,
            uint &position, const vector_view<T, Slice> &term, kernel_generator_state&)
    {
        assert(device == 0);

        Slice::setArgs(kernel, device, index_offset, position, term);
    }
};

template <typename T, class Slice>
struct expression_properties< vector_view<T, Slice> > {
    static void get(const vector_view<T, Slice> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.base.queue_list();
        partition  = term.base.partition();
        size       = term.slice.size();

        assert(partition.size() == 2);
        partition.back() = size;
    }
};

/// \endcond

/// Generalized slice selector.
/**
 * This is very similar to std::gslice.
 *
 * Index to base vector is obtained as start + sum(i_k * stride[k]), where i_k
 * is coordinate along each dimension of gslice.
 */
template <size_t NDIM>
struct gslice {
    static_assert(NDIM > 0, "Incorrect dimension for gslice");

    cl_ulong start;
    cl_ulong length[NDIM];
    cl_long  stride[NDIM]; // Signed type allows reverse slicing.

#ifndef BOOST_NO_INITIALIZER_LISTS
    template <typename T1, typename T2>
    gslice(cl_ulong start,
           const std::initializer_list<T1> &p_length,
           const std::initializer_list<T2> &p_stride
          ) : start(start)
    {
        assert(p_length.size() == NDIM);
        assert(p_stride.size() == NDIM);

        std::copy(p_length.begin(), p_length.end(), length);
        std::copy(p_stride.begin(), p_stride.end(), stride);
    }
#endif

    template <typename T1, typename T2>
    gslice(cl_ulong start,
           const std::array<T1, NDIM> &p_length,
           const std::array<T2, NDIM> &p_stride
          ) : start(start)
    {
        std::copy(p_length.begin(), p_length.end(), length);
        std::copy(p_stride.begin(), p_stride.end(), stride);
    }

    template <typename T1, typename T2>
    gslice(cl_ulong start,
           const T1 *p_length,
           const T2 *p_stride
          ) : start(start)
    {
        std::copy(p_length, p_length + NDIM, length);
        std::copy(p_stride, p_stride + NDIM, stride);
    }

    size_t size() const {
        return std::accumulate(length, length + NDIM, 1UL, std::multiplies<size_t>());
    }

    static std::string indexing_function(int component, int position) {
        std::ostringstream s;

        s << "ulong slice_" << component << "_" << position << "(\n\tulong start";
        for(size_t k = 0; k < NDIM; ++k)
            s << ",\n\tulong length" << k << ",\n\tlong stride" << k;
        s << ",\n\tulong idx)\n{\n";

        if (NDIM == 1) {
            s << "    return start + idx * stride0;\n";
        } else {
            s << "    size_t ptr = start + (idx % length" << NDIM - 1 <<  ") * stride" << NDIM - 1 << ";\n";
            for(size_t k = NDIM - 1; k-- > 0;) {
                s << "    idx /= length" << k + 1 << ";\n"
                     "    ptr += (idx % length" << k <<  ") * stride" << k <<  ";\n";
            }
            s << "    return ptr;\n";
        }
        s << "}\n\n";

        return s.str();
    }

    static std::string partial_expression(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;

        s << prm.str() << "base["
          << "slice_" << component << "_" << position << "("
          << prm.str() << "start";
        for(size_t k = 0; k < NDIM; ++k)
            s << ", " << prm.str() << "length" << k
              << ", " << prm.str() << "stride" << k;
        s << ", idx)]";

        return s.str();
    }

    template <typename T>
    static std::string parameter_declaration(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;

        s << ",\n\tglobal " << type_name<T>() << " * " << prm.str() << "base"
          << ", ulong " << prm.str() << "start";

        for(size_t k = 0; k < NDIM; ++k)
            s << ", ulong " << prm.str() << "length" << k
              << ", long  " << prm.str() << "stride" << k;

        return s.str();
    }

    template <typename T>
    static void setArgs(cl::Kernel &kernel, uint device, size_t/*index_offset*/, uint &position, const vector_view<T, gslice> &term) {
        kernel.setArg(position++, term.base(device));
        kernel.setArg(position++, term.slice.start);
        for(size_t k = 0; k < NDIM; ++k) {
            kernel.setArg(position++, term.slice.length[k]);
            kernel.setArg(position++, term.slice.stride[k]);
        }
    }

    /// Returns sliced vector.
    template <typename T>
    vector_view<T, gslice> operator()(const vector<T> &base) const {
        assert(base.queue_list().size() == 1);
        return vector_view<T, gslice>(base, *this);
    }
};


/// An index range for use with slicer class.
struct range {
    size_t start;
    size_t stride;
    size_t stop;

    /// All elements in this dimension.
    range () : start(0), stride(0), stop(0) {}

    /// Elements from open interval with given stride.
    range(size_t start, size_t stride, size_t stop)
        : start(start), stride(stride), stop(stop) {}

    /// Every element from open interval.
    range(size_t start, size_t stop)
        : start(start), stride(1), stop(stop) {}

    bool empty() const {
        return !(start || stride || stop);
    }
};

/// Slicing operator.
/**
 * Slices multi-dimensional array stored in vex::vector in row-major order.
 * Usage:
 * \code
 * using vex::range;
 *
 * vex::vector<double> x(ctx, n * n)
 * vex::vector<double> y(ctx, n)
 * vex::vector<double> z(ctx, n / 2);
 *
 * vex::slicer<2> slice({n, n});
 *
 * y = slice[42](x);                // Put 42-th row of x into y.
 * y = slice[range()][42](x);       // Put 42-th column of x into y.
 * z = slice[range(0, 2, n)][5](x); // Put even elements of 5-th column of x into z.
 * \endcode
 */
template <size_t NDIM>
class slicer {
    private:
        std::array<size_t, NDIM> dim;

    public:
        template <size_t CDIM>
        struct slice : public gslice<NDIM> {
            std::array<size_t, NDIM> dim;

#ifndef BOOST_NO_INITIALIZER_LISTS
            template <typename T1, typename T2>
            slice(size_t start,
                  const std::initializer_list<T1> &length,
                  const std::initializer_list<T2> &stride,
                  const std::array<size_t, NDIM> &dim
                 ) : gslice<NDIM>(start, length, stride), dim(dim) {}
#endif

            template <typename T1, typename T2>
            slice(size_t start,
                  const std::array<T1, NDIM> &length,
                  const std::array<T2, NDIM> &stride,
                  const std::array<size_t, NDIM> &dim
                 ) : gslice<NDIM>(start, length, stride), dim(dim) {}

            template <typename T1, typename T2>
            slice(size_t start,
                  const T1 *length,
                  const T2 *stride,
                  const std::array<size_t, NDIM> &dim
                 ) : gslice<NDIM>(start, length, stride), dim(dim) {}

            slice(const slice<CDIM - 1> &parent, const range &r)
                : gslice<NDIM>(parent.start, parent.length, parent.stride), dim(parent.dim)
            {
                this->start += r.start * this->stride[CDIM];
                this->length[CDIM] = (r.stop - r.start + r.stride - 1) / r.stride;
                this->stride[CDIM] *= r.stride;
            }

            slice<CDIM + 1> operator[](const range &r) const {
                static_assert(CDIM + 1 < NDIM, "Incorrect dimensions in vex::slicer[]");

                if (r.empty())
                    return slice<CDIM + 1>(*this, range(0, dim[CDIM + 1]));
                else
                    return slice<CDIM + 1>(*this, r);
            };

            slice<CDIM + 1> operator[](size_t i) const {
                return this->operator[](range(i, i + 1));
            }
        };

#ifndef BOOST_NO_INITIALIZER_LISTS
        template <typename T>
        slicer(const std::initializer_list<T> &target_dimensions) {
            std::copy(target_dimensions.begin(), target_dimensions.end(), dim.begin());
        }
#endif
        template <typename T>
        slicer(const std::array<T, NDIM> &target_dimensions) {
            std::copy(target_dimensions.begin(), target_dimensions.end(), dim.begin());
        }

        template <typename T>
        slicer(const T *target_dimensions) {
            std::copy(target_dimensions, target_dimensions + NDIM, dim.begin());
        }

        slice<0> operator[](const range &r) const {
            if (r.empty())
                return get_slice(range(0, dim[0]));
            else
                return get_slice(r);
        }

        slice<0> operator[](size_t i) const {
            return this->operator[](range(i, i + 1));
        }
    private:
        slice<0> get_slice(const range &r) const {
            std::array<size_t, NDIM> stride;

            stride[NDIM - 1] = 1;
            for(size_t j = 1, i = NDIM - 2; j < NDIM; ++j, --i)
                stride[i] = stride[i + 1] * dim[j - 1];

            size_t start = r.start * stride[0];
            stride[0] *= r.stride;

            std::array<size_t, NDIM> length = {{(r.stop - r.start + r.stride - 1) / r.stride}};
            std::copy(dim.begin() + 1, dim.end(), length.begin() + 1);

            return slice<0>(start, length, stride, dim);
        }
};


/// Permutation operator.
struct permutation {
    const vector<size_t> &index;

    permutation(const vector<size_t> &index) : index(index) {
        assert(index.queue_list().size() == 1);
    }

    size_t size() const {
        return index.size();
    }

    static std::string partial_expression(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;
        s << prm.str() << "base[" << prm.str() << "index[idx]]";

        return s.str();
    }

    static std::string indexing_function(int/*component*/, int/*position*/) {
        return "";
    }

    template <typename T>
    static std::string parameter_declaration(int component, int position) {
        std::ostringstream prm;
        prm << "prm_" << component << "_" << position << "_";

        std::ostringstream s;

        s << ",\n\tglobal " << type_name<T>() << " * " << prm.str() << "base"
          << ", global ulong * " << prm.str() << "index";

        return s.str();
    }

    template <typename T>
    static void setArgs(cl::Kernel &kernel, uint device, size_t/*index_offset*/, uint &position, const vector_view<T, permutation> &term) {
        kernel.setArg(position++, term.base(device));
        kernel.setArg(position++, term.slice.index(device));
    }

    template <typename T>
    vector_view<T, permutation> operator()(const vector<T> &base) const {
        assert(base.queue_list().size() == 1);
        return vector_view<T, permutation>(base, *this);
    }
};

} // namespace vex

#endif
