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

#include <vector>
#include <array>
#include <string>
#include <sstream>
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
    {
        precondition(
                base.nparts() == 1,
                "Base vector should reside on a single compute device"
                );
    }

    // Expression assignments (copy assignment needs to be explicitly defined
    // to allow vector_view to vector_view assignment).
#define ASSIGNMENT(cop, op) \
    template <class Expr> \
    typename std::enable_if< \
        boost::proto::matches< \
            typename boost::proto::result_of::as_expr<Expr>::type, \
            vector_expr_grammar \
        >::value, \
        const vector_view& \
    >::type \
    operator cop(const Expr &expr) { \
        std::vector<size_t> part(2, 0); \
        part.back() = slice.size(); \
        detail::assign_expression<op>(*this, expr, base.queue_list(), part); \
        return *this; \
    } \
    const vector_view& operator cop(const vector_view &other) { \
        std::vector<size_t> part(2, 0); \
        part.back() = slice.size(); \
        detail::assign_expression<op>(*this, other, base.queue_list(), part); \
        return *this; \
    }

    ASSIGNMENT(=,   assign::SET);
    ASSIGNMENT(+=,  assign::ADD);
    ASSIGNMENT(-=,  assign::SUB);
    ASSIGNMENT(*=,  assign::MUL);
    ASSIGNMENT(/=,  assign::DIV);
    ASSIGNMENT(%=,  assign::MOD);
    ASSIGNMENT(&=,  assign::AND);
    ASSIGNMENT(|=,  assign::OR);
    ASSIGNMENT(^=,  assign::XOR);
    ASSIGNMENT(<<=, assign::LSH);
    ASSIGNMENT(>>=, assign::RSH);

#undef ASSIGNMENT
};

/// \endcond

// Allow vector_view to participate in vector expressions:
namespace traits {

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
    static std::string get(const cl::Device&, int component, int position,
            detail::kernel_generator_state&)
    {
        return Slice::partial_expression(component, position);
    }
};

template <typename T, class Slice>
struct terminal_preamble< vector_view<T, Slice> > {
    static std::string get(const cl::Device&, int component, int position,
            detail::kernel_generator_state&)
    {
        return Slice::indexing_function(component, position);
    }
};

template <typename T, class Slice>
struct kernel_param_declaration< vector_view<T, Slice> > {
    static std::string get(const cl::Device&, int component, int position,
            detail::kernel_generator_state&)
    {
        return Slice::template parameter_declaration<T>(component, position);
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(cl::Kernel &kernel, unsigned device, size_t index_offset,
            unsigned &position, const vector_view<T, Slice> &term,
            detail::kernel_generator_state&)
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

} // namespace traits

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

    size_t    start;
    size_t    length[NDIM];
    ptrdiff_t stride[NDIM]; // Signed type allows reverse slicing.

#ifndef BOOST_NO_INITIALIZER_LISTS
    template <typename T1, typename T2>
    gslice(size_t start,
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
    gslice(size_t start,
           const std::array<T1, NDIM> &p_length,
           const std::array<T2, NDIM> &p_stride
          ) : start(start)
    {
        std::copy(p_length.begin(), p_length.end(), length);
        std::copy(p_stride.begin(), p_stride.end(), stride);
    }

    template <typename T1, typename T2>
    gslice(size_t start,
           const T1 *p_length,
           const T2 *p_stride
          ) : start(start)
    {
        std::copy(p_length, p_length + NDIM, length);
        std::copy(p_stride, p_stride + NDIM, stride);
    }

    size_t size() const {
        return std::accumulate(length, length + NDIM,
	    static_cast<size_t>(1), std::multiplies<size_t>());
    }

    static std::string indexing_function(int component, int position) {
        std::ostringstream s;

        s << type_name<size_t>() << " slice_" << component << "_" << position
          << "(\n\t" << type_name<size_t>() << " start";
        for(size_t k = 0; k < NDIM; ++k)
            s << ",\n\t" << type_name<size_t>() << " length" << k
              << ",\n\t" << type_name<ptrdiff_t>() << " stride" << k;
        s << ",\n\t" << type_name<size_t>() << " idx)\n{\n";

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
          << ", " << type_name<size_t>() << " " << prm.str() << "start";

        for(size_t k = 0; k < NDIM; ++k)
            s << ", " << type_name<size_t>()    << " " << prm.str() << "length" << k
              << ", " << type_name<ptrdiff_t>() << " " << prm.str() << "stride" << k;

        return s.str();
    }

    template <typename T>
    static void setArgs(cl::Kernel &kernel, unsigned device, size_t/*index_offset*/,
	    unsigned &position, const vector_view<T, gslice> &term)
    {
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

    /// Range with a single element.
    range(size_t i)
        : start(i), stride(1), stop(i + 1) {}

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

template <size_t NDIM>
struct extent_gen {
    std::array<size_t, NDIM> dim;

    extent_gen() {}

    extent_gen<NDIM + 1> operator[](size_t new_dim) const {
        extent_gen<NDIM + 1> new_extent;
        std::copy(dim.begin(), dim.end(), new_extent.dim.begin());
        new_extent.dim.back() = new_dim;
        return new_extent;
    }
};

const extent_gen<0> extents;

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
template <size_t NR>
class slicer {
    private:
        std::array<size_t, NR> dim;
        std::array<size_t, NR> stride;

    public:
        template <typename T>
        slicer(const std::array<T, NR> &target_dimensions) {
            init(target_dimensions.data());
        }

        template <typename T>
        slicer(const T *target_dimensions) {
            init(target_dimensions);
        }

        slicer(const extent_gen<NR> &ext) {
            init(ext.dim.data());
        }

        template <size_t C>
        struct slice : public gslice<NR> {
            const slicer &parent;

            slice(const slicer &parent, const range &r)
                : gslice<NR>(r.start * parent.stride[0], parent.dim, parent.stride),
                  parent(parent)
            {
                static_assert(C == 0, "Wrong slice constructor!");

                this->length[0] = (r.stop - r.start + r.stride - 1) / r.stride;
                this->stride[0] *= r.stride;
            }

            slice(const slice<C-1> &parent, const range &r)
                : gslice<NR>(parent),
                  parent(parent.parent)
            {
                static_assert(C > 0, "Wrong slice constructor!");

                this->start += r.start * this->stride[C];
                this->length[C] = (r.stop - r.start + r.stride - 1) / r.stride;
                this->stride[C] *= r.stride;
            }

            slice<C+1> operator[](const range &r) const {
                return slice<C+1>(*this, r.empty() ? range(0, parent.dim[C + 1]) : r);
            }
        };

        slice<0> operator[](const range &r) const {
            return slice<0>(*this, r.empty() ? range(0, dim[0]) : r);
        }

    private:
        template <typename T>
        void init(const T *target_dim) {
            std::copy(target_dim, target_dim + NR, dim.begin());

            stride.back() = 1;
            for(size_t j = 1, i = NR - 2; j < NR; ++j, --i)
                stride[i] = stride[i + 1] * dim[j - 1];
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
          << ", global " << type_name<size_t>() << " * " << prm.str() << "index";

        return s.str();
    }

    template <typename T>
    static void setArgs(cl::Kernel &kernel, unsigned device, size_t/*index_offset*/,
	    unsigned &position, const vector_view<T, permutation> &term)
    {
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
