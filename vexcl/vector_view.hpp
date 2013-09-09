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

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>

namespace vex {

/// \cond INTERNAL

struct vector_view_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< vector_view_terminal >::type
    > vector_view_terminal_expression;

template <typename T, class Slice>
struct vector_view : public vector_view_terminal_expression
{
    typedef T value_type;

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
        if (part.back() == 0) part.back() = base.size(); \
        detail::assign_expression<op>(*this, expr, base.queue_list(), part); \
        return *this; \
    } \
    const vector_view& operator cop(const vector_view &other) { \
        std::vector<size_t> part(2, 0); \
        part.back() = slice.size(); \
        if (part.back() == 0) part.back() = base.size(); \
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
struct is_vector_expr_terminal< vector_view_terminal > : std::true_type {};

template <>
struct proto_terminal_is_value< vector_view_terminal > : std::true_type {};

template <typename T, class Slice>
struct terminal_preamble< vector_view<T, Slice> > {
    static std::string get(const vector_view<T, Slice> &term,
            const cl::Device &device, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        return term.slice.indexing_function(prm_name, device);
    }
};

template <typename T, class Slice>
struct kernel_param_declaration< vector_view<T, Slice> > {
    static std::string get(const vector_view<T, Slice> &term,
            const cl::Device &device, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        return term.slice.template parameter_declaration<T>(prm_name, device);
    }
};

template <typename T, class Slice>
struct partial_vector_expr< vector_view<T, Slice> > {
    static std::string get(const vector_view<T, Slice> &term,
            const cl::Device &device, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        return term.slice.partial_expression(prm_name, device);
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(const vector_view<T, Slice> &term,
            cl::Kernel &kernel, unsigned device, size_t index_offset,
            unsigned &position, detail::kernel_generator_state&)
    {
        assert(device == 0);

        kernel.setArg(position++, term.base(device));
        term.slice.setArgs(kernel, device, index_offset, position);
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

    std::string indexing_function(const std::string &prm_name,
            const cl::Device&) const
    {
        std::ostringstream s;

        s << type_name<size_t>() << " slice_" << prm_name
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

    std::string partial_expression(const std::string &prm_name,
            const cl::Device&) const
    {
        std::ostringstream s;

        s << prm_name << "_base[" << "slice_" << prm_name << "("
          << prm_name << "_start";
        for(size_t k = 0; k < NDIM; ++k)
            s << ", " << prm_name << "_length" << k
              << ", " << prm_name << "_stride" << k;
        s << ", idx)]";

        return s.str();
    }

    template <typename T>
    std::string parameter_declaration(const std::string &prm_name,
            const cl::Device&) const
    {
        std::ostringstream s;

        s << ",\n\tglobal " << type_name<T>() << " * " << prm_name << "_base"
          << ", " << type_name<size_t>() << " " << prm_name << "_start";

        for(size_t k = 0; k < NDIM; ++k)
            s << ", " << type_name<size_t>()    << " " << prm_name << "_length" << k
              << ", " << type_name<ptrdiff_t>() << " " << prm_name << "_stride" << k;

        return s.str();
    }

    void setArgs(cl::Kernel &kernel, unsigned device, size_t/*index_offset*/,
	    unsigned &position) const
    {
        kernel.setArg(position++, start);
        for(size_t k = 0; k < NDIM; ++k) {
            kernel.setArg(position++, length[k]);
            kernel.setArg(position++, stride[k]);
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

const range _;

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

    size_t size() const {
        return std::accumulate(dim.begin(), dim.end(),
                static_cast<size_t>(1), std::multiplies<size_t>());
    }
};

const extent_gen<0> extents;

template <size_t NR, class Dimensions = boost::fusion::vector<> >
struct index_gen {
    std::array<range, NR> ranges;

    index_gen() {}

    typedef
        typename boost::fusion::result_of::as_vector<
            typename boost::fusion::result_of::push_back<
                Dimensions,
                boost::mpl::size_t<NR>
            >::type
        >::type next_dim;

    index_gen<NR+1, next_dim> operator[](const range &r) const {
        return append_range<next_dim>(r);
    }

    index_gen<NR+1, Dimensions> operator[](size_t i) const {
        return append_range<Dimensions>(i);
    }

    private:
        template <class Dim>
        index_gen<NR+1, Dim> append_range(const range &r) const {
            index_gen<NR+1, Dim> idx;
            std::copy(ranges.begin(), ranges.end(), idx.ranges.begin());
            idx.ranges.back() = r;
            return idx;
        }
};

const index_gen<0> indices;

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
struct slicer {
    std::array<size_t, NR> dim;
    std::array<size_t, NR> stride;

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

    template <class Dimensions>
    gslice<NR> operator()(const index_gen<NR, Dimensions> &idx) const {
        size_t start = 0;
        std::array<size_t, NR> len;
        std::array<size_t, NR> str;

        for(size_t i = 0; i < NR; ++i) {
            range r = idx.ranges[i].empty() ? range(0, dim[i]) : idx.ranges[i];

            start += r.start * stride[i];
            len[i] = (r.stop - r.start + r.stride - 1) / r.stride;
            str[i] = r.stride * stride[i];
        }

        return gslice<NR>(start, len, str);
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

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4307)
#endif
        slice<C+1> operator[](const range &r) const {
            return slice<C+1>(*this, r.empty() ? range(0, parent.dim[C + 1]) : r);
        }
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
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

    std::string partial_expression(const std::string &prm_name,
            const cl::Device&) const
    {
        std::ostringstream s;
        s << prm_name << "_base[" << prm_name << "_index[idx]]";

        return s.str();
    }

    std::string indexing_function(const std::string &/*prm_name*/,
            const cl::Device&) const
    {
        return "";
    }

    template <typename T>
    std::string parameter_declaration(const std::string &prm_name,
            const cl::Device&) const
    {
        std::ostringstream s;

        s << ",\n\tglobal " << type_name<T>() << " * " << prm_name << "_base"
          << ", global " << type_name<size_t>() << " * " << prm_name << "_index";

        return s.str();
    }

    void setArgs(cl::Kernel &kernel, unsigned device, size_t/*index_offset*/,
	    unsigned &position) const
    {
        kernel.setArg(position++, index(device));
    }

    template <typename T>
    vector_view<T, permutation> operator()(const vector<T> &base) const {
        assert(base.queue_list().size() == 1);
        return vector_view<T, permutation>(base, *this);
    }
};

/// Expression-based permutation operator.
template <class Expr>
struct expr_slice {
    const Expr expr;

    expr_slice(const Expr &expr) : expr(expr) {}

    size_t size() const {
        detail::get_expression_properties prop;
        detail::extract_terminals()(expr, prop);
        return prop.size;
    }

    std::string indexing_function(const std::string &prm_name,
            const cl::Device &dev) const
    {
        std::ostringstream s;

        detail::output_terminal_preamble ctx(s, dev, 1, prm_name + "_");
        boost::proto::eval(boost::proto::as_child(expr), ctx);

        return s.str();
    }

    std::string partial_expression(const std::string &prm_name,
            const cl::Device &dev) const
    {
        // TODO: local preamble?
        std::ostringstream s;
        s << prm_name << "_base[";
        detail::vector_expr_context ctx(s, dev, 1, prm_name + "_");
        boost::proto::eval(boost::proto::as_child(expr), ctx);
        s << "]";

        return s.str();
    }

    template <typename T>
    std::string parameter_declaration(const std::string &prm_name,
            const cl::Device &dev) const
    {
        std::ostringstream s;

        s << ",\n\tglobal " << type_name<T>() << " * " << prm_name << "_base";

        detail::declare_expression_parameter ctx(s, dev, 1, prm_name + "_");
        detail::extract_terminals()(boost::proto::as_child(expr), ctx);

        return s.str();
    }

    void setArgs(cl::Kernel &kernel, unsigned device, size_t index_offset,
	    unsigned &position) const
    {
        detail::extract_terminals()( boost::proto::as_child(expr),
                detail::set_expression_argument(kernel, device, position, index_offset));
    }

    template <typename T>
    vector_view<T, expr_slice> operator()(const vector<T> &base) const {
        assert(base.queue_list().size() == 1);
        return vector_view<T, expr_slice>(base, *this);
    }
};

/// Returns permutation functor which is based on an integral expression.
/**
 * Example:
 * \code
 * auto reverse = vex::eslice(N - 1 - vex::element_index());
 * Y = reverse(X);
 * \endcode
 */
template <class Expr>
typename std::enable_if<
    std::is_integral<typename detail::return_type<Expr>::type>::value,
    expr_slice<Expr>
>::type
eslice(const Expr &expr) {
    return expr_slice<Expr>(expr);
}

//---------------------------------------------------------------------------
// Slice reduction
//---------------------------------------------------------------------------
/// \cond INTERNAL
struct reduced_vector_view_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< reduced_vector_view_terminal >::type
    > reduced_vector_view_terminal_expression;

template <typename T, size_t NDIM, size_t NR, class RDC>
struct reduced_vector_view : public reduced_vector_view_terminal_expression
{
    const vector<T>        &base;
    gslice<NDIM>           slice;
    std::array<size_t, NR> reduce_dims;

    reduced_vector_view(
            const vector_view< T, gslice<NDIM> > &view,
            std::array<size_t, NR> dims
            ) : base(view.base), slice(view.slice)
    {
        std::copy(dims.begin(), dims.end(), reduce_dims.begin());
        std::sort(reduce_dims.begin(), reduce_dims.end());
    }
};

namespace traits {

template <>
struct is_vector_expr_terminal< reduced_vector_view_terminal >
    : std::true_type
{ };

template <typename T, size_t NDIM, size_t NR, class RDC>
struct terminal_preamble< reduced_vector_view<T, NDIM, NR, RDC> > {
    static std::string get(const reduced_vector_view<T, NDIM, NR, RDC>&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;
        std::ostringstream rdc_name;

        rdc_name << "reduce_op_" << prm_name;

        typedef typename RDC::template function<T> fun;
        fun::define(s, rdc_name.str());

        s << type_name<T>()
          << " reduce_" << prm_name << "(\n"
          "\tglobal " << type_name<T>() << " * base,\n"
          "\t" << type_name<size_t>() << " start";
        for(size_t k = 0; k < NDIM; ++k)
            s << ",\n\t" << type_name<size_t>() << " length" << k
              << ", " << type_name<ptrdiff_t>() << " stride" << k;
        s << ",\n\t" << type_name<size_t>() << " idx)\n{\n"
          "\tsize_t ptr" << NDIM - NR - 1 << " = start + (idx % length" << NDIM - NR - 1
          << ") * stride" << NDIM - NR - 1 << ";\n";
        for(size_t k = NDIM - NR - 1; k-- > 0;)
            s << "\tidx /= length" << k + 1 << ";\n"
              "\tptr" << NDIM - NR - 1 << " += (idx % length" << k
              << ") * stride" << k << ";\n";

        s << "\t" << type_name<T>() << " sum = " << RDC::template initial<T>() << ";\n";

        std::ostringstream indent;
        indent << "\t";

        for(size_t k = NDIM - NR; k < NDIM; ++k) {
            s << indent.str() << "for(size_t i" << k << " = 0, ptr" << k
              << " = ptr" << k - 1 << "; i" << k << " < length" << k << "; ++i"
              << k << ", ptr" << k << " += stride" << k << ")\n";
            indent << "\t";
        }

        s << indent.str() << "sum = " << rdc_name.str()
          << "(sum, base[ptr" << NDIM - 1 << "]);\n"
          "\treturn sum;\n}\n";

        return s.str();
    }
};

template <typename T, size_t NDIM, size_t NR, class RDC>
struct kernel_param_declaration< reduced_vector_view<T, NDIM, NR, RDC> > {
    static std::string get(const reduced_vector_view<T, NDIM, NR, RDC> &term,
            const cl::Device &device, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        return term.slice.template parameter_declaration<T>(prm_name, device);
    }
};

template <typename T, size_t NDIM, size_t NR, class RDC>
struct partial_vector_expr< reduced_vector_view<T, NDIM, NR, RDC> > {
    static std::string get(const reduced_vector_view<T, NDIM, NR, RDC>&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;
        s << "reduce_" << prm_name << "("
          << prm_name << "_base, "
          << prm_name << "_start, ";
        for(size_t k = 0; k < NDIM; ++k)
            s << prm_name << "_length" << k << ", "
              << prm_name << "_stride" << k << ", ";
        s << "idx)";

        return s.str();
    }
};

template <typename T, size_t NDIM, size_t NR, class RDC>
struct kernel_arg_setter< reduced_vector_view<T, NDIM, NR, RDC> > {
    static void set(const reduced_vector_view<T, NDIM, NR, RDC> &term,
            cl::Kernel &kernel, unsigned device, size_t/*index_offset*/,
            unsigned &position, detail::kernel_generator_state&)
    {
        kernel.setArg(position++, term.base(device));
        kernel.setArg(position++, term.slice.start);

        for(size_t k = 0; k < NDIM; ++k) {
            if (!std::binary_search(term.reduce_dims.begin(), term.reduce_dims.end(), k)) {
                kernel.setArg(position++, term.slice.length[k]);
                kernel.setArg(position++, term.slice.stride[k]);
            }
        }

        for(size_t k = 0; k < NR; ++k) {
            kernel.setArg(position++, term.slice.length[term.reduce_dims[k]]);
            kernel.setArg(position++, term.slice.stride[term.reduce_dims[k]]);
        }
    }
};

template <typename T, size_t NDIM, size_t NR, class RDC>
struct expression_properties< reduced_vector_view<T, NDIM, NR, RDC> > {
    static void get(const reduced_vector_view<T, NDIM, NR, RDC> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.base.queue_list();
        partition  = std::vector<size_t>(2, 0);
        size       = 1;

        for(size_t k = 0; k < NDIM; ++k)
            if (!std::binary_search(term.reduce_dims.begin(), term.reduce_dims.end(), k))
                size *= term.slice[k];

        partition.back() = size;
    }
};

} // namespace traits
/// \endcond

/// Reduce vector_view along specified dimensions.
template <class RDC, typename T, size_t NDIM, size_t NR>
reduced_vector_view<T, NDIM, NR, RDC> reduce(
        const vector_view<T, gslice<NDIM> > &view,
        const std::array<size_t, NR> &reduce_dims
        )
{
    return reduced_vector_view<T, NDIM, NR, RDC>(view, reduce_dims);
}

/// Reduce vector_view along specified dimension.
template <class RDC, typename T, size_t NDIM>
reduced_vector_view<T, NDIM, 1, RDC> reduce(
        const vector_view<T, gslice<NDIM> > &view,
        size_t reduce_dim
        )
{
    std::array<size_t, 1> dim = {{reduce_dim}};
    return reduced_vector_view<T, NDIM, 1, RDC>(view, dim);
}

} // namespace vex

#endif
