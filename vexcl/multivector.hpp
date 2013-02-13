#ifndef VEXCL_MULTIVECTOR_HPP
#define VEXCL_MULTIVECTOR_HPP

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
 * \file   vexcl/multivector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device multi-vector.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <functional>
#include <boost/proto/proto.hpp>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <vexcl/vector.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

/// \cond INTERNAL

template <typename T, bool own>
struct multivector_storage { };

template <typename T>
struct multivector_storage<T, true> {
    typedef std::unique_ptr<vex::vector<T>> type;
};

template <typename T>
struct multivector_storage<T, false> {
    typedef vex::vector<T>* type;
};

/// \endcond

typedef multivector_expression<
    typename boost::proto::terminal< multivector_terminal >::type
    > multivector_terminal_expression;


/// Container for several vex::vectors.
/**
 * This class allows to synchronously operate on several vex::vectors of the
 * same type and size.
 */
template <typename T, size_t N, bool own>
class multivector : public multivector_terminal_expression {
    public:
        typedef vex::vector<T>  subtype;
        typedef std::array<T,N> value_type;

        /// Proxy class.
        class element {
            public:
                operator const value_type () const {
                    value_type val;
                    for(uint i = 0; i < N; i++) val[i] = vec(i)[index];
                    return val;
                }

                const value_type operator=(value_type val) {
                    for(uint i = 0; i < N; i++) vec(i)[index] = val[i];
                    return val;
                }
            private:
                element(multivector &vec, size_t index)
                    : vec(vec), index(index) {}

                multivector &vec;
                const size_t      index;

                friend class multivector;
        };

        /// Proxy class.
        class const_element {
            public:
                operator const value_type () const {
                    value_type val;
                    for(uint i = 0; i < N; i++) val[i] = vec(i)[index];
                    return val;
                }
            private:
                const_element(const multivector &vec, size_t index)
                    : vec(vec), index(index) {}

                const multivector &vec;
                const size_t      index;

                friend class multivector;
        };

        template <class V, class E>
        class iterator_type {
            public:
                E operator*() const {
                    return E(vec, pos);
                }

                iterator_type& operator++() {
                    pos++;
                    return *this;
                }

                iterator_type operator+(ptrdiff_t d) const {
                    return iterator_type(vec, pos + d);
                }

                ptrdiff_t operator-(const iterator_type &it) const {
                    return pos - it.pos;
                }

                bool operator==(const iterator_type &it) const {
                    return pos == it.pos;
                }

                bool operator!=(const iterator_type &it) const {
                    return pos != it.pos;
                }
            private:
                iterator_type(V &vec, size_t pos) : vec(vec), pos(pos) {}

                V      &vec;
                size_t pos;

                friend class multivector;
        };

        typedef iterator_type<multivector, element> iterator;
        typedef iterator_type<const multivector, const_element> const_iterator;

        multivector() {
            static_assert(own,
                    "Empty constructor unavailable for referenced-type multivector");

            for(uint i = 0; i < N; i++) vec[i].reset(new vex::vector<T>());
        };

        /// Constructor.
        /**
         * If host pointer is not NULL, it is copied to the underlying vector
         * components of the multivector.
         * \param queue queue list to be shared between all components.
         * \param host  Host vector that holds data to be copied to
         *              the components. Size of host vector should be divisible
         *              by N. Components of the created multivector will have
         *              size equal to host.size() / N. The data will be
         *              partitioned equally between all components.
         * \param flags cl::Buffer creation flags.
         */
        multivector(const std::vector<cl::CommandQueue> &queue,
                const std::vector<T> &host,
                cl_mem_flags flags = CL_MEM_READ_WRITE)
        {
            static_assert(own, "Wrong constructor for non-owning multivector");
            static_assert(N > 0, "What's the point?");

            size_t size = host.size() / N;
            assert(N * size == host.size());

            for(uint i = 0; i < N; i++)
                vec[i].reset(new vex::vector<T>(
                        queue, size, host.data() + i * size, flags
                        ) );
        }

        /// Constructor.
        /**
         * If host pointer is not NULL, it is copied to the underlying vector
         * components of the multivector.
         * \param queue queue list to be shared between all components.
         * \param size  Size of each component.
         * \param host  Pointer to host buffer that holds data to be copied to
         *              the components. Size of the buffer should be equal to
         *              N * size. The data will be partitioned equally between
         *              all components.
         * \param flags cl::Buffer creation flags.
         */
        multivector(const std::vector<cl::CommandQueue> &queue, size_t size,
                const T *host = 0, cl_mem_flags flags = CL_MEM_READ_WRITE)
        {
            static_assert(own, "Wrong constructor for non-owning multivector");
            static_assert(N > 0, "What's the point?");

            for(uint i = 0; i < N; i++)
                vec[i].reset(new vex::vector<T>(
                        queue, size, host ? host + i * size : 0, flags
                        ) );
        }

        /// Construct from size.
        multivector(size_t size) {
            static_assert(own, "Wrong constructor for non-owning multivector");
            static_assert(N > 0, "What's the point?");

            for(uint i = 0; i < N; i++) vec[i].reset(new vex::vector<T>(size));
        }

        /// Copy constructor.
        multivector(const multivector &mv) {
            copy_components<own>(mv);
        }

        /// Constructor.
        /**
         * Copies references to component vectors.
         */
        multivector(std::array<vex::vector<T>*,N> components)
            : vec(components)
        {
            static_assert(!own, "Wrong constructor");
        }

        /// Resize multivector.
        void resize(const std::vector<cl::CommandQueue> &queue, size_t size) {
            for(uint i = 0; i < N; i++) vec[i]->resize(queue, size);
        }

        /// Resize multivector.
        void resize(size_t size) {
            for(uint i = 0; i < N; i++) vec[i]->resize(size);
        }

        /// Fills multivector with zeros.
        void clear() {
            *this = static_cast<T>(0);
        }

        /// Return size of a multivector (equals size of individual components).
        size_t size() const {
            return vec[0]->size();
        }

        /// Returns multivector component.
        const vex::vector<T>& operator()(uint i) const {
            return *vec[i];
        }

        /// Returns multivector component.
        vex::vector<T>& operator()(uint i) {
            return *vec[i];
        }

        /// Const iterator to beginning.
        const_iterator begin() const {
            return const_iterator(*this, 0);
        }

        /// Iterator to beginning.
        iterator begin() {
            return iterator(*this, 0);
        }

        /// Const iterator to end.
        const_iterator end() const {
            return const_iterator(*this, size());
        }

        /// Iterator to end.
        iterator end() {
            return iterator(*this, size());
        }

        /// Returns elements of all vectors, packed in std::array.
        const_element operator[](size_t i) const {
            return const_element(*this, i);
        }

        /// Assigns elements of all vectors to a std::array value.
        element operator[](size_t i) {
            return element(*this, i);
        }

        /// Return reference to multivector's queue list
        const std::vector<cl::CommandQueue>& queue_list() const {
            return vec[0]->queue_list();
        }

        /// Assignment to a multivector.
        const multivector& operator=(const multivector &mv) {
            if (this != &mv) {
                for(uint i = 0; i < N; i++)
                    *vec[i] = mv(i);
            }
            return *this;
        }

        /** \name Expression assignments.
         * @{
         * All operations are delegated to components of the multivector.
         */
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                multivector_expr_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr& expr) {
            const std::vector<cl::CommandQueue> &queue = vec[0]->queue_list();

            for(auto q = queue.begin(); q != queue.end(); q++) {
                cl::Context context = qctx(*q);
                cl::Device  device  = qdev(*q);

                if (!exdata<Expr>::compiled[context()]) {

                    std::ostringstream kernel_name;
                    kernel_name << "multi_";
                    vector_name_context name_ctx(kernel_name);
                    boost::proto::eval(boost::proto::as_child(expr), name_ctx);

                    std::ostringstream kernel;
                    kernel << standard_kernel_header;

                    extract_user_functions()(
                            boost::proto::as_child(expr),
                            declare_user_function(kernel)
                            );

                    kernel << "kernel void " << kernel_name.str()
                           << "(\n\t" << type_name<size_t>() << " n";

                    for(size_t i = 0; i < N; )
                        kernel << ",\n\tglobal " << type_name<T>()
                               << " *res_" << ++i;

                    build_param_list<N>(boost::proto::as_child(expr), kernel);

                    kernel <<
                        "\n)\n{\n\t"
                        "for(size_t idx = get_global_id(0); idx < n; "
                        "idx += get_global_size(0)) {\n";

                    build_expr_list(boost::proto::as_child(expr), kernel);

                    kernel << "\t}\n}\n";

                    auto program = build_sources(context, kernel.str());

                    exdata<Expr>::kernel[context()]   = cl::Kernel(program, kernel_name.str().c_str());
                    exdata<Expr>::compiled[context()] = true;
                    exdata<Expr>::wgsize[context()]   = kernel_workgroup_size(
                            exdata<Expr>::kernel[context()], device);

                }
            }

            for(uint d = 0; d < queue.size(); d++) {
                if (size_t psize = vec[0]->part_size(d)) {
                    cl::Context context = qctx(queue[d]);
                    cl::Device  device  = qdev(queue[d]);

                    size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
                        alignup(psize, exdata<Expr>::wgsize[context()]) :
                        device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * exdata<Expr>::wgsize[context()] * 4;

                    uint pos = 0;
                    exdata<Expr>::kernel[context()].setArg(pos++, psize);

                    for(uint i = 0; i < N; i++)
                        exdata<Expr>::kernel[context()].setArg(pos++, vec[i]->operator()(d));

                    set_kernel_args<N>(
                            boost::proto::as_child(expr),
                            exdata<Expr>::kernel[context()],
                            d, pos, vec[0]->part_start(d)
                            );

                    queue[d].enqueueNDRangeKernel(
                            exdata<Expr>::kernel[context()],
                            cl::NullRange,
                            g_size, exdata<Expr>::wgsize[context()]
                            );
                }
            }

            return *this;
        }

        /// Multi-expression assignments.
#ifndef BOOST_NO_VARIADIC_TEMPLATES
        template <class... Args>
        typename std::enable_if<N == sizeof...(Args), const multivector&>::type
        operator=(const std::tuple<Args...> &expr) {
            typedef std::tuple<Args...> ExprTuple;
#else
        template <class ExprTuple>
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<ExprTuple>::type,
                multivector_full_grammar
            >::value
#if !defined(_MSC_VER) || _MSC_VER >= 1700
            && N == std::tuple_size<ExprTuple>::value
#endif
            , const multivector&
        >::type
        operator=(const ExprTuple &expr) {
#endif
            const std::vector<cl::CommandQueue> &queue = vec[0]->queue_list();

            for(auto q = queue.begin(); q != queue.end(); q++) {
                cl::Context context = qctx(*q);
                cl::Device  device  = qdev(*q);

                if (!exdata<ExprTuple>::compiled[context()]) {
                    std::ostringstream kernel;

                    kernel << standard_kernel_header;

                    {
                        get_header f(kernel);
                        for_each<0>(expr, f);
                    }

                    kernel <<
                        "kernel void multi_expr_tuple(\n"
                        "\t" << type_name<size_t>() << " n";

                    for(uint i = 1; i <= N; i++)
                        kernel << ",\n\tglobal " << type_name<T>() << " *res_" << i;

                    {
                        get_params f(kernel);
                        for_each<0>(expr, f);
                    }

                    kernel <<
                        "\n)\n{\n\t"
                        "for(size_t idx = get_global_id(0); idx < n; "
                        "idx += get_global_size(0)) {\n";

                    {
                        get_expressions f(kernel);
                        for_each<0>(expr, f);
                    }

                    kernel << "\n";

                    for(uint i = 1; i <= N; i++)
                        kernel << "\t\tres_" << i << "[idx] = buf_" << i << ";\n";

                    kernel << "\t}\n}\n";

                    auto program = build_sources(context, kernel.str());

                    exdata<ExprTuple>::kernel[context()]   = cl::Kernel(program, "multi_expr_tuple");
                    exdata<ExprTuple>::compiled[context()] = true;
                    exdata<ExprTuple>::wgsize[context()]   = kernel_workgroup_size(
                            exdata<ExprTuple>::kernel[context()], device);

                }
            }

            for(uint d = 0; d < queue.size(); d++) {
                if (size_t psize = vec[0]->part_size(d)) {
                    cl::Context context = qctx(queue[d]);
                    cl::Device  device  = qdev(queue[d]);

                    size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
                        alignup(psize, exdata<ExprTuple>::wgsize[context()]) :
                        device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * exdata<ExprTuple>::wgsize[context()] * 4;

                    uint pos = 0;
                    exdata<ExprTuple>::kernel[context()].setArg(pos++, psize);

                    for(uint i = 0; i < N; i++)
                        exdata<ExprTuple>::kernel[context()].setArg(pos++, (*vec[i])(d));

                    {
                        set_arguments f(exdata<ExprTuple>::kernel[context()], d, pos, vec[0]->part_start(d));
                        for_each<0>(expr, f);
                    }

                    queue[d].enqueueNDRangeKernel(
                            exdata<ExprTuple>::kernel[context()],
                            cl::NullRange,
                            g_size, exdata<ExprTuple>::wgsize[context()]
                            );
                }
            }

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_multivector_transform_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr &expr) {
            apply_additive_transform</*append=*/false>(*this,
                    simplify_additive_transform()( expr ));
            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                multivector_full_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                multivector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_multivector_transform_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr &expr) {
            *this = extract_multivector_expressions()( expr );

            apply_additive_transform</*append=*/true>(*this,
                    simplify_additive_transform()(
                        extract_additive_multivector_transforms()( expr )
                        )
                    );
            return *this;
        }

#define COMPOUND_ASSIGNMENT(cop, op) \
        template <class Expr> \
        const multivector& operator cop(const Expr &expr) { \
            return *this = *this op expr; \
        }

        COMPOUND_ASSIGNMENT(+=, +);
        COMPOUND_ASSIGNMENT(-=, -);
        COMPOUND_ASSIGNMENT(*=, *);
        COMPOUND_ASSIGNMENT(/=, /);
        COMPOUND_ASSIGNMENT(%=, %);
        COMPOUND_ASSIGNMENT(&=, &);
        COMPOUND_ASSIGNMENT(|=, |);
        COMPOUND_ASSIGNMENT(^=, ^);
        COMPOUND_ASSIGNMENT(<<=, <<);
        COMPOUND_ASSIGNMENT(>>=, >>);

#undef COMPOUND_ASSIGNMENT

        /** @} */
    private:
        template <size_t I, class Expr>
            typename std::enable_if<I == N>::type
            expr_list_loop(const Expr &, std::ostream &) { }

        template <size_t I, class Expr>
            typename std::enable_if<I < N>::type
            expr_list_loop(const Expr &expr, std::ostream &os) {
                multivector_expr_context<N, I> ctx(os);
                os << "\t\tres_" << I + 1 << "[idx] = ";
                boost::proto::eval(expr, ctx);
                os << ";\n";

                expr_list_loop<I+1, Expr>(expr, os);
            }

        template <class Expr>
            void build_expr_list(const Expr &expr, std::ostream &os) {
                expr_list_loop<0, Expr>(expr, os);
            }


        template <bool own_components>
        typename std::enable_if<own_components,void>::type
        copy_components(const multivector &mv) {
            for(uint i = 0; i < N; i++)
                vec[i].reset(new vex::vector<T>(mv(i)));
        }

        template <bool own_components>
        typename std::enable_if<!own_components,void>::type
        copy_components(const multivector &mv) {
            for(uint i = 0; i < N; i++)
                vec[i] = mv.vec[i];
        }

        struct get_header {
            std::ostream &os;
            mutable int cmp_idx;

            get_header(std::ostream &os) : os(os), cmp_idx(0) {}

            template <class Expr>
            void operator()(const Expr &expr) const {
                extract_user_functions()(
                        boost::proto::as_expr(expr),
                        declare_user_function(os, ++cmp_idx)
                        );
            }
        };

        struct get_params {
            std::ostream &os;
            mutable int cmp_idx;

            get_params(std::ostream &os) : os(os), cmp_idx(0) {}

            template <class Expr>
            void operator()(const Expr &expr) const {
                extract_terminals()(
                        boost::proto::as_child(expr),
                        declare_expression_parameter(os, ++cmp_idx)
                        );
            }
        };

        struct get_expressions {
            std::ostream &os;
            mutable int cmp_idx;

            get_expressions(std::ostream &os) : os(os), cmp_idx(0) {}

            template <class Expr>
            void operator()(const Expr &expr) const {
                vector_expr_context ctx(os, ++cmp_idx);
                os << "\t\t" << type_name<T>() << " buf_" << cmp_idx << " = ";
                boost::proto::eval(boost::proto::as_child(expr), ctx);
                os << ";\n";
            }
        };

        struct set_arguments {
            cl::Kernel &krn;
            uint d, &pos;
            size_t part_start;

            set_arguments(cl::Kernel &krn, uint d, uint &pos, size_t part_start)
                : krn(krn), d(d), pos(pos), part_start(part_start) {}

            template <class Expr>
            void operator()(const Expr &expr) const {
                extract_terminals()(
                        boost::proto::as_child(expr),
                        set_expression_argument(krn, d, pos, part_start)
                        );
            }
        };

        std::array<typename multivector_storage<T, own>::type,N> vec;

        template <class Expr>
        struct exdata {
            static std::map<cl_context,bool>       compiled;
            static std::map<cl_context,cl::Kernel> kernel;
            static std::map<cl_context,size_t>     wgsize;
        };
};

template <class T, size_t N, bool own> template <class Expr>
std::map<cl_context,bool> multivector<T,N,own>::exdata<Expr>::compiled;

template <class T, size_t N, bool own> template <class Expr>
std::map<cl_context,cl::Kernel> multivector<T,N,own>::exdata<Expr>::kernel;

template <class T, size_t N, bool own> template <class Expr>
std::map<cl_context,size_t> multivector<T,N,own>::exdata<Expr>::wgsize;

/// Copy multivector to host vector.
template <class T, size_t N, bool own>
void copy(const multivector<T,N,own> &mv, std::vector<T> &hv) {
    for(uint i = 0; i < N; i++)
        vex::copy(mv(i).begin(), mv(i).end(), hv.begin() + i * mv.size());
}

/// Copy host vector to multivector.
template <class T, size_t N, bool own>
void copy(const std::vector<T> &hv, multivector<T,N,own> &mv) {
    for(uint i = 0; i < N; i++)
        vex::copy(hv.begin() + i * mv.size(), hv.begin() + (i + 1) * mv.size(),
                mv(i).begin());
}

#ifndef BOOST_NO_VARIADIC_TEMPLATES
/// Ties several vex::vectors into a multivector.
/**
 * The following example results in a single kernel:
 * \code
 * vex::vector<double> x(ctx, 1024);
 * vex::vector<double> y(ctx, 1024);
 *
 * vex::tie(x,y) = std::tie( x + y, y - x );
 * \endcode
 * This is functionally equivalent to
 * \code
 * tmp_x = x + y;
 * tmp_y = y - x;
 * x = tmp_x;
 * y = tmp_y;
 * \endcode
 * but does not use temporaries and is more efficient.
 */
template<typename T, class... Tail>
typename std::enable_if<
    And<std::is_same<T,Tail>...>::value,
    multivector<T, sizeof...(Tail) + 1, false>
    >::type
tie(vex::vector<T> &head, vex::vector<Tail>&... tail) {
    std::array<vex::vector<T>*, sizeof...(Tail) + 1> ptr = {{&head, (&tail)...}};

    return multivector<T, sizeof...(Tail) + 1, false>(ptr);
}
#else

#define TIE_VECTORS(z, n, data) \
template<typename T> \
multivector<T, n, false> tie( BOOST_PP_ENUM_PARAMS(n, vex::vector<T> &v) ) { \
    std::array<vex::vector<T>*, n> ptr = {{ BOOST_PP_ENUM_PARAMS(n, &v) }}; \
    return multivector<T, n, false>(ptr); \
}

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TIE_VECTORS, ~)

#undef TIE_VECTORS

#endif

} // namespace vex

namespace boost { namespace fusion { namespace traits {

template <class T, size_t N, bool own>
struct is_sequence< vex::multivector<T, N, own> > : std::false_type
{};

} } }


// vim: et
#endif
