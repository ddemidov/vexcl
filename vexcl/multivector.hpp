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

#ifdef _MSC_VER
#  define NOMINMAX
#endif

#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <boost/proto/proto.hpp>

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

struct multivector_terminal {};

template <typename T, size_t N, bool own = true> class multivector;

namespace traits {

// Extract component directly from terminal rather than from value(terminal):
template <>
struct proto_terminal_is_value< multivector_terminal >
    : std::true_type
{ };

template <>
struct is_multivector_expr_terminal< multivector_terminal >
    : std::true_type
{ };

// Hold multivector terminals by reference:
template <class T>
struct hold_terminal_by_reference< T,
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr< T >::type,
                boost::proto::terminal< multivector_terminal >
            >::value
        >::type
    >
    : std::true_type
{ };

template <typename T, size_t N, bool own>
struct number_of_components< multivector<T, N, own> >
    : boost::mpl::size_t<N>
{};

template <size_t I, typename T, size_t N, bool own>
struct component< I, multivector<T, N, own> > {
    typedef const vector<T>& type;
};

} // namespace traits

template <size_t I, typename T, size_t N, bool own>
const vector<T>& get(const multivector<T, N, own> &mv) {
    static_assert(I < N, "Component number out of bounds");

    return mv(I);
}

template <size_t I, typename T, size_t N, bool own>
vector<T>& get(multivector<T, N, own> &mv) {
    static_assert(I < N, "Component number out of bounds");

    return mv(I);
}


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
	typedef T               sub_value_type;

        /// Proxy class.
        class element {
            public:
                operator const value_type () const {
                    value_type val;
                    for(unsigned i = 0; i < N; i++) val[i] = vec(i)[index];
                    return val;
                }

                const value_type operator=(value_type val) {
                    for(unsigned i = 0; i < N; i++) vec(i)[index] = val[i];
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
                    for(unsigned i = 0; i < N; i++) val[i] = vec(i)[index];
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

            for(unsigned i = 0; i < N; i++) vec[i].reset(new vex::vector<T>());
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

            for(unsigned i = 0; i < N; i++)
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

            for(unsigned i = 0; i < N; i++)
                vec[i].reset(new vex::vector<T>(
                        queue, size, host ? host + i * size : 0, flags
                        ) );
        }

        /// Construct from size.
        multivector(size_t size) {
            static_assert(own, "Wrong constructor for non-owning multivector");
            static_assert(N > 0, "What's the point?");

            for(unsigned i = 0; i < N; i++) vec[i].reset(new vex::vector<T>(size));
        }

        /// Copy constructor.
        multivector(const multivector &mv) {
            copy_components<own>(mv);
        }

        /// Move constructor.
        multivector(multivector &&mv) noexcept {
            std::swap(vec, mv.vec);
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
            for(unsigned i = 0; i < N; i++) vec[i]->resize(queue, size);
        }

        /// Resize multivector.
        void resize(size_t size) {
            for(unsigned i = 0; i < N; i++) vec[i]->resize(size);
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
        const vex::vector<T>& operator()(size_t i) const {
            return *vec[i];
        }

        /// Returns multivector component.
        vex::vector<T>& operator()(size_t i) {
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
                for(unsigned i = 0; i < N; i++)
                    *vec[i] = mv(i);
            }
            return *this;
        }

#ifdef DOXYGEN
#  define ASSIGNMENT(cop, op) \
        /** \brief Multiector expression assignment.
         \details All operations are delegated to components of the multivector.
         */ \
        template <class Expr> \
            const multivector& \
        operator cop(const Expr &expr);
#else
#  define ASSIGNMENT(cop, op) \
        template <class Expr> \
        typename std::enable_if< \
            boost::proto::matches< \
                typename boost::proto::result_of::as_expr<Expr>::type, \
                multivector_expr_grammar \
            >::value || is_tuple<Expr>::value, \
            const multivector& \
        >::type \
        operator cop(const Expr &expr) { \
            assign_expression<op>(expr); \
            return *this; \
        }
#endif

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

#ifndef DOXYGEN
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_multivector_transform_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/false>(*this,
                    detail::simplify_additive_transform()( expr ));
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
        operator+=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/true>(*this,
                    detail::simplify_additive_transform()( expr ));
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
        operator-=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/true>(*this,
                    detail::simplify_additive_transform()( -expr ));
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
            *this  = detail::extract_multivector_expressions()( expr );
            *this += detail::extract_additive_multivector_transforms()( expr );

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
        operator+=(const Expr &expr) {
            *this += detail::extract_multivector_expressions()( expr );
            *this += detail::extract_additive_multivector_transforms()( expr );

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
        operator-=(const Expr &expr) {
            *this -= detail::extract_multivector_expressions()( expr );
            *this -= detail::extract_additive_multivector_transforms()( expr );

            return *this;
        }
#endif

    private:
        template <bool own_components>
        typename std::enable_if<own_components,void>::type
        copy_components(const multivector &mv) {
            for(unsigned i = 0; i < N; i++)
                vec[i].reset(new vex::vector<T>(mv(i)));
        }

        template <bool own_components>
        typename std::enable_if<!own_components,void>::type
        copy_components(const multivector &mv) {
            for(unsigned i = 0; i < N; i++)
                vec[i] = mv.vec[i];
        }

        template <class OP, class Expr>
        struct subexpression_assigner {
            multivector &result;
            const Expr &expr;

            subexpression_assigner(multivector &result, const Expr &expr)
                : result(result), expr(expr) { }

            template <long I>
            void apply() const {
                detail::assign_expression<OP>(
                        result(I), detail::subexpression<I>::get(expr),
                        result(I).queue, result(I).part);
            }
        };

        template <class Expr>
        struct preamble_constructor {
            const Expr   &expr;
            std::ostream &source;
            const cl::Device &device;

            preamble_constructor(const Expr &expr, std::ostream &source, const cl::Device &device)
                : expr(expr), source(source), device(device)
            { }

            template <long I>
            void apply() const {
                detail::construct_preamble(detail::subexpression<I>::get(expr),
                        source, device, I + 1);
            }
        };

        template <class Expr>
        struct parameter_declarator {
            const Expr   &expr;
            std::ostream &source;
            const cl::Device &device;

            parameter_declarator(const Expr &expr, std::ostream &source, const cl::Device &device)
                : expr(expr), source(source), device(device)
            { }

            template <long I>
            void apply() const {
                detail::extract_terminals()(
                        boost::proto::as_child(detail::subexpression<I>::get(expr)),
                        detail::declare_expression_parameter(source, device, I + 1)
                        );
            }
        };

        template <class Expr>
        struct expression_builder {
            const Expr   &expr;
            std::ostream &source;
            const cl::Device &device;

            expression_builder(const Expr &expr, std::ostream &source, const cl::Device &device)
                : expr(expr), source(source), device(device)
            { }

            template <long I>
            void apply() const {
                source << "\t\t" << type_name<T>() << " buf_" << I + 1 << " = ";

                detail::vector_expr_context expr_ctx(source, device, I + 1);
                boost::proto::eval(
                        boost::proto::as_child(detail::subexpression<I>::get(expr)),
                        expr_ctx);

                source << ";\n";
            }
        };

        template <class Expr>
        struct kernel_arg_setter {
            const Expr   &expr;
            cl::Kernel   &krn;
            unsigned     dev;
            size_t       offset;
            unsigned     &pos;

            kernel_arg_setter(const Expr &expr, cl::Kernel &krn, unsigned dev, size_t offset, unsigned &pos)
                : expr(expr), krn(krn), dev(dev), offset(offset), pos(pos) { }

            template <long I>
            void apply() const {
                detail::extract_terminals()(
                            boost::proto::as_child(detail::subexpression<I>::get(expr)),
                            detail::set_expression_argument(krn, dev, pos, offset)
                            );

            }
        };

        // Static for loop
        template <long Begin, long End>
        class static_for {
            public:
                template <class Func>
                static void loop(Func &&f) {
                    iterate<Begin>(f);
                }

            private:
                template <long I, class Func>
                static typename std::enable_if<(I < End)>::type
                iterate(Func &&f) {
                    f.template apply<I>();
                    iterate<I + 1>(f);
                }

                template <long I, class Func>
                static typename std::enable_if<(I >= End)>::type
                iterate(Func&&)
                { }
        };

        template <class OP, class Expr>
        void assign_expression(const Expr &expr) {
            using namespace detail;

            static kernel_cache cache;

            const std::vector<cl::CommandQueue> &queue = vec[0]->queue_list();

            // If any device in context is CPU, then do not fuse the kernel,
            // but assign components individually.
            if (
                    std::any_of(queue.begin(), queue.end(),
                        [](const cl::CommandQueue &q) {
                            return is_cpu(qdev(q));\
                        })
               )
            {
                static_for<0, N>::loop(
                        subexpression_assigner<OP, Expr>(*this, expr)
                        );
                return;
            }

            for(unsigned d = 0; d < queue.size(); d++) {
                cl::Context context = qctx(queue[d]);
                cl::Device  device  = qdev(queue[d]);

                auto kernel = cache.find( context() );

                if (kernel == cache.end()) {
                    std::ostringstream source;

                    source << standard_kernel_header(device);

                    static_for<0, N>::loop(
                            preamble_constructor<Expr>(expr, source, device)
                            );

                    source << "kernel void multiex_kernel(\n\t"
                           << type_name<size_t>() << " n";

                    for(size_t i = 0; i < N; )
                        source << ",\n\tglobal " << type_name<T>()
                               << " *res_" << ++i;

                    static_for<0, N>::loop(
                            parameter_declarator<Expr>(expr, source, device)
                            );

                    source <<
                        "\n)\n{\n"
                        "\tfor(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n";

                    static_for<0, N>::loop(
                            expression_builder<Expr>(expr, source, device)
                            );

                    source << "\n";

                    for(unsigned i = 1; i <= N; ++i)
                        source << "\t\tres_" << i << "[idx] " << OP::string() << " buf_" << i << ";\n";

                    source << "\t}\n}\n";

                    auto program = build_sources(context, source.str());

                    cl::Kernel krn(program, "multiex_kernel");
                    size_t wgs = kernel_workgroup_size(krn, device);

                    kernel = cache.insert(std::make_pair(
                                context(), kernel_cache_entry(krn, wgs)
                                )).first;
                }

                if (size_t psize = vec[0]->part_size(d)) {
                    size_t w_size = kernel->second.wgsize;
                    size_t g_size = num_workgroups(device) * w_size;

                    unsigned pos = 0;
                    kernel->second.kernel.setArg(pos++, psize);

                    for(unsigned i = 0; i < N; i++)
                        kernel->second.kernel.setArg(pos++, vec[i]->operator()(d));

                    static_for<0, N>::loop(
                            kernel_arg_setter<Expr>(expr, kernel->second.kernel, d, vec[0]->part_start(d), pos)
                            );

                    queue[d].enqueueNDRangeKernel(
                            kernel->second.kernel, cl::NullRange, g_size, w_size
                            );
                }
            }
        }

        std::array<typename multivector_storage<T, own>::type,N> vec;
};

namespace traits {

} // namespace traits

/// Copy multivector to host vector.
template <class T, size_t N, bool own>
void copy(const multivector<T,N,own> &mv, std::vector<T> &hv) {
    for(unsigned i = 0; i < N; i++)
        vex::copy(mv(i).begin(), mv(i).end(), hv.begin() + i * mv.size());
}

/// Copy host vector to multivector.
template <class T, size_t N, bool own>
void copy(const std::vector<T> &hv, multivector<T,N,own> &mv) {
    for(unsigned i = 0; i < N; i++)
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
    traits::And<std::is_same<T,Tail>...>::value,
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


#endif
