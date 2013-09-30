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

struct multivector_terminal {};

template <typename T, size_t N> class multivector;

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

template <typename T, size_t N>
struct number_of_components< multivector<T, N> > : boost::mpl::size_t<N> {};

template <size_t I, typename T, size_t N>
struct component< I, multivector<T, N> > {
    typedef const vector<T>& type;
};

} // namespace traits

template <size_t I, typename T, size_t N>
const vector<T>& get(const multivector<T, N> &mv) {
    static_assert(I < N, "Component number out of bounds");

    return mv(I);
}

template <size_t I, typename T, size_t N>
vector<T>& get(multivector<T, N> &mv) {
    static_assert(I < N, "Component number out of bounds");

    return mv(I);
}

namespace detail {

template <class OP, class LHS, class RHS>
void assign_multiexpression(LHS &lhs, const RHS &rhs,
        const std::vector<cl::CommandQueue> &queue,
        const std::vector<size_t> &part
        );
}


typedef multivector_expression<
    typename boost::proto::terminal< multivector_terminal >::type
    > multivector_terminal_expression;

/// \endcond

/// Container for several vex::vectors.
/**
 * This class allows to synchronously operate on several vex::vectors of the
 * same type and size.
 */
template <typename T, size_t N>
class multivector : public multivector_terminal_expression {
    public:
        typedef vex::vector<T>  subtype;
        typedef std::array<T,N> value_type;
	typedef T               sub_value_type;

        const static size_t NDIM = N;

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

        multivector() {};

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
            static_assert(N > 0, "What's the point?");

            size_t size = host.size() / N;
            assert(N * size == host.size());

            for(size_t i = 0; i < N; ++i)
                vec[i].resize(queue, size, host.data() + i * size, flags);
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
            static_assert(N > 0, "What's the point?");

            for(size_t i = 0; i < N; ++i)
                vec[i].resize(queue, size, host ? host + i * size : 0, flags);
        }

        /// Construct from size.
        multivector(size_t size) {
            static_assert(N > 0, "What's the point?");

            for(size_t i = 0; i < N; ++i) vec[i].resize(size);
        }

        /// Copy constructor.
        multivector(const multivector &mv) {
#ifdef VEXCL_SHOW_COPIES
            std::cout << "Copying vex::multivector<" << type_name<T>()
                      << ", " << N << "> of size " << size() << std::endl;
#endif
            for(size_t i = 0; i < N; ++i) vec[i].resize(mv(i));
        }

        /// Move constructor.
        multivector(multivector &&mv) noexcept {
            for(size_t i = 0; i < N; ++i) vec[i].swap(mv.vec[i]);
        }

        /// Resize multivector.
        void resize(const std::vector<cl::CommandQueue> &queue, size_t size) {
            for(unsigned i = 0; i < N; i++) vec[i].resize(queue, size);
        }

        /// Resize multivector.
        void resize(size_t size) {
            for(unsigned i = 0; i < N; i++) vec[i].resize(size);
        }

        /// Fills multivector with zeros.
        void clear() {
            *this = static_cast<T>(0);
        }

        /// Return size of a multivector (equals size of individual components).
        size_t size() const {
            return vec[0].size();
        }

        /// Returns multivector component.
        const vex::vector<T>& operator()(size_t i) const {
            return vec[i];
        }

        /// Returns multivector component.
        vex::vector<T>& operator()(size_t i) {
            return vec[i];
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
            return vec[0].queue_list();
        }

        /// Assignment to a multivector.
        const multivector& operator=(const multivector &mv) {
            if (this != &mv)
                for(size_t i = 0; i < N; ++i) vec[i] = mv(i);
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
            detail::assign_multiexpression<op>(*this, expr, vec[0].queue_list(), vec[0].partition()); \
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
        std::array<vector<T>,N> vec;
};

/// Copy multivector to host vector.
template <class T, size_t N>
void copy(const multivector<T,N> &mv, std::vector<T> &hv) {
    for(size_t i = 0; i < N; ++i)
        vex::copy(mv(i).begin(), mv(i).end(), hv.begin() + i * mv.size());
}

/// Copy host vector to multivector.
template <class T, size_t N>
void copy(const std::vector<T> &hv, multivector<T,N> &mv) {
    for(size_t i = 0; i < N; ++i)
        vex::copy(hv.begin() + i * mv.size(), hv.begin() + (i + 1) * mv.size(),
                mv(i).begin());
}

} // namespace vex

namespace boost { namespace fusion { namespace traits {

template <class T, size_t N>
struct is_sequence< vex::multivector<T, N> > : std::false_type
{};

} } }


#endif
