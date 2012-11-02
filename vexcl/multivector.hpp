#ifndef VEXCL_MULTIVECTOR_HPP
#define VEXCL_MULTIVECTOR_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device multi-vector.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#ifndef _MSC_VER
#  define VEXCL_VARIADIC_TEMPLATES
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
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
#include <boost/mpl/max.hpp>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <vexcl/vector.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

// TODO: remove this
namespace proto = boost::proto;
using proto::_;

/// \cond INTERNAL

template <class T, class Enable = void>
struct is_multiscalar : std::false_type
{};

// Arithmetic scalars

template <class T>
struct is_multiscalar< T, 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

template <class T>
struct number_of_components : boost::mpl::size_t<1>
{};

template <size_t I, class T, class Enable = void>
struct component {
    typedef T type;
};

template <size_t I, typename T>
inline T& get(T &t) {
    return t;
}

#ifdef VEXCL_VARIADIC_TEMPLATES

// std::tuple<...>
template <typename... T>
struct And : std::true_type {};

template <typename Head, typename... Tail>
struct And<Head, Tail...>
    : std::conditional<Head::value, And<Tail...>, std::false_type>::type
{};

template <class... Args>
struct is_multiscalar<std::tuple<Args...>,
    typename std::enable_if<And< std::is_arithmetic<Args>... >::type::value >::type >
    : std::true_type
{};

template <class... Args>
struct number_of_components< std::tuple<Args...> >
    : boost::mpl::size_t<sizeof...(Args)>
{};

template <size_t I, class... Args>
struct component< I, std::tuple<Args...> >
    : std::tuple_element< I, std::tuple<Args...> >
{};

#endif

// std::array<T,N>

template <class T, size_t N>
struct is_multiscalar< std::array<T, N>, 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

template <class T, size_t N>
struct number_of_components< std::array<T, N> >
    : boost::mpl::size_t<N>
{};

template <size_t I, class T, size_t N>
struct component< I, std::array<T, N> > {
    typedef T type;
};

// C-style arrays
template <class T, size_t N>
struct is_multiscalar< T[N], 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

template <class T, size_t N>
struct number_of_components< T[N] >
    : boost::mpl::size_t<N>
{};

template <size_t I, class T, size_t N>
struct component< I, T[N] > {
    typedef T type;
};

template <size_t I, typename T, size_t N>
inline const T& get(const T t[N]) {
    static_assert(I < N, "Component number out of bounds");
    return t[I];
}

template <size_t I, typename T, size_t N>
inline T& get(T t[N]) {
    static_assert(I < N, "Component number out of bounds");
    return t[I];
}

struct multivector_terminal {};

template <typename T, size_t N, bool own = true>
struct multivector;

template <typename T, size_t N, bool own>
struct number_of_components< multivector<T, N, own> >
    : boost::mpl::size_t<N>
{};

template <size_t I, typename T, size_t N, bool own>
struct component< I, multivector<T, N, own> > {
    typedef vector<T> type;
};

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

template 
struct mutltiex_dimension
        : proto::or_ <
            proto::when <
		proto::terminal< _ >,
		number_of_components<_>()
	    > ,
	    proto::when <
		proto::nary_expr<_, proto::vararg<_> >,
		proto::fold<_,
		    boost::mpl::size_t<0>(),
		    boost::mpl::max<mutltiex_dimension, proto::_state>()>()
	    >
        >
{};

struct multivector_expr_grammar
    : proto::or_<
	  proto::or_<
	      proto::terminal< multivector_terminal >,
	      proto::and_<
	          proto::terminal< _ >,
		  proto::if_< is_multiscalar< proto::_value >() >
	      >
          >,
	  BUILTIN_OPERATIONS(multivector_expr_grammar),
	  USER_FUNCTIONS(multivector_expr_grammar)
      >
{};

template <class Expr>
struct multivector_expression;

struct multivector_domain
    : proto::domain<
	proto::generator<multivector_expression>,
	multivector_expr_grammar
      >
{
    template <class T>
    struct as_child
	: proto_base_domain::as_child<T> {};

    template <class T>
    struct as_child< typename std::enable_if< std::is_arithmetic<T>::value, T > >
	: proto_base_domain::as_expr<T> {};
};

// TODO: allow vector subexpressions in multivector expressions.
template <class Expr>
struct multivector_expression
    : proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
{
    typedef
	proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
	base_type;

    multivector_expression(const Expr &expr = Expr()) : base_type(expr) {}
};

template <size_t I, class T>
struct component< I, T,
    typename std::enable_if<
	!is_multiscalar<T>::value &&
	is_multiscalar<
	    typename proto::result_of::value<
		typename proto::result_of::as_expr<T>::type
	    >::type
	>::value >::type
    >
{
    typedef typename proto::result_of::value<
		typename proto::result_of::as_expr<T>::type
	    >::type value_type;

    typedef typename proto::result_of::as_child<
	typename component<I, value_type>::type
	>::type type;
};

template <size_t I, typename T>
inline const
typename std::enable_if<
    !is_multiscalar<T>::value &&
    is_multiscalar<
	typename proto::result_of::value<
	    typename proto::result_of::as_expr<T>::type
	>::type
    >::value,
    typename component<I, T>::type
>::type
get(const T &t) {
    return proto::as_child(get<I>(proto::value(t)));
}


//---------------------------------------------------------------------------
// Multivector contexts
//---------------------------------------------------------------------------

// Builds textual representation for a vector expression.
template <size_t N, size_t C>
struct multivector_expr_context {
    std::ostream &os;
    int prm_idx, fun_idx;

    multivector_expr_context(std::ostream &os) : os(os), prm_idx(0), fun_idx(0) {}

    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {};

#define BINARY_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( "; \
	    proto::eval(proto::left(expr), ctx); \
	    ctx.os << " " #the_op " "; \
	    proto::eval(proto::right(expr), ctx); \
	    ctx.os << " )"; \
	} \
    }

    BINARY_OPERATION(plus,          +);
    BINARY_OPERATION(minus,         -);
    BINARY_OPERATION(multiplies,    *);
    BINARY_OPERATION(divides,       /);
    BINARY_OPERATION(modulus,       %);
    BINARY_OPERATION(shift_left,   <<);
    BINARY_OPERATION(shift_right,  >>);
    BINARY_OPERATION(less,          <);
    BINARY_OPERATION(greater,       >);
    BINARY_OPERATION(less_equal,   <=);
    BINARY_OPERATION(greater_equal,>=);
    BINARY_OPERATION(equal_to,     ==);
    BINARY_OPERATION(not_equal_to, !=);
    BINARY_OPERATION(logical_and,  &&);
    BINARY_OPERATION(logical_or,   ||);
    BINARY_OPERATION(bitwise_and,   &);
    BINARY_OPERATION(bitwise_or,    |);
    BINARY_OPERATION(bitwise_xor,   ^);

#undef BINARY_OPERATION

#define UNARY_PRE_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( " #the_op "( "; \
	    proto::eval(proto::child(expr), ctx); \
	    ctx.os << " ) )"; \
	} \
    }

    UNARY_PRE_OPERATION(unary_plus,   +);
    UNARY_PRE_OPERATION(negate,       -);
    UNARY_PRE_OPERATION(logical_not,  !);
    UNARY_PRE_OPERATION(pre_inc,     ++);
    UNARY_PRE_OPERATION(pre_dec,     --);

#undef UNARY_PRE_OPERATION

#define UNARY_POST_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( ( "; \
	    proto::eval(proto::child(expr), ctx); \
	    ctx.os << " )" #the_op " )"; \
	} \
    }

    UNARY_POST_OPERATION(post_inc, ++);
    UNARY_POST_OPERATION(post_dec, --);

#undef UNARY_POST_OPERATION

    template <typename Expr>
    struct eval<Expr, proto::tag::function> {
	typedef void result_type;

	struct do_eval {
	    mutable int pos;
	    multivector_expr_context &ctx;

	    do_eval(multivector_expr_context &ctx) : pos(0), ctx(ctx) {}

	    template <typename Arg>
	    void operator()(const Arg &arg) const {
		if (pos++) ctx.os << ", ";
		proto::eval(arg, ctx);
	    }
	};

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		builtin_function,
		typename proto::result_of::value<
		    typename proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, multivector_expr_context &ctx) const {
	    ctx.os << proto::value(proto::child_c<0>(expr)).name() << "( ";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr),
		    do_eval(ctx)
		    );
	    ctx.os << " )";
	}

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		user_function,
		typename proto::result_of::value<
		    typename proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, multivector_expr_context &ctx) const {
	    ctx.os << "func_1_" << ++ctx.fun_idx << "( ";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr),
		    do_eval(ctx)
		    );
	    ctx.os << " )";
	}
    };

    template <typename Expr>
    struct eval<Expr, proto::tag::terminal> {
	typedef void result_type;

	template <typename T, size_t M, bool own>
	void operator()(const multivector<T,M,own> &term, multivector_expr_context &ctx) const {
	    static_assert(M == N, "Wrong number of components in a multivector");

	    ctx.os << "prm_" << C + 1 << "_" << ++ctx.prm_idx << "[idx]";
	}

	template <typename Term>
	void operator()(const Term &term, multivector_expr_context &ctx) const {
	    typedef typename proto::result_of::value<Term>::type term_type;

	    static_assert(
		    number_of_components<term_type>::value == 1 ||
		    number_of_components<term_type>::value == N,
		    "Wrong number of components in a multiscalar"
		    );

	    ctx.prm_idx++;

	    if (number_of_components<term_type>::value > 1) {
		ctx.os << "prm_" << C + 1 << "_" << ctx.prm_idx;
	    } else {
		ctx.os << "prm_1_" << ctx.prm_idx;
	    }
	}
    };
};


template <size_t N, size_t C>
struct declare_multiex_parameter {
    std::ostream &os;
    mutable int prm_idx;

    declare_multiex_parameter(std::ostream &os) : os(os), prm_idx(0) { }

    template <typename T, size_t M, bool own>
    void operator()(const multivector<T, M, own> &term) const {
	static_assert(M == N, "Wrong number of components in a multivector");

	os << ",\n\tglobal " << type_name<T>() << " *prm_"
	   << C + 1 << "_" << ++prm_idx;
    }

    template <typename Term>
    void operator()(const Term &term) const {
	typedef typename proto::result_of::value<Term>::type term_type;

	typedef
	    typename component<
		C, typename proto::result_of::value<Term>::type
		>::type
	    component_type;

	static_assert(
		number_of_components<term_type>::value == 1 ||
		number_of_components<term_type>::value == N,
		"Wrong number of components in a multiscalar"
		);

	prm_idx++;

	if (number_of_components<term_type>::value > 1) {
	    os << ",\n\t"
	       << type_name< component_type >()
	       << " prm_" << C + 1 << "_" << prm_idx;
	} else if (C == 0) {
	    os << ",\n\t"
	       << type_name< component_type >()
	       << " prm_1_" << prm_idx;
	}
    }
};

template <size_t I, size_t N, class Expr>
typename std::enable_if<I == N>::type
mv_param_list_loop(const Expr &expr, std::ostream &os) { }

template <size_t I, size_t N, class Expr>
typename std::enable_if<I < N>::type
mv_param_list_loop(const Expr &expr, std::ostream &os) {
    extract_terminals()( expr,
	    declare_multiex_parameter<N, I>(os)
	    );

    mv_param_list_loop<I+1, N, Expr>(expr, os);
}

template <size_t N, class Expr>
void build_param_list(const Expr &expr, std::ostream &os) {
    mv_param_list_loop<0, N, Expr>(expr, os);
}




template <size_t N, size_t C>
struct set_multiex_argument {
    cl::Kernel &krn;
    uint dev, &pos;

    set_multiex_argument(cl::Kernel &krn, uint dev, uint &pos)
	: krn(krn), dev(dev), pos(pos) {}

    template <typename T, size_t M, bool own>
    void operator()(const multivector<T, M, own> &term) const {
	static_assert(M == N, "Wrong number of components in a multivector");
	krn.setArg(pos++, term(C)(dev));
    }

    template <typename Term>
    void operator()(const Term &term) const {
	typedef typename proto::result_of::value<Term>::type term_type;

	static_assert(
		number_of_components<term_type>::value == 1 ||
		number_of_components<term_type>::value == N,
		"Wrong number of components in a multiscalar"
		);

	if ((number_of_components<term_type>::value > 1) || (C == 0))
	    krn.setArg(pos++, get<C>(proto::value(term)));
    }
};

template <size_t I, size_t N, class Expr>
typename std::enable_if<I == N>::type
mv_kernel_args_loop(const Expr &expr, cl::Kernel &krn, uint d, uint &pos) { }

template <size_t I, size_t N, class Expr>
typename std::enable_if<I < N>::type
mv_kernel_args_loop(const Expr &expr, cl::Kernel &krn, uint d, uint &pos) {
    extract_terminals()( expr,
	    set_multiex_argument<N, I>(krn, d, pos)
	    );

    mv_kernel_args_loop<I+1, N, Expr>(expr, krn, d, pos);
}

template <size_t N, class Expr>
void set_kernel_args(const Expr &expr, cl::Kernel &krn, uint d, uint &pos) {
    mv_kernel_args_loop<0, N, Expr>(expr, krn, d, pos);
}



template <size_t C>
struct extract_component
    : boost::proto::transform < extract_component<C> >
{
    template<typename Expr, typename Unused1, typename Unused2>
    struct impl : boost::proto::transform_impl<Expr, Unused1, Unused2>
    {
	typedef
	    typename vex::component<C,
		typename boost::remove_const<
		    typename boost::remove_reference<
			typename impl::expr
		    >::type
		>::type
	    >::type result_type;

	result_type operator ()(
              typename impl::expr_param term
            , typename impl::state_param
            , typename impl::data_param) const
        {
	    using namespace vex;
	    using namespace std;

	    return get<C>(term);
        }

    };
};

template <size_t C>
struct extract_subexpression
    : boost::proto::or_ <
	boost::proto::when <
	    boost::proto::terminal<boost::proto::_>,
	    extract_component<C>
	> ,
	boost::proto::function<
	    boost::proto::_,
	    boost::proto::vararg< extract_subexpression<C> >
	> ,
	boost::proto::when <
	    boost::proto::nary_expr<
		boost::proto::_,
		boost::proto::vararg< extract_subexpression<C> >
	    >
	>
    >
{};








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

template <typename T, size_t N, bool own>
struct multivector
    : multivector_expression<
	typename proto::terminal< multivector_terminal >::type
      >
{
    public:
	typedef std::integral_constant<size_t, N> dim;

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
	    static_assert(own, "Wrong constructor for referenced-type multivector");
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
	    static_assert(own, "Wrong constructor for referenced-type multivector");
	    static_assert(N > 0, "What's the point?");

	    for(uint i = 0; i < N; i++)
		vec[i].reset(new vex::vector<T>(
			queue, size, host ? host + i * size : 0, flags
			) );
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
	const multivector& operator=(const Expr& expr) {
	    const std::vector<cl::CommandQueue> &queue = vec[0]->queue_list();

	    for(auto q = queue.begin(); q != queue.end(); q++) {
		cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();
		cl::Device  device  = q->getInfo<CL_QUEUE_DEVICE>();

		if (!exdata<Expr>::compiled[context()]) {

		    std::ostringstream kernel_name;
		    kernel_name << "multi_";
		    vector_name_context name_ctx(kernel_name);
		    proto::eval(proto::as_child(expr), name_ctx);

		    std::ostringstream kernel;
		    kernel << standard_kernel_header;

		    extract_user_functions()(
			    proto::as_child(expr),
			    declare_user_function(kernel)
			    );

		    kernel << "kernel void " << kernel_name.str()
		           << "(\n\t" << type_name<size_t>() << " n";

		    for(size_t i = 0; i < N; )
			kernel << ",\n\tglobal " << type_name<T>()
			       << " *res_" << ++i;

		    build_param_list<N>(proto::as_child(expr), kernel);

		    kernel <<
			"\n)\n{\n\t"
			"for(size_t idx = get_global_id(0); idx < n; "
			"idx += get_global_size(0)) {\n";

		    build_expr_list(proto::as_child(expr), kernel);

		    kernel << "\t}\n}\n";

#ifdef VEXCL_SHOW_KERNELS
		    std::cout << kernel.str() << std::endl;
#endif

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
			    proto::as_child(expr),
			    exdata<Expr>::kernel[context()],
			    d,
			    pos
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

#ifdef VEXCL_VARIADIC_TEMPLATES
	/// Multi-expression assignments.
	template <class... Expr>
	typename std::enable_if<N == sizeof...(Expr), const multivector& >::type
	operator=(const std::tuple<Expr...> &expr) {
#if 0
	    assign_components assign(vec);
	    for_each(expr, assign);
#else
	    typedef std::tuple<Expr...> MultiExpr;

	    const std::vector<cl::CommandQueue> &queue = vec[0]->queue_list();

	    for(auto q = queue.begin(); q != queue.end(); q++) {
		cl::Context context = qctx(*q);
		cl::Device  device  = qdev(*q);

		if (!exdata<MultiExpr>::compiled[context()]) {
		    std::ostringstream kernel;

		    kernel << standard_kernel_header;

		    {
			get_header f(kernel);
			for_each(expr, f);
		    }

		    kernel <<
			"kernel void multi_expr_tuple(\n"
			"\t" << type_name<size_t>() << " n";

		    for(uint i = 1; i <= N; i++)
			kernel << ",\n\tglobal " << type_name<T>() << " *res_" << i;

		    {
			get_params f(kernel);
			for_each(expr, f);
		    }

		    kernel <<
			"\n)\n{\n\t"
			"for(size_t idx = get_global_id(0); idx < n; "
			"idx += get_global_size(0)) {\n";

		    {
			get_expressions f(kernel);
			for_each(expr, f);
		    }

		    kernel << "\n";

		    for(uint i = 1; i <= N; i++)
			kernel << "\t\tres_" << i << "[idx] = buf_" << i << ";\n";

		    kernel << "\t}\n}\n";

#ifdef VEXCL_SHOW_KERNELS
		    std::cout << kernel.str() << std::endl;
#endif

		    auto program = build_sources(context, kernel.str());

		    exdata<MultiExpr>::kernel[context()]   = cl::Kernel(program, "multi_expr_tuple");
		    exdata<MultiExpr>::compiled[context()] = true;
		    exdata<MultiExpr>::wgsize[context()]   = kernel_workgroup_size(
			    exdata<MultiExpr>::kernel[context()], device);

		}
	    }

	    for(uint d = 0; d < queue.size(); d++) {
		if (size_t psize = vec[0]->part_size(d)) {
		    cl::Context context = qctx(queue[d]);
		    cl::Device  device  = qdev(queue[d]);

		    size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
			alignup(psize, exdata<MultiExpr>::wgsize[context()]) :
			device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * exdata<MultiExpr>::wgsize[context()] * 4;

		    uint pos = 0;
		    exdata<MultiExpr>::kernel[context()].setArg(pos++, psize);

		    for(uint i = 0; i < N; i++)
			exdata<MultiExpr>::kernel[context()].setArg(pos++, vec[i]->operator()(d));

		    {
			set_arguments f(exdata<MultiExpr>::kernel[context()], d, pos);
			for_each(expr, f);
		    }

		    queue[d].enqueueNDRangeKernel(
			    exdata<MultiExpr>::kernel[context()],
			    cl::NullRange,
			    g_size, exdata<MultiExpr>::wgsize[context()]
			    );
		}
	    }
#endif

	    return *this;
	}
#endif

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
	    expr_list_loop(const Expr &expr, std::ostream &os) { }

	template <size_t I, class Expr>
	    typename std::enable_if<I < N>::type
	    expr_list_loop(const Expr &expr, std::ostream &os) {
		multivector_expr_context<N, I> ctx(os);
		os << "\t\tres_" << I + 1 << "[idx] = ";
		proto::eval(expr, ctx);
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

#ifdef VEXCL_VARIADIC_TEMPLATES
	struct get_header {
	    std::ostream &os;
	    mutable int cmp_idx;

	    get_header(std::ostream &os) : os(os), cmp_idx(0) {}

	    template <class Expr>
	    void operator()(const Expr &expr) const {
		extract_user_functions()(
			proto::as_expr(expr),
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
			proto::as_child(expr),
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
		proto::eval(proto::as_child(expr), ctx);
		os << ";\n";
	    }
	};

	struct set_arguments {
	    cl::Kernel &krn;
	    uint d, &pos;

	    set_arguments(cl::Kernel &krn, uint d, uint &pos)
		: krn(krn), d(d), pos(pos) {}

	    template <class Expr>
	    void operator()(const Expr &expr) const {
		extract_terminals()(
			proto::as_child(expr),
			set_expression_argument(krn, d, pos)
			);
	    }
	};
#endif

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

#ifdef VEXCL_VARIADIC_TEMPLATES
/// Ties several vex::vectors into a multivector.
/**
 * The following example results in a single kernel:
 * \code
 * vex::vector<double> x(ctx.queue(), 1024);
 * vex::vector<double> y(ctx.queue(), 1024);
 *
 * vex::tie(x,y) = std::make_tuple(
 *			x + y,
 *			y - x
 *			);
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
#endif

} // namespace vex

namespace boost { namespace fusion { namespace traits {
template <class T, size_t N, bool own>
struct is_sequence< vex::multivector<T, N, own> > : std::false_type
{};
} } }

#endif
