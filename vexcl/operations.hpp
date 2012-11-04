#ifndef VEXCL_OPERATIONS_HPP
#define VEXCL_OPERATIONS_HPP

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
 * \file   operations.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Set of operations to be used in vector expressions.
 */

#ifdef WIN32
#  pragma warning(push)
#  define NOMINMAX
#endif

#include <boost/proto/proto.hpp>
#include <boost/mpl/max.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

/// \cond INTERNAL

struct builtin_function {};
struct user_function {};

// TODO compare compilation speed with boost::proto::switch_

//--- Standard grammar (no terminals) ---------------------------------------
#define BUILTIN_OPERATIONS(grammar) \
    boost::proto::or_< \
	boost::proto::unary_plus< grammar >, \
	boost::proto::negate< grammar >, \
	boost::proto::logical_not< grammar >, \
	boost::proto::pre_inc< grammar >, \
	boost::proto::pre_dec< grammar >, \
	boost::proto::post_inc< grammar >, \
	boost::proto::post_dec< grammar > \
    >, \
    boost::proto::or_< \
	boost::proto::or_< \
	    boost::proto::plus          < grammar, grammar >, \
	    boost::proto::minus         < grammar, grammar >, \
	    boost::proto::multiplies    < grammar, grammar >, \
	    boost::proto::divides       < grammar, grammar >, \
	    boost::proto::modulus       < grammar, grammar >, \
	    boost::proto::shift_left    < grammar, grammar >, \
	    boost::proto::shift_right   < grammar, grammar > \
	>, \
	boost::proto::or_< \
	    boost::proto::less          < grammar, grammar >, \
	    boost::proto::greater       < grammar, grammar >, \
	    boost::proto::less_equal    < grammar, grammar >, \
	    boost::proto::greater_equal < grammar, grammar >, \
	    boost::proto::equal_to      < grammar, grammar >, \
	    boost::proto::not_equal_to  < grammar, grammar > \
	>, \
	boost::proto::or_< \
	    boost::proto::logical_and   < grammar, grammar >, \
	    boost::proto::logical_or    < grammar, grammar > \
	>, \
	boost::proto::or_< \
	    boost::proto::bitwise_and   < grammar, grammar >, \
	    boost::proto::bitwise_or    < grammar, grammar >, \
	    boost::proto::bitwise_xor   < grammar, grammar > \
	> \
    >, \
    boost::proto::function< \
	boost::proto::terminal< \
	    boost::proto::convertible_to<builtin_function> \
	>, \
	boost::proto::vararg<grammar> \
    >

#define USER_FUNCTIONS(grammar) \
    boost::proto::function< \
	boost::proto::terminal< \
	    boost::proto::convertible_to<user_function> \
	>, \
	boost::proto::vararg<grammar> \
    >

//--- Builtin function ------------------------------------------------------

#define BUILTIN_FUNCTION_1(func) \
struct func##_func : builtin_function { \
    static const char* name() { \
	return #func; \
    } \
}; \
template <typename Arg> \
typename boost::proto::result_of::make_expr< \
    boost::proto::tag::function, \
    func##_func, \
    const Arg& \
>::type const \
func(const Arg &arg) { \
    return boost::proto::make_expr<boost::proto::tag::function>( \
	    func##_func(), \
	    boost::ref(arg) \
	    ); \
}

#define BUILTIN_FUNCTION_2(func) \
struct func##_func : builtin_function { \
    static const char* name() { \
	return #func; \
    } \
}; \
template <typename Arg1, typename Arg2> \
typename boost::proto::result_of::make_expr< \
    boost::proto::tag::function, \
    func##_func, \
    const Arg1&, \
    const Arg2& \
>::type const \
func(const Arg1 &arg1, const Arg2 &arg2) { \
    return boost::proto::make_expr<boost::proto::tag::function>( \
	    func##_func(), \
	    boost::ref(arg1), \
	    boost::ref(arg2) \
	    ); \
}

#define BUILTIN_FUNCTION_3(func) \
struct func##_func : builtin_function { \
    static const char* name() { \
	return #func; \
    } \
}; \
template <typename Arg1, typename Arg2, typename Arg3> \
typename boost::proto::result_of::make_expr< \
    boost::proto::tag::function, \
    func##_func, \
    const Arg1&, \
    const Arg2&, \
    const Arg3& \
>::type const \
func(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3) { \
    return boost::proto::make_expr<boost::proto::tag::function>( \
	    func##_func(), \
	    boost::ref(arg1), \
	    boost::ref(arg2), \
	    boost::ref(arg3) \
	    ); \
}

BUILTIN_FUNCTION_1(acos);
BUILTIN_FUNCTION_1(acosh);
BUILTIN_FUNCTION_1(acospi);
BUILTIN_FUNCTION_1(asin);
BUILTIN_FUNCTION_1(asinh);
BUILTIN_FUNCTION_1(asinpi);
BUILTIN_FUNCTION_1(atan);
BUILTIN_FUNCTION_2(atan2);
BUILTIN_FUNCTION_1(atanh);
BUILTIN_FUNCTION_1(atanpi);
BUILTIN_FUNCTION_2(atan2pi);
BUILTIN_FUNCTION_1(cbrt);
BUILTIN_FUNCTION_1(ceil);
BUILTIN_FUNCTION_2(copysign);
BUILTIN_FUNCTION_1(cos);
BUILTIN_FUNCTION_1(cosh);
BUILTIN_FUNCTION_1(cospi);
BUILTIN_FUNCTION_1(erfc);
BUILTIN_FUNCTION_1(erf);
BUILTIN_FUNCTION_1(exp);
BUILTIN_FUNCTION_1(exp2);
BUILTIN_FUNCTION_1(exp10);
BUILTIN_FUNCTION_1(expm1);
BUILTIN_FUNCTION_1(fabs);
BUILTIN_FUNCTION_2(fdim);
BUILTIN_FUNCTION_1(floor);
BUILTIN_FUNCTION_3(fma);
BUILTIN_FUNCTION_2(fmax);
BUILTIN_FUNCTION_2(fmin);
BUILTIN_FUNCTION_2(fmod);
BUILTIN_FUNCTION_2(fract);
BUILTIN_FUNCTION_2(frexp);
BUILTIN_FUNCTION_2(hypot);
BUILTIN_FUNCTION_1(ilogb);
BUILTIN_FUNCTION_2(ldexp);
BUILTIN_FUNCTION_1(lgamma);
BUILTIN_FUNCTION_2(lgamma_r);
BUILTIN_FUNCTION_1(log);
BUILTIN_FUNCTION_1(log2);
BUILTIN_FUNCTION_1(log10);
BUILTIN_FUNCTION_1(log1p);
BUILTIN_FUNCTION_1(logb);
BUILTIN_FUNCTION_3(mad);
BUILTIN_FUNCTION_2(maxmag);
BUILTIN_FUNCTION_2(minmag);
BUILTIN_FUNCTION_2(modf);
BUILTIN_FUNCTION_1(nan);
BUILTIN_FUNCTION_2(nextafter);
BUILTIN_FUNCTION_2(pow);
BUILTIN_FUNCTION_2(pown);
BUILTIN_FUNCTION_2(powr);
BUILTIN_FUNCTION_2(remainder);
BUILTIN_FUNCTION_3(remquo);
BUILTIN_FUNCTION_1(rint);
BUILTIN_FUNCTION_2(rootn);
BUILTIN_FUNCTION_1(round);
BUILTIN_FUNCTION_1(rsqrt);
BUILTIN_FUNCTION_1(sin);
BUILTIN_FUNCTION_2(sincos);
BUILTIN_FUNCTION_1(sinh);
BUILTIN_FUNCTION_1(sinpi);
BUILTIN_FUNCTION_1(sqrt);
BUILTIN_FUNCTION_1(tan);
BUILTIN_FUNCTION_1(tanh);
BUILTIN_FUNCTION_1(tanpi);
BUILTIN_FUNCTION_1(tgamma);
BUILTIN_FUNCTION_1(trunc);

#undef BUILTIN_FUNCTION_1
#undef BUILTIN_FUNCTION_2
#undef BUILTIN_FUNCTION_3

#ifndef BOOST_NO_VARIADIC_TEMPLATES

//--- User Function ---------------------------------------------------------
template <const char *body, class T>
struct UserFunction {};

template<const char *body, class RetType, class... ArgType>
struct UserFunction<body, RetType(ArgType...)> : user_function
{
    template <class... Arg>
    typename boost::proto::result_of::make_expr<
	boost::proto::tag::function,
	UserFunction,
	const Arg&...
    >::type const
    operator()(const Arg&... arg) {
	return boost::proto::make_expr<boost::proto::tag::function>(
		UserFunction(), boost::ref(arg)...
		);
    }
    
    static void define(std::ostream &os, const std::string &name) {
	os << type_name<RetType>() << " " << name << "(";
	show_arg<ArgType...>(os, 1);
	os << "\n)\n{\n" << body << "\n}\n\n";
    }

    template <class Head>
    static void show_arg(std::ostream &os, uint pos) {
	if (pos > 1) os << ",";
	os << "\n\t" << type_name<Head>() << " prm" << pos;
    }

    template <class Head, class... Tail>
    static typename std::enable_if<sizeof...(Tail), void>::type
    show_arg(std::ostream &os, uint pos) {
	if (pos > 1) os << ",";
	show_arg<Tail...>(
		os << "\n\t" << type_name<Head>() << " prm" << pos, pos + 1
		);
    }
};

#else

//--- User Function ---------------------------------------------------------
template <const char *body, class T>
struct UserFunction {};

template<const char *body, class RetType, class Arg1Type>
struct UserFunction<body, RetType(Arg1Type)> : user_function
{
    template <class Arg1>
    typename boost::proto::result_of::make_expr<
	boost::proto::tag::function,
	UserFunction,
	const Arg1&
    >::type const
    operator()(const Arg1& arg1) {
	return boost::proto::make_expr<boost::proto::tag::function>(
		UserFunction(), boost::ref(arg1)
		);
    }
    
    static void define(std::ostream &os, const std::string &name) {
	os << type_name<RetType>() << " " << name << "("
	   << "\n\t" << type_name<Arg1Type>() << " prm1"
	   << "\n)\n{\n" << body << "\n}\n\n";
    }
};

template<const char *body, class RetType,
    class Arg1Type,
    class Arg2Type
    >
struct UserFunction<body, RetType(
	Arg1Type,
	Arg2Type
	)> : user_function
{
    template <
	class Arg1,
	class Arg2
	>
    typename boost::proto::result_of::make_expr<
	boost::proto::tag::function,
	UserFunction,
	const Arg1&,
	const Arg2&
    >::type const
    operator()(
	    const Arg1& arg1,
	    const Arg2& arg2
	    )
    {
	return boost::proto::make_expr<boost::proto::tag::function>(
		UserFunction(),
		boost::ref(arg1),
		boost::ref(arg2)
		);
    }
    
    static void define(std::ostream &os, const std::string &name) {
	os << type_name<RetType>() << " " << name << "("
	   << "\n\t" << type_name<Arg1Type>() << " prm1,"
	   << "\n\t" << type_name<Arg2Type>() << " prm2"
	   << "\n)\n{\n" << body << "\n}\n\n";
    }
};

template<const char *body, class RetType,
    class Arg1Type,
    class Arg2Type,
    class Arg3Type
    >
struct UserFunction<body, RetType(
	Arg1Type,
	Arg2Type,
	Arg3Type
	)> : user_function
{
    template <
	class Arg1,
	class Arg2,
	class Arg3
	>
    typename boost::proto::result_of::make_expr<
	boost::proto::tag::function,
	UserFunction,
	const Arg1&,
	const Arg2&,
	const Arg3&
    >::type const
    operator()(
	    const Arg1& arg1,
	    const Arg2& arg2,
	    const Arg3& arg3
	    )
    {
	return boost::proto::make_expr<boost::proto::tag::function>(
		UserFunction(),
		boost::ref(arg1),
		boost::ref(arg2),
		boost::ref(arg3)
		);
    }
    
    static void define(std::ostream &os, const std::string &name) {
	os << type_name<RetType>() << " " << name << "("
	   << "\n\t" << type_name<Arg1Type>() << " prm1,"
	   << "\n\t" << type_name<Arg2Type>() << " prm2,"
	   << "\n\t" << type_name<Arg3Type>() << " prm3"
	   << "\n)\n{\n" << body << "\n}\n\n";
    }
};

#endif

//---------------------------------------------------------------------------
// Expression Transforms
//---------------------------------------------------------------------------
template <class Context>
struct do_eval {
    Context &ctx;

    do_eval(Context &ctx) : ctx(ctx) {}

    template <class Expr>
    void operator()(const Expr &expr) const {
	boost::proto::eval(expr, ctx);
    }
};

struct process_terminal
    : boost::proto::transform < process_terminal >
{
    template<typename Expr, typename Unused1, typename Unused2>
    struct impl : boost::proto::transform_impl<Expr, Unused1, Unused2>
    {
	typedef typename impl::expr_param result_type;

        result_type operator ()(
              typename impl::expr_param term
            , typename impl::state_param process
            , typename impl::data_param) const
        {
	    process(term);
	    return term;
        }
    };
};

struct extract_terminals
    : boost::proto::or_ <
	boost::proto::when <
	    boost::proto::terminal<boost::proto::_>,
	    process_terminal
	> ,
	boost::proto::function<
	    boost::proto::_,
	    boost::proto::vararg< extract_terminals >
	> ,
	boost::proto::when <
	    boost::proto::nary_expr<
		boost::proto::_,
		boost::proto::vararg< extract_terminals >
	    >
	>
    >
{};

struct extract_user_functions
    : boost::proto::or_ <
	boost::proto::terminal<boost::proto::_>,
	boost::proto::when <
	    boost::proto::function<
		boost::proto::terminal <
		    boost::proto::convertible_to<vex::user_function>
		>,
		boost::proto::vararg< extract_user_functions >
	    >,
	    process_terminal(boost::proto::_child0)
	> ,
	boost::proto::when <
	    boost::proto::nary_expr<
		boost::proto::_,
		boost::proto::vararg< extract_user_functions >
	    >
	>
    >
{};


//---------------------------------------------------------------------------
// Additive transforms
//---------------------------------------------------------------------------

struct additive_vector_transform {};

struct additive_vector_transform_grammar
    : boost::proto::or_<
        boost::proto::terminal< additive_vector_transform >,
	boost::proto::plus<
	    additive_vector_transform_grammar,
	    additive_vector_transform_grammar
	>,
	boost::proto::minus<
	    additive_vector_transform_grammar,
	    additive_vector_transform_grammar
	>,
	boost::proto::negate<
	    additive_vector_transform_grammar
	>
      >
{};

//---------------------------------------------------------------------------
// Elementwise vector operations
//---------------------------------------------------------------------------

struct vector_terminal {};
template <typename T> struct vector;

//--- Vector grammar --------------------------------------------------------
struct vector_expr_grammar
    : boost::proto::or_<
	  boost::proto::or_<
	      boost::proto::terminal< vector_terminal >,
	      boost::proto::and_<
	          boost::proto::terminal< boost::proto::_ >,
		  boost::proto::if_< std::is_arithmetic< boost::proto::_value >() >
	      >
          >,
	  BUILTIN_OPERATIONS(vector_expr_grammar),
	  USER_FUNCTIONS(vector_expr_grammar)
      >
{};

struct vector_full_grammar
    : boost::proto::or_<
	vector_expr_grammar,
	boost::proto::terminal< additive_vector_transform >,
	boost::proto::plus< vector_full_grammar, vector_full_grammar >,
	boost::proto::minus< vector_full_grammar, vector_full_grammar >,
	boost::proto::negate< vector_full_grammar >
      >
{};

struct extract_vector_expressions
    : boost::proto::or_<
	  boost::proto::when<vector_expr_grammar, boost::proto::_>,
	  boost::proto::when<
	     boost::proto::plus<
		vector_full_grammar,
		additive_vector_transform_grammar
	     >,
	     extract_vector_expressions(boost::proto::_left)
	  >,
	  boost::proto::when<
	     boost::proto::plus<
		additive_vector_transform_grammar,
		vector_full_grammar
	     >,
	     extract_vector_expressions(boost::proto::_right)
	  >,
	  boost::proto::when<
	     boost::proto::minus<
		vector_full_grammar,
		additive_vector_transform_grammar
	     >,
	     extract_vector_expressions(boost::proto::_left)
	  >,
	  boost::proto::when<
	     boost::proto::minus<
		additive_vector_transform_grammar,
		vector_full_grammar
	     >,
	     boost::proto::_make_negate(
		    extract_vector_expressions(boost::proto::_right)
		    )
	  >,
	  boost::proto::when<
	     boost::proto::binary_expr<boost::proto::_,
		extract_vector_expressions,
		extract_vector_expressions
	     >
	  >
      >
{};

struct extract_additive_vector_transforms
    : boost::proto::or_<
	  boost::proto::when<additive_vector_transform_grammar, boost::proto::_>
	, boost::proto::when<
	     boost::proto::plus<vector_full_grammar, vector_expr_grammar >,
	     extract_additive_vector_transforms(boost::proto::_left)
	  >
	, boost::proto::when<
	     boost::proto::plus< vector_expr_grammar, vector_full_grammar >,
	     extract_additive_vector_transforms(boost::proto::_right)
	  >
	, boost::proto::when<
	     boost::proto::minus<vector_full_grammar, vector_expr_grammar >,
	     extract_additive_vector_transforms(boost::proto::_left)
	  >
	, boost::proto::when<
	     boost::proto::minus<vector_expr_grammar, vector_full_grammar >,
	     boost::proto::_make_negate(
		     extract_additive_vector_transforms(boost::proto::_right)
		     )
	  >
	, boost::proto::when<
	     boost::proto::binary_expr<boost::proto::_,
		extract_additive_vector_transforms,
		extract_additive_vector_transforms
	     >
	  >
      >
{};

template <class Expr>
struct vector_expression;

struct vector_domain
    : boost::proto::domain<
	boost::proto::generator<vector_expression>,
	vector_full_grammar
	>
{ };

template <class Expr>
struct vector_expression
    : boost::proto::extends< Expr, vector_expression<Expr>, vector_domain>
{
    vector_expression(const Expr &expr = Expr())
	: boost::proto::extends< Expr, vector_expression<Expr>, vector_domain>(expr) {}
};

//--- Vector contexts and transform helpers ---------------------------------

// Builds kernel name for a vector expression.
struct vector_name_context {
    std::ostream &os;

    vector_name_context(std::ostream &os) : os(os) {}

    // Any expression except function or terminal is only interesting for its
    // children:
    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {
	typedef void result_type;

	void operator()(const Expr &expr, vector_name_context &ctx) const {
	    ctx.os << Tag() << "_";
	    boost::fusion::for_each(expr, do_eval<vector_name_context>(ctx));
	}
    };

    // We only need to look at parameters of a function:
    template <typename Expr>
    struct eval<Expr, boost::proto::tag::function> {
	typedef void result_type;

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		builtin_function,
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, vector_name_context &ctx) const {
	    ctx.os << boost::proto::value(boost::proto::child_c<0>(expr)).name() << "_";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr),
		    do_eval<vector_name_context>(ctx)
		    );
	}

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		user_function,
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, vector_name_context &ctx) const {
	    ctx.os << "func" << boost::fusion::size(expr) - 1 <<  "_";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr),
		    do_eval<vector_name_context>(ctx)
		    );
	}
    };

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::terminal> {
	typedef void result_type;

	void operator()(const Expr &expr, vector_name_context &ctx) const {
	    ctx.os << "term_";
	}
    };
};


// Builds textual representation for a vector expression.
struct vector_expr_context {
    std::ostream &os;
    int cmp_idx, prm_idx, fun_idx;

    vector_expr_context(std::ostream &os, int cmp_idx = 1)
	: os(os), cmp_idx(cmp_idx), prm_idx(0), fun_idx(0) {}

    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {};

#define BINARY_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, vector_expr_context &ctx) const { \
	    ctx.os << "( "; \
	    boost::proto::eval(boost::proto::left(expr), ctx); \
	    ctx.os << " " #the_op " "; \
	    boost::proto::eval(boost::proto::right(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, vector_expr_context &ctx) const { \
	    ctx.os << "( " #the_op "( "; \
	    boost::proto::eval(boost::proto::child(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, vector_expr_context &ctx) const { \
	    ctx.os << "( ( "; \
	    boost::proto::eval(boost::proto::child(expr), ctx); \
	    ctx.os << " )" #the_op " )"; \
	} \
    }

    UNARY_POST_OPERATION(post_inc, ++);
    UNARY_POST_OPERATION(post_dec, --);

#undef UNARY_POST_OPERATION

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::function> {
	typedef void result_type;

	struct do_eval {
	    mutable int pos;
	    vector_expr_context &ctx;

	    do_eval(vector_expr_context &ctx) : pos(0), ctx(ctx) {}

	    template <typename Arg>
	    void operator()(const Arg &arg) const {
		if (pos++) ctx.os << ", ";
		boost::proto::eval(arg, ctx);
	    }
	};

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		builtin_function,
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, vector_expr_context &ctx) const {
	    ctx.os << boost::proto::value(boost::proto::child_c<0>(expr)).name() << "( ";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr), do_eval(ctx)
		    );
	    ctx.os << " )";
	}

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		user_function,
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, vector_expr_context &ctx) const {
	    ctx.os << "func_" << ctx.cmp_idx << "_" << ++ctx.fun_idx << "( ";
	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr), do_eval(ctx)
		    );
	    ctx.os << " )";
	}
    };

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::terminal> {
	typedef void result_type;

	template <typename T>
	void operator()(const vector<T> &term, vector_expr_context &ctx) const {
	    ctx.os << "prm_" << ctx.cmp_idx << "_" << ++ctx.prm_idx << "[idx]";
	}

	template <typename Term>
	void operator()(const Term &term, vector_expr_context &ctx) const {
	    ctx.os << "prm_" << ctx.cmp_idx << "_" << ++ctx.prm_idx;
	}
    };
};

struct declare_user_function {
    std::ostream &os;
    int cmp_idx;
    mutable int fun_idx;

    declare_user_function(std::ostream &os, int cmp_idx = 1)
	: os(os), cmp_idx(cmp_idx), fun_idx(0) {}

	template <class FunCall>
	void operator()(const FunCall &expr) const {
	    std::ostringstream name;
	    name << "func_" << cmp_idx << "_" << ++fun_idx;

	    // Output function definition and continue with parameters.
	    boost::proto::value(expr).define(os, name.str());
	}
};

struct declare_expression_parameter {
    std::ostream &os;
    int cmp_idx;
    mutable int prm_idx;

    declare_expression_parameter(std::ostream &os, int cmp_idx = 1)
    : os(os), cmp_idx(cmp_idx), prm_idx(0) {}

    template <typename T>
    void operator()(const vector<T> &term) const {
	os << ",\n\tglobal " << type_name<T>() << " *prm_"
	   << cmp_idx << "_" << ++prm_idx;
    }

    template <typename Term>
    void operator()(const Term &term) const {
	os << ",\n\t"
	   << type_name< typename boost::proto::result_of::value<Term>::type >()
	   << " prm_" << cmp_idx << "_" << ++prm_idx;
    }
};

struct set_expression_argument {
    cl::Kernel &krn;
    uint dev, &pos;

    set_expression_argument(cl::Kernel &krn, uint dev, uint &pos)
	: krn(krn), dev(dev), pos(pos) {}

    template <typename T>
    void operator()(const vector<T> &term) const {
	krn.setArg(pos++, term(dev));
    }

    template <typename Term>
    void operator()(const Term &term) const {
	krn.setArg(pos++, boost::proto::value(term));
    }
};

struct get_expression_properties {
    mutable std::vector<cl::CommandQueue> const* queue;
    mutable std::vector<size_t> const* part;
    mutable size_t size;

    get_expression_properties() : queue(0), part(0), size(0) {}

    size_t part_size(uint d) const {
	return part ?
	    part->operator[](d + 1) - part->operator[](d) :
	    0;
    }

    template <typename T>
    void operator()(const vector<T> &term) const {
	if (!queue) {
	    queue = &( term.queue_list() );
	    part  = &( term.partition() );
	    size  = term.size();
	}
    }

    template <typename Term>
    void operator()(const Term &term) const { }
};

//---------------------------------------------------------------------------
// Elementwise multi-vector operations
//---------------------------------------------------------------------------

//--- Scalars and helper types/functions used in multivector expressions ----
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

#ifndef BOOST_NO_VARIADIC_TEMPLATES

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

struct mutltiex_dimension
        : boost::proto::or_ <
            boost::proto::when <
		boost::proto::terminal< boost::proto::_ >,
		number_of_components<boost::proto::_>()
	    > ,
	    boost::proto::when <
		boost::proto::nary_expr<boost::proto::_, boost::proto::vararg<boost::proto::_> >,
		boost::proto::fold<boost::proto::_,
		    boost::mpl::size_t<0>(),
		    boost::mpl::max<mutltiex_dimension, boost::proto::_state>()>()
	    >
        >
{};

template <size_t I, class T>
struct component< I, T,
    typename std::enable_if<
	!is_multiscalar<T>::value &&
	is_multiscalar<
	    typename boost::proto::result_of::value<
		typename boost::proto::result_of::as_expr<T>::type
	    >::type
	>::value >::type
    >
{
    typedef typename boost::proto::result_of::value<
		typename boost::proto::result_of::as_expr<T>::type
	    >::type value_type;

    typedef typename boost::proto::result_of::as_child<
	typename component<I, value_type>::type
	>::type type;
};

template <size_t I, typename T>
inline const
typename std::enable_if<
    !is_multiscalar<T>::value &&
    is_multiscalar<
	typename boost::proto::result_of::value<
	    typename boost::proto::result_of::as_expr<T>::type
	>::type
    >::value,
    typename component<I, T>::type
>::type
get(const T &t) {
    return boost::proto::as_child(get<I>(boost::proto::value(t)));
}

//--- Multivector grammar ---------------------------------------------------

struct multivector_expr_grammar
    : boost::proto::or_<
	  boost::proto::or_<
	      boost::proto::terminal< multivector_terminal >,
	      boost::proto::and_<
	          boost::proto::terminal< boost::proto::_ >,
		  boost::proto::if_< is_multiscalar< boost::proto::_value >() >
	      >
          >,
	  BUILTIN_OPERATIONS(multivector_expr_grammar),
	  USER_FUNCTIONS(multivector_expr_grammar)
      >
{};

template <class Expr>
struct multivector_expression;

struct multivector_domain
    : boost::proto::domain<
	boost::proto::generator<multivector_expression>,
	multivector_expr_grammar
      >
{ };

template <class Expr>
struct multivector_expression
    : boost::proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
{
    multivector_expression(const Expr &expr = Expr())
	: boost::proto::extends< Expr, multivector_expression<Expr>, multivector_domain>(expr) {}
};

//--- Multivector contexts and transform helpers ----------------------------

// Builds textual representation for a multi-vector expression.
template <size_t N, size_t C>
struct multivector_expr_context {
    std::ostream &os;
    int prm_idx, fun_idx;

    multivector_expr_context(std::ostream &os) : os(os), prm_idx(0), fun_idx(0) {}

    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {};

#define BINARY_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( "; \
	    boost::proto::eval(boost::proto::left(expr), ctx); \
	    ctx.os << " " #the_op " "; \
	    boost::proto::eval(boost::proto::right(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( " #the_op "( "; \
	    boost::proto::eval(boost::proto::child(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, multivector_expr_context &ctx) const { \
	    ctx.os << "( ( "; \
	    boost::proto::eval(boost::proto::child(expr), ctx); \
	    ctx.os << " )" #the_op " )"; \
	} \
    }

    UNARY_POST_OPERATION(post_inc, ++);
    UNARY_POST_OPERATION(post_dec, --);

#undef UNARY_POST_OPERATION

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::function> {
	typedef void result_type;

	struct do_eval {
	    mutable int pos;
	    multivector_expr_context &ctx;

	    do_eval(multivector_expr_context &ctx) : pos(0), ctx(ctx) {}

	    template <typename Arg>
	    void operator()(const Arg &arg) const {
		if (pos++) ctx.os << ", ";
		boost::proto::eval(arg, ctx);
	    }
	};

	template <class FunCall>
	typename std::enable_if<
	    std::is_base_of<
		builtin_function,
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
		>::type
	    >::value,
	void
	>::type
	operator()(const FunCall &expr, multivector_expr_context &ctx) const {
	    ctx.os << boost::proto::value(boost::proto::child_c<0>(expr)).name() << "( ";
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
		typename boost::proto::result_of::value<
		    typename boost::proto::result_of::child_c<FunCall,0>::type
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
    struct eval<Expr, boost::proto::tag::terminal> {
	typedef void result_type;

	template <typename T, size_t M, bool own>
	void operator()(const multivector<T,M,own> &term, multivector_expr_context &ctx) const {
	    static_assert(M == N, "Wrong number of components in a multivector");

	    ctx.os << "prm_" << C + 1 << "_" << ++ctx.prm_idx << "[idx]";
	}

	template <typename Term>
	void operator()(const Term &term, multivector_expr_context &ctx) const {
	    typedef typename boost::proto::result_of::value<Term>::type term_type;

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
	typedef typename boost::proto::result_of::value<Term>::type term_type;

	typedef
	    typename component<
		C, typename boost::proto::result_of::value<Term>::type
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
	typedef typename boost::proto::result_of::value<Term>::type term_type;

	static_assert(
		number_of_components<term_type>::value == 1 ||
		number_of_components<term_type>::value == N,
		"Wrong number of components in a multiscalar"
		);

	if ((number_of_components<term_type>::value > 1) || (C == 0))
	    krn.setArg(pos++, get<C>(boost::proto::value(term)));
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
struct extract_component : boost::proto::callable {
    template <class T>
    struct result;

    template <class This, class T>
    struct result< This(T) > {
	typedef const typename component< C,
		typename boost::remove_const<
		    typename boost::remove_reference<T>::type
		>::type
	    >::type& type;
    };

    template <class T>
    typename result<extract_component(const T&)>::type
    operator()(const T &t) const {
	using namespace std;
	return get<C>(t);
    }
};

template <size_t C>
struct extract_subexpression
    : boost::proto::or_ <
	boost::proto::when <
	    boost::proto::terminal< multivector_terminal >,
	    extract_component<C>( boost::proto::_ )
	> ,
	boost::proto::when <
	    boost::proto::terminal<boost::proto::_>,
	    boost::proto::_make_terminal (
		    extract_component<C>(
			boost::proto::_value(boost::proto::_)
			)
		    )
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

/// \endcond

} // namespace vex;

#endif
