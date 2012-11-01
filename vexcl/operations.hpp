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

#ifndef _MSC_VER
#  define VEXCL_VARIADIC_TEMPLATES
#endif

#include <boost/proto/proto.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

/// \cond INTERNAL

struct builtin_function {};
struct user_function {};

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

#ifndef VEXCL_VARIADIC_TEMPLATES

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

/// \endcond

} // namespace vex;

#endif
