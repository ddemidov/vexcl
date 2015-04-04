#ifndef VEXCL_FUNCTION_HPP
#define VEXCL_FUNCTION_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/function.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  User-defined device functions.
 */

#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/tuple.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <vexcl/operations.hpp>

namespace vex {

/// Stringizes compute kernel source code.
/**
 * Example:
\code
VEX_FUNCTION_V1(diff_cube, double(double, double),
    VEX_STRINGIZE_SOURCE(
        double d = prm1 - prm2;
        return d * d * d;
        )
    );
\endcode
*/
#define VEX_STRINGIZE_SOURCE(...) #__VA_ARGS__

//---------------------------------------------------------------------------
// VEX_FUNCTION (v1) macros
//---------------------------------------------------------------------------
/// Macro to declare a user function type.
/**
 \code
 VEX_FUNCTION_V1_TYPE(pow3_t, double(double), "", "return pow(prm1, 3.0);");
 pow3_t pow3;
 output = pow3(input);
 \endcode

 \deprecated

 \note This version of the macro uses function call signature in order to
 define the function paramaters. Parameters are named automatically (prm1,
 prm2, ...), which reduces readability of the code. Use of VEX_FUNCTION is
 recommended instead.

 \note Should be used in case same function is used in several places (to
 save on OpenCL kernel recompilations). Otherwise VEX_FUNCTION should
 be used locally.
 */
#define VEX_FUNCTION_V1_TYPE(fname, signature, preamble_str, body_str)         \
  struct vex_function_##fname                                                  \
    : vex::UserFunction<vex_function_##fname, signature>                       \
  {                                                                            \
    vex_function_##fname() {}                                                  \
    static std::string name() { return #fname; }                               \
    static std::string preamble() { return preamble_str; }                     \
    static std::string body() { return body_str; }                             \
  }

/// Macro to declare a user function.
/**
 \code
 VEX_FUNCTION_V1(pow3, double(double), "return pow(prm1, 3.0);");
 output = pow3(input);
 \endcode

 \deprecated

 \note This version of the macro uses function call signature in order to
 define the function paramaters. Parameters are named automatically (prm1,
 prm2, ...), which reduces readability of the code. Use of VEX_FUNCTION is
 recommended instead.
 */
#define VEX_FUNCTION_V1(name, signature, body)                                 \
  VEX_FUNCTION_V1_TYPE(name, signature, "", body) const name


/// Macro to declare a user function with preamble.
/**
 * The preamble may be used to define helper functions or macros.
 \code
 VEX_FUNCTION_V1_WITH_PREAMBLE(one, double(double),
         "double sin2(double x) { return pow(sin(x), 2.0); }\n"
         "double cos2(double x) { return pow(cos(x), 2.0); }\n",
         "return sin2(prm1) + cos2(prm1);"
         );
 y = one(x);
 \endcode

 \deprecated

 \note This version of the macro uses function call signature in order to
 define the function paramaters. Parameters are named automatically (prm1,
 prm2, ...), which reduces readability of the code. Use of VEX_FUNCTION is
 recommended instead.
 */
#define VEX_FUNCTION_V1_WITH_PREAMBLE(name, signature, preamble, body)         \
  VEX_FUNCTION_V1_TYPE(name, signature, preamble, body) const name



//---------------------------------------------------------------------------
// VEX_FUNCTION (v2) macros
//---------------------------------------------------------------------------
/// \cond INTERNAL
#define VEXCL_FUNCTION_ARG_TYPE(s, data, arg) BOOST_PP_TUPLE_ELEM(2, 0, arg)

#define VEXCL_FUNCTION_ARG_TYPES(args) \
    BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(VEXCL_FUNCTION_ARG_TYPE, ~, args))

#define VEXCL_FUNCTION_NTH_ARG_TYPE(n, args)                                   \
    BOOST_PP_TUPLE_ELEM(2, 0, BOOST_PP_SEQ_ELEM(n, args))

#define VEXCL_FUNCTION_NTH_ARG_NAME(n, args)                                   \
    BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(2, 1, BOOST_PP_SEQ_ELEM(n, args)))

#define VEXCL_FUNCTION_DEF_ARG(z, n, args)                                     \
    src.parameter<VEXCL_FUNCTION_NTH_ARG_TYPE(n, args)>(                       \
            VEXCL_FUNCTION_NTH_ARG_NAME(n, args));

#define VEXCL_FUNCTION_DEFINE_DEP(z, data, dep)                                \
    {                                                                          \
        typedef decltype(dep) dep_type;                                        \
        dep_type::define(src);                                                 \
    }

#define VEX_FUNCTION_SINK(rtype, func_name, nargs, args, deps, body)           \
struct vex_function_##func_name                                                \
    : vex::UserFunction<                                                       \
        vex_function_##func_name,                                              \
        rtype( VEXCL_FUNCTION_ARG_TYPES(args) )                                \
      >                                                                        \
{                                                                              \
    vex_function_##func_name() {}                                              \
    static std::string name() { return #func_name; }                           \
    static void define(vex::backend::source_generator &src) {                  \
        define(src, name());                                                   \
    }                                                                          \
    static void define(vex::backend::source_generator &src,                    \
            const std::string &fname)                                          \
    {                                                                          \
        BOOST_PP_SEQ_FOR_EACH(VEXCL_FUNCTION_DEFINE_DEP, ~, deps)              \
        src.function< rtype >(fname).open("(");                                \
        BOOST_PP_REPEAT(nargs, VEXCL_FUNCTION_DEF_ARG, args)                   \
        src.close(")").open("{").new_line() << body;                           \
        src.close("}");                                                        \
    }                                                                          \
} const func_name

#define VEXCL_FUNCTION_MAKE_SEQ_0(...) ((__VA_ARGS__)) VEXCL_FUNCTION_MAKE_SEQ_1
#define VEXCL_FUNCTION_MAKE_SEQ_1(...) ((__VA_ARGS__)) VEXCL_FUNCTION_MAKE_SEQ_0
#define VEXCL_FUNCTION_MAKE_SEQ_0_END
#define VEXCL_FUNCTION_MAKE_SEQ_1_END

#define VEXCL_FUNCTION_MAKE_SEQ(args)                                          \
    BOOST_PP_CAT(VEXCL_FUNCTION_MAKE_SEQ_0 args,_END)

#define VEXCL_DUAL_FUNCTOR_ARG(s, data, arg)                                   \
    BOOST_PP_TUPLE_ELEM(2, 0, arg) BOOST_PP_TUPLE_ELEM(2, 1, arg)

#define VEXCL_DUAL_FUNCTOR_ARGS(args) \
    BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(VEXCL_DUAL_FUNCTOR_ARG, ~, args))

#define VEX_DUAL_FUNCTOR_SINK(rtype, nargs, args, ...)                         \
rtype operator()(VEXCL_DUAL_FUNCTOR_ARGS(args)) const {                        \
    __VA_ARGS__                                                                \
}
/// \endcond

/// Create a user-defined function with dependencies.
/**
 \param type Return type of the function.
 \param name Name of the function.
 \param args Arguments of the function. Specified as a preprocessor sequence
             of tuples. In each of the tuples the first element argument
             type, and the second element defines argument name.
 \param deps User-defined functions that are called inside the body of the
             function that is being defined. Specified as a preprocessor
             sequence of function names.
 \param body Body of the function specified as string.

 Example:
 \code
 VEX_FUNCTION_SD(double, foo, (double, x)(double, y), (bar)(baz),
    "return bar(x + y) * baz(x - y);");

 vex::vector<double> x(ctx, n), y(ctx, n), z(ctx, n);
 z = foo(x, y);
 \endcode
 */
#define VEX_FUNCTION_SD(type, name, args, deps, body)                          \
    VEX_FUNCTION_SINK(type, name,                                              \
            BOOST_PP_SEQ_SIZE(VEXCL_FUNCTION_MAKE_SEQ(args)),                  \
            VEXCL_FUNCTION_MAKE_SEQ(args), deps, body)

/// Alias for VEX_FUNCTION_SD
/** \copydoc VEX_FUNCTION_SD */
#define VEX_FUNCTION_DS VEX_FUNCTION_SD

/// Create a user-defined function.
/**
 \param type Return type of the function.
 \param name Name of the function.
 \param args Arguments of the function. Specified as a preprocessor sequence
             of tuples. In each of the tuples the first element argument
             type, and the second element defines argument name.
 \param body Body of the function specified as string.

 Example:
 \code
 VEX_FUNCTION_S(double, foo, (double, x)(double, y),
    "return (x + y) * (x - y);");

 vex::vector<double> x(ctx, n), y(ctx, n), z(ctx, n);
 z = foo(x, y);
 \endcode
 */
#define VEX_FUNCTION_S(type, name, args, body)                                 \
    VEX_FUNCTION_SD(type, name, args, , body)

/// Create a user-defined function with dependencies.
/**
 \param type Return type of the function.
 \param name Name of the function.
 \param args Arguments of the function. Specified as a preprocessor sequence
             of tuples. In each of the tuples the first element argument
             type, and the second element defines argument name.
 \param deps User-defined functions that are called inside the body of the
             function that is being defined. Specified as a preprocessor
             sequence of function names.

 \note Body of the function is specified as unquoted C source at the end of the
       macro.

 Example:
 \code
 VEX_FUNCTION_D(double, foo, (double, x)(double, y), (bar)(baz),
    return bar(x + y) * baz(x - y);
    );

 vex::vector<double> x(ctx, n), y(ctx, n), z(ctx, n);
 z = foo(x, y);
 \endcode
 */
#define VEX_FUNCTION_D(type, name, args, deps, ...)                            \
    VEX_FUNCTION_SD(type, name, args, deps, VEX_STRINGIZE_SOURCE(__VA_ARGS__) )


/// Create a user-defined function.
/**
 \param type Return type of the function.
 \param name Name of the function.
 \param args Arguments of the function. Specified as a preprocessor sequence
             of tuples. In each of the tuples the first element argument
             type, and the second element defines argument name.

 \note Body of the function is specified as unquoted C source at the end of the
       macro.

 Example:
 \code
 VEX_FUNCTION(double, foo, (double, x)(double, y),
    return (x + y) * (x - y);
    );

 vex::vector<double> x(ctx, n), y(ctx, n), z(ctx, n);
 z = foo(x, y);
 \endcode
 */
#define VEX_FUNCTION(type, name, args, ...)                                    \
    VEX_FUNCTION_S(type, name, args, VEX_STRINGIZE_SOURCE(__VA_ARGS__))

/// Defines both device and host versions of a function call operator.
/**
 The intended use is the creation of comparison and reduction functors for
 use with scan/sort/reduce algorithms.

 Example:
 \code
 template <typename T>
 struct less {
     VEX_DUAL_FUNCTOR(bool, (T, a)(T, b),
         return a < b;
         )
 };
 \endcode
 */
#define VEX_DUAL_FUNCTOR(type, args, ...) \
    VEX_FUNCTION(type, device, args, __VA_ARGS__);                             \
    VEX_DUAL_FUNCTOR_SINK(type,                                                \
            BOOST_PP_SEQ_SIZE(VEXCL_FUNCTION_MAKE_SEQ(args)),                  \
            VEXCL_FUNCTION_MAKE_SEQ(args), __VA_ARGS__)

//---------------------------------------------------------------------------
// Builtin functions
//---------------------------------------------------------------------------
#ifdef DOXYGEN

/// Define builtin function.
#define VEX_BUILTIN_FUNCTION(nargs, func)                                      \
  expression func(const Arg0 & arg0, ... const ArgN & argN);

#else

#define VEXCL_BUILTIN_PRINT_BOOST_REF(z, n, data) boost::ref(arg##n)

#define VEX_BUILTIN_FUNCTION(nargs, func)                                      \
    struct func##_func : vex::builtin_function {                               \
        static const char *name() { return #func; }                            \
    };                                                                         \
    template <BOOST_PP_ENUM_PARAMS(nargs, class Arg)>                          \
    typename boost::proto::result_of::make_expr<                               \
        boost::proto::tag::function, func##_func,                              \
        BOOST_PP_ENUM_BINARY_PARAMS(nargs, const Arg,                          \
                                    &BOOST_PP_INTERCEPT)>::type const          \
    func(BOOST_PP_ENUM_BINARY_PARAMS(nargs, const Arg, &arg)) {                \
        return boost::proto::make_expr<boost::proto::tag::function>(           \
            func##_func(), BOOST_PP_ENUM(                                      \
                nargs, VEXCL_BUILTIN_PRINT_BOOST_REF, ~));                     \
    }

#endif

/// \defgroup builtins Builtin device functions
/** @{ */
VEX_BUILTIN_FUNCTION( 2, abs_diff )
VEX_BUILTIN_FUNCTION( 1, acos )
VEX_BUILTIN_FUNCTION( 1, acosh )
VEX_BUILTIN_FUNCTION( 1, acospi )
VEX_BUILTIN_FUNCTION( 2, add_sat )
VEX_BUILTIN_FUNCTION( 1, all )
VEX_BUILTIN_FUNCTION( 1, any )
VEX_BUILTIN_FUNCTION( 1, asin )
VEX_BUILTIN_FUNCTION( 1, asinh )
VEX_BUILTIN_FUNCTION( 1, asinpi )
VEX_BUILTIN_FUNCTION( 1, atan )
VEX_BUILTIN_FUNCTION( 2, atan2 )
VEX_BUILTIN_FUNCTION( 2, atan2pi )
VEX_BUILTIN_FUNCTION( 1, atanh )
VEX_BUILTIN_FUNCTION( 1, atanpi )
VEX_BUILTIN_FUNCTION( 3, bitselect )
VEX_BUILTIN_FUNCTION( 1, cbrt )
VEX_BUILTIN_FUNCTION( 1, ceil )
VEX_BUILTIN_FUNCTION( 3, clamp )
VEX_BUILTIN_FUNCTION( 1, clz )
VEX_BUILTIN_FUNCTION( 2, copysign )
VEX_BUILTIN_FUNCTION( 1, cos )
VEX_BUILTIN_FUNCTION( 1, cosh )
VEX_BUILTIN_FUNCTION( 1, cospi )
VEX_BUILTIN_FUNCTION( 2, cross )
VEX_BUILTIN_FUNCTION( 1, degrees )
VEX_BUILTIN_FUNCTION( 2, distance )
VEX_BUILTIN_FUNCTION( 2, dot )
VEX_BUILTIN_FUNCTION( 1, erf )
VEX_BUILTIN_FUNCTION( 1, erfc )
VEX_BUILTIN_FUNCTION( 1, exp )
VEX_BUILTIN_FUNCTION( 1, exp10 )
VEX_BUILTIN_FUNCTION( 1, exp2 )
VEX_BUILTIN_FUNCTION( 1, expm1 )
VEX_BUILTIN_FUNCTION( 1, fabs )
VEX_BUILTIN_FUNCTION( 2, fast_distance )
VEX_BUILTIN_FUNCTION( 1, fast_length )
VEX_BUILTIN_FUNCTION( 1, fast_normalize )
VEX_BUILTIN_FUNCTION( 2, fdim )
VEX_BUILTIN_FUNCTION( 1, floor )
VEX_BUILTIN_FUNCTION( 3, fma )
VEX_BUILTIN_FUNCTION( 2, fmax )
VEX_BUILTIN_FUNCTION( 2, fmin )
VEX_BUILTIN_FUNCTION( 2, fmod )
VEX_BUILTIN_FUNCTION( 2, fract )
VEX_BUILTIN_FUNCTION( 2, frexp )
VEX_BUILTIN_FUNCTION( 2, hadd )
VEX_BUILTIN_FUNCTION( 2, hypot )
VEX_BUILTIN_FUNCTION( 1, ilogb )
VEX_BUILTIN_FUNCTION( 2, isequal )
VEX_BUILTIN_FUNCTION( 1, isfinite )
VEX_BUILTIN_FUNCTION( 2, isgreater )
VEX_BUILTIN_FUNCTION( 2, isgreaterequal )
VEX_BUILTIN_FUNCTION( 1, isinf )
VEX_BUILTIN_FUNCTION( 2, isless )
VEX_BUILTIN_FUNCTION( 2, islessequal )
VEX_BUILTIN_FUNCTION( 2, islessgreater )
VEX_BUILTIN_FUNCTION( 1, isnan )
VEX_BUILTIN_FUNCTION( 1, isnormal )
VEX_BUILTIN_FUNCTION( 2, isnotequal )
VEX_BUILTIN_FUNCTION( 2, isordered )
VEX_BUILTIN_FUNCTION( 2, isunordered )
VEX_BUILTIN_FUNCTION( 2, ldexp )
VEX_BUILTIN_FUNCTION( 1, length )
VEX_BUILTIN_FUNCTION( 1, lgamma )
VEX_BUILTIN_FUNCTION( 2, lgamma_r )
VEX_BUILTIN_FUNCTION( 1, log )
VEX_BUILTIN_FUNCTION( 1, log10 )
VEX_BUILTIN_FUNCTION( 1, log1p )
VEX_BUILTIN_FUNCTION( 1, log2 )
VEX_BUILTIN_FUNCTION( 1, logb )
VEX_BUILTIN_FUNCTION( 3, mad )
VEX_BUILTIN_FUNCTION( 3, mad24 )
VEX_BUILTIN_FUNCTION( 3, mad_hi )
VEX_BUILTIN_FUNCTION( 3, mad_sat )
VEX_BUILTIN_FUNCTION( 2, max )
VEX_BUILTIN_FUNCTION( 2, maxmag )
VEX_BUILTIN_FUNCTION( 2, min )
VEX_BUILTIN_FUNCTION( 2, minmag )
VEX_BUILTIN_FUNCTION( 3, mix )
VEX_BUILTIN_FUNCTION( 2, modf )
VEX_BUILTIN_FUNCTION( 2, mul_hi )
VEX_BUILTIN_FUNCTION( 1, nan )
VEX_BUILTIN_FUNCTION( 2, nextafter )
VEX_BUILTIN_FUNCTION( 1, normalize )
#if defined(VEXCL_BACKEND_CUDA)
VEX_BUILTIN_FUNCTION( 1, __popc )
VEX_BUILTIN_FUNCTION( 1, __popcll )
#else
VEX_BUILTIN_FUNCTION( 1, popcount )
#endif
VEX_BUILTIN_FUNCTION( 2, pow )
VEX_BUILTIN_FUNCTION( 2, pown )
VEX_BUILTIN_FUNCTION( 2, powr )
VEX_BUILTIN_FUNCTION( 1, radians )
VEX_BUILTIN_FUNCTION( 2, remainder )
VEX_BUILTIN_FUNCTION( 3, remquo )
VEX_BUILTIN_FUNCTION( 2, rhadd )
VEX_BUILTIN_FUNCTION( 1, rint )
VEX_BUILTIN_FUNCTION( 2, rootn )
VEX_BUILTIN_FUNCTION( 2, rotate )
VEX_BUILTIN_FUNCTION( 1, round )
VEX_BUILTIN_FUNCTION( 1, rsqrt )
VEX_BUILTIN_FUNCTION( 3, select )
VEX_BUILTIN_FUNCTION( 2, shuffle )
VEX_BUILTIN_FUNCTION( 3, shuffle2 )
VEX_BUILTIN_FUNCTION( 1, sign )
VEX_BUILTIN_FUNCTION( 1, signbit )
VEX_BUILTIN_FUNCTION( 1, sin )
VEX_BUILTIN_FUNCTION( 2, sincos )
VEX_BUILTIN_FUNCTION( 1, sinh )
VEX_BUILTIN_FUNCTION( 1, sinpi )
VEX_BUILTIN_FUNCTION( 3, smoothstep )
VEX_BUILTIN_FUNCTION( 1, sqrt )
VEX_BUILTIN_FUNCTION( 2, step )
VEX_BUILTIN_FUNCTION( 2, sub_sat )
VEX_BUILTIN_FUNCTION( 1, tan )
VEX_BUILTIN_FUNCTION( 1, tanh )
VEX_BUILTIN_FUNCTION( 1, tanpi )
VEX_BUILTIN_FUNCTION( 1, tgamma )
VEX_BUILTIN_FUNCTION( 1, trunc )
VEX_BUILTIN_FUNCTION( 2, upsample )

// Special case: abs() overloaded with floating point arguments should call
// fabs in the OpenCL code
struct abs_func : builtin_function {
    static const char* name() {
        return "abs";
    }
};


namespace detail {
    template <class Expr> struct return_type;
}

#ifdef DOXYGEN
expression
#else
template <typename Arg>
typename std::enable_if<
    std::is_integral<
        typename cl_scalar_of<
            typename detail::return_type<Arg>::type
        >::type
    >::value,
    typename boost::proto::result_of::make_expr<
        boost::proto::tag::function,
        abs_func,
        const Arg&
    >::type const
>::type
#endif
abs(const Arg &arg) {
    return boost::proto::make_expr<boost::proto::tag::function>(
            abs_func(),
            boost::ref(arg)
            );
}

#ifdef DOXYGEN
expression
#else
template <typename Arg>
typename std::enable_if<
    !std::is_integral<
        typename cl_scalar_of<
            typename detail::return_type<Arg>::type
        >::type
    >::value,
    typename boost::proto::result_of::make_expr<
        boost::proto::tag::function,
        fabs_func,
        const Arg&
    >::type const
>::type
#endif
abs(const Arg &arg) {
    return boost::proto::make_expr<boost::proto::tag::function>(
            fabs_func(),
            boost::ref(arg)
            );
}

/** @} */

} // namespace vex

#endif
