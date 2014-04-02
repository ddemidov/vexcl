#ifndef VEXCL_FUNCTION_HPP
#define VEXCL_FUNCTION_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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
#define VEX_STRINGIZE_SOURCE(...) BOOST_PP_STRINGIZE(__VA_ARGS__)

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
        BOOST_PP_CAT(vex_function_, dep)::define(src);

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
 \param deps User-defined functions that are called inside the body of the
             function that is being defined. Specified as a preprocessor
             sequence of function names.
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
    VEX_FUNCTION_SD(type, name, args, deps, BOOST_PP_STRINGIZE(__VA_ARGS__) )


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
    VEX_FUNCTION_S(type, name, args, BOOST_PP_STRINGIZE(__VA_ARGS__))

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

#endif
