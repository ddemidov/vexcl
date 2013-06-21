#ifndef VEXCL_OPERATIONS_HPP
#define VEXCL_OPERATIONS_HPP

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
 * \file   vexcl/operations.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Set of operations to be used in vector expressions.
 */

#ifdef WIN32
#  pragma warning(push)
#  define NOMINMAX
#endif

#include <array>
#include <tuple>
#include <boost/proto/proto.hpp>
#include <boost/mpl/max.hpp>
#include <boost/any.hpp>


// Include boost.preprocessor header if variadic templates are not available.
// Also include it if we use gcc v4.6.
// This is required due to bug http://gcc.gnu.org/bugzilla/show_bug.cgi?id=35722
#if defined(BOOST_NO_VARIADIC_TEMPLATES) || (defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 4 && __GNUC_MINOR__ == 6)
#  include <boost/preprocessor/repetition.hpp>
#  ifndef VEXCL_MAX_ARITY
#    define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#  endif
#endif

/// Vector expression template library for OpenCL.
namespace vex {

/// \cond INTERNAL
//---------------------------------------------------------------------------
// Assignment operators.
//---------------------------------------------------------------------------
namespace assign {

#define ASSIGN_OP(name, op) \
    struct name { \
        static std::string string() { \
            return #op; \
        } \
    };

    ASSIGN_OP(SET, =);
    ASSIGN_OP(ADD, +=);
    ASSIGN_OP(SUB, -=);
    ASSIGN_OP(MUL, *=);
    ASSIGN_OP(DIV, /=);
    ASSIGN_OP(MOD, %=);
    ASSIGN_OP(AND, &=);
    ASSIGN_OP(OR,  |=);
    ASSIGN_OP(XOR, ^=);
    ASSIGN_OP(LSH, <<=);
    ASSIGN_OP(RSH, >>=);

#undef ASSIGN_OP

}

//---------------------------------------------------------------------------
// Scalable expressions
//---------------------------------------------------------------------------
/* Type trait to determine if an expression is scalable.
 *
 * The expression should have a type `value_type` and a field `scale` of that
 * type, this enables operator* and operator/.
 */
template <class T> struct is_scalable : std::false_type {};

// Scalable expressions may be multiplied by a scalar:
template <class T>
typename std::enable_if<vex::is_scalable<T>::value, T>::type
operator*(const T &expr, const typename T::value_type &factor) {
    T scaled_expr(expr);
    scaled_expr.scale *= factor;
    return scaled_expr;
}

// Scalable expressions may be multiplied by a scalar:
template <class T>
typename std::enable_if<vex::is_scalable<T>::value, T>::type
operator*(const typename T::value_type &factor, const T &expr) {
    return expr * factor;
}

// Scalable expressions may be divided by a scalar:
template <class T> typename std::enable_if<vex::is_scalable<T>::value, T>::type
operator/(const T &expr, const typename T::value_type &factor) {
    T scaled_expr(expr);
    scaled_expr.scale /= factor;
    return scaled_expr;
}

//---------------------------------------------------------------------------
// Standard grammar (no terminals)
//---------------------------------------------------------------------------
struct builtin_function {};
struct user_function {};

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

//---------------------------------------------------------------------------
// Builtin functions
//---------------------------------------------------------------------------
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

BUILTIN_FUNCTION_2(cross);
BUILTIN_FUNCTION_2(dot);
BUILTIN_FUNCTION_2(distance);
BUILTIN_FUNCTION_1(length);
BUILTIN_FUNCTION_1(normalize);

#undef BUILTIN_FUNCTION_1
#undef BUILTIN_FUNCTION_2
#undef BUILTIN_FUNCTION_3

#define VEXCL_VECTOR_EXPR_EXTRACTOR(name, VG, AG, FG) \
struct name \
    : boost::proto::or_< \
          boost::proto::when<VG, boost::proto::_>, \
          boost::proto::when< \
             boost::proto::plus< FG, AG >, \
             name(boost::proto::_left) \
          >, \
          boost::proto::when< \
             boost::proto::plus< AG, FG >, \
             name(boost::proto::_right) \
          >, \
          boost::proto::when< \
             boost::proto::minus< FG, AG >, \
             name(boost::proto::_left) \
          >, \
          boost::proto::when< \
             boost::proto::minus< AG, FG >, \
             boost::proto::_make_negate( name(boost::proto::_right) ) \
          >, \
          boost::proto::when< \
             boost::proto::binary_expr<boost::proto::_, name, name > \
          > \
      > \
{}

#define VEXCL_ADDITIVE_EXPR_EXTRACTOR(name, VG, AG, FG) \
struct name \
    : boost::proto::or_< \
          boost::proto::when<AG, boost::proto::_> \
        , boost::proto::when< \
             boost::proto::plus<FG, VG >, \
             name(boost::proto::_left) \
          > \
        , boost::proto::when< \
             boost::proto::plus< VG, FG >, \
             name(boost::proto::_right) \
          > \
        , boost::proto::when< \
             boost::proto::minus<FG, VG >, \
             name(boost::proto::_left) \
          > \
        , boost::proto::when< \
             boost::proto::minus<VG, FG >, \
             boost::proto::_make_negate( name(boost::proto::_right) ) \
          > \
        , boost::proto::when< \
             boost::proto::binary_expr<boost::proto::_, name, name > \
          > \
      > \
{}

//---------------------------------------------------------------------------
// User-defined functions
//---------------------------------------------------------------------------
template <class C, class T>
struct UserFunction {};

// Workaround for gcc bug http://gcc.gnu.org/bugzilla/show_bug.cgi?id=35722
#if !defined(BOOST_NO_VARIADIC_TEMPLATES) && ((!defined(__GNUC__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 6)) || defined(__clang__))

template<class Impl, class RetType, class... ArgType>
struct UserFunction<Impl, RetType(ArgType...)> : user_function
{
    template <class... Arg>
    typename boost::proto::result_of::make_expr<
        boost::proto::tag::function,
        Impl,
        const Arg&...
    >::type const
    operator()(const Arg&... arg) {
        return boost::proto::make_expr<boost::proto::tag::function>(
                Impl(), boost::ref(arg)...
                );
    }

    static std::string preamble() {
        return "";
    }

    static void define(std::ostream &os, const std::string &name) {
        os << Impl::preamble() << "\n"
           << type_name<RetType>() << " " << name << "(";
        show_arg<ArgType...>(os, 1);
        os << "\n)\n{\n" << Impl::body() << "\n}\n\n";
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

#define PRINT_ARG_REF(z, n, data) const Arg ## n&
#define PRINT_PARAM(z, n, data) const Arg ## n &arg ## n
#define PRINT_BOOST_REF(z, n, data) boost::ref(arg ## n)
#define PRINT_PRM_DEF(z, n, data) \
    << "\n\t" << type_name<ArgType ## n>() \
    << " prm" << n + 1 BOOST_PP_EXPR_IF(BOOST_PP_LESS(BOOST_PP_ADD(n, 1), data), << ",")
#define USER_FUNCTION(z, n, data) \
template< class Impl, class RetType, BOOST_PP_ENUM_PARAMS(n, class ArgType) > \
struct UserFunction<Impl, RetType( BOOST_PP_ENUM_PARAMS(n, ArgType) )> : user_function \
{ \
    template < BOOST_PP_ENUM_PARAMS(n, class Arg) > \
    typename boost::proto::result_of::make_expr< \
        boost::proto::tag::function, \
        Impl, \
        BOOST_PP_ENUM(n, PRINT_ARG_REF, ~) \
    >::type const \
    operator()( BOOST_PP_ENUM(n, PRINT_PARAM, ~) ) { \
        return boost::proto::make_expr<boost::proto::tag::function>( \
                Impl(), BOOST_PP_ENUM(n, PRINT_BOOST_REF, ~) \
                ); \
    } \
    static std::string preamble() { \
        return ""; \
    } \
    static void define(std::ostream &os, const std::string &name) { \
        os << Impl::preamble() << "\n" \
           << type_name<RetType>() << " " << name << "(" \
           BOOST_PP_REPEAT(n, PRINT_PRM_DEF, n) \
           << "\n)\n{\n" << Impl::body() << "\n}\n\n"; \
    } \
};

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, USER_FUNCTION, ~)

#undef PRINT_ARG_REF
#undef PRINT_PARAM
#undef PRINT_BOOST_REF
#undef PRINT_PRM_DEF
#undef USER_FUNCTION

#endif

/// \endcond

/// Macro to declare a user function type.
/**
 * \code
 * VEX_FUNCTION_TYPE(pow3_t, double(double), "", "return pow(prm1, 3.0);");
 * pow3_t pow3;
 * output = pow3(input);
 * \endcode
 *
 * \note Should be used in case same function is used in several places (to
 * save on OpenCL kernel recompilations). Otherwise VEX_FUNCTION should
 * be used locally.
 */
#define VEX_FUNCTION_TYPE(name, signature, preamble_str, body_str) \
    struct name : vex::UserFunction<name, signature> { \
        static std::string preamble() { return preamble_str; } \
        static std::string body()     { return body_str;     } \
    }

/// Macro to declare a user function.
/**
 * \code
 * VEX_FUNCTION(pow3, double(double), "return pow(prm1, 3.0);");
 * output = pow3(input);
 * \endcode
 */
#define VEX_FUNCTION(name, signature, body) \
    VEX_FUNCTION_TYPE(user_function_##name##_body, signature, "", body) name


/// Macro to declare a user function with preamble.
/**
 * The preamble may be used to define helper functions or macros.
 * \code
 * VEX_FUNCTION_WITH_PREAMBLE(one, double(double),
 *         "double sin2(double x) { return pow(sin(x), 2.0); }\n"
 *         "double cos2(double x) { return pow(cos(x), 2.0); }\n",
 *         "return sin2(prm1) + cos2(prm1);"
 *         );
 * y = one(x);
 * \endcode
 */
#define VEX_FUNCTION_WITH_PREAMBLE(name, signature, preamble, body) \
    VEX_FUNCTION_TYPE(user_function_##name##_body, signature, preamble, body) name


/// \cond INTERNAL

//---------------------------------------------------------------------------
// Expression Transforms
//---------------------------------------------------------------------------

// Helper functor for use with boost::fusion::for_each
template <class Context>
struct do_eval {
    Context &ctx;

    do_eval(Context &ctx) : ctx(ctx) {}

    template <class Expr>
    void operator()(const Expr &expr) const {
        boost::proto::eval(expr, ctx);
    }
};

//---------------------------------------------------------------------------
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
                boost::proto::when <
                    boost::proto::terminal <
                        boost::proto::convertible_to<vex::user_function>
                    >,
                    process_terminal(boost::proto::_)
                >,
                boost::proto::vararg< extract_user_functions >
            >
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
// Elementwise vector operations
//---------------------------------------------------------------------------

// Terminals allowed in vector expressions.
template <class Term, class Enable = void>
struct is_vector_expr_terminal
    : std::false_type
{ };

template <class T>
struct is_vector_expr_terminal< T, typename std::enable_if< is_cl_native< T >::value >::type >
    : std::true_type
{ };


// Grammar for vector expressions that may be processed with single kernel:
struct vector_expr_grammar
    : boost::proto::or_<
          boost::proto::and_<
              boost::proto::terminal< boost::proto::_ >,
              boost::proto::if_< is_vector_expr_terminal< boost::proto::_value >() >
          >,
          BUILTIN_OPERATIONS(vector_expr_grammar),
          USER_FUNCTIONS(vector_expr_grammar)
      >
{};

// Grammar for additive expressions (each additive term requires separate
// kernel):
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

// Joined grammar for vector expressions and additive expressions.
struct vector_full_grammar
    : boost::proto::or_<
        vector_expr_grammar,
        boost::proto::terminal< additive_vector_transform >,
        boost::proto::plus< vector_full_grammar, vector_full_grammar >,
        boost::proto::minus< vector_full_grammar, vector_full_grammar >,
        boost::proto::negate< vector_full_grammar >
      >
{};


// Boost.Proto domain for vector expressions.
template <class Expr>
struct vector_expression;

template <class T, class Enable = void>
struct hold_terminal_by_reference : std::false_type {};

struct vector_domain
    : boost::proto::domain<
        boost::proto::generator<vector_expression>,
        vector_full_grammar
        >
{
    // Store everything by value inside expressions...
    template <typename T, class Enable = void>
    struct as_child : proto_base_domain::as_expr<T>
    {};

    // ... except for terminals that explicitly request storage by reference:
    template <typename T>
    struct as_child< T,
        typename std::enable_if< hold_terminal_by_reference<T>::value >::type
        > : proto_base_domain::as_child< T >
    {};
};

template <class Expr>
struct vector_expression
    : boost::proto::extends< Expr, vector_expression<Expr>, vector_domain>
{
    vector_expression(const Expr &expr = Expr())
        : boost::proto::extends< Expr, vector_expression<Expr>, vector_domain>(expr) {}
};


//---------------------------------------------------------------------------
// Vector contexts and transform helpers
//---------------------------------------------------------------------------

// Representation of a terminal in a kernel name
template <class T, class Enable = void>
struct kernel_name {
    static std::string get() {
        return "term_";
    }
};


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

        template <typename Term>
        void operator()(const Term&, vector_name_context &ctx) const {
            ctx.os << kernel_name<Term>::get();
        }
    };
};

// Used as a state parameter in kernel generation functions.
typedef std::map<std::string, boost::any> kernel_generator_state;

struct expression_context {
    mutable kernel_generator_state state;
};

// Partial expression for a terminal:
template <class Term, class Enable = void>
struct partial_vector_expr {
    static std::string get(const cl::Device&, int component, int position, kernel_generator_state&) {
        std::ostringstream s;
        s << "prm_" << component << "_" << position;
        return s.str();
    }
};

// Builds textual representation for a vector expression.
struct vector_expr_context : public expression_context {
    std::ostream &os;
    const cl::Device &device;
    int cmp_idx, prm_idx, fun_idx;

    vector_expr_context(std::ostream &os, const cl::Device &device, int cmp_idx = 1)
        : os(os), device(device), cmp_idx(cmp_idx), prm_idx(0), fun_idx(0) {}

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

        template <typename Term>
        void operator()(const Term&, vector_expr_context &ctx) const {
            ctx.os << partial_vector_expr<Term>::get(ctx.device, ctx.cmp_idx, ++ctx.prm_idx, ctx.state);
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

// Some terminals need preamble (e.g. struct declaration or helper function).
// But most of them do not:
template <class T>
struct terminal_preamble {
    static std::string get(const cl::Device&, int/*component*/, int/*position*/, kernel_generator_state&)
    {
        return "";
    }
};

struct output_terminal_preamble : expression_context {
    std::ostream &os;
    const cl::Device &device;
    int cmp_idx;
    mutable int prm_idx;

    output_terminal_preamble(std::ostream &os, const cl::Device &device, int cmp_idx = 1)
        : os(os), device(device), cmp_idx(cmp_idx), prm_idx(0) {}

        template <class Term>
        void operator()(const Term&) const {
            os << terminal_preamble<Term>::get(device, cmp_idx, ++prm_idx, state) << std::endl;
        }
};

template <class Expr>
void construct_preamble(const Expr &expr, std::ostream &kernel_source, const cl::Device &device, int component = 1) {

    extract_user_functions()(
            boost::proto::as_child(expr),
            declare_user_function(kernel_source, component)
            );

    extract_terminals()(
            boost::proto::as_child(expr),
            output_terminal_preamble(kernel_source, device, component)
            );

}

template <class Term, class Enable = void>
struct kernel_param_declaration {
    static std::string get(const cl::Device&, int component, int position, kernel_generator_state&) {
        std::ostringstream s;
        s << ",\n\t" << type_name<typename boost::proto::result_of::value<Term>::type>()
          << " prm_" << component << "_" << position;
        return s.str();
    }
};

struct declare_expression_parameter : expression_context {
    std::ostream &os;
    const cl::Device &device;
    int cmp_idx;
    mutable int prm_idx;

    declare_expression_parameter(std::ostream &os, const cl::Device &device, int cmp_idx = 1)
        : os(os), device(device), cmp_idx(cmp_idx), prm_idx(0) {}

    template <typename T>
    void operator()(const T&) const {
        os << kernel_param_declaration<T>::get(device, cmp_idx, ++prm_idx, state);
    }
};

template <class Term, class Enable = void>
struct kernel_arg_setter {
    static void set(cl::Kernel &kernel, uint/*device*/, size_t/*index_offset*/,
            uint &position, const Term &term, kernel_generator_state&)
    {
        kernel.setArg(position++, boost::proto::value(term));
    }
};

struct set_expression_argument : expression_context {
    cl::Kernel &krn;
    uint dev, &pos;
    size_t part_start;

    set_expression_argument(cl::Kernel &krn, uint dev, uint &pos, size_t part_start)
        : krn(krn), dev(dev), pos(pos), part_start(part_start) {}

    template <typename T>
    void operator()(const T &term) const {
        kernel_arg_setter<T>::set(krn, dev, part_start, pos, term, state);
    }
};

template <class T, class Enable = void>
struct expression_properties {
    static void get(const T &/*term*/,
            std::vector<cl::CommandQueue> &/*queue_list*/,
            std::vector<size_t> &/*partition*/,
            size_t &/*size*/
            )
    { }
};

struct get_expression_properties {
    mutable std::vector<cl::CommandQueue> queue;
    mutable std::vector<size_t> part;
    mutable size_t size;

    get_expression_properties() : size(0) {}

    size_t part_start(uint d) const {
        return part.empty() ? 0 : part[d];
    }

    size_t part_size(uint d) const {
        return part.empty() ? 0 : part[d + 1] - part[d];
    }

    template <typename T>
    void operator()(const T &term) const {
        if (queue.empty())
            expression_properties<T>::get(term, queue, part, size);
    }
};

//---------------------------------------------------------------------------
VEXCL_VECTOR_EXPR_EXTRACTOR(extract_vector_expressions,
        vector_expr_grammar,
        additive_vector_transform_grammar,
        vector_full_grammar
        );

VEXCL_ADDITIVE_EXPR_EXTRACTOR(extract_additive_vector_transforms,
        vector_expr_grammar,
        additive_vector_transform_grammar,
        vector_full_grammar
        );

struct simplify_additive_transform
    : boost::proto::or_<
          boost::proto::terminal< boost::proto::_ >,
          boost::proto::when<
             boost::proto::negate< boost::proto::terminal< boost::proto::_ > >,
             boost::proto::_
          >,
          boost::proto::when<
             boost::proto::negate< boost::proto::negate< boost::proto::_ > >,
             simplify_additive_transform(boost::proto::_child(boost::proto::_child))
          >,
          boost::proto::plus< simplify_additive_transform, simplify_additive_transform >,
          boost::proto::when<
            boost::proto::minus< boost::proto::_, boost::proto::_ >,
            boost::proto::_make_plus(
                    simplify_additive_transform(boost::proto::_left),
                    simplify_additive_transform(boost::proto::_make_negate(boost::proto::_right))
                    )
          >,
          boost::proto::when<
             boost::proto::negate< boost::proto::plus<boost::proto::_, boost::proto::_> >,
             boost::proto::_make_plus(
                     simplify_additive_transform(boost::proto::_make_negate(boost::proto::_left(boost::proto::_child))),
                     simplify_additive_transform(boost::proto::_make_negate(boost::proto::_right(boost::proto::_child)))
                     )
          >,
          boost::proto::when<
             boost::proto::negate< boost::proto::minus<boost::proto::_, boost::proto::_> >,
             boost::proto::_make_plus(
                     simplify_additive_transform(boost::proto::_make_negate(boost::proto::_left(boost::proto::_child))),
                     simplify_additive_transform(boost::proto::_right(boost::proto::_child))
                     )
          >
      >
{};

template <bool append, class Vector>
struct additive_applicator {
    Vector &dest;

    additive_applicator(Vector &dest) : dest(dest) {}

    template <typename Expr>
    typename std::enable_if<
        boost::proto::matches<
            typename boost::proto::result_of::as_expr<Expr>::type,
            boost::proto::terminal<boost::proto::_>
        >::value,
        void
    >::type
    operator()(const Expr &expr) const {
        expr.template apply</*negate=*/false, append>(dest);
    }

    template <typename Expr>
    typename std::enable_if<
        boost::proto::matches<
            typename boost::proto::result_of::as_expr<Expr>::type,
            boost::proto::negate<boost::proto::_>
        >::value,
        void
    >::type
    operator()(const Expr &expr) const {
        boost::proto::child(expr).template apply</*negate=*/true, append>(dest);
    }
};

template <bool append, class Vector, class Expr>
typename std::enable_if<
    boost::proto::matches<
        typename boost::proto::result_of::as_expr<Expr>::type,
        boost::proto::terminal<boost::proto::_>
    >::value ||
    boost::proto::matches<
        typename boost::proto::result_of::as_expr<Expr>::type,
        boost::proto::negate<boost::proto::_>
    >::value,
    void
>::type apply_additive_transform(Vector &dest, const Expr &expr) {
    (additive_applicator<append, Vector>(dest))(expr);
}

template <bool append, class Vector, class Expr>
typename std::enable_if<
    !boost::proto::matches<
        typename boost::proto::result_of::as_expr<Expr>::type,
        boost::proto::terminal<boost::proto::_>
    >::value &&
    !boost::proto::matches<
        typename boost::proto::result_of::as_expr<Expr>::type,
        boost::proto::negate<boost::proto::_>
    >::value,
    void
>::type apply_additive_transform(Vector &dest, const Expr &expr) {
    auto flat_expr = boost::proto::flatten(expr);

    (additive_applicator<append, Vector>(dest))(boost::fusion::front(flat_expr));

    boost::fusion::for_each(boost::fusion::pop_front(flat_expr),
            additive_applicator</*append=*/true, Vector>(dest)
            );
}

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
    typename std::enable_if< is_cl_native<T>::value >::type >
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
    typename std::enable_if<And< is_cl_native<Args>... >::type::value >::type >
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

#else

#define TUPLE_IS_MS(z, n, unused)                                      \
  template < BOOST_PP_ENUM_PARAMS(n, class Arg) >                      \
  struct is_multiscalar< std::tuple < BOOST_PP_ENUM_PARAMS(n, Arg) > > \
    : std::true_type                                                   \
  {};

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TUPLE_IS_MS, ~)

#undef TUPLE_IS_MS

#define TUPLE_COMP(z, n, unused)                                          \
  template < size_t I, BOOST_PP_ENUM_PARAMS(n, class Arg) >               \
  struct component< I, std::tuple < BOOST_PP_ENUM_PARAMS(n, Arg) > >      \
    : std::tuple_element< I, std::tuple< BOOST_PP_ENUM_PARAMS(n, Arg) > > \
  {};

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TUPLE_COMP, ~)

#undef TUPLE_COMP

#endif

// std::array<T,N>

template <class T, size_t N>
struct is_multiscalar< std::array<T, N>,
    typename std::enable_if< is_cl_native<T>::value >::type >
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
    typename std::enable_if< is_cl_native<T>::value >::type >
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

template <typename T, size_t N, bool own = true>
class multivector;

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

//--- Multivector grammar ---------------------------------------------------

// Terminals allowed in multivector expressions.
template <class Term, class Enable = void>
struct is_multivector_expr_terminal
    : std::false_type
{ };

template <class T>
struct is_multivector_expr_terminal< T,
    typename std::enable_if< is_multiscalar<T>::value >::type
    >
    : std::true_type
{ };

struct multivector_expr_grammar
    : boost::proto::or_<
          boost::proto::and_<
              boost::proto::terminal< boost::proto::_ >,
              boost::proto::if_< is_multivector_expr_terminal< boost::proto::_value >() >
          >,
          BUILTIN_OPERATIONS(multivector_expr_grammar),
          USER_FUNCTIONS(multivector_expr_grammar)
      >
{};

struct additive_multivector_transform {};

struct additive_multivector_transform_grammar
    : boost::proto::or_<
        boost::proto::terminal< additive_multivector_transform >,
        boost::proto::plus<
            additive_multivector_transform_grammar,
            additive_multivector_transform_grammar
        >,
        boost::proto::minus<
            additive_multivector_transform_grammar,
            additive_multivector_transform_grammar
        >,
        boost::proto::negate<
            additive_multivector_transform_grammar
        >
      >
{};

struct multivector_full_grammar
    : boost::proto::or_<
        multivector_expr_grammar,
        boost::proto::terminal< additive_multivector_transform >,
        boost::proto::plus< multivector_full_grammar, multivector_full_grammar >,
        boost::proto::minus< multivector_full_grammar, multivector_full_grammar >,
        boost::proto::negate< multivector_full_grammar >
      >
{};

template <class Expr>
struct multivector_expression;

struct multivector_domain
    : boost::proto::domain<
        boost::proto::generator<multivector_expression>,
        multivector_full_grammar
      >
{
    // Store everything by value inside expressions...
    template <typename T, class Enable = void>
    struct as_child : proto_base_domain::as_expr<T>
    {};

    // ... except for terminals that explicitly request storage by reference:
    template <typename T>
    struct as_child< T,
        typename std::enable_if< hold_terminal_by_reference<T>::value >::type
        > : proto_base_domain::as_child< T >
    {};
};

template <class Expr>
struct multivector_expression
    : boost::proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
{
    multivector_expression(const Expr &expr = Expr())
        : boost::proto::extends< Expr, multivector_expression<Expr>, multivector_domain>(expr) {}
};

//---------------------------------------------------------------------------
// Multiexpression component extractor
//---------------------------------------------------------------------------
template <class T, class Enable = void>
struct proto_terminal_is_value : std::false_type { };

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
            >::type type;
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
            boost::proto::and_<
                boost::proto::terminal< boost::proto::_ >,
                boost::proto::if_< proto_terminal_is_value< boost::proto::_value >() >
            >,
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

template <size_t C>
struct subexpression {
    template<class Expr>
    static auto get(const Expr &expr) ->
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                multivector_expr_grammar
            >::value,
            decltype(extract_subexpression<C>()(boost::proto::as_child(expr)))
        >::type
    {
        return extract_subexpression<C>()(boost::proto::as_child(expr));
    }

    // If expression does not match multivector_expr_grammar, assume its a
    // tuple of vector expressions.
    template<class Expr>
    static auto get(const Expr &expr) ->
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                multivector_expr_grammar
            >::value,
            decltype(std::get<C>(expr))
        >::type
    {
        return std::get<C>(expr);
    }
};

VEXCL_VECTOR_EXPR_EXTRACTOR(extract_multivector_expressions,
        multivector_expr_grammar,
        additive_multivector_transform_grammar,
        multivector_full_grammar
        );

VEXCL_ADDITIVE_EXPR_EXTRACTOR(extract_additive_multivector_transforms,
        multivector_expr_grammar,
        additive_multivector_transform_grammar,
        multivector_full_grammar
        );


//---------------------------------------------------------------------------
// Assign expression to lhs
//---------------------------------------------------------------------------
template <class OP, class LHS, class Expr>
void assign_expression(LHS &lhs, const Expr &expr,
        const std::vector<cl::CommandQueue> &queue,
        const std::vector<size_t> &part
        )
{
    static kernel_cache cache;

    for(uint d = 0; d < queue.size(); d++) {
        cl::Context context = qctx(queue[d]);
        cl::Device  device  = qdev(queue[d]);

        auto kernel = cache.find(context());

        if (kernel == cache.end()) {
            std::ostringstream source;

            std::ostringstream kernel_name;
            vector_name_context name_ctx(kernel_name);
            boost::proto::eval(boost::proto::as_child(expr), name_ctx);

            source << standard_kernel_header(device);

            declare_user_function declfun(source);

            extract_user_functions()(boost::proto::as_child(lhs),  declfun);
            extract_user_functions()(boost::proto::as_child(expr), declfun);

            output_terminal_preamble termpream(source, device);

            extract_terminals()(boost::proto::as_child(lhs),  termpream);
            extract_terminals()(boost::proto::as_child(expr), termpream);

            source << "kernel void " << kernel_name.str()
                << "(\n\t" << type_name<size_t>() << " n";

            declare_expression_parameter declare(source, device);

            extract_terminals()(boost::proto::as_child(lhs),  declare);
            extract_terminals()(boost::proto::as_child(expr), declare);

            source << "\n)\n{\n";

            if ( is_cpu(device) ) {
                source <<
                    "\tsize_t chunk_size  = (n + get_global_size(0) - 1) / get_global_size(0);\n"
                    "\tsize_t chunk_start = get_global_id(0) * chunk_size;\n"
                    "\tsize_t chunk_end   = min(n, chunk_start + chunk_size);\n"
                    "\tfor(size_t idx = chunk_start; idx < chunk_end; ++idx) {\n";
            } else {
                source <<
                    "\tfor(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n";
            }

            vector_expr_context expr_ctx(source, device);

            source << "\t\t";

            boost::proto::eval(boost::proto::as_child(lhs), expr_ctx);
            source << " " << OP::string() << " ";
            boost::proto::eval(boost::proto::as_child(expr), expr_ctx);

            source << ";\n\t}\n}\n";

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, kernel_name.str().c_str());
            size_t wgs = kernel_workgroup_size(krn, device);

            kernel = cache.insert(std::make_pair(
                        context(), kernel_cache_entry(krn, wgs)
                        )).first;
        }

        if (size_t psize = part[d + 1] - part[d]) {
            size_t w_size = kernel->second.wgsize;
            size_t g_size = num_workgroups(device) * w_size;

            uint pos = 0;
            kernel->second.kernel.setArg(pos++, psize);

            set_expression_argument setarg(kernel->second.kernel, d, pos, part[d]);

            extract_terminals()( boost::proto::as_child(lhs),  setarg);
            extract_terminals()( boost::proto::as_child(expr), setarg);

            queue[d].enqueueNDRangeKernel(
                    kernel->second.kernel, cl::NullRange, g_size, w_size
                    );
        }
    }
}

/// \endcond

} // namespace vex;


// vim: et
#endif
