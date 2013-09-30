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
 * \brief  Templates used for expression tree traversal and kernel generation.
 */

#ifdef _MSC_VER
#  define NOMINMAX
#endif

#include <array>
#include <tuple>
#include <deque>
#include <memory>

#include <boost/proto/proto.hpp>
#include <boost/mpl/max.hpp>
#include <boost/any.hpp>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

#include <vexcl/types.hpp>
#include <vexcl/util.hpp>

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

namespace detail {

// Used as a state parameter in kernel generation functions.
typedef std::map<std::string, boost::any> kernel_generator_state;
typedef std::shared_ptr<kernel_generator_state> kernel_generator_state_ptr;

inline kernel_generator_state_ptr empty_state() {
    return std::make_shared<kernel_generator_state>();
}

} // namespace detail

namespace traits {

// Terminals allowed in vector expressions
template <class Term, class Enable = void>
struct is_vector_expr_terminal : std::false_type { };

template <class T>
struct is_vector_expr_terminal< T,
    typename std::enable_if< is_cl_native< T >::value >::type
    > : std::true_type
{ };

// Hold everything by value inside proto expressions unless explicitly
// specified otherwise.
template <class T, class Enable = void>
struct hold_terminal_by_reference : std::false_type {};

// Value type of a terminal
template <class T, class Enable = void>
struct value_type { typedef T type; };

// If a terminal has typedef'ed value_type, then use it:
template <class T>
struct value_type<T,
    typename std::enable_if<
        std::is_same<typename T::value_type, typename T::value_type>::value
    >::type>
{
    typedef typename T::value_type type;
};


//---------------------------------------------------------------------------
// Kernel source generation
//---------------------------------------------------------------------------

// Some terminals need preamble (e.g. struct declaration or helper function).
// But most of them do not:
template <class T>
struct terminal_preamble {
    static std::string get(const T&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

// How to declare OpenCL kernel parameters for a terminal:
template <class Term, class Enable = void>
struct kernel_param_declaration {
    static std::string get(const Term&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        std::ostringstream s;
        s << ",\n\t" << type_name<Term>() << " " << prm_name;
        return s.str();
    }
};

// Local terminal initialization (e.g. temporary declaration)
template <class Term, class Enable = void>
struct local_terminal_init {
    static std::string get(const Term&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

// Partial expression for a terminal:
template <class Term, class Enable = void>
struct partial_vector_expr {
    static std::string get(const Term&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        return prm_name;
    }
};

// How to set OpenCL kernel arguments for a terminal:
template <class Term, class Enable = void>
struct kernel_arg_setter {
    static void set(const Term &term,
            cl::Kernel &kernel, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &position, detail::kernel_generator_state_ptr)
    {
        kernel.setArg(position++, term);
    }
};

// How to deduce queue list, partitioning and size from a terminal:
template <class T, class Enable = void>
struct expression_properties {
    static void get(const T &/*term*/,
            std::vector<cl::CommandQueue> &/*queue_list*/,
            std::vector<size_t> &/*partition*/,
            size_t &/*size*/
            )
    { }
};

//---------------------------------------------------------------------------
// Scalars and helper types/functions used in multivector expressions
//---------------------------------------------------------------------------
template <class T, class Enable = void>
struct is_multiscalar : std::false_type {};

// Arithmetic scalars
template <class T>
struct is_multiscalar< T,
    typename std::enable_if< is_cl_native<T>::value >::type
    > : std::true_type
{};

// Number of components in a multivector expression terminal.
template <class T>
struct number_of_components : boost::mpl::size_t<1> {};

// Type of I-th component of a multivector expression terminal.
template <size_t I, class T, class Enable = void>
struct component { typedef T type; };

#ifndef BOOST_NO_VARIADIC_TEMPLATES
template <typename... T>
struct And : std::true_type {};

template <typename Head, typename... Tail>
struct And<Head, Tail...>
    : std::conditional<Head::value, And<Tail...>, std::false_type>::type
{};
#endif

// std::tuple<...>
#ifndef BOOST_NO_VARIADIC_TEMPLATES
template <class... Args>
struct is_multiscalar<std::tuple<Args...>,
    typename std::enable_if<And< is_cl_native<Args>... >::type::value >::type
    > : std::true_type
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
    typename std::enable_if< is_cl_native<T>::value >::type
    > : std::true_type
{};

template <class T, size_t N>
struct number_of_components< std::array<T, N> > : boost::mpl::size_t<N> {};

template <size_t I, class T, size_t N>
struct component< I, std::array<T, N> > { typedef T type; };

// C-style arrays
template <class T, size_t N>
struct is_multiscalar< T[N],
    typename std::enable_if< is_cl_native<T>::value >::type
    > : std::true_type
{};

template <class T, size_t N>
struct number_of_components< T[N] > : boost::mpl::size_t<N> {};

template <size_t I, class T, size_t N>
struct component< I, T[N] > { typedef T type; };

// Terminals allowed in multivector expressions.
template <class Term, class Enable = void>
struct is_multivector_expr_terminal : std::false_type { };

template <class T>
struct is_multivector_expr_terminal< T,
    typename std::enable_if< is_multiscalar<T>::value >::type
    >
    : std::true_type
{ };

// Extract component directly from terminal rather than from value(terminal):
template <class T, class Enable = void>
struct proto_terminal_is_value : std::false_type { };

template <class T>
struct terminal_is_value :
    boost::proto::matches<
            typename boost::proto::result_of::as_expr<T>::type,
            boost::proto::and_<
                boost::proto::terminal< boost::proto::_ >,
                boost::proto::if_< proto_terminal_is_value< boost::proto::_value >() >
            >
    >
{};

/* Type trait to determine if an expression is scalable.
 *
 * The expression should have a type `value_type` and a field `scale` of that
 * type, this enables operator* and operator/.
 */
template <class T> struct is_scalable : std::false_type {};

} // namespace traits

//---------------------------------------------------------------------------
// Extracting components from multivector expression terminals
//---------------------------------------------------------------------------
template <size_t I, typename T, size_t N>
inline T& get(T t[N]) {
    static_assert(I < N, "Component number out of bounds");
    return t[I];
}

template <size_t I, typename T, size_t N>
inline const T& get(const T t[N]) {
    static_assert(I < N, "Component number out of bounds");
    return t[I];
}

template <size_t I, typename T>
inline T& get(T &t) {
    return t;
}

// Scalable expressions may be multiplied by a scalar:
template <class T>
typename std::enable_if<vex::traits::is_scalable<T>::value, T>::type
operator*(const T &expr, const typename T::value_type &factor) {
    T scaled_expr(expr);
    scaled_expr.scale *= factor;
    return scaled_expr;
}

// Scalable expressions may be multiplied by a scalar:
template <class T>
typename std::enable_if<vex::traits::is_scalable<T>::value, T>::type
operator*(const typename T::value_type &factor, const T &expr) {
    return expr * factor;
}

// Scalable expressions may be divided by a scalar:
template <class T> typename std::enable_if<vex::traits::is_scalable<T>::value, T>::type
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
        >, \
        boost::proto::or_< \
            boost::proto::address_of < grammar >, \
            boost::proto::dereference< grammar > \
        >, \
        boost::proto::or_< \
            boost::proto::if_else_< grammar, grammar, grammar > \
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

BUILTIN_FUNCTION_2( abs_diff );
BUILTIN_FUNCTION_1( acos );
BUILTIN_FUNCTION_1( acosh );
BUILTIN_FUNCTION_1( acospi );
BUILTIN_FUNCTION_2( add_sat );
BUILTIN_FUNCTION_1( all );
BUILTIN_FUNCTION_1( any );
BUILTIN_FUNCTION_1( asin );
BUILTIN_FUNCTION_1( asinh );
BUILTIN_FUNCTION_1( asinpi );
BUILTIN_FUNCTION_1( atan );
BUILTIN_FUNCTION_2( atan2 );
BUILTIN_FUNCTION_2( atan2pi );
BUILTIN_FUNCTION_1( atanh );
BUILTIN_FUNCTION_1( atanpi );
BUILTIN_FUNCTION_3( bitselect );
BUILTIN_FUNCTION_1( cbrt );
BUILTIN_FUNCTION_1( ceil );
BUILTIN_FUNCTION_3( clamp )
BUILTIN_FUNCTION_1( clz );
BUILTIN_FUNCTION_2( copysign );
BUILTIN_FUNCTION_1( cos );
BUILTIN_FUNCTION_1( cosh );
BUILTIN_FUNCTION_1( cospi );
BUILTIN_FUNCTION_2( cross );
BUILTIN_FUNCTION_1( degrees )
BUILTIN_FUNCTION_2( distance );
BUILTIN_FUNCTION_2( dot );
BUILTIN_FUNCTION_1( erf );
BUILTIN_FUNCTION_1( erfc );
BUILTIN_FUNCTION_1( exp );
BUILTIN_FUNCTION_1( exp10 );
BUILTIN_FUNCTION_1( exp2 );
BUILTIN_FUNCTION_1( expm1 );
BUILTIN_FUNCTION_1( fabs );
BUILTIN_FUNCTION_2( fast_distance );
BUILTIN_FUNCTION_1( fast_length );
BUILTIN_FUNCTION_1( fast_normalize );
BUILTIN_FUNCTION_2( fdim );
BUILTIN_FUNCTION_1( floor );
BUILTIN_FUNCTION_3( fma );
BUILTIN_FUNCTION_2( fmax );
BUILTIN_FUNCTION_2( fmin );
BUILTIN_FUNCTION_2( fmod );
BUILTIN_FUNCTION_2( fract );
BUILTIN_FUNCTION_2( frexp );
BUILTIN_FUNCTION_2( hadd );
BUILTIN_FUNCTION_2( hypot );
BUILTIN_FUNCTION_1( ilogb );
BUILTIN_FUNCTION_2( isequal );
BUILTIN_FUNCTION_1( isfinite );
BUILTIN_FUNCTION_2( isgreater );
BUILTIN_FUNCTION_2( isgreaterequal );
BUILTIN_FUNCTION_1( isinf );
BUILTIN_FUNCTION_2( isless );
BUILTIN_FUNCTION_2( islessequal );
BUILTIN_FUNCTION_2( islessgreater );
BUILTIN_FUNCTION_1( isnan );
BUILTIN_FUNCTION_1( isnormal );
BUILTIN_FUNCTION_2( isnotequal );
BUILTIN_FUNCTION_2( isordered );
BUILTIN_FUNCTION_2( isunordered );
BUILTIN_FUNCTION_2( ldexp );
BUILTIN_FUNCTION_1( length );
BUILTIN_FUNCTION_1( lgamma );
BUILTIN_FUNCTION_2( lgamma_r );
BUILTIN_FUNCTION_1( log );
BUILTIN_FUNCTION_1( log10 );
BUILTIN_FUNCTION_1( log1p );
BUILTIN_FUNCTION_1( log2 );
BUILTIN_FUNCTION_1( logb );
BUILTIN_FUNCTION_3( mad );
BUILTIN_FUNCTION_3( mad24 );
BUILTIN_FUNCTION_3( mad_hi );
BUILTIN_FUNCTION_3( mad_sat );
BUILTIN_FUNCTION_2( max );
BUILTIN_FUNCTION_2( maxmag );
BUILTIN_FUNCTION_2( min );
BUILTIN_FUNCTION_2( minmag );
BUILTIN_FUNCTION_3( mix );
BUILTIN_FUNCTION_2( modf );
BUILTIN_FUNCTION_2( mul_hi );
BUILTIN_FUNCTION_1( nan );
BUILTIN_FUNCTION_2( nextafter );
BUILTIN_FUNCTION_1( normalize );
BUILTIN_FUNCTION_1( popcount );
BUILTIN_FUNCTION_2( pow );
BUILTIN_FUNCTION_2( pown );
BUILTIN_FUNCTION_2( powr );
BUILTIN_FUNCTION_1( radians );
BUILTIN_FUNCTION_2( remainder );
BUILTIN_FUNCTION_3( remquo );
BUILTIN_FUNCTION_2( rhadd );
BUILTIN_FUNCTION_1( rint );
BUILTIN_FUNCTION_2( rootn );
BUILTIN_FUNCTION_2( rotate );
BUILTIN_FUNCTION_1( round );
BUILTIN_FUNCTION_1( rsqrt );
BUILTIN_FUNCTION_3( select );
BUILTIN_FUNCTION_2( shuffle );
BUILTIN_FUNCTION_3( shuffle2 );
BUILTIN_FUNCTION_1( sign );
BUILTIN_FUNCTION_1( signbit );
BUILTIN_FUNCTION_1( sin );
BUILTIN_FUNCTION_2( sincos );
BUILTIN_FUNCTION_1( sinh );
BUILTIN_FUNCTION_1( sinpi );
BUILTIN_FUNCTION_3( smoothstep );
BUILTIN_FUNCTION_1( sqrt );
BUILTIN_FUNCTION_2( step );
BUILTIN_FUNCTION_2( sub_sat );
BUILTIN_FUNCTION_1( tan );
BUILTIN_FUNCTION_1( tanh );
BUILTIN_FUNCTION_1( tanpi );
BUILTIN_FUNCTION_1( tgamma );
BUILTIN_FUNCTION_1( trunc );
BUILTIN_FUNCTION_2( upsample );

#undef BUILTIN_FUNCTION_1
#undef BUILTIN_FUNCTION_2
#undef BUILTIN_FUNCTION_3

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
abs(const Arg &arg) {
    return boost::proto::make_expr<boost::proto::tag::function>(
            abs_func(),
            boost::ref(arg)
            );
}

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
abs(const Arg &arg) {
    return boost::proto::make_expr<boost::proto::tag::function>(
            fabs_func(),
            boost::ref(arg)
            );
}

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
    typedef RetType value_type;

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
    static void show_arg(std::ostream &os, unsigned pos) {
        if (pos > 1) os << ",";
        os << "\n\t" << type_name<Head>() << " prm" << pos;
    }

    template <class Head, class... Tail>
    static typename std::enable_if<sizeof...(Tail), void>::type
    show_arg(std::ostream &os, unsigned pos) {
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
    typedef RetType value_type; \
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
// Vector grammar
//---------------------------------------------------------------------------
// Grammar for vector expressions that may be processed with single kernel:
struct vector_expr_grammar
    : boost::proto::or_<
          boost::proto::and_<
              boost::proto::terminal< boost::proto::_ >,
              boost::proto::if_< traits::is_vector_expr_terminal< boost::proto::_value >() >
          >,
          BUILTIN_OPERATIONS(vector_expr_grammar),
          USER_FUNCTIONS(vector_expr_grammar)
      >
{};

// Grammar for additive expressions
// (each additive term requires separate kernel):
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
        typename std::enable_if<
                traits::hold_terminal_by_reference<T>::value
            >::type
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
// Multivector grammar
//---------------------------------------------------------------------------
struct multivector_expr_grammar
    : boost::proto::or_<
          boost::proto::and_<
              boost::proto::terminal< boost::proto::_ >,
              boost::proto::if_< traits::is_multivector_expr_terminal< boost::proto::_value >() >
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
        typename std::enable_if<
                traits::hold_terminal_by_reference<T>::value
            >::type
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

namespace traits {

// Number of components in a multivector expression.
struct multiex_dimension :
    boost::proto::or_ <
        boost::proto::when <
            boost::proto::terminal< boost::proto::_ >,
            traits::number_of_components<boost::proto::_>()
        > ,
        boost::proto::when <
            boost::proto::nary_expr<boost::proto::_, boost::proto::vararg<boost::proto::_> >,
            boost::proto::fold<boost::proto::_,
                boost::mpl::size_t<0>(),
                boost::mpl::max<multiex_dimension, boost::proto::_state>()>()
        >
    >
{};

template <class Expr, class Enable = void>
struct get_dimension {};

template <class Expr>
struct get_dimension<Expr, typename std::enable_if<
        boost::proto::matches<
            typename boost::proto::result_of::as_expr<Expr>::type,
            multivector_expr_grammar
        >::value &&
        !is_tuple<typename std::decay<Expr>::type>::value
    >::type>
{
    const static size_t value = std::result_of<traits::multiex_dimension(Expr)>::type::value;
};

template <class Expr>
struct get_dimension<Expr, typename std::enable_if<
        is_tuple<typename std::decay<Expr>::type>::value
    >::type>
{
    const static size_t value = std::tuple_size<typename std::decay<Expr>::type>::value;
};

} // namespace traits

//---------------------------------------------------------------------------
// Expression Transforms and evaluation contexts
//---------------------------------------------------------------------------
namespace detail {

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

// Generic terminal processing functor
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

// Extract (and process) terminals from a vector expression.
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

// Base class for stateful expression evaluation contexts .
struct expression_context {
    std::ostream &os;
    const cl::Device &device;
    mutable int prm_idx;
    int fun_idx;
    std::string prefix;
    kernel_generator_state_ptr state;

    expression_context(
            std::ostream &os, const cl::Device &device,
            const std::string &prefix,
            kernel_generator_state_ptr state
            )
        : os(os), device(device), prm_idx(0), fun_idx(0),
          prefix(prefix), state(state)
    {}
};

// Outputs kernel preamble.
struct output_terminal_preamble : public expression_context {

    output_terminal_preamble(
            std::ostream &os, const cl::Device &device,
            const std::string &prefix,
            kernel_generator_state_ptr state
            )
        : expression_context(os, device, prefix, state)
    {}

    // Any expression except user function or terminal is only interesting
    // for its children:
    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {
        typedef void result_type;

        void operator()(const Expr &expr, output_terminal_preamble &ctx) const
        {
            boost::fusion::for_each( expr,
                    do_eval<output_terminal_preamble>(ctx));
        }
    };

    // Function is either builtin (not interesting) or user-defined:
    template <typename Expr>
    struct eval<Expr, boost::proto::tag::function> {
        typedef void result_type;

        // Builtin function is only interesting for its children:
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
        operator()(const FunCall &expr, output_terminal_preamble &ctx) const
        {
            boost::fusion::for_each(
                    boost::fusion::pop_front(expr),
                    do_eval<output_terminal_preamble>(ctx)
                    );
        }

        // User-defined function needs to be defined.
        // Then look at its children:
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
        operator()(const FunCall &expr, output_terminal_preamble &ctx) const {
            std::ostringstream name;
            name << ctx.prefix << "_func_" << ++ctx.fun_idx;

            // Output function definition and continue with parameters.
            boost::proto::result_of::value<
                typename boost::proto::result_of::child_c<FunCall,0>::type
            >::type::define(ctx.os, name.str());

            boost::fusion::for_each(
                    boost::fusion::pop_front(expr),
                    do_eval<output_terminal_preamble>(ctx)
                    );
        }
    };

    // Some terminals have preambles too:
    template <typename T>
    struct eval<T, boost::proto::tag::terminal> {
        typedef void result_type;

        template <class Term>
        typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, output_terminal_preamble &ctx) const
        {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::terminal_preamble<
                typename std::decay<Term>::type
                >::get(term, ctx.device, prm_name.str(), ctx.state);
        }

        template <class Term>
        typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, output_terminal_preamble &ctx) const
        {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::terminal_preamble<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
                >::get(boost::proto::value(term), ctx.device, prm_name.str(), ctx.state);
        }
    };
};

// Performs local initialization (such as declaring and initializing temporary values).
struct output_local_preamble : public expression_context {

    output_local_preamble(
            std::ostream &os, const cl::Device &device,
            const std::string &prefix,
            kernel_generator_state_ptr state
            )
        : expression_context(os, device, prefix, state)
    {}

    // Any expression except user function or terminal is only interesting
    // for its children:
    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {
        typedef void result_type;

        void operator()(const Expr &expr, output_local_preamble &ctx) const
        {
            boost::fusion::for_each( expr,
                    do_eval<output_local_preamble>(ctx));
        }
    };

    // Functions are only interesting for their parameters:
    template <typename Expr>
    struct eval<Expr, boost::proto::tag::function> {
        typedef void result_type;

        // Builtin function is only interesting for its children:
        template <class FunCall>
        void operator()(const FunCall &expr, output_local_preamble &ctx) const
        {
            boost::fusion::for_each(
                    boost::fusion::pop_front(expr),
                    do_eval<output_local_preamble>(ctx)
                    );
        }
    };

    // Some terminals need to be initialized:
    template <typename T>
    struct eval<T, boost::proto::tag::terminal> {
        typedef void result_type;

        template <class Term>
        typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, output_local_preamble &ctx) const
        {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::local_terminal_init<
                typename std::decay<Term>::type
                >::get(term, ctx.device, prm_name.str(), ctx.state);
        }

        template <class Term>
        typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, output_local_preamble &ctx) const
        {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::local_terminal_init<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
                >::get(boost::proto::value(term), ctx.device, prm_name.str(), ctx.state);
        }
    };
};

// Builds textual representation for a vector expression.
struct vector_expr_context : public expression_context {

    vector_expr_context(
            std::ostream &os, const cl::Device &device,
            const std::string &prefix,
            kernel_generator_state_ptr state
            )
        : expression_context(os, device, prefix, state)
    {}

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
    struct eval<Expr, boost::proto::tag::address_of> {
        typedef void result_type;
        void operator()(const Expr &expr, vector_expr_context &ctx) const {
            ctx.os << "( &( ";
            boost::proto::eval(boost::proto::child(expr), ctx);
            ctx.os << " ) )";
        }
    };

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::dereference> {
        typedef void result_type;
        void operator()(const Expr &expr, vector_expr_context &ctx) const {
            ctx.os << "( *( ";
            boost::proto::eval(boost::proto::child(expr), ctx);
            ctx.os << " ) )";
        }
    };

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::if_else_> {
        typedef void result_type;
        void operator()(const Expr &expr, vector_expr_context &ctx) const {
            ctx.os << "( ";
            boost::proto::eval(boost::proto::child_c<0>(expr), ctx);
            ctx.os << " ? ";
            boost::proto::eval(boost::proto::child_c<1>(expr), ctx);
            ctx.os << " : ";
            boost::proto::eval(boost::proto::child_c<2>(expr), ctx);
            ctx.os << " )";
        }
    };

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
            ctx.os << ctx.prefix << "_func_" << ++ctx.fun_idx << "( ";
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
        typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, vector_expr_context &ctx) const {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::partial_vector_expr<
                    typename std::decay<Term>::type
                >::get(term, ctx.device, prm_name.str(), ctx.state);
        }

        template <typename Term>
        typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
        operator()(const Term &term, vector_expr_context &ctx) const {
            std::ostringstream prm_name;
            prm_name << ctx.prefix << "_" << ++ctx.prm_idx;

            ctx.os << traits::partial_vector_expr<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
                >::get(boost::proto::value(term), ctx.device, prm_name.str(), ctx.state);
        }
    };
};

struct declare_expression_parameter : expression_context {

    declare_expression_parameter(std::ostream &os, const cl::Device &device,
            const std::string &prefix,
            kernel_generator_state_ptr state
            )
        : expression_context(os, device, prefix, state)
    {}

    template <typename Term>
    typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        std::ostringstream prm_name;
        prm_name << prefix << "_" << ++prm_idx;

        os << traits::kernel_param_declaration<
                typename std::decay<Term>::type
            >::get(term, device, prm_name.str(), state);
    }

    template <typename Term>
    typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        std::ostringstream prm_name;
        prm_name << prefix << "_" << ++prm_idx;

        os << traits::kernel_param_declaration<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
            >::get(boost::proto::value(term), device, prm_name.str(), state);
    }
};

struct set_expression_argument {
    cl::Kernel &krn;
    unsigned dev, &pos;
    size_t part_start;
    kernel_generator_state_ptr state;

    set_expression_argument(cl::Kernel &krn, unsigned dev, unsigned &pos, size_t part_start,
            kernel_generator_state_ptr state
            )
        : krn(krn), dev(dev), pos(pos), part_start(part_start), state(state)
    {}

    template <typename Term>
    typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        traits::kernel_arg_setter<
            typename std::decay<Term>::type
            >::set(term, krn, dev, part_start, pos, state);
    }

    template <typename Term>
    typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        traits::kernel_arg_setter<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
            >::set(boost::proto::value(term), krn, dev, part_start, pos, state);
    }
};

struct get_expression_properties {
    mutable std::vector<cl::CommandQueue> queue;
    mutable std::vector<size_t> part;
    mutable size_t size;

    get_expression_properties() : size(0) {}

    size_t part_start(unsigned d) const {
        return part.empty() ? 0 : part[d];
    }

    size_t part_size(unsigned d) const {
        return part.empty() ? 0 : part[d + 1] - part[d];
    }

    template <typename Term>
    typename std::enable_if<traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        if (queue.empty())
            traits::expression_properties<
                typename std::decay<Term>::type
                >::get(term, queue, part, size);
    }

    template <typename Term>
    typename std::enable_if<!traits::terminal_is_value<Term>::value, void>::type
    operator()(const Term &term) const {
        if (queue.empty())
            traits::expression_properties<
                    typename std::decay<
                        typename boost::proto::result_of::value<
                            typename std::decay<Term>::type
                        >::type
                    >::type
                >::get(boost::proto::value(term), queue, part, size);
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
// Multiexpression component extractor
//---------------------------------------------------------------------------
template <size_t C>
struct extract_component : boost::proto::callable {
    template <class T>
    struct result;

    template <class This, class T>
    struct result< This(T) > {
        typedef const typename traits::component< C,
                typename std::decay<T>::type
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
                boost::proto::if_< traits::proto_terminal_is_value< boost::proto::_value >() >
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
// Expression result type deduction
//---------------------------------------------------------------------------

// Proxy for value_type<>
struct get_value_type : boost::proto::callable {
    template <class T> struct result;

    template <class This, class T>
    struct result< This(T) > {
        typedef
            typename traits::value_type< typename std::decay<T>::type >::type
            type;
    };
};

// Proxy for std::common_type<>
struct common_type : boost::proto::callable {
    template <class T, class Enable = void> struct result;

    template <class This, class T1, class T2>
    struct result< This(T1, T2) >
    {
        typedef typename std::decay<T1>::type D1;
        typedef typename std::decay<T2>::type D2;

        static_assert(
                cl_vector_length<D1>::value == 1 ||
                cl_vector_length<D2>::value == 1 ||
                cl_vector_length<D1>::value == cl_vector_length<D2>::value,
                "Operations with vectors of different lengths are not supported"
                );

        typedef
            typename cl_vector_of<
                typename std::common_type<
                    typename cl_scalar_of<D1>::type,
                    typename cl_scalar_of<D2>::type
                >::type,
                boost::mpl::max<
                    boost::mpl::size_t<cl_vector_length<D1>::value>,
                    boost::mpl::size_t<cl_vector_length<D2>::value>
                >::type::value
            >::type type;
    };
};


struct deduce_value_type
    : boost::proto::or_<
        // Terminals are passed to value_type<>
        boost::proto::when <
            boost::proto::and_<
                boost::proto::terminal< boost::proto::_ >,
                boost::proto::if_< traits::proto_terminal_is_value< boost::proto::_value >() >
            >,
            get_value_type( boost::proto::_ )
        > ,
        boost::proto::when <
            boost::proto::terminal< boost::proto::_ >,
            get_value_type( boost::proto::_value )
        >,
        // Result of logical operations is bool for scalars and long for vector
        // types. Lets keep it simple and return long.
        boost::proto::when <
            boost::proto::or_<
                boost::proto::or_<
                    boost::proto::less          < boost::proto::_, boost::proto::_ >,
                    boost::proto::greater       < boost::proto::_, boost::proto::_ >,
                    boost::proto::less_equal    < boost::proto::_, boost::proto::_ >,
                    boost::proto::greater_equal < boost::proto::_, boost::proto::_ >,
                    boost::proto::equal_to      < boost::proto::_, boost::proto::_ >,
                    boost::proto::not_equal_to  < boost::proto::_, boost::proto::_ >
                >,
                boost::proto::or_<
                    boost::proto::logical_and   < boost::proto::_, boost::proto::_ >,
                    boost::proto::logical_or    < boost::proto::_, boost::proto::_ >,
                    boost::proto::logical_not   < boost::proto::_ >
                >
            >,
            cl_long()
        >,
        boost::proto::when <
            boost::proto::if_else_< boost::proto::_, boost::proto::_, boost::proto::_ >,
            common_type( deduce_value_type(boost::proto::_child1), deduce_value_type(boost::proto::_child2) )
        >,
        // We assume that type of builtin function is the common type of its
        // arguments (TODO: this could be wrong for some functions).
        boost::proto::when <
            boost::proto::function<
                boost::proto::terminal<
                    boost::proto::convertible_to< builtin_function >
                >,
                boost::proto::vararg< boost::proto::_ >
                >,
            boost::proto::fold<
                boost::proto::functional::pop_front( boost::proto::_ ),
                char(),
                common_type(deduce_value_type, boost::proto::_state)
            >()
        >,
        // User-defined functions know their return type
        boost::proto::when <
            boost::proto::function<
                boost::proto::terminal<
                    boost::proto::convertible_to< user_function >
                >,
                boost::proto::vararg< boost::proto::_ >
                >,
            get_value_type( boost::proto::functional::value(boost::proto::_child0) )
        >,
        // Fold the operands of nary epxressions with std::common_type<>
        boost::proto::when <
            boost::proto::nary_expr<boost::proto::_, boost::proto::vararg<boost::proto::_> >,
            boost::proto::fold<
                boost::proto::_,
                char(),
                common_type(deduce_value_type, boost::proto::_state)
            >()
        >
      >
{};

// Hide the ugly type deduction details in an easy to use metafunction:
template <class Expr>
struct return_type {
    typedef
        typename std::decay<
                typename boost::result_of<
                    deduce_value_type(
                        typename boost::proto::result_of::as_expr<
                            typename std::decay<Expr>::type
                        >::type
                        )
                >::type
            >::type
        type;
};


// Kernel cache (is a map from context handle to a kernel)
struct kernel_cache_entry {
    cl::Kernel kernel;
    size_t     wgsize;

    kernel_cache_entry(const cl::Kernel &kernel, size_t wgsize)
        : kernel(kernel), wgsize(wgsize)
    {}
};

struct kernel_cache;

template <bool dummy = true>
struct cache_register {
    static_assert(dummy, "Dummy parameter should be true");

    static std::deque<kernel_cache*> caches;

    static void add(kernel_cache *cache) {
        caches.push_back(cache);
    }

    static void clear();
    static void erase(cl_context key);
};

template <bool dummy>
std::deque<kernel_cache*> cache_register<dummy>::caches;

struct kernel_cache {
    typedef std::map<cl_context, kernel_cache_entry> store_type;

    store_type store;

    kernel_cache() {
        cache_register<>::add(this);
    }

    template <typename T>
    std::pair<store_type::iterator, bool> insert(T&& item) {
        return store.insert(std::forward<T>(item));
    }

    store_type::const_iterator end() const {
        return store.end();
    }

    template <typename T>
    store_type::iterator find(T&& key) {
        return store.find( std::forward<T>(key) );
    }

    void clear() {
        store.clear();
    }

    void erase(cl_context key) {
        store.erase(key);
    }
};

template <bool dummy>
void cache_register<dummy>::clear() {
    for(auto c = caches.begin(); c != caches.end(); ++c)
        (*c)->clear();
}

template <bool dummy>
void cache_register<dummy>::erase(cl_context key) {
    for(auto c = caches.begin(); c != caches.end(); ++c)
        (*c)->erase(key);
}

//---------------------------------------------------------------------------
// Assign expression to lhs
//---------------------------------------------------------------------------
template <class OP, class LHS, class RHS>
void assign_expression(LHS &lhs, const RHS &rhs,
        const std::vector<cl::CommandQueue> &queue,
        const std::vector<size_t> &part
        )
{
    static kernel_cache cache;

    for(unsigned d = 0; d < queue.size(); d++) {
        cl::Context context = qctx(queue[d]);
        cl::Device  device  = qdev(queue[d]);

        auto kernel = cache.find(context());

        if (kernel == cache.end()) {
            std::ostringstream source;

            source << standard_kernel_header(device);

            output_terminal_preamble termpream(source, device, "prm", empty_state());

            boost::proto::eval(boost::proto::as_child(lhs), termpream);
            boost::proto::eval(boost::proto::as_child(rhs), termpream);

            source << "kernel void vexcl_vector_kernel(\n"
                   "\t" << type_name<size_t>() << " n";

            declare_expression_parameter declare(source, device, "prm", empty_state());

            extract_terminals()(boost::proto::as_child(lhs), declare);
            extract_terminals()(boost::proto::as_child(rhs), declare);

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

            output_local_preamble loc_init(source, device, "prm", empty_state());
            boost::proto::eval(boost::proto::as_child(lhs), loc_init);
            boost::proto::eval(boost::proto::as_child(rhs), loc_init);

            vector_expr_context expr_ctx(source, device, "prm", empty_state());

            source << "\t\t";

            boost::proto::eval(boost::proto::as_child(lhs), expr_ctx);
            source << " " << OP::string() << " ";
            boost::proto::eval(boost::proto::as_child(rhs), expr_ctx);

            source << ";\n\t}\n}\n";

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, "vexcl_vector_kernel");
            size_t wgs = kernel_workgroup_size(krn, device);

            kernel = cache.insert(std::make_pair(
                        context(), kernel_cache_entry(krn, wgs)
                        )).first;
        }

        if (size_t psize = part[d + 1] - part[d]) {
            size_t w_size = kernel->second.wgsize;
            size_t g_size = num_workgroups(device) * w_size;

            unsigned pos = 0;
            kernel->second.kernel.setArg(pos++, psize);

            set_expression_argument setarg(kernel->second.kernel, d, pos, part[d], empty_state());

            extract_terminals()( boost::proto::as_child(lhs), setarg);
            extract_terminals()( boost::proto::as_child(rhs), setarg);

            queue[d].enqueueNDRangeKernel(
                    kernel->second.kernel, cl::NullRange, g_size, w_size
                    );
        }
    }
}

// Static for loop
template <size_t Begin, size_t End>
class static_for {
    public:
        template <class Func>
        static void loop(Func &&f) {
            iterate<Begin>(f);
        }

    private:
        template <size_t I, class Func>
        static typename std::enable_if<(I < End)>::type
        iterate(Func &&f) {
            f.template apply<I>();
            iterate<I + 1>(f);
        }

        template <size_t I, class Func>
        static typename std::enable_if<(I >= End)>::type
        iterate(Func&&)
        { }
};

template <class OP, class LHS, class RHS>
struct subexpression_assigner {
    const LHS &lhs;
    const RHS &rhs;
    const std::vector<cl::CommandQueue> &queue;
    const std::vector<size_t> &part;

    subexpression_assigner(LHS &lhs, const RHS &rhs,
            const std::vector<cl::CommandQueue> &queue,
            const std::vector<size_t> &part
            )
        : lhs(lhs), rhs(rhs), queue(queue), part(part) {}

    template <size_t I>
    void apply() const {
        detail::assign_expression<OP>(
                subexpression<I>::get(lhs),
                subexpression<I>::get(rhs),
                queue, part);
    }
};

template <class LHS, class RHS>
struct preamble_constructor {
    const LHS &lhs;
    const RHS &rhs;

    kernel_generator_state_ptr state;
    mutable detail::output_terminal_preamble lhs_ctx;
    mutable detail::output_terminal_preamble rhs_ctx;

    preamble_constructor(const LHS &lhs, const RHS &rhs,
            std::ostream &source, const cl::Device &device
            )
        : lhs(lhs), rhs(rhs), state(empty_state()),
          lhs_ctx(source, device, "lhs", state),
          rhs_ctx(source, device, "rhs", state)
    { }

    template <size_t I>
    void apply() const {
        boost::proto::eval(subexpression<I>::get(lhs), lhs_ctx);
        boost::proto::eval(subexpression<I>::get(rhs), rhs_ctx);
    }
};

template <class LHS, class RHS>
struct parameter_declarator {
    const LHS &lhs;
    const RHS &rhs;

    kernel_generator_state_ptr state;
    mutable detail::declare_expression_parameter lhs_ctx;
    mutable detail::declare_expression_parameter rhs_ctx;

    parameter_declarator(const LHS &lhs, const RHS &rhs,
            std::ostream &source, const cl::Device &device)
        : lhs(lhs), rhs(rhs), state(empty_state()),
          lhs_ctx(source, device, "lhs", state),
          rhs_ctx(source, device, "rhs", state)
    { }

    template <size_t I>
    void apply() const {
        extract_terminals()(subexpression<I>::get(lhs), lhs_ctx);
        extract_terminals()(subexpression<I>::get(rhs), rhs_ctx);
    }
};

template <class LHS, class RHS>
struct expression_init {
    const LHS &lhs;
    const RHS &rhs;

    std::ostream &source;

    kernel_generator_state_ptr state;
    mutable detail::output_local_preamble rhs_pre;
    mutable detail::vector_expr_context   rhs_ctx;

    expression_init(const LHS &lhs, const RHS &rhs,
            std::ostream &source, const cl::Device &device)
        : lhs(lhs), rhs(rhs), source(source), state(empty_state()),
          rhs_pre(source, device, "rhs", state),
          rhs_ctx(source, device, "rhs", state)
    { }

    template <size_t I>
    void apply() const {
        boost::proto::eval(subexpression<I>::get(rhs), rhs_pre);

        typedef
            typename return_type<decltype(subexpression<I>::get(lhs))>::type
            RT;

        source << "\t\t" << type_name<RT>() << " buf_" << I + 1 << " = ";

        boost::proto::eval(subexpression<I>::get(rhs), rhs_ctx);
        source << ";\n";
    }
};

template <class OP, class LHS>
struct expression_finalize {
    const LHS &lhs;

    std::ostream &source;

    kernel_generator_state_ptr state;
    mutable detail::output_local_preamble lhs_pre;
    mutable detail::vector_expr_context   lhs_ctx;

    expression_finalize(const LHS &lhs,
            std::ostream &source, const cl::Device &device)
        : lhs(lhs), source(source), state(empty_state()),
          lhs_pre(source, device, "lhs", state),
          lhs_ctx(source, device, "lhs", state)
    { }

    template <size_t I>
    void apply() const {
        boost::proto::eval(subexpression<I>::get(lhs), lhs_pre);
        source << "\t\t";
        boost::proto::eval(subexpression<I>::get(lhs), lhs_ctx);
        source << " " << OP::string() << " buf_" << I + 1 << ";\n";
    }
};

template <class LHS, class RHS>
struct kernel_arg_setter {
    const LHS &lhs;
    const RHS &rhs;

    mutable detail::set_expression_argument ctx;

    kernel_arg_setter(const LHS &lhs, const RHS &rhs,
            cl::Kernel &krn, unsigned dev, size_t offset, unsigned &pos)
        : lhs(lhs), rhs(rhs), ctx(krn, dev, pos, offset, empty_state())
    { }

    template <size_t I>
    void apply() const {
        detail::extract_terminals()(subexpression<I>::get(lhs), ctx);
        detail::extract_terminals()(subexpression<I>::get(rhs), ctx);
    }
};

template <class OP, class LHS, class RHS>
void assign_multiexpression( LHS &lhs, const RHS &rhs,
        const std::vector<cl::CommandQueue> &queue,
        const std::vector<size_t> &part
        )
{

    typedef traits::get_dimension<LHS> N;

    static kernel_cache cache;

    // 1. If any device in context is CPU, then do not fuse the kernel,
    //    but assign components individually (this works better with CPU
    //    caches).
    // 2. If dimension of the multiexpression is 1, then assign_expression()
    //    would work better as well (no need to spend registers on temp
    //    variables).
    if (
            (N::value == 1) ||
            std::any_of(queue.begin(), queue.end(),
                [](const cl::CommandQueue &q) { return is_cpu(qdev(q)); })
       )
    {
        static_for<0, N::value>::loop(
                subexpression_assigner<OP, LHS, RHS>(lhs, rhs, queue, part)
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

            static_for<0, N::value>::loop(
                    preamble_constructor<LHS, RHS>(lhs, rhs, source, device)
                    );

            source << "kernel void vexcl_multivector_kernel(\n\t"
                   << type_name<size_t>() << " n";

            static_for<0, N::value>::loop(
                    parameter_declarator<LHS, RHS>(lhs, rhs, source, device)
                    );

            source <<
                "\n)\n{\n"
                "\tfor(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n";

            static_for<0, N::value>::loop(expression_init<LHS, RHS>(lhs, rhs, source, device));
            static_for<0, N::value>::loop(expression_finalize<OP, LHS>(lhs, source, device));

            source << "\t}\n}\n";

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, "vexcl_multivector_kernel");
            size_t wgs = kernel_workgroup_size(krn, device);

            kernel = cache.insert(std::make_pair(
                        context(), kernel_cache_entry(krn, wgs)
                        )).first;
        }

        if (size_t psize = part[d + 1] - part[d]) {
            size_t w_size = kernel->second.wgsize;
            size_t g_size = num_workgroups(device) * w_size;

            unsigned pos = 0;
            kernel->second.kernel.setArg(pos++, psize);

            static_for<0, N::value>::loop(
                    kernel_arg_setter<LHS, RHS>(lhs, rhs, kernel->second.kernel, d, part[d], pos)
                    );

            queue[d].enqueueNDRangeKernel(
                    kernel->second.kernel, cl::NullRange, g_size, w_size
                    );
        }
    }
}

} // namespace detail

/// Assignable tuple of expressions
template <class LHS>
struct expression_tuple {
    static const size_t NDIM = std::tuple_size<LHS>::value;

    const LHS lhs;

    expression_tuple(const LHS &lhs) : lhs(lhs) {}

#ifdef DOXYGEN
#  define ASSIGNMENT(cop, op) \
        /** \brief Multiexpression assignment.
         \details All operations are delegated to components of the multivector.
         */ \
        template <class RHS> \
        expression_tuple& operator cop(const RHS &rhs);
#else
#  define ASSIGNMENT(cop, op) \
    template <class RHS> \
    typename std::enable_if< \
        boost::proto::matches< \
            typename boost::proto::result_of::as_expr<RHS>::type, \
            multivector_expr_grammar \
        >::value || is_tuple<RHS>::value, \
        const expression_tuple& \
    >::type \
    operator cop(const RHS &rhs) const { \
        detail::get_expression_properties prop; \
        detail::extract_terminals()(detail::subexpression<0>::get(lhs), prop); \
        detail::assign_multiexpression<op>(lhs, rhs, prop.queue, prop.part); \
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
};

/// \endcond

#ifndef BOOST_NO_VARIADIC_TEMPLATES
/// Ties several vector expressions into writeable tuple.
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
template<class... Expr>
expression_tuple< std::tuple<const Expr&...> >
tie(const Expr&... expr) {
    return expression_tuple< std::tuple<const Expr&...> >( std::tie(expr...) );
}
#else

#define PRINT_TYPES(z, n, data) const Expr ## n &
#define PRINT_PARAM(z, n, data) const Expr ## n &expr ## n

#define TIE_VECTORS(z, n, data) \
template<BOOST_PP_ENUM_PARAMS(n, class Expr)> \
expression_tuple< std::tuple<BOOST_PP_ENUM(n, PRINT_TYPES, ~)> > \
tie( BOOST_PP_ENUM(n, PRINT_PARAM, ~) ) { \
    return expression_tuple< std::tuple<BOOST_PP_ENUM(n, PRINT_TYPES, ~)> >( std::tie(BOOST_PP_ENUM_PARAMS(n, expr)) ); \
}

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TIE_VECTORS, ~)

#undef TIE_VECTORS
#undef PRINT_PARAM
#undef PRINT_TYPES

#endif

/// Clears cached OpenCL kernels, allowing to release OpenCL contexts.
inline void purge_kernel_caches() {
    detail::cache_register<>::clear();
}

/// Clears cached OpenCL kernels, allowing to release OpenCL contexts.
inline void purge_kernel_caches(const cl::Context &context) {
    detail::cache_register<>::erase(context());
}

/// Clears cached OpenCL kernels, allowing to release OpenCL contexts.
inline void purge_kernel_caches(const std::vector<cl::CommandQueue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        detail::cache_register<>::erase( qctx(*q)() );
}

} // namespace vex;


#endif
