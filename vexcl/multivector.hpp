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
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/builtins.hpp>
#include <vexcl/vector_proto.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

// TODO: remove this
namespace proto = boost::proto;
using proto::_;

/// \cond INTERNAL

template <typename... T>
struct And : std::true_type {};

template <typename Head, typename... Tail>
struct And<Head, Tail...>
    : std::conditional<Head::value, And<Tail...>, std::false_type>::type
{};

template <class T, class Enable = void>
struct is_multiscalar : std::false_type
{};

template <class... Args>
struct is_multiscalar<std::tuple<Args...>,
    typename std::enable_if<And< std::is_arithmetic<Args>... >::type::value >::type >
    : std::true_type
{};

template <class T, size_t N>
struct is_multiscalar< std::array<T, N>, 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

template <class T>
struct is_multiscalar< T, 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

template <class T>
struct is_multiscalar< T[], 
    typename std::enable_if< std::is_arithmetic<T>::value >::type >
    : std::true_type
{};

struct multivector_terminal {};

template <size_t N, typename T>
inline T get(const T &t) {
    return t;
}

template <size_t N, typename T>
inline T get(const T t[]) {
    return t[N];
}

struct multivector_expr_grammar
    : proto::or_<
	  proto::or_<
	      proto::terminal< multivector_terminal >,
	      proto::and_<
	          proto::terminal< _ >,
		  proto::if_< is_multiscalar< proto::_value >() >
	      >
          >,
	  VEX_OPERATIONS(multivector_expr_grammar)
      >
{};

template <class Expr>
struct multivector_expression;

struct multivector_domain
    : proto::domain<
	proto::generator<multivector_expression>,
	multivector_expr_grammar
      >
{};

template <class Expr>
struct multivector_expression
    : proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
{
    typedef
	proto::extends< Expr, multivector_expression<Expr>, multivector_domain>
	base_type;

    multivector_expression(const Expr &expr = Expr()) : base_type(expr) {}
};

template <typename T, size_t N>
struct multivector
    : multivector_expression<
	typename proto::terminal< multivector_terminal >::type
      >
{};

/// \endcond

} // namespace vex

#endif
