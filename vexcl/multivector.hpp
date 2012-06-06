#ifndef MULTIVECTOR_HPP
#define MULTIVECTOR_HPP

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
 * \file   multivector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Pack of device vectors.
 */

#ifdef WIN32
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#ifndef _MSC_VER
#  define VEXCL_VARIADIC_TEMPLATES
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <array>
#include <vector>
#include <type_traits>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/vector.hpp>

namespace vex {

template <typename T, uint N>
class multivector {
    public:
	static const bool is_multiexpression = true;
	static const uint dim = N;
	typedef vex::vector<T> subtype;

	multivector(const std::vector<cl::CommandQueue> &queue, size_t size,
		cl_mem_flags flags = CL_MEM_READ_WRITE)
	{
	    static_assert(N > 0, "What's the point?");

	    for(uint i = 0; i < N; i++)
		vec[i] = vex::vector<T>(queue, size, 0, flags);
	}

	size_t size() const {
	    return vec[0].size();
	}

	const vex::vector<T>& operator[](uint i) const {
	    return vec[i];
	}

	vex::vector<T>& operator[](uint i) {
	    return vec[i];
	}

	template <class Expr>
	const multivector& operator=(const Expr& expr) {
	    for(uint i = 0; i < N; i++)
		vec[i] = expr[i];
	}

    private:
	std::array<vex::vector<T>,N> vec;
};

template<class T, class Enable = void>
struct multiex_traits {
    static const uint dim = T::dim;
    typedef typename T::subtype subtype;
};

template<class T>
struct multiex_traits<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    static const uint dim = 0;
    typedef T subtype;
};

template <class Expr>
typename std::enable_if<Expr::is_multiexpression, const typename Expr::subtype&>::type
extract_component(const Expr &expr, uint i) {
    return expr[i];
}

template <class Expr>
typename std::enable_if<std::is_arithmetic<Expr>::value, const Expr&>::type
extract_component(const Expr &expr, uint i) {
    return expr;
}

template <class T, class Enable = void>
struct valid_multiexpression {
    static const bool value = false;
};

template <typename T>
struct valid_multiexpression<T, typename std::enable_if<T::is_multiexpression>::type> {
    static const bool value = true;
};

template <typename T>
struct valid_multiexpression<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    static const bool value = true;
};

template <class T1, class T2, class Enable = void>
struct compatible_multiexpressions {
    static const bool value = false;
};

template <class T1, class T2>
struct compatible_multiexpressions<T1, T2,
    typename std::enable_if<
	T1::is_multiexpression &&
	T2::is_multiexpression &&
	T1::dim == T2::dim>::type
	>
{
    static const bool value = true;
};

template <class T1, class T2>
struct compatible_multiexpressions<T1, T2,
    typename std::enable_if<
	T1::is_multiexpression &&
	std::is_arithmetic<T2>::value>::type
	>
{
    static const bool value = true;
};

template <class T1, class T2>
struct compatible_multiexpressions<T1, T2,
    typename std::enable_if<
	std::is_arithmetic<T1>::value &&
	T2::is_multiexpression>::type
	>
{
    static const bool value = true;
};

template <class LHS, char OP, class RHS>
struct BinaryMultiExpression {
    static const bool is_multiexpression = true;
    static const uint dim = multiex_traits<LHS>::dim > multiex_traits<RHS>::dim ?
	multiex_traits<LHS>::dim : multiex_traits<RHS>::dim;

    typedef BinaryExpression<
			typename multiex_traits<LHS>::subtype,
			OP,
			typename multiex_traits<RHS>::subtype
			> subtype;

    BinaryMultiExpression(const LHS &lhs, const RHS &rhs) {
	for(uint i = 0; i < dim; i++)
	    expr[i].reset(new subtype(extract_component(lhs, i), extract_component(rhs, i)));
    }

    const subtype& operator[](uint i) const {
	return *expr[i];
    }

    std::array<std::unique_ptr<subtype>, dim> expr;
};

template <class LHS, class RHS>
typename std::enable_if<compatible_multiexpressions<LHS, RHS>::value,
	 BinaryMultiExpression<LHS, '+', RHS>
	 >::type
operator+(const LHS &lhs, const RHS &rhs) {
    return BinaryMultiExpression<LHS, '+', RHS>(lhs, rhs);
}

template <class LHS, class RHS>
typename std::enable_if<compatible_multiexpressions<LHS, RHS>::value,
	 BinaryMultiExpression<LHS, '-', RHS>
	 >::type
operator-(const LHS &lhs, const RHS &rhs) {
    return BinaryMultiExpression<LHS, '-', RHS>(lhs, rhs);
}

template <class LHS, class RHS>
typename std::enable_if<compatible_multiexpressions<LHS, RHS>::value,
	 BinaryMultiExpression<LHS, '*', RHS>
	 >::type
operator*(const LHS &lhs, const RHS &rhs) {
    return BinaryMultiExpression<LHS, '*', RHS>(lhs, rhs);
}

template <class LHS, class RHS>
typename std::enable_if<compatible_multiexpressions<LHS, RHS>::value,
	 BinaryMultiExpression<LHS, '/', RHS>
	 >::type
operator/(const LHS &lhs, const RHS &rhs) {
    return BinaryMultiExpression<LHS, '/', RHS>(lhs, rhs);
}

}


#endif
