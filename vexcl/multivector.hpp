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
#include <cassert>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/vector.hpp>

namespace vex {

/// Container for several vex::vectors.
/**
 * This class allows to synchronously operate on several vex::vectors of the
 * same type and size.
 */
template <typename T, uint N>
class multivector {
    public:
	static const bool is_multiexpression = true;
	static const uint dim = N;

	typedef vex::vector<T> subtype;
	typedef std::array<T,N> value_type;

	/// Proxy class.
	class element {
	    public:
		operator value_type () const {
		    value_type val;
		    for(uint i = 0; i < N; i++) val[i] = vec(i)[index];
		    return val;
		}

		value_type operator=(value_type val) {
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
		operator value_type () const {
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

	    for(uint i = 0; i < N; i++)
		vec[i] = vex::vector<T>(
			queue, size, host.data() + i * size, flags
			);
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

	    for(uint i = 0; i < N; i++)
		vec[i] = vex::vector<T>(
			queue, size, host ? host + i * size : 0, flags
			);
	}

	/// Resize multivector.
	void resize(const std::vector<cl::CommandQueue> &queue, size_t size) {
	    for(uint i = 0; i < N; i++) vec[i].resize(queue, size);
	}

	/// Return size of a multivector (equals size of individual components).
	size_t size() const {
	    return vec[0].size();
	}

	/// Returns multivector component.
	const vex::vector<T>& operator()(uint i) const {
	    return vec[i];
	}

	/// Returns multivector component.
	vex::vector<T>& operator()(uint i) {
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

	/** \name Expression assignments.
	 * @{
	 * All operations are delegated to components of the multivector.
	 */
	template <class Expr>
	const multivector& operator=(const Expr& expr) {
	    for(uint i = 0; i < N; i++) vec[i] = expr(i);
	}

	template <class Expr>
	const multivector& operator+=(const Expr &expr) {
	    return *this = *this + expr;
	}

	template <class Expr>
	const multivector& operator*=(const Expr &expr) {
	    return *this = *this * expr;
	}

	template <class Expr>
	const multivector& operator/=(const Expr &expr) {
	    return *this = *this / expr;
	}

	template <class Expr>
	const multivector& operator-=(const Expr &expr) {
	    return *this = *this - expr;
	}
	/// @}

    private:
	std::array<vex::vector<T>,N> vec;
};

/// Copy multivector to host vector.
template <class T, uint N>
void copy(const multivector<T,N> &mv, std::vector<T> &hv) {
    for(uint i = 0; i < N; i++)
	vex::copy(mv(i).begin(), mv(i).end(), hv.begin() + i * mv.size());
}

/// Copy host vector to multivector.
template <class T, uint N>
void copy(const std::vector<T> &hv, multivector<T,N> &mv) {
    for(uint i = 0; i < N; i++)
	vex::copy(hv.begin() + i * mv.size(), hv.begin() + (i + 1) * mv.size(),
		mv(i).begin());
}

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
    return expr(i);
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

    const subtype& operator()(uint i) const {
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
