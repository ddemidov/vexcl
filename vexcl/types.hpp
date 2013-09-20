#ifndef VEXCL_TYPES_HPP
#define VEXCL_TYPES_HPP

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
 * \file   vexcl/types.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Support for using native C++ and OpenCL types in expressions.
 */

#include <string>
#include <type_traits>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

typedef unsigned int  uint;
typedef unsigned char uchar;

namespace vex {

    /// Get the corresponding scalar type for a CL vector (or scalar) type.
    /** \code cl_scalar_of<cl_float4>::type == cl_float \endcode */
    template <class T>
    struct cl_scalar_of {};

    /// Get the corresponding vector type for a CL scalar type.
    /** \code cl_vector_of<cl_float, 4>::type == cl_float4 \endcode */
    template <class T, int dim>
    struct cl_vector_of {};

    /// Get the number of values in a CL vector (or scalar) type.
    /** \code cl_vector_length<cl_float4>::value == 4 \endcode */
    template <class T>
    struct cl_vector_length {};

} // namespace vex

#define BIN_OP(base_type, len, op) \
inline cl_##base_type##len &operator op##= (cl_##base_type##len &a, const cl_##base_type##len &b) { \
    for(size_t i = 0 ; i < len ; i++) a.s[i] op##= b.s[i]; \
    return a; \
} \
inline cl_##base_type##len operator op(const cl_##base_type##len &a, const cl_##base_type##len &b) { \
    cl_##base_type##len res = a; return res op##= b; \
}

// `scalar OP vector` acts like `(vector_t)(scalar) OP vector` in OpenCl:
// all components are set to the scalar value.
#define BIN_SCALAR_OP(base_type, len, op) \
inline cl_##base_type##len &operator op##= (cl_##base_type##len &a, const cl_##base_type &b) { \
    for(size_t i = 0 ; i < len ; i++) a.s[i] op##= b; \
    return a; \
} \
inline cl_##base_type##len operator op(const cl_##base_type##len &a, const cl_##base_type &b) { \
    cl_##base_type##len res = a; return res op##= b; \
} \
inline cl_##base_type##len operator op(const cl_##base_type &a, const cl_##base_type##len &b) { \
    cl_##base_type##len res = b; return res op##= a; \
}

#define CL_VEC_TYPE(base_type, len) \
BIN_OP(base_type, len, +) \
BIN_OP(base_type, len, -) \
BIN_OP(base_type, len, *) \
BIN_OP(base_type, len, /) \
BIN_SCALAR_OP(base_type, len, +) \
BIN_SCALAR_OP(base_type, len, -) \
BIN_SCALAR_OP(base_type, len, *) \
BIN_SCALAR_OP(base_type, len, /) \
inline cl_##base_type##len operator -(const cl_##base_type##len &a) { \
    cl_##base_type##len res; \
    for(size_t i = 0 ; i < len ; i++) res.s[i] = -a.s[i]; \
    return res; \
} \
inline std::ostream &operator<<(std::ostream &os, const cl_##base_type##len &value) { \
    os << "(" #base_type #len ")("; \
    for(std::size_t i = 0 ; i < len ; i++) { \
        if(i != 0) os << ','; \
        os << value.s[i]; \
    } \
    return os << ')'; \
} \
namespace vex { \
    template <> struct cl_scalar_of<cl_##base_type##len> { typedef cl_##base_type type; }; \
    template <> struct cl_vector_of<cl_##base_type, len> { typedef cl_##base_type##len type; }; \
    template <> struct cl_vector_length<cl_##base_type##len> : std::integral_constant<unsigned, len>{ }; \
}

#define CL_TYPES(base_type) \
CL_VEC_TYPE(base_type, 2); \
CL_VEC_TYPE(base_type, 4); \
CL_VEC_TYPE(base_type, 8); \
CL_VEC_TYPE(base_type, 16); \
namespace vex { \
    template <> struct cl_scalar_of<cl_##base_type> { typedef cl_##base_type type; }; \
    template <> struct cl_vector_of<cl_##base_type, 1> { typedef cl_##base_type type; }; \
    template <> struct cl_vector_length<cl_##base_type> : std::integral_constant<unsigned, 1> { }; \
}

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4146)
#endif
CL_TYPES(float);
CL_TYPES(double);
CL_TYPES(char);  CL_TYPES(uchar);
CL_TYPES(short); CL_TYPES(ushort);
CL_TYPES(int);   CL_TYPES(uint);
CL_TYPES(long);  CL_TYPES(ulong);
#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#undef BIN_OP
#undef CL_VEC_TYPE
#undef CL_TYPES


namespace vex {

/// Convert each element of the vector to another type.
template<class To, class From>
inline To cl_convert(const From &val) {
    const size_t n = cl_vector_length<To>::value;
    static_assert(n == cl_vector_length<From>::value, "Vectors must be same length.");
    To out;
    for(size_t i = 0 ; i != n ; i++)
        out.s[i] = val.s[i];
    return out;
}

/// Declares a type as CL native, allows using it as a literal.
template <class T> struct is_cl_native : std::false_type {};

/// Convert typename to string.
template <class T> inline std::string type_name() {
    throw std::logic_error("Trying to use an undefined type in a kernel.");
}


#define STRINGIFY(type) \
template <> inline std::string type_name<cl_##type>() { return #type; } \
template <> struct is_cl_native<cl_##type> : std::true_type {};

// enable use of OpenCL vector types as literals
#define CL_VEC_TYPE(type, len) \
template <> inline std::string type_name<cl_##type##len>() { return #type #len; } \
template <> struct is_cl_native<cl_##type##len> : std::true_type {};

#define CL_TYPES(type) \
STRINGIFY(type); \
CL_VEC_TYPE(type, 2); \
CL_VEC_TYPE(type, 4); \
CL_VEC_TYPE(type, 8); \
CL_VEC_TYPE(type, 16);

CL_TYPES(float);
CL_TYPES(double);
CL_TYPES(char);  CL_TYPES(uchar);
CL_TYPES(short); CL_TYPES(ushort);
CL_TYPES(int);   CL_TYPES(uint);
CL_TYPES(long);  CL_TYPES(ulong);

#undef CL_TYPES
#undef CL_VEC_TYPE
#undef STRINGIFY

// char and cl_char are different types. Hence, special handling is required:
template <> inline std::string type_name<char>() { return "char"; }
template <> struct is_cl_native<char> : std::true_type {};
template <> struct cl_vector_length<char> : std::integral_constant<unsigned, 1> {};
template <> struct cl_scalar_of<char> { typedef char type; };

// One can not pass bool to the kernel, but the overload is needed for type
// deduction:
template <> inline std::string type_name<bool>() { return "bool"; }


#if defined(__APPLE__)
template <> inline std::string type_name<size_t>() {
    return sizeof(std::size_t) == sizeof(uint) ? "uint" : "ulong";
}

template <> inline std::string type_name<ptrdiff_t>() {
    return sizeof(std::size_t) == sizeof(uint) ? "int" : "long";
}

template <> struct is_cl_native<size_t>    : std::true_type {};
template <> struct is_cl_native<ptrdiff_t> : std::true_type {};

template <> struct cl_vector_length<size_t>    : std::integral_constant<unsigned, 1> {};
template <> struct cl_vector_length<ptrdiff_t> : std::integral_constant<unsigned, 1> {};
#endif

template <class T>
struct is_cl_scalar :
    std::integral_constant<
        bool,
        is_cl_native<T>::value && (cl_vector_length<T>::value == 1)
        >
{};

template <class T>
struct is_cl_vector :
    std::integral_constant<
        bool,
        is_cl_native<T>::value && (cl_vector_length<T>::value > 1)
        >
{};

}


#endif
