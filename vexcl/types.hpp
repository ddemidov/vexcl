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
 * \file   types.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  C++ sugar for OpenCL vector types, eg. cl::float4, operator+.
 */

namespace cl {
    /// Get the corresponding scalar type for a CL vector (or scalar) type.
    /// \code scalar_of<cl_float4>::type == cl_float \endcode
    template <class T>
    struct scalar_of {};

    /// Get the corresponding vector type for a CL scalar type.
    /// \code vector_of<cl_float, 4>::type == cl_float4 \endcode
    template <class T, int dim>
    struct vector_of {};
}


#define BIN_OP(base_type, len, op) \
cl_##base_type##len &operator op##= (cl_##base_type##len &a, const cl_##base_type##len &b) { \
    for(size_t i = 0 ; i < len ; i++) a.s[i] op##= b.s[i]; \
    return a; \
} \
cl_##base_type##len operator op(const cl_##base_type##len &a, const cl_##base_type##len &b) { \
    cl_##base_type##len res = a; return res op##= b; \
}

#define CL_VEC_TYPE(base_type, len) \
BIN_OP(base_type, len, +) \
BIN_OP(base_type, len, -) \
BIN_OP(base_type, len, *) \
BIN_OP(base_type, len, /) \
std::ostream &operator<<(std::ostream &os, const cl_##base_type##len &value) { \
    os << "(" #base_type #len ")("; \
    for(std::size_t i = 0 ; i < len ; i++) { \
        if(i != 0) os << ','; \
        os << value.s[i]; \
    } \
    return os << ')'; \
} \
namespace cl { \
    typedef cl_##base_type##len base_type##len; \
    template <> struct scalar_of<cl_##base_type##len> { typedef cl_##base_type type; }; \
    template <> struct vector_of<cl_##base_type, len> { typedef cl_##base_type##len type; }; \
}

#define CL_TYPES(base_type) \
CL_VEC_TYPE(base_type, 2); \
CL_VEC_TYPE(base_type, 4); \
CL_VEC_TYPE(base_type, 8); \
CL_VEC_TYPE(base_type, 16); \
namespace cl { \
    template <> struct scalar_of<cl_##base_type> { typedef cl_##base_type type; }; \
}

CL_TYPES(float);
CL_TYPES(double);
CL_TYPES(char);  CL_TYPES(uchar);
CL_TYPES(short); CL_TYPES(ushort);
CL_TYPES(int);   CL_TYPES(uint);
CL_TYPES(long);  CL_TYPES(ulong);


#undef BIN_OP
#undef CL_VEC_TYPE
#undef CL_TYPES


#endif
