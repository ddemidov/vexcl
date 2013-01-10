#ifndef VEXCL_TYPES_HPP
#define VEXCL_TYPES_HPP

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
 * \file   types.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  C++ sugar for OpenCL vector types, eg. cl::float4, operator+.
 */

#define BIN_OP(type, len, op) \
cl_##type##len &operator op##= (cl_##type##len &a, const cl_##type##len &b) { \
    for(size_t i = 0 ; i < len ; i++) a.s[i] op##= b.s[i]; \
    return a; \
} \
cl_##type##len operator op(const cl_##type##len &a, const cl_##type##len &b) { \
    cl_##type##len res; return res op##= b; \
}

#define CL_VEC_TYPE(type, len) \
namespace std { \
    BIN_OP(type, len, +) \
    BIN_OP(type, len, -) \
    BIN_OP(type, len, *) \
    BIN_OP(type, len, /) \
    ostream &operator<<(ostream &os, const cl_##type##len &value) { \
        os << "(" #type #len ")("; \
        for(size_t i = 0 ; i < len ; i++) { \
            if(i != 0) os << ','; \
            os << value.s[i]; \
        } \
        return os << ')'; \
    } \
} \
namespace cl { \
    typedef cl_##type##len type##len; \
}

#define CL_TYPES(type) \
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


#undef BIN_OP
#undef CL_VEC_TYPE
#undef CL_TYPES


#endif
