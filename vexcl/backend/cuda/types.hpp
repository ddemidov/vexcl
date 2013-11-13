#ifndef VEXCL_BACKEND_CUDA_TYPES_HPP
#define VEXCL_BACKEND_CUDA_TYPES_HPP

typedef unsigned char       uchar;
typedef unsigned int        uint;
typedef unsigned short      ushort;
typedef unsigned long long  ulong;

#define REGISTER_VECTOR_TYPE(name, base, len)                                  \
  struct cl_##name##len {                                                      \
    base s[len];                                                               \
  }

#define REGISTER_CL_TYPES(name, base)                                          \
  typedef base cl_##name;                                                      \
  REGISTER_VECTOR_TYPE(name, base, 2);                                         \
  REGISTER_VECTOR_TYPE(name, base, 4);                                         \
  REGISTER_VECTOR_TYPE(name, base, 8);                                         \
  REGISTER_VECTOR_TYPE(name, base, 16);

REGISTER_CL_TYPES(float,  float)
REGISTER_CL_TYPES(double, double)
REGISTER_CL_TYPES(char,   char)
REGISTER_CL_TYPES(uchar,  uchar)
REGISTER_CL_TYPES(short,  short)
REGISTER_CL_TYPES(ushort, ushort)
REGISTER_CL_TYPES(int,    int)
REGISTER_CL_TYPES(uint,   uint)
REGISTER_CL_TYPES(long,   long long)
REGISTER_CL_TYPES(ulong,  ulong)

#undef REGISTER_CL_TYPES
#undef REGISTER_VECTOR_TYPE

#include <vexcl/types.hpp>

#endif
