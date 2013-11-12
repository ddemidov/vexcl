#ifndef VEXCL_BACKEND_CUDA_TYPES_HPP
#define VEXCL_BACKEND_CUDA_TYPES_HPP

#define REGICTER_VECTOR_TYPE(name, base, len)                                  \
  struct cl_##name##len {                                                      \
    base s[len];                                                               \
  }

#define REGISTER_CL_TYPES(name, base)                                          \
  typedef base cl_##name;                                                      \
  REGICTER_VECTOR_TYPE(name, base, 2);                                         \
  REGICTER_VECTOR_TYPE(name, base, 4);                                         \
  REGICTER_VECTOR_TYPE(name, base, 8);                                         \
  REGICTER_VECTOR_TYPE(name, base, 16);

typedef unsigned char  uchar;
typedef unsigned int   uint;
typedef unsigned short ushort;

REGISTER_CL_TYPES(float,  float)
REGISTER_CL_TYPES(double, double)
REGISTER_CL_TYPES(char,   int8_t)
REGISTER_CL_TYPES(uchar,  uint8_t)
REGISTER_CL_TYPES(short,  short)
REGISTER_CL_TYPES(ushort, ushort)
REGISTER_CL_TYPES(int,    int)
REGISTER_CL_TYPES(uint,   uint)
REGISTER_CL_TYPES(long,   long)
REGISTER_CL_TYPES(ulong,  ulong)

#undef REGISTER_CL_TYPES
#undef REGICTER_VECTOR_TYPE

#endif
