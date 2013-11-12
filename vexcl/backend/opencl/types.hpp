#ifndef VEXCL_BACKEND_OPENCL_TYPES_HPP
#define VEXCL_BACKEND_OPENCL_TYPES_HPP

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

#include <vexcl/types.hpp>

typedef unsigned char  uchar;
typedef unsigned int   uint;
typedef unsigned short ushort;

namespace vex {

// char and cl_char are different types. Hence, special handling is required:
template <> struct type_name_impl<char> {
    static std::string get() { return "char"; }
};
template <> struct is_cl_native<char> : std::true_type {};
template <> struct cl_vector_length<char> : std::integral_constant<unsigned, 1> {};
template <> struct cl_scalar_of<char> { typedef char type; };

}

#endif
