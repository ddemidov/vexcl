#ifndef VEXCL_UTIL_HPP
#define VEXCL_UTIL_HPP

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
 * \file   vexcl/util.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL general utilities.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <climits>
#include <stdexcept>
#include <limits>
#include <boost/config.hpp>
#include <boost/type_traits/is_same.hpp>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>
#include <vexcl/types.hpp>

typedef unsigned int  uint;
typedef unsigned char uchar;

namespace vex {

/// Convert typename to string.
template <class T> inline std::string type_name() {
    throw std::logic_error("Trying to use an undefined type in a kernel.");
}

/// Declares a type as CL native, allows using it as a literal.
template <class T> struct is_cl_native : std::false_type {};


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

#if defined(__clang__) && defined(__APPLE__)
template <> inline std::string type_name<size_t>() {
    return std::numeric_limits<std::size_t>::max() ==
        std::numeric_limits<uint>::max() ? "uint" : "ulong";
}
template <> struct is_cl_native<size_t> : std::true_type {};
template <> inline std::string type_name<ptrdiff_t>() {
    return std::numeric_limits<std::ptrdiff_t>::max() ==
        std::numeric_limits<int>::max() ? "int" : "long";
}
template <> struct is_cl_native<ptrdiff_t> : std::true_type {};
#endif

const std::string standard_kernel_header = std::string(
        "#if defined(cl_khr_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
        "#elif defined(cl_amd_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
        "#endif\n"
        );

/// \cond INTERNAL

/// Binary operations with their traits.
namespace binop {
    enum kind {
        Add,
        Subtract,
        Multiply,
        Divide,
        Remainder,
        Greater,
        Less,
        GreaterEqual,
        LessEqual,
        Equal,
        NotEqual,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        LogicalAnd,
        LogicalOr,
        RightShift,
        LeftShift
    };

    template <kind> struct traits {};

#define BOP_TRAITS(kind, op, nm)   \
    template <> struct traits<kind> {  \
        static std::string oper() { return op; } \
        static std::string name() { return nm; } \
    };

    BOP_TRAITS(Add,          "+",  "Add_")
    BOP_TRAITS(Subtract,     "-",  "Sub_")
    BOP_TRAITS(Multiply,     "*",  "Mul_")
    BOP_TRAITS(Divide,       "/",  "Div_")
    BOP_TRAITS(Remainder,    "%",  "Mod_")
    BOP_TRAITS(Greater,      ">",  "Gtr_")
    BOP_TRAITS(Less,         "<",  "Lss_")
    BOP_TRAITS(GreaterEqual, ">=", "Geq_")
    BOP_TRAITS(LessEqual,    "<=", "Leq_")
    BOP_TRAITS(Equal,        "==", "Equ_")
    BOP_TRAITS(NotEqual,     "!=", "Neq_")
    BOP_TRAITS(BitwiseAnd,   "&",  "BAnd_")
    BOP_TRAITS(BitwiseOr,    "|",  "BOr_")
    BOP_TRAITS(BitwiseXor,   "^",  "BXor_")
    BOP_TRAITS(LogicalAnd,   "&&", "LAnd_")
    BOP_TRAITS(LogicalOr,    "||", "LOr_")
    BOP_TRAITS(RightShift,   ">>", "Rsh_")
    BOP_TRAITS(LeftShift,    "<<", "Lsh_")

#undef BOP_TRAITS
}

/// \endcond

/// Return next power of 2.
inline size_t nextpow2(size_t x) {
    --x;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    return ++x;
}

/// Align n to the next multiple of m.
inline size_t alignup(size_t n, size_t m = 16U) {
    return n % m ? n - n % m + m : n;
}

/// Iterate over tuple elements.
template <size_t I, class Function, class Tuple>
typename std::enable_if<(I == std::tuple_size<Tuple>::value), void>::type
for_each(const Tuple &, Function &)
{ }

/// Iterate over tuple elements.
template <size_t I, class Function, class Tuple>
typename std::enable_if<(I < std::tuple_size<Tuple>::value), void>::type
for_each(const Tuple &v, Function &f)
{
    f( std::get<I>(v) );

    for_each<I + 1>(v, f);
}


/// Create and build a program from source string.
inline cl::Program build_sources(
        const cl::Context &context, const std::string &source,
        const std::string &options = ""
        )
{
#ifdef VEXCL_SHOW_KERNELS
    std::cout << source << std::endl;
#endif

    cl::Program program(context, cl::Program::Sources(
                1, std::make_pair(source.c_str(), source.size())
                ));

    auto device = context.getInfo<CL_CONTEXT_DEVICES>();

    try {
        program.build(device, options.c_str());
    } catch(const cl::Error&) {
        std::cerr << source
                  << std::endl
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
                  << std::endl;
        throw;
    }

    return program;
}

/// Get maximum possible workgroup size for given kernel.
inline uint kernel_workgroup_size(
        const cl::Kernel &kernel,
        const cl::Device &device
        )
{
    size_t wgsz = 1024U;

    uint dev_wgsz = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    while(wgsz > dev_wgsz) wgsz /= 2;

    return wgsz;
}

/// Shortcut for q.getInfo<CL_QUEUE_CONTEXT>()
inline cl::Context qctx(const cl::CommandQueue& q) {
    cl::Context ctx;
    q.getInfo(CL_QUEUE_CONTEXT, &ctx);
    return ctx;
}

/// Shortcut for q.getInfo<CL_QUEUE_DEVICE>()
inline cl::Device qdev(const cl::CommandQueue& q) {
    cl::Device dev;
    q.getInfo(CL_QUEUE_DEVICE, &dev);
    return dev;
}

struct column_owner {
    const std::vector<size_t> &part;

    column_owner(const std::vector<size_t> &part) : part(part) {}

    size_t operator()(size_t c) const {
        return std::upper_bound(part.begin(), part.end(), c)
            - part.begin() - 1;
    }
};

} // namespace vex

/// Output description of an OpenCL error to a stream.
inline std::ostream& operator<<(std::ostream &os, const cl::Error &e) {
    os << e.what() << "(";

    switch (e.err()) {
        case 0:
            os << "Success";
            break;
        case -1:
            os << "Device not found";
            break;
        case -2:
            os << "Device not available";
            break;
        case -3:
            os << "Compiler not available";
            break;
        case -4:
            os << "Mem object allocation failure";
            break;
        case -5:
            os << "Out of resources";
            break;
        case -6:
            os << "Out of host memory";
            break;
        case -7:
            os << "Profiling info not available";
            break;
        case -8:
            os << "Mem copy overlap";
            break;
        case -9:
            os << "Image format mismatch";
            break;
        case -10:
            os << "Image format not supported";
            break;
        case -11:
            os << "Build program failure";
            break;
        case -12:
            os << "Map failure";
            break;
        case -13:
            os << "Misaligned sub buffer offset";
            break;
        case -14:
            os << "Exec status error for events in wait list";
            break;
        case -30:
            os << "Invalid value";
            break;
        case -31:
            os << "Invalid device type";
            break;
        case -32:
            os << "Invalid platform";
            break;
        case -33:
            os << "Invalid device";
            break;
        case -34:
            os << "Invalid context";
            break;
        case -35:
            os << "Invalid queue properties";
            break;
        case -36:
            os << "Invalid command queue";
            break;
        case -37:
            os << "Invalid host ptr";
            break;
        case -38:
            os << "Invalid mem object";
            break;
        case -39:
            os << "Invalid image format descriptor";
            break;
        case -40:
            os << "Invalid image size";
            break;
        case -41:
            os << "Invalid sampler";
            break;
        case -42:
            os << "Invalid binary";
            break;
        case -43:
            os << "Invalid build options";
            break;
        case -44:
            os << "Invalid program";
            break;
        case -45:
            os << "Invalid program executable";
            break;
        case -46:
            os << "Invalid kernel name";
            break;
        case -47:
            os << "Invalid kernel definition";
            break;
        case -48:
            os << "Invalid kernel";
            break;
        case -49:
            os << "Invalid arg index";
            break;
        case -50:
            os << "Invalid arg value";
            break;
        case -51:
            os << "Invalid arg size";
            break;
        case -52:
            os << "Invalid kernel args";
            break;
        case -53:
            os << "Invalid work dimension";
            break;
        case -54:
            os << "Invalid work group size";
            break;
        case -55:
            os << "Invalid work item size";
            break;
        case -56:
            os << "Invalid global offset";
            break;
        case -57:
            os << "Invalid event wait list";
            break;
        case -58:
            os << "Invalid event";
            break;
        case -59:
            os << "Invalid operation";
            break;
        case -60:
            os << "Invalid gl object";
            break;
        case -61:
            os << "Invalid buffer size";
            break;
        case -62:
            os << "Invalid mip level";
            break;
        case -63:
            os << "Invalid global work size";
            break;
        case -64:
            os << "Invalid property";
            break;
        default:
            os << "Unknown error";
            break;
    }

    return os << ")";
}

#ifdef WIN32
#  pragma warning(pop)
#endif

// vim: et
#endif
