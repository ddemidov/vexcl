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

#if defined(_MSC_VER) && ( defined(min) || defined(max) )
#  error Please define NOMINMAX macro globally in your project
#endif

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

#include <boost/config.hpp>

#ifdef BOOST_NO_VARIADIC_TEMPLATES
#  include <boost/proto/proto.hpp>
#  include <boost/preprocessor/repetition.hpp>
#  ifndef VEXCL_MAX_ARITY
#    define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#  endif
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

namespace vex {

/// Check run-time condition.
/** Throws std::runtime_error if condition is false */
template <class Condition, class Message>
inline void precondition(const Condition &condition, const Message &fail_message) {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4800)
#endif
    if (!condition) throw std::runtime_error(fail_message);
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}

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
    return (n + m - 1) / m * m;
}

template <class T>
struct is_tuple : std::false_type {};


#ifndef BOOST_NO_VARIADIC_TEMPLATES

template <class... Elem>
struct is_tuple < std::tuple<Elem...> > : std::true_type {};

#else

#define IS_TUPLE(z, n, unused)                                                 \
  template <BOOST_PP_ENUM_PARAMS(n, class Elem)>                               \
  struct is_tuple<                                                             \
      std::tuple<BOOST_PP_ENUM_PARAMS(n, Elem)> > : std::true_type {           \
  };

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, IS_TUPLE, ~)

#undef IS_TUPLE

#endif

#ifndef BOOST_NO_VARIADIC_TEMPLATES
/// Create std::array from arguments
template <class T, class... Tail>
std::array<T, 1 + sizeof...(Tail)>
make_array(T t, Tail... tail) {
    std::array<T, 1 + sizeof...(Tail)> a = {{t, static_cast<T>(tail)...}};
    return a;
}
#else

#define PRINT_PARAM(z, n, data) T ## n t ## n
#define INIT_ARRAY(z, n, data) static_cast<T0>(t ## n)
#define MAKE_ARRAY(z, n, data)                                                 \
  template <BOOST_PP_ENUM_PARAMS(n, class T)>                                  \
  std::array<T0, n> make_array(BOOST_PP_ENUM(n, PRINT_PARAM, ~)) {             \
    std::array<T0, n> a = { { BOOST_PP_ENUM(n, INIT_ARRAY, ~) } };             \
    return a;                                                                  \
  }

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, MAKE_ARRAY, ~)

#undef MAKE_ARRAY
#undef INIT_ARRAY
#undef PRINT_PARAM

#endif

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

/// Checks if the compute device is CPU.
inline bool is_cpu(const cl::Device &d) {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4800)
#endif
    return d.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU;
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}

struct column_owner {
    const std::vector<size_t> &part;

    column_owner(const std::vector<size_t> &part) : part(part) {}

    size_t operator()(size_t c) const {
        return std::upper_bound(part.begin(), part.end(), c)
            - part.begin() - 1;
    }
};

/// Helper function for generating LocalSpaceArg objects.
/**
 * This is a copy of cl::Local that is absent in some of cl.hpp versions.
 */
inline cl::LocalSpaceArg
Local(size_t size) {
    cl::LocalSpaceArg ret = { size };
    return ret;
}

} // namespace vex

/// Output description of an OpenCL error to a stream.
inline std::ostream& operator<<(std::ostream &os, const cl::Error &e) {
    os << e.what() << "(";

#define CL_ERR2TXT(num, msg) case (num): os << (msg); break

    switch (e.err()) {
        CL_ERR2TXT(  0, "Success");
        CL_ERR2TXT( -1, "Device not found");
        CL_ERR2TXT( -2, "Device not available");
        CL_ERR2TXT( -3, "Compiler not available");
        CL_ERR2TXT( -4, "Mem object allocation failure");
        CL_ERR2TXT( -5, "Out of resources");
        CL_ERR2TXT( -6, "Out of host memory");
        CL_ERR2TXT( -7, "Profiling info not available");
        CL_ERR2TXT( -8, "Mem copy overlap");
        CL_ERR2TXT( -9, "Image format mismatch");
        CL_ERR2TXT(-10, "Image format not supported");
        CL_ERR2TXT(-11, "Build program failure");
        CL_ERR2TXT(-12, "Map failure");
        CL_ERR2TXT(-13, "Misaligned sub buffer offset");
        CL_ERR2TXT(-14, "Exec status error for events in wait list");
        CL_ERR2TXT(-30, "Invalid value");
        CL_ERR2TXT(-31, "Invalid device type");
        CL_ERR2TXT(-32, "Invalid platform");
        CL_ERR2TXT(-33, "Invalid device");
        CL_ERR2TXT(-34, "Invalid context");
        CL_ERR2TXT(-35, "Invalid queue properties");
        CL_ERR2TXT(-36, "Invalid command queue");
        CL_ERR2TXT(-37, "Invalid host ptr");
        CL_ERR2TXT(-38, "Invalid mem object");
        CL_ERR2TXT(-39, "Invalid image format descriptor");
        CL_ERR2TXT(-40, "Invalid image size");
        CL_ERR2TXT(-41, "Invalid sampler");
        CL_ERR2TXT(-42, "Invalid binary");
        CL_ERR2TXT(-43, "Invalid build options");
        CL_ERR2TXT(-44, "Invalid program");
        CL_ERR2TXT(-45, "Invalid program executable");
        CL_ERR2TXT(-46, "Invalid kernel name");
        CL_ERR2TXT(-47, "Invalid kernel definition");
        CL_ERR2TXT(-48, "Invalid kernel");
        CL_ERR2TXT(-49, "Invalid arg index");
        CL_ERR2TXT(-50, "Invalid arg value");
        CL_ERR2TXT(-51, "Invalid arg size");
        CL_ERR2TXT(-52, "Invalid kernel args");
        CL_ERR2TXT(-53, "Invalid work dimension");
        CL_ERR2TXT(-54, "Invalid work group size");
        CL_ERR2TXT(-55, "Invalid work item size");
        CL_ERR2TXT(-56, "Invalid global offset");
        CL_ERR2TXT(-57, "Invalid event wait list");
        CL_ERR2TXT(-58, "Invalid event");
        CL_ERR2TXT(-59, "Invalid operation");
        CL_ERR2TXT(-60, "Invalid gl object");
        CL_ERR2TXT(-61, "Invalid buffer size");
        CL_ERR2TXT(-62, "Invalid mip level");
        CL_ERR2TXT(-63, "Invalid global work size");
        CL_ERR2TXT(-64, "Invalid property");

        default:
            os << "Unknown error";
            break;
    }

#undef CL_ERR2TXT

    return os << ")";
}

#endif
