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

#ifdef _MSC_VER
#  define NOMINMAX
#endif

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <stdexcept>
#include <algorithm>

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

#define IS_TUPLE(z, n, unused)                                    \
  template < BOOST_PP_ENUM_PARAMS(n, class Elem) >                \
  struct is_tuple< std::tuple < BOOST_PP_ENUM_PARAMS(n, Elem) > > \
    : std::true_type                                              \
  {};

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, IS_TUPLE, ~)

#undef IS_TUPLE

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

enum device_options_kind {
    compile_options,
    program_header
};

/// Global program options holder
template <device_options_kind kind>
struct device_options {
    static const std::string& get(const cl::Device &dev) {
        if (options[dev()].empty()) options[dev()].push_back("");

        return options[dev()].back();
    }

    static void push(const cl::Device &dev, const std::string &str) {
        options[dev()].push_back(str);
    }

    static void pop(const cl::Device &dev) {
        if (!options[dev()].empty()) options[dev()].pop_back();
    }

    private:
        static std::map<cl_device_id, std::vector<std::string> > options;
};

template <device_options_kind kind>
std::map<cl_device_id, std::vector<std::string> > device_options<kind>::options;

inline std::string get_compile_options(const cl::Device &dev) {
    return device_options<compile_options>::get(dev);
}

inline std::string get_program_header(const cl::Device &dev) {
    return device_options<program_header>::get(dev);
}

/// Set global OpenCL compilation options for a given device.
/**
 * This replaces any previously set options. To roll back, call
 * pop_compile_options().
 */
inline void push_compile_options(const cl::Device &dev, const std::string &str) {
    device_options<compile_options>::push(dev, str);
}

/// Rolls back changes to compile options.
inline void pop_compile_options(const cl::Device &dev) {
    device_options<compile_options>::pop(dev);
}

/// Set global OpenCL program header for a given device.
/**
 * This replaces any previously set header. To roll back, call
 * pop_program_header().
 */
inline void push_program_header(const cl::Device &dev, const std::string &str) {
    device_options<program_header>::push(dev, str);
}

/// Rolls back changes to compile options.
inline void pop_program_header(const cl::Device &dev) {
    device_options<program_header>::pop(dev);
}

/// Set global OpenCL compilation options for each device in queue list.
inline void push_compile_options(const std::vector<cl::CommandQueue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::push(qdev(*q), str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_compile_options(const std::vector<cl::CommandQueue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::pop(qdev(*q));
}

/// Set global OpenCL program header for each device in queue list.
inline void push_program_header(const std::vector<cl::CommandQueue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::push(qdev(*q), str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_program_header(const std::vector<cl::CommandQueue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::pop(qdev(*q));
}

/// Returns standard OpenCL program header.
/**
 * Defines pragmas necessary to work with double precision and anything
 * provided by the user with help of push_program_header().
 */
inline std::string standard_kernel_header(const cl::Device &dev) {
    return std::string(
        "#if defined(cl_khr_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
        "#elif defined(cl_amd_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
        "#endif\n"
        ) + get_program_header(dev);
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
        program.build(device, (options + " " + get_compile_options(device[0])).c_str());
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
inline unsigned kernel_workgroup_size(
        const cl::Kernel &kernel,
        const cl::Device &device
        )
{
    unsigned wgsz = 1024U;

    unsigned dev_wgsz = static_cast<unsigned>(
        kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device));
    while(wgsz > dev_wgsz) wgsz /= 2;

    return wgsz;
}

/// Standard number of workgroups to launch on a device.
inline size_t num_workgroups(const cl::Device &device) {
    // This is a simple heuristic-based estimate. More advanced technique may
    // be employed later.
    return 4 * device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
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
