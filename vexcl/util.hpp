#ifndef VEXCL_UTIL_HPP
#define VEXCL_UTIL_HPP

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
 * \file   util.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL general utilities.
 */

#ifdef WIN32
#  pragma warning(disable : 4290)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <CL/cl.hpp>

typedef unsigned int  uint;
typedef unsigned char uchar;

namespace vex {

/// Convert typename to string.
template <class T> std::string type_name()  { return "undefined_type"; }
template <> std::string type_name<float>()  { return "float"; }
template <> std::string type_name<double>() { return "double"; }
template <> std::string type_name<int>()    { return "int"; }
template <> std::string type_name<char>()   { return "char"; }
template <> std::string type_name<uint>()   { return "unsigned int"; }
template <> std::string type_name<uchar>()  { return "unsigned char"; }

template <> std::string type_name<size_t>() {
    static_assert(sizeof(size_t) == 4 || sizeof(size_t) == 8,
	    "Only 32bit or 64bit architectures are supported");
    return sizeof(size_t) == 4 ? "uint" : "ulong";
}

template <> std::string type_name<ptrdiff_t>() {
    static_assert(sizeof(size_t) == 4 || sizeof(size_t) == 8,
	    "Only 32bit or 64bit architectures are supported");
    return sizeof(size_t) == 4 ? "int" : "long";
}

std::string standard_kernel_header = std::string(
	"#if defined(cl_khr_fp64)\n"
	"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	"#elif defined(cl_amd_fp64)\n"
	"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	"#endif\n"
	);

/// Return next power of 2.
size_t nextpow2(size_t x) {
    --x;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    return ++x;
}

/// Align n to the next multiple of m.
size_t alignup(size_t n, size_t m = 16U) {
    return n % m ? n - n % m + m : n;
}

/// Partitions vector wrt to vector performance of devices.
/**
 * Launches the following kernel on each device:
 * \code
 * a = b + c;
 * \endcode
 * where a, b and c are device vectors. Each device gets portion of the vector
 * proportional to the performance of this operation.
 */
std::vector<size_t> partition_by_vector_perf(
	size_t n, const std::vector<cl::CommandQueue> &queue);

/// Partitions vector wrt to spmv performance of devices.
/**
 * Launches the following kernel on each device:
 * \code
 * y = A * x;
 * \endcode
 * where x and y are vectors, and A is matrix for 3D Poisson problem in square
 * domain. Each device gets portion of the vector proportional to the
 * performance of this operation.
 */
std::vector<size_t> partition_by_spmv_perf(
	size_t n, const std::vector<cl::CommandQueue> &queue);

/// Partitions vector equally.
std::vector<size_t> partition_equally(
	size_t n, const std::vector<cl::CommandQueue> &queue)
{
    size_t m = queue.size();

    std::vector<size_t> part(m + 1);
    part[0] = 0;

    if (queue.size() > 1) {
	for(size_t d = 0, chunk_size = alignup((n + m - 1) / m); d < m; d++)
	    part[d + 1] = std::min(n, part[d] + chunk_size);
    } else {
	part.back() = n;
    }

    return part;
}


/// Partitioning scheme for vectors and matrices.
/**
 * Should be set once before any object of vector or matrix type is declared.
 * Otherwise default parttioning function (partition_by_vector_perf) is
 * selected.
 */
struct partitioning_scheme {
    typedef std::function<
	std::vector<size_t>(size_t, const std::vector<cl::CommandQueue>&)
	> function_type;

    static void set(function_type f) {
	if (!is_set) {
	    pfun = f;
	    is_set = true;
	} else {
	    std::cerr <<
		"Warning: "
		"partitioning function is already set and will be left as is."
		<< std::endl;
	}
    }

    std::vector<size_t> operator()(size_t n,
	    const std::vector<cl::CommandQueue> &queue) const
    {
	if (!is_set) {
	    pfun = partition_by_vector_perf;
	    is_set = true;
	}
	return pfun(n, queue);
    }

    private:
	static bool is_set;
	static function_type pfun;
} partition;

bool partitioning_scheme::is_set = false;
partitioning_scheme::function_type partitioning_scheme::pfun;

/// Create and build a program from source string.
cl::Program build_sources(
	const cl::Context &context, const std::string &source
	)
{
    cl::Program program(context, cl::Program::Sources(
		1, std::make_pair(source.c_str(), source.size())
		));

    auto device = context.getInfo<CL_CONTEXT_DEVICES>();

    try {
	program.build(device);
    } catch(const cl::Error&) {
	std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
	          << std::endl;
	throw;
    }

    return program;
}

/// Get maximum possible workgroup size for given kernel.
size_t kernel_workgroup_size(
	const cl::Kernel &kernel,
	const std::vector<cl::Device> &device
	)
{
    size_t wgsz = 1024U;

    for(auto d = device.begin(); d != device.end(); d++) {
	size_t dev_wgsz = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*d);
	while(wgsz > dev_wgsz) wgsz /= 2;
    }

    return wgsz;
}

} // namespace vex

/// Output description of an OpenCL error to a stream.
std::ostream& operator<<(std::ostream &os, const cl::Error &e) {
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

#endif
