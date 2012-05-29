#ifndef VEXCL_UTIL_HPP
#define VEXCL_UTIL_HPP

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
#include <string>
#include <algorithm>
#include <type_traits>
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

/// Return next power of 2.
template <class T>
typename std::enable_if<std::is_integral<T>::value, T>::type
nextpow2(T x) {
    --x;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    return ++x;
}

/// Align n to the next multiple of m.
template <class T>
typename std::enable_if<std::is_integral<T>::value, T>::type
alignup(T n, T m = 16U) {
    return n % m ? n - n % m + m : n;
}

/// Partition n into m almost equal parts.
inline std::vector<uint> partition(uint n, uint m) {
    std::vector<uint> part(m + 1);

    uint chunk_size = alignup((n + m - 1) / m);

    part[0] = 0;

    for(uint i = 0; i < m; i++)
	part[i + 1] = std::min(n, part[i] + chunk_size);

    return part;
}

/// Create and build a program from source string.
inline cl::Program build_sources(
	cl::Context context, const std::string &source
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
inline uint kernel_workgroup_size(
	const cl::Kernel &kernel,
	const std::vector<cl::Device> &device
	)
{
    uint wgsz = 1024U;

    for(auto d = device.begin(); d != device.end(); d++) {
	uint dev_wgsz = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*d);
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
