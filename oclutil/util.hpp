#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <string>
#include <CL/cl.hpp>

typedef unsigned int  uint;
typedef unsigned char uchar;

namespace clu {

/// Convert typename to string.
template <class T> std::string type_name()  { return "undefined_type"; }
template <> std::string type_name<float>()  { return "float"; }
template <> std::string type_name<double>() { return "double"; }
template <> std::string type_name<int>()    { return "int"; }
template <> std::string type_name<char>()   { return "char"; }
template <> std::string type_name<uint>()   { return "unsigned int"; }
template <> std::string type_name<uchar>()  { return "unsigned char"; }

/// Return next power of 2.
inline uint nextpow2(uint x) {
    --x;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    return ++x;
}

/// Align n to the next multiple of m.
inline uint alignup(uint n, uint m = 16U) {
    return n % m ? n - n % m + m : n;
}

/// Partition n into m almost equal parts.
inline std::vector<size_t> partition(size_t n, size_t m) {
    std::vector<size_t> part(m + 1);

    size_t chunk_size = alignup((n + m - 1) / m);

    part[0] = 0;

    for(size_t i = 0; i < m; i++)
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

} // namespace clu

#endif
