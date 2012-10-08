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
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <CL/cl.hpp>

typedef unsigned int  uint;
typedef unsigned char uchar;

namespace vex {

/// Convert typename to string.
template <class T> inline std::string type_name()  { return "undefined_type"; }
template <> inline std::string type_name<float>()  { return "float"; }
template <> inline std::string type_name<double>() { return "double"; }
template <> inline std::string type_name<int>()    { return "int"; }
template <> inline std::string type_name<char>()   { return "char"; }
template <> inline std::string type_name<bool>()   { return "bool"; }
template <> inline std::string type_name<uint>()   { return "unsigned int"; }
template <> inline std::string type_name<uchar>()  { return "unsigned char"; }

template <> inline
typename std::enable_if<sizeof(size_t) == 8, std::string>::type
type_name<size_t>() {
    return sizeof(size_t) == 4 ? "uint" : "ulong";
}

template <> inline
typename std::enable_if<sizeof(size_t) == 8, std::string>::type
type_name<ptrdiff_t>() {
    return sizeof(size_t) == 4 ? "int" : "long";
}

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

/// Weights device wrt to vector performance.
/**
 * Launches the following kernel on each device:
 * \code
 * a = b + c;
 * \endcode
 * where a, b and c are device vectors. Each device gets portion of the vector
 * proportional to the performance of this operation.
 */
inline double device_vector_perf(
	const cl::Context &context, const cl::Device &device
	);

/// Weights device wrt to spmv performance.
/**
 * Launches the following kernel on each device:
 * \code
 * y = A * x;
 * \endcode
 * where x and y are vectors, and A is matrix for 3D Poisson problem in square
 * domain. Each device gets portion of the vector proportional to the
 * performance of this operation.
 */
inline double device_spmv_perf(
	const cl::Context &context, const cl::Device &device
	);

/// Assigns equal weight to each device.
/**
 * This results in equal partitioning.
 */
inline double equal_weights(
	const cl::Context &context, const cl::Device &device
	)
{
    return 1;
}


/// Partitioning scheme for vectors and matrices.
/**
 * Should be set once before any object of vector or matrix type is declared.
 * Otherwise default parttioning function (partition_by_vector_perf) is
 * selected.
 */
template <bool dummy = true>
struct partitioning_scheme {
    typedef std::function<
	double(const cl::Context&, const cl::Device&)
	> weight_function;

    static void set(weight_function f) {
	if (!is_set) {
	    weight = f;
	    is_set = true;
	} else {
	    std::cerr <<
		"Warning: "
		"device weighting function is already set and will be left as is."
		<< std::endl;
	}
    }

    static std::vector<size_t> get(size_t n, const std::vector<cl::CommandQueue> &queue);

    private:
	static bool is_set;
	static weight_function weight;
	static std::map<cl_device_id, double> device_weight;
};

template <bool dummy>
bool partitioning_scheme<dummy>::is_set = false;

template <bool dummy>
std::map<cl_device_id, double> partitioning_scheme<dummy>::device_weight;

template <bool dummy>
std::vector<size_t> partitioning_scheme<dummy>::get(size_t n,
	const std::vector<cl::CommandQueue> &queue)
{
    if (!is_set) {
	weight = device_vector_perf;
	is_set = true;
    }

    std::vector<size_t> part;
    part.reserve(queue.size() + 1);
    part.push_back(0);

    if (queue.size() > 1) {
	std::vector<double> cumsum;
	cumsum.reserve(queue.size() + 1);
	cumsum.push_back(0);

	for(auto q = queue.begin(); q != queue.end(); q++) {
	    cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();
	    cl::Device  device  = q->getInfo<CL_QUEUE_DEVICE>();

	    auto dw = device_weight.find(device());

	    double w = (dw == device_weight.end()) ?
		(device_weight[device()] = weight(context, device)) :
		dw->second;

	    cumsum.push_back(cumsum.back() + w);
	}

	for(uint d = 1; d < queue.size(); d++)
	    part.push_back(
		    std::min(n,
			alignup(static_cast<size_t>(n * cumsum[d] / cumsum.back()))
			)
		    );
    }

    part.push_back(n);
    return part;
}

template <bool dummy>
typename partitioning_scheme<dummy>::weight_function partitioning_scheme<dummy>::weight;

inline std::vector<size_t> partition(size_t n,
	    const std::vector<cl::CommandQueue> &queue)
{
    return partitioning_scheme<true>::get(n, queue);
}

/// Create and build a program from source string.
inline cl::Program build_sources(
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

} // namespace vex

#ifdef WIN32
#  pragma warning(pop)
#endif
#endif
