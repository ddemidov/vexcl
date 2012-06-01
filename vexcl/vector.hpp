#ifndef VEXCL_VECTOR_HPP
#define VEXCL_VECTOR_HPP

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
 * \file   vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device vector.
 */

#ifdef WIN32
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <cassert>

/// OpenCL convenience utilities.
namespace vex {

template<class T> struct SpMV;
template <class Expr, typename T> struct ExSpMV;

inline std::vector<uint> partition(
	uint n, const std::vector<cl::CommandQueue> &queue);

/// Device vector.
template<class T>
class vector {
    public:
	static const bool is_expression = true;

	/// Proxy class.
	/**
	 * Instances of this class are returned from vector::operator[]. These
	 * may be used to read or write single element of a vector, although
	 * this operations are too expensive to be used extensively and should
	 * be reserved for debugging purposes.
	 */
	class element {
	    public:
		/// Read associated element of a vector.
		operator T() const {
		    T val;
		    queue.enqueueReadBuffer(
			    buf, CL_TRUE,
			    index * sizeof(T), sizeof(T),
			    &val
			    );
		    return val;
		}

		/// Write associated element of a vector.
		T operator=(T val) {
		    queue.enqueueWriteBuffer(
			    buf, CL_TRUE,
			    index * sizeof(T), sizeof(T),
			    &val
			    );
		    return val;
		}
	    private:
		element(cl::CommandQueue q, cl::Buffer b, uint i)
		    : queue(q), buf(b), index(i) {}

		const cl::CommandQueue  &queue;
		cl::Buffer              buf;
		const uint              index;

		friend class vector;
		friend typename vector::iterator;
	};

	/// Iterator class.
	/**
	 * This class may in principle be used with standard template library,
	 * although its main purpose is range specification for vector copy
	 * operations.
	 */
	template <class vector_type, class element_type>
	class iterator_type
	    : public std::iterator<std::random_access_iterator_tag, T>
	{
	    public:
		static const bool device_iterator = true;

		element_type operator*() const {
		    return element_type(
			    vec.queue[part], vec.buf[part],
			    pos - vec.part[part]
			    );
		}

		iterator_type& operator++() {
		    pos++;
		    while (pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		    return *this;
		}

		iterator_type operator+(ptrdiff_t d) const {
		    return iterator_type(vec, pos + d);
		}

		ptrdiff_t operator-(iterator_type it) const {
		    return pos - it.pos;
		}

		bool operator==(const iterator_type &it) const {
		    return pos == it.pos;
		}

		bool operator!=(const iterator_type &it) const {
		    return pos != it.pos;
		}

		vector_type &vec;
		size_t  pos;
		size_t  part;

	    private:
		iterator_type(vector_type &vec, size_t pos)
		    : vec(vec), pos(pos), part(0)
		{
		    while(pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		}

		friend class vector;
	};

	typedef iterator_type<vector, element> iterator;
	typedef iterator_type<const vector, const element> const_iterator;

	/// Empty constructor.
	vector() {}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue, cl_mem_flags flags,
		uint size, const T *host = 0
	      ) : queue(queue), part(vex::partition(size, queue)),
	          buf(queue.size()), event(queue.size())
	{
	    if (size) allocate_buffers(flags, host);
	}

	/// Move constructor
	vector(vector &&v)
	    : queue(std::move(v.queue)), part(std::move(v.part)),
	      buf(std::move(v.buf)), event(std::move(v.event))
	{
	}

	/// Move assignment
	const vector& operator=(vector &&v) {
	    std::swap(queue,   v.queue);
	    std::swap(part,    v.part);
	    std::swap(buf,     v.buf);
	    std::swap(event,   v.event);

	    return *this;
	}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue, cl_mem_flags flags,
		const std::vector<T> &host
	      ) : queue(queue), part(vex::partition(host.size(), queue)),
		  buf(queue.size()), event(queue.size())
	{
	    if (!host.empty()) allocate_buffers(flags, host.data());
	}

	/// Return cl::Buffer object located on a given device.
	cl::Buffer operator()(uint d = 0) const {
	    return buf[d];
	}

	/// Const iterator to beginning.
	const_iterator begin() const {
	    return const_iterator(*this, 0);
	}

	/// Const iterator to end.
	const_iterator end() const {
	    return const_iterator(*this, size());
	}

	/// Iterator to beginning.
	iterator begin() {
	    return iterator(*this, 0);
	}

	/// Iterator to end.
	iterator end() {
	    return iterator(*this, size());
	}

	/// Access element.
	const element operator[](uint index) const {
	    uint p = 0;
	    while(index >= part[p + 1] && p < nparts()) p++;
	    return element(queue[p], buf[p], index - part[p]);
	}

	/// Access element.
	element operator[](uint index) {
	    uint p = 0;
	    while(index >= part[p + 1] && p < nparts()) p++;
	    return element(queue[p], buf[p], index - part[p]);
	}

	/// Return size .
	uint size() const {
	    return part.back();
	}

	/// Return number of parts (devices).
	uint nparts() const {
	    return queue.size();
	}

	/// Return size of part on a given device.
	uint part_size(uint d) const {
	    return part[d + 1] - part[d];
	}

	/// Copies data from device vector.
	const vector& operator=(const vector &x) {
	    if (&x != this) {
		for(uint i = 0; i < queue.size(); i++)
		    queue[i].enqueueCopyBuffer(x.buf[i], buf[i], 0, 0,
			    (part[i + 1] - part[i]) * sizeof(T));
	    }

	    return *this;
	}

	/** \name Expression assignments.
	 * @{
	 * The appropriate kernel is compiled first time the assignment is
	 * made. Vectors participating in expression should have same number of
	 * parts; corresponding parts of the vectors should reside on the same
	 * compute devices.
	 */
	template <class Expr>
	    const vector& operator=(const Expr &expr) {
		for(auto q = queue.begin(); q != queue.end(); q++) {
		    cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();

		    if (!exdata<Expr>::compiled[context()]) {
			std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

			bool device_is_cpu = (
				device[0].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU
				);

			std::ostringstream kernel;

			std::string kernel_name = expr.kernel_name();

			kernel <<
			    "#if defined(cl_khr_fp64)\n"
			    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
			    "#elif defined(cl_amd_fp64)\n"
			    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
			    "#endif\n" <<
			    "kernel void " << kernel_name << "(\n"
			    "\tunsigned int n,\n"
			    "\tglobal " << type_name<T>() << " *res";

			expr.kernel_prm(kernel);

			kernel <<
			    "\n\t)\n{\n"
			    "\tunsigned int i = get_global_id(0);\n";
			if (device_is_cpu) {
			    kernel <<
				"\tif (i < n) {\n"
				"\t\tres[i] = ";
			} else {
			    kernel <<
				"\tunsigned int grid_size = get_num_groups(0) * get_local_size(0);\n"
				"\twhile (i < n) {\n"
				"\t\tres[i] = ";
			}

			expr.kernel_expr(kernel);

			if (device_is_cpu) {
			    kernel <<
				";\n"
				"\t}\n"
				"}" << std::endl;
			} else {
			    kernel <<
				";\n"
				"\t\ti += grid_size;\n"
				"\t}\n"
				"}" << std::endl;
			}

			auto program = build_sources(context, kernel.str());

			exdata<Expr>::kernel[context()]   = cl::Kernel(program, kernel_name.c_str());
			exdata<Expr>::compiled[context()] = true;
			exdata<Expr>::wgsize[context()]   = kernel_workgroup_size(
				exdata<Expr>::kernel[context()], device);
		    }
		}

		for(uint d = 0; d < queue.size(); d++) {
		    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();
		    cl::Device  device  = queue[d].getInfo<CL_QUEUE_DEVICE>();

		    uint psize = part[d + 1] - part[d];

		    uint g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
			alignup(psize, exdata<Expr>::wgsize[context()]) :
			device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * exdata<Expr>::wgsize[context()] * 4;


		    uint pos = 0;
		    exdata<Expr>::kernel[context()].setArg(pos++, psize);
		    exdata<Expr>::kernel[context()].setArg(pos++, buf[d]);
		    expr.kernel_args(exdata<Expr>::kernel[context()], d, pos);

		    queue[d].enqueueNDRangeKernel(
			    exdata<Expr>::kernel[context()],
			    cl::NullRange,
			    g_size, exdata<Expr>::wgsize[context()]
			    );
		}

		return *this;
	    }

	template <class Expr>
	const vector& operator+=(const Expr &expr) {
	    return *this = *this + expr;
	}

	template <class Expr>
	const vector& operator*=(const Expr &expr) {
	    return *this = *this * expr;
	}

	template <class Expr>
	const vector& operator/=(const Expr &expr) {
	    return *this = *this / expr;
	}

	template <class Expr>
	const vector& operator-=(const Expr &expr) {
	    return *this = *this - expr;
	}

	template <class Expr>
	const vector& operator=(const ExSpMV<Expr,T> &xmv);

	const vector& operator=(const SpMV<T> &spmv);
	const vector& operator+=(const SpMV<T> &spmv);
	const vector& operator-=(const SpMV<T> &spmv);
	/// @}

	/** \name Service methods used for kernel generation.
	 * @{
	 */
	std::string kernel_name() const {
	    return "v";
	}

	void kernel_expr(std::ostream &os, std::string name = "v") const {
	    os << name << "[i]";
	}

	void kernel_prm(std::ostream &os, std::string name = "v") const {
	    os << ",\n\tglobal " << type_name<T>() << " *" << name;
	}

	void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	    k.setArg(pos++, buf[devnum]);
	}
	/// @}

	/// Copy data from host buffer to device(s).
	void write_data(uint offset, uint size, const T *hostptr, cl_bool blocking) {
	    if (!size) return;

	    for(uint d = 0; d < queue.size(); d++) {
		uint start = std::max(offset,        part[d]);
		uint stop  = std::min(offset + size, part[d + 1]);

		if (stop <= start) continue;

		queue[d].enqueueWriteBuffer(buf[d], CL_FALSE,
			sizeof(T) * (start - part[d]),
			sizeof(T) * (stop - start),
			hostptr + start - offset,
			0, &event[d]
			);
	    }

	    if (blocking)
		for(auto e = event.begin(); e != event.end(); e++)
		    e->wait();
	}

	/// Copy data from device(s) to host buffer .
	void read_data(uint offset, uint size, T *hostptr, cl_bool blocking) const {
	    if (!size) return;

	    for(uint d = 0; d < queue.size(); d++) {
		uint start = std::max(offset,        part[d]);
		uint stop  = std::min(offset + size, part[d + 1]);

		if (stop <= start) continue;

		queue[d].enqueueReadBuffer(buf[d], CL_FALSE,
			sizeof(T) * (start - part[d]),
			sizeof(T) * (stop - start),
			hostptr + start - offset,
			0, &event[d]
			);
	    }

	    if (blocking)
		for(auto e = event.begin(); e != event.end(); e++)
		    e->wait();
	}
    private:
	template <class Expr>
	struct exdata {
	    static std::map<cl_context,bool>       compiled;
	    static std::map<cl_context,cl::Kernel> kernel;
	    static std::map<cl_context,uint>       wgsize;
	};

	std::vector<cl::CommandQueue>	queue;
	std::vector<uint>               part;
	std::vector<cl::Buffer>		buf;
	mutable std::vector<cl::Event>  event;

	void allocate_buffers(cl_mem_flags flags, const T *hostptr) {
	    for(uint d = 0; d < queue.size(); d++) {
		cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

		buf[d] = cl::Buffer(context, flags,
				    (part[d + 1] - part[d]) * sizeof(T));
	    }

	    if (hostptr) write_data(0, size(), hostptr, CL_TRUE);
	}
};

template <class T> template <class Expr>
std::map<cl_context, bool> vector<T>::exdata<Expr>::compiled;

template <class T> template <class Expr>
std::map<cl_context, cl::Kernel> vector<T>::exdata<Expr>::kernel;

template <class T> template <class Expr>
std::map<cl_context, uint> vector<T>::exdata<Expr>::wgsize;

/// Copy device vector to host vector.
template <class T>
void copy(const vex::vector<T> &dv, std::vector<T> &hv, cl_bool blocking = CL_TRUE) {
    dv.read_data(0, dv.size(), hv.data(), blocking);
}

/// Copy host vector to device vector.
template <class T>
void copy(const std::vector<T> &hv, vex::vector<T> &dv, cl_bool blocking = CL_TRUE) {
    dv.write_data(0, dv.size(), hv.data(), blocking);
}

template<class Iterator, class Enable = void>
struct stored_on_device {
    static const bool value = false;
};

template<class Iterator>
struct stored_on_device<Iterator, typename std::enable_if<Iterator::device_iterator>::type> {
    static const bool value = true;
};

/// Copy range from device vector to host vector.
template<class InputIterator, class OutputIterator>
typename std::enable_if<
    std::is_same<
	typename std::iterator_traits<InputIterator>::value_type,
	typename std::iterator_traits<OutputIterator>::value_type
	>::value &&
    stored_on_device<InputIterator>::value &&
    !stored_on_device<OutputIterator>::value,
    OutputIterator
    >::type
copy(InputIterator first, InputIterator last,
	OutputIterator result, cl_bool blocking = CL_TRUE)
{
    first.vec.read_data(first.pos, last - first, &result[0], blocking);
    return result + (last - first);
}

/// Copy range from host vector to device vector.
template<class InputIterator, class OutputIterator>
typename std::enable_if<
    std::is_same<
	typename std::iterator_traits<InputIterator>::value_type,
	typename std::iterator_traits<OutputIterator>::value_type
	>::value &&
    !stored_on_device<InputIterator>::value &&
    stored_on_device<OutputIterator>::value,
    OutputIterator
    >::type
copy(InputIterator first, InputIterator last,
	OutputIterator result, cl_bool blocking = CL_TRUE)
{
    result.vec.write_data(result.pos, last - first, &first[0], blocking);
    return result + (last - first);
}

/// Expression template.
template <class LHS, char OP, class RHS>
struct BinaryExpression {
    static const bool is_expression = true;

    BinaryExpression(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    std::string kernel_name() const {
	// Polish notation.
	switch (OP) {
	    case '+':
		return "p" + lhs.kernel_name() + rhs.kernel_name();
	    case '-':
		return "m" + lhs.kernel_name() + rhs.kernel_name();
	    case '*':
		return "t" + lhs.kernel_name() + rhs.kernel_name();
	    case '/':
		return "d" + lhs.kernel_name() + rhs.kernel_name();
	    default:
		throw "unknown operation";
	}
    }

    void kernel_prm(std::ostream &os, std::string name = "") const {
	lhs.kernel_prm(os, name + "l");
	rhs.kernel_prm(os, name + "r");
    }

    void kernel_expr(std::ostream &os, std::string name = "") const {
	os << "(";
	lhs.kernel_expr(os, name + "l");
	os << " " << OP << " ";
	rhs.kernel_expr(os, name + "r");
	os << ")";
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	lhs.kernel_args(k, devnum, pos);
	rhs.kernel_args(k, devnum, pos);
    }

    uint part_size(uint dev) const {
	return std::max(lhs.part_size(dev), rhs.part_size(dev));
    }

    const LHS &lhs;
    const RHS &rhs;
};

/// Sum of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '+', RHS>
    >::type
    operator+(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'+',RHS>(lhs, rhs);
    }

/// Difference of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '-', RHS>
    >::type
    operator-(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'-',RHS>(lhs, rhs);
    }

/// Product of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '*', RHS>
    >::type
    operator*(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'*',RHS>(lhs, rhs);
    }

/// Division of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '/', RHS>
    >::type
    operator/(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'/',RHS>(lhs, rhs);
    }

/// Constant for use in vector expressions.
template <class T>
struct Constant {
    static const bool is_expression = true;

    Constant(T value) : value(value) {}

    std::string kernel_name() const {
	return "c";
    }

    void kernel_expr(std::ostream &os, std::string name = "c") const {
	os << name;
    }

    void kernel_prm(std::ostream &os, std::string name = "c") const {
	os << ",\n\t" << type_name<T>() << " " << name;
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	k.setArg(pos++, value);
    }

    uint part_size(uint dev) const {
	return 1;
    }

    private:
	T value;
};

/// Constant for use in vector expressions.
template <class T>
Constant<T> Const(T value) { return Constant<T>(value); }

struct UnaryFunction;

/// \internal Unary expression template.
template <class Expr>
struct UnaryExpression {
    static const bool is_expression = true;

    UnaryExpression(const Expr &expr, const UnaryFunction &fun)
	: expr(expr), fun(fun) {}

    std::string kernel_name() const;

    void kernel_expr(std::ostream &os, std::string name = "f") const;

    void kernel_prm(std::ostream &os, std::string name = "f") const {
	expr.kernel_prm(os, name);
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	expr.kernel_args(k, devnum, pos);
    }

    uint part_size(uint dev) const {
	return expr.part_size(dev);
    }

    private:
	const Expr &expr;
	const UnaryFunction &fun;
};

struct UnaryFunction {
    UnaryFunction(const std::string &name) : name(name) {}

    template <class Expr>
    typename std::enable_if<Expr::is_expression, UnaryExpression<Expr>>::type
    operator()(const Expr &expr) const {
	return UnaryExpression<Expr>(expr, *this);
    }

    std::string name;
};

template <class Expr>
std::string UnaryExpression<Expr>::kernel_name() const {
    return fun.name + expr.kernel_name();
}

template <class Expr>
void UnaryExpression<Expr>::kernel_expr(std::ostream &os, std::string name) const {
    os << fun.name << "(";
    expr.kernel_expr(os, name);
    os << ")";
}

static const UnaryFunction Sqrt("sqrt");
static const UnaryFunction Abs ("fabs");
static const UnaryFunction Absf("fabsf");
static const UnaryFunction Sin ("sin");
static const UnaryFunction Cos ("cos");
static const UnaryFunction Tan ("tan");

#ifndef VEXCL_DUMB_PARTITIONING

/// Returns device weight after simple bandwidth test
inline double device_weight(
	const cl::Context &context, const cl::Device &device,
	uint test_size = 1048576U
	)
{
    std::vector<cl::CommandQueue> queue(1,
	    cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE)
	    );

    // Allocate test vectors on current device and measure execution
    // time of a simple kernel.
    vex::vector<float> a(queue, CL_MEM_READ_WRITE, test_size);
    vex::vector<float> b(queue, CL_MEM_READ_WRITE, test_size);
    vex::vector<float> c(queue, CL_MEM_READ_WRITE, test_size);

    b = Const(1);
    c = Const(2);

    // Skip the first run.
    a = b + c;

    // Measure the second run.
    cl::Event beg, end;

    queue[0].enqueueMarker(&beg);
    a = b + c;
    queue[0].enqueueMarker(&end);

    beg.wait();
    end.wait();

    return 1.0 / (
	    end.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
	    beg.getProfilingInfo<CL_PROFILING_COMMAND_END>()
	    );
}

/// Returns partitions for vector of size n.
inline std::vector<uint> partition(uint n, const std::vector<cl::CommandQueue> &queue)
{
    static std::map<cl_device_id, double> dev_weights;

    std::vector<uint> part(queue.size() + 1);
    part[0] = 0;

    if (queue.size() > 1) {
	double tot_weight = 0;
	for(auto q = queue.begin(); q != queue.end(); q++) {
	    cl::Device device = q->getInfo<CL_QUEUE_DEVICE>();

	    auto dw = dev_weights.find(device());

	    if (dw == dev_weights.end()) {
		cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();
		tot_weight += (dev_weights[device()] = device_weight(context, device));
	    } else {
		tot_weight += dw->second;
	    }
	}

	double sum = 0;
	for(uint i = 0; i < queue.size(); i++) {
	    cl::Device device = queue[i].getInfo<CL_QUEUE_DEVICE>();

	    sum += dev_weights[device()] / tot_weight;

	    part[i + 1] = std::min(n, alignup(static_cast<uint>(sum * n)));
	}
    }

    part.back() = n;

    return part;
}

#else

/// Returns partitions for vector of size n.
inline std::vector<uint> partition(uint n, const std::vector<cl::CommandQueue> &queue)
{
    uint m = queue.size();

    std::vector<uint> part(m + 1);
    part[0] = 0;

    if (queue.size() > 1) {
	for(uint i = 0, chunk_size = alignup((n + m - 1) / m); i < m; i++)
	    part[i + 1] = std::min(n, part[i] + chunk_size);
    } else {
	part.back() = n;
    }

    return part;
}

#endif

} // namespace vex

#endif
