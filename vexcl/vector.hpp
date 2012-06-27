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
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#ifndef _MSC_VER
#  define VEXCL_VARIADIC_TEMPLATES
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif

#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <cassert>
#include <CL/cl.hpp>
#include <vexcl/util.hpp>
#include <vexcl/profiler.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

/// \cond INTERNAL

template<class T, typename column_t> struct SpMV;
template <class Expr, typename T, typename column_t> struct ExSpMV;
template<class T, typename column_t, uint N> struct MultiSpMV;
template <class Expr, typename T, typename column_t, uint N> struct MultiExSpMV;
template <class T> struct Conv;
template <class f, class T> struct GConv;
template <class Expr, class T> struct ExConv;
template <class Expr, class f, class T> struct ExGConv;
template <class T, uint N> struct MultiConv;
template <class Expr, class T, uint N> struct MultiExConv;
template <class func, class T, uint N> struct MultiGConv;
template <class Expr, class func, class T, uint N> struct MultiExGConv;
template <class T> struct GStencilProd;
template <class T, uint N> struct MultiGStencilProd;

/// Base class for a member of an expression.
/**
 * Each vector expression results in a single kernel. Name of the kernel,
 * parameter list and the body are formed automatically with the help of
 * members of the expression. This class is an abstract interface which any
 * expression member should implement.
 */
struct expression {
    static const bool is_expr = true;

    /// Preamble.
    /**
     * An expression might need to put something prior the kernel definition.
     * This could be a typedef, a helper function definition or anything else.
     * Most expressions don't need it and so default empty function is
     * provided.
     * \param os   Output stream which holds kernel source.
     * \param name Name of current node in an expression tree. Should be used
     *		   as prefix when naming any objects created to escape
     *		   possible ambiguities with other nodes.
     */
    virtual void preamble(std::ostream &os, std::string name) const {
    }

    /// Kernel name.
    /**
     * Name of the kernel is formed by its members through calls to their
     * kernel_name() functions. For example, expression
     * \code
     *   x = 3 * y + z;
     * \endcode
     * will result in kernel named "ptcvv" (plus times constant vector vector;
     * polish notation is used).
     * This naming scheme is not strictly necessary, as each expression
     * template holds its own cl::Program object and no ambiguity is possible.
     * But it helps when you profiling your program performance.
     */
    virtual std::string kernel_name() const = 0;

    /// Kernel parameter list.
    /**
     * Each terminal expression should output type and name of kernel
     * parameters it needs here.
     * \param os   Output stream which holds kernel source.
     * \param name Name of current node in an expression tree. Should be used
     *             directly or as a prefix to form parameter name(s).
     */
    virtual void kernel_prm(std::ostream &os, std::string name) const = 0;

    /// Kernel arguments.
    /**
     * This function is called at the time of actual kernel launch.
     * Each terminal expression should set kernel arguments for the parameters
     * it needs at specified position. Position should be incremented
     * afterwards.
     * \param k      OpenCL kernel that is being prepared to launch.
     * \param devnum Number of queue in queue list for which the kernel is
     *               launched.
     * \param pos    Current position in parameter stack.
     */
    virtual void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const = 0;

    /// Kernel body.
    /**
     * The actual expression which forms the kernel body.
     * \param os   Output stream which holds kernel source.
     * \param name Name of current node in an expression tree. Should be used
     *             directly or as a prefix to form parameter name(s).
     */
    virtual void kernel_expr(std::ostream &os, std::string name) const = 0;

    /// Size of vectors forming the expression.
    /**
     * \param dev Position in active queue list for which to return the size.
     */
    virtual size_t part_size(uint dev) const = 0;

    virtual ~expression() {}
};

/// Default kernel generation helper.
/**
 * Works on top of classes inheriting expression interface;
 */
template <class T, class Enable = void>
struct KernelGenerator {
    KernelGenerator(const T &value) : value(value) {}

    void preamble(std::ostream &os, std::string name) const {
	value.preamble(os, name);
    }

    std::string kernel_name() const {
	return value.kernel_name();
    }

    void kernel_prm(std::ostream &os, std::string name) const {
	value.kernel_prm(os, name);
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	value.kernel_args(k, devnum, pos);
    }

    void kernel_expr(std::ostream &os, std::string name) const {
	value.kernel_expr(os, name);
    }

    size_t part_size(uint dev) const {
	return value.part_size(dev);
    }

    private:
	const T &value;
};

/// Kernel generation helper for arithmetic types.
template <typename T>
struct KernelGenerator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    KernelGenerator(const T &value) : value(value) {}

    void preamble(std::ostream &os, std::string name) const {}

    std::string kernel_name() const {
	return "c";
    }

    void kernel_expr(std::ostream &os, std::string name) const {
	os << name;
    }

    void kernel_prm(std::ostream &os, std::string name) const {
	os << ",\n\t" << type_name<T>() << " " << name;
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	k.setArg(pos++, value);
    }

    size_t part_size(uint dev) const {
	return 0;
    }

    private:
	const T &value;
};

/// \endcond

/// Device vector.
template<class T>
class vector : public expression {
    public:
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
		element(const cl::CommandQueue &q, cl::Buffer b, size_t i)
		    : queue(q), buf(b), index(i) {}

		const cl::CommandQueue  &queue;
		cl::Buffer              buf;
		const size_t            index;

		friend class vector;
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
		    while (part < vec.nparts() && pos >= vec.part[part + 1])
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
		    if (!vec.part.empty()) {
			part = std::upper_bound(
				vec.part.begin(), vec.part.end(), pos
				) - vec.part.begin() - 1;
		    }
		}

		friend class vector;
	};

	typedef iterator_type<vector, element> iterator;
	typedef iterator_type<const vector, const element> const_iterator;

	/// Empty constructor.
	vector() {}

	/// Copy constructor.
	vector(const vector &v)
	    : queue(v.queue), part(v.part),
	      buf(queue.size()), event(queue.size())
	{
	    if (size()) allocate_buffers(CL_MEM_READ_WRITE, 0);
	    *this = v;
	}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue,
		size_t size, const T *host = 0,
		cl_mem_flags flags = CL_MEM_READ_WRITE
	      ) : queue(queue), part(vex::partition(size, queue)),
	          buf(queue.size()), event(queue.size())
	{
	    if (size) allocate_buffers(flags, host);
	}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue,
		const std::vector<T> &host,
		cl_mem_flags flags = CL_MEM_READ_WRITE
	      ) : queue(queue), part(vex::partition(host.size(), queue)),
		  buf(queue.size()), event(queue.size())
	{
	    if (!host.empty()) allocate_buffers(flags, host.data());
	}

	/// Move constructor
	vector(vector &&v)
	    : queue(std::move(v.queue)), part(std::move(v.part)),
	      buf(std::move(v.buf)), event(std::move(v.event))
	{
	}

	/// Move assignment
	const vector& operator=(vector &&v) {
	    swap(v);
	    return *this;
	}

	/// Swap function.
	void swap(vector &v) {
	    std::swap(queue,   v.queue);
	    std::swap(part,    v.part);
	    std::swap(buf,     v.buf);
	    std::swap(event,   v.event);
	}

	/// Resize vector.
	void resize(const vector &v, cl_mem_flags flags = CL_MEM_READ_WRITE)
	{
	    *this = std::move(vector(v.queue, v.size(), 0, flags));
	    *this = v;
	}

	/// Resize vector.
	void resize(const std::vector<cl::CommandQueue> &queue,
		size_t size, const T *host = 0,
		cl_mem_flags flags = CL_MEM_READ_WRITE
		)
	{
	    *this = std::move(vector(queue, size, host, flags));
	}

	/// Resize vector.
	void resize(const std::vector<cl::CommandQueue> &queue,
		const std::vector<T> &host,
		cl_mem_flags flags = CL_MEM_READ_WRITE
	      )
	{
	    *this = std::move(vector(queue, host, flags));
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
	const element operator[](size_t index) const {
	    uint d = std::upper_bound(
		    part.begin(), part.end(), index) - part.begin() - 1;
	    return element(queue[d], buf[d], index - part[d]);
	}

	/// Access element.
	element operator[](size_t index) {
	    uint d = static_cast<uint>(
		std::upper_bound(part.begin(), part.end(), index) - part.begin() - 1
		);
	    return element(queue[d], buf[d], index - part[d]);
	}

	/// Return size .
	size_t size() const {
	    return part.empty() ? 0 : part.back();
	}

	/// Return number of parts (devices).
	uint nparts() const {
	    return queue.size();
	}

	/// Return size of part on a given device.
	size_t part_size(uint d) const {
	    return part[d + 1] - part[d];
	}

	/// Return part start for a given device.
	size_t part_start(uint d) const {
	    return part[d];
	}

	/// Return reference to vector's queue list
	const std::vector<cl::CommandQueue>& queue_list() const {
	    return queue;
	}

	/// Copies data from device vector.
	const vector& operator=(const vector &x) {
	    if (&x != this) {
		for(uint d = 0; d < queue.size(); d++)
		    if (size_t psize = part[d + 1] - part[d]) {
			queue[d].enqueueCopyBuffer(x.buf[d], buf[d], 0, 0,
				psize * sizeof(T));
		    }
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
		KernelGenerator<Expr> kgen(expr);

		for(auto q = queue.begin(); q != queue.end(); q++) {
		    cl::Context context = q->getInfo<CL_QUEUE_CONTEXT>();
		    cl::Device  device  = q->getInfo<CL_QUEUE_DEVICE>();

		    if (!exdata<Expr>::compiled[context()]) {

			bool device_is_cpu = (
				device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU
				);

			std::ostringstream kernel;

			std::string kernel_name = kgen.kernel_name();

			kernel << standard_kernel_header;

			kgen.preamble(kernel, "prm");

			kernel <<
			    "kernel void " << kernel_name << "(\n"
			    "\t" << type_name<size_t>() << " n,\n"
			    "\tglobal " << type_name<T>() << " *res";

			kgen.kernel_prm(kernel, "prm");

			kernel <<
			    "\n\t)\n{\n"
			    "\tsize_t i = get_global_id(0);\n";
			if (device_is_cpu) {
			    kernel <<
				"\tif (i < n) {\n"
				"\t\tres[i] = ";
			} else {
			    kernel <<
				"\tsize_t grid_size = get_num_groups(0) * get_local_size(0);\n"
				"\twhile (i < n) {\n"
				"\t\tres[i] = ";
			}

			kgen.kernel_expr(kernel, "prm");

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

#ifdef VEXCL_SHOW_KERNELS
			std::cout << kernel.str() << std::endl;
#endif

			auto program = build_sources(context, kernel.str());

			exdata<Expr>::kernel[context()]   = cl::Kernel(program, kernel_name.c_str());
			exdata<Expr>::compiled[context()] = true;
			exdata<Expr>::wgsize[context()]   = kernel_workgroup_size(
				exdata<Expr>::kernel[context()], device);
		    }
		}

		for(uint d = 0; d < queue.size(); d++) {
		    if (size_t psize = part[d + 1] - part[d]) {
			cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();
			cl::Device  device  = queue[d].getInfo<CL_QUEUE_DEVICE>();

			size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
			    alignup(psize, exdata<Expr>::wgsize[context()]) :
			    device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * exdata<Expr>::wgsize[context()] * 4;

			uint pos = 0;
			exdata<Expr>::kernel[context()].setArg(pos++, psize);
			exdata<Expr>::kernel[context()].setArg(pos++, buf[d]);
			kgen.kernel_args(exdata<Expr>::kernel[context()], d, pos);

			queue[d].enqueueNDRangeKernel(
				exdata<Expr>::kernel[context()],
				cl::NullRange,
				g_size, exdata<Expr>::wgsize[context()]
				);
		    }
		}

		return *this;
	    }

#define COMPOUND_ASSIGNMENT(cop, op) \
	template <class Expr> \
	const vector& operator cop(const Expr &expr) { \
	    return *this = *this op expr; \
	}

	COMPOUND_ASSIGNMENT(+=, +);
	COMPOUND_ASSIGNMENT(-=, -);
	COMPOUND_ASSIGNMENT(*=, *);
	COMPOUND_ASSIGNMENT(/=, /);
	COMPOUND_ASSIGNMENT(%=, %);
	COMPOUND_ASSIGNMENT(&=, &);
	COMPOUND_ASSIGNMENT(|=, |);
	COMPOUND_ASSIGNMENT(^=, ^);
	COMPOUND_ASSIGNMENT(<<=, <<);
	COMPOUND_ASSIGNMENT(>>=, >>);

#undef COMPOUND_ASSIGNMENT

	template <class Expr, typename column_t>
	const vector& operator=(const ExSpMV<Expr,T,column_t> &xmv);

	template <typename column_t>
	const vector& operator=(const SpMV<T,column_t> &spmv);

	const vector& operator=(const Conv<T> &cnv);

	template <class f>
	const vector& operator=(const GConv<f,T> &cnv);

	template <class Expr>
	const vector& operator=(const ExConv<Expr, T> &xc);

	template <class Expr, class f>
	const vector& operator=(const ExGConv<Expr,f,T> &xc);
	/// @}

	/// \cond INTERNAL

	/** \name Service methods used for kernel generation.
	 * @{
	 */
	std::string kernel_name() const {
	    return "v";
	}

	void kernel_expr(std::ostream &os, std::string name) const {
	    os << name << "[i]";
	}

	void kernel_prm(std::ostream &os, std::string name) const {
	    os << ",\n\tglobal " << type_name<T>() << " *" << name;
	}

	void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	    k.setArg(pos++, buf[devnum]);
	}
	/// @}

	/// Copy data from host buffer to device(s).
	void write_data(size_t offset, size_t size, const T *hostptr, cl_bool blocking,
		std::vector<cl::Event> *uevent = 0)
	{
	    if (!size) return;

	    std::vector<cl::Event> &ev = uevent ? *uevent : event;

	    for(uint d = 0; d < queue.size(); d++) {
		size_t start = std::max(offset,        part[d]);
		size_t stop  = std::min(offset + size, part[d + 1]);

		if (stop <= start) continue;

		queue[d].enqueueWriteBuffer(buf[d], CL_FALSE,
			sizeof(T) * (start - part[d]),
			sizeof(T) * (stop - start),
			hostptr + start - offset,
			0, &ev[d]
			);
	    }

	    if (blocking)
		for(size_t d = 0; d < queue.size(); d++) {
		    size_t start = std::max(offset,        part[d]);
		    size_t stop  = std::min(offset + size, part[d + 1]);

		    if (start < stop) ev[d].wait();
		}
	}

	/// Copy data from device(s) to host buffer .
	void read_data(size_t offset, size_t size, T *hostptr, cl_bool blocking,
		std::vector<cl::Event> *uevent = 0) const
	{
	    if (!size) return;

	    std::vector<cl::Event> &ev = uevent ? *uevent : event;

	    for(uint d = 0; d < queue.size(); d++) {
		size_t start = std::max(offset,        part[d]);
		size_t stop  = std::min(offset + size, part[d + 1]);

		if (stop <= start) continue;

		queue[d].enqueueReadBuffer(buf[d], CL_FALSE,
			sizeof(T) * (start - part[d]),
			sizeof(T) * (stop - start),
			hostptr + start - offset,
			0, &ev[d]
			);
	    }

	    if (blocking)
		for(uint d = 0; d < queue.size(); d++) {
		    size_t start = std::max(offset,        part[d]);
		    size_t stop  = std::min(offset + size, part[d + 1]);

		    if (start < stop) ev[d].wait();
		}
	}

	/// \endcond

    private:
	template <class Expr>
	struct exdata {
	    static std::map<cl_context,bool>       compiled;
	    static std::map<cl_context,cl::Kernel> kernel;
	    static std::map<cl_context,size_t>     wgsize;
	};

	std::vector<cl::CommandQueue>	queue;
	std::vector<size_t>             part;
	std::vector<cl::Buffer>		buf;
	mutable std::vector<cl::Event>  event;

	void allocate_buffers(cl_mem_flags flags, const T *hostptr) {
	    for(uint d = 0; d < queue.size(); d++) {
		if (size_t psize = part[d + 1] - part[d]) {
		    cl::Context context = queue[d].getInfo<CL_QUEUE_CONTEXT>();

		    buf[d] = cl::Buffer(context, flags, psize * sizeof(T));
		}
	    }
	    if (hostptr) write_data(0, size(), hostptr, CL_TRUE);
	}
};

template <class T> template <class Expr>
std::map<cl_context, bool> vector<T>::exdata<Expr>::compiled;

template <class T> template <class Expr>
std::map<cl_context, cl::Kernel> vector<T>::exdata<Expr>::kernel;

template <class T> template <class Expr>
std::map<cl_context, size_t> vector<T>::exdata<Expr>::wgsize;

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

/// \cond INTERNAL

template<class Iterator, class Enable = void>
struct stored_on_device : std::false_type {};

template<class Iterator>
struct stored_on_device<Iterator,
    typename std::enable_if<Iterator::device_iterator>::type
    > : std::true_type {};

/// \endcond

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

/// Swap two vectors.
template <typename T>
void swap(vector<T> &x, vector<T> &y) {
    x.swap(y);
}

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

/// Expression template.
template <class LHS, binop::kind OP, class RHS>
struct BinaryExpression : public expression {
    BinaryExpression(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    void preamble(std::ostream &os, std::string name) const {
	lhs.preamble(os, name + "l");
	rhs.preamble(os, name + "r");
    }

    std::string kernel_name() const {
	return binop::traits<OP>::name() + lhs.kernel_name() + rhs.kernel_name();
    }

    void kernel_prm(std::ostream &os, std::string name = "") const {
	lhs.kernel_prm(os, name + "l");
	rhs.kernel_prm(os, name + "r");
    }

    void kernel_expr(std::ostream &os, std::string name = "") const {
	os << "(";
	lhs.kernel_expr(os, name + "l");
	os << " " << binop::traits<OP>::oper() << " ";
	rhs.kernel_expr(os, name + "r");
	os << ")";
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	lhs.kernel_args(k, devnum, pos);
	rhs.kernel_args(k, devnum, pos);
    }

    size_t part_size(uint dev) const {
	return std::max(lhs.part_size(dev), rhs.part_size(dev));
    }

    const KernelGenerator<LHS> lhs;
    const KernelGenerator<RHS> rhs;
};

template <class T, class Enable = void>
struct valid_expr : std::false_type {};

template <typename T>
struct valid_expr<T,
    typename std::enable_if<T::is_expr>::type
    > : std::true_type {};

template <typename T>
struct valid_expr<T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type
    > : std::true_type{};

//---------------------------------------------------------------------------
// Multivector
//---------------------------------------------------------------------------
template <class Expr>
typename std::enable_if<Expr::is_multiex, const typename Expr::subtype&>::type
extract_component(const Expr &expr, uint i) {
    return expr(i);
}

template <class Expr>
typename std::enable_if<std::is_arithmetic<Expr>::value, const Expr&>::type
extract_component(const Expr &expr, uint i) {
    return expr;
}

template <class real, size_t N>
const real& extract_component(const std::array<real,N> &expr, uint i) {
    return expr[i];
}

template <class T>
struct is_std_array : std::false_type {};

template <class real, size_t N>
struct is_std_array<std::array<real,N>> : std::true_type {};

template <class T1, class T2, class Enable = void>
struct compatible_multiex : std::false_type {};

template <class T1, class T2>
struct compatible_multiex<T1, T2,
    typename std::enable_if<
	T1::is_multiex &&
	T2::is_multiex &&
	T1::dim == T2::dim>::type
	> : std::true_type {};

template <class T1, class T2>
struct compatible_multiex<T1, T2,
    typename std::enable_if<
	T1::is_multiex &&
	std::is_arithmetic<T2>::value>::type
	> : std::true_type {};

template <class T1, class T2>
struct compatible_multiex<T1, T2,
    typename std::enable_if<
	std::is_arithmetic<T1>::value &&
	T2::is_multiex>::type
	> : std::true_type {};

template <class T1, class T2, size_t N>
struct compatible_multiex<
	T1, std::array<T2,N>,
	typename std::enable_if<T1::is_multiex && T1::dim == N>::type
    > : std::true_type {};

template <class T1, size_t N, class T2>
struct compatible_multiex<
	std::array<T1,N>, T2,
	typename std::enable_if<T2::is_multiex && T2::dim == N>::type
    > : std::true_type {};

/// \endcond

/// Container for several vex::vectors.
/**
 * This class allows to synchronously operate on several vex::vectors of the
 * same type and size.
 */
template <typename T, uint N>
class multivector {
    public:
	static const bool is_multiex = true;
	static const uint dim = N;

	typedef vex::vector<T> subtype;
	typedef std::array<T,N> value_type;

	/// Proxy class.
	class element {
	    public:
		operator value_type () const {
		    value_type val;
		    for(uint i = 0; i < N; i++) val[i] = vec(i)[index];
		    return val;
		}

		value_type operator=(value_type val) {
		    for(uint i = 0; i < N; i++) vec(i)[index] = val[i];
		    return val;
		}
	    private:
		element(multivector &vec, size_t index)
		    : vec(vec), index(index) {}

		multivector &vec;
		const size_t      index;

		friend class multivector;
	};

	/// Proxy class.
	class const_element {
	    public:
		operator value_type () const {
		    value_type val;
		    for(uint i = 0; i < N; i++) val[i] = vec(i)[index];
		    return val;
		}
	    private:
		const_element(const multivector &vec, size_t index)
		    : vec(vec), index(index) {}

		const multivector &vec;
		const size_t      index;

		friend class multivector;
	};

	template <class V, class E>
	class iterator_type {
	    public:
		E operator*() const {
		    return E(vec, pos);
		}

		iterator_type& operator++() {
		    pos++;
		    return *this;
		}

		iterator_type operator+(ptrdiff_t d) const {
		    return iterator_type(vec, pos + d);
		}

		ptrdiff_t operator-(const iterator_type &it) const {
		    return pos - it.pos;
		}

		bool operator==(const iterator_type &it) const {
		    return pos == it.pos;
		}

		bool operator!=(const iterator_type &it) const {
		    return pos != it.pos;
		}
	    private:
		iterator_type(V &vec, size_t pos) : vec(vec), pos(pos) {}

		V      &vec;
		size_t pos;

		friend class multivector;
	};

	typedef iterator_type<multivector, element> iterator;
	typedef iterator_type<const multivector, const_element> const_iterator;

	multivector() {};

	/// Constructor.
	/**
	 * If host pointer is not NULL, it is copied to the underlying vector
	 * components of the multivector.
	 * \param queue queue list to be shared between all components.
	 * \param host  Host vector that holds data to be copied to
	 *              the components. Size of host vector should be divisible
	 *              by N. Components of the created multivector will have
	 *              size equal to host.size() / N. The data will be
	 *              partitioned equally between all components.
	 * \param flags cl::Buffer creation flags.
	 */
	multivector(const std::vector<cl::CommandQueue> &queue,
		const std::vector<T> &host,
		cl_mem_flags flags = CL_MEM_READ_WRITE)
	{
	    static_assert(N > 0, "What's the point?");

	    size_t size = host.size() / N;
	    assert(N * size == host.size());

	    for(uint i = 0; i < N; i++)
		vec[i] = vex::vector<T>(
			queue, size, host.data() + i * size, flags
			);
	}

	/// Constructor.
	/**
	 * If host pointer is not NULL, it is copied to the underlying vector
	 * components of the multivector.
	 * \param queue queue list to be shared between all components.
	 * \param size  Size of each component.
	 * \param host  Pointer to host buffer that holds data to be copied to
	 *              the components. Size of the buffer should be equal to
	 *              N * size. The data will be partitioned equally between
	 *              all components.
	 * \param flags cl::Buffer creation flags.
	 */
	multivector(const std::vector<cl::CommandQueue> &queue, size_t size,
		const T *host = 0, cl_mem_flags flags = CL_MEM_READ_WRITE)
	{
	    static_assert(N > 0, "What's the point?");

	    for(uint i = 0; i < N; i++)
		vec[i] = vex::vector<T>(
			queue, size, host ? host + i * size : 0, flags
			);
	}

	/// Resize multivector.
	void resize(const std::vector<cl::CommandQueue> &queue, size_t size) {
	    for(uint i = 0; i < N; i++) vec[i].resize(queue, size);
	}

	/// Return size of a multivector (equals size of individual components).
	size_t size() const {
	    return vec[0].size();
	}

	/// Returns multivector component.
	const vex::vector<T>& operator()(uint i) const {
	    return vec[i];
	}

	/// Returns multivector component.
	vex::vector<T>& operator()(uint i) {
	    return vec[i];
	}

	/// Const iterator to beginning.
	const_iterator begin() const {
	    return const_iterator(*this, 0);
	}

	/// Iterator to beginning.
	iterator begin() {
	    return iterator(*this, 0);
	}

	/// Const iterator to end.
	const_iterator end() const {
	    return const_iterator(*this, size());
	}

	/// Iterator to end.
	iterator end() {
	    return iterator(*this, size());
	}

	/// Returns elements of all vectors, packed in std::array.
	const_element operator[](size_t i) const {
	    return const_element(*this, i);
	}

	/// Assigns elements of all vectors to a std::array value.
	element operator[](size_t i) {
	    return element(*this, i);
	}

	/// Return reference to multivector's queue list
	const std::vector<cl::CommandQueue>& queue_list() const {
	    return vec[0].queue_list();
	}

	/** \name Expression assignments.
	 * @{
	 * All operations are delegated to components of the multivector.
	 */
	template <class Expr>
	typename std::enable_if<compatible_multiex<multivector, Expr>::value,
	    const multivector&
	    >::type
	operator=(const Expr& expr) {
	    for(uint i = 0; i < N; i++) vec[i] = extract_component(expr, i);
	    return *this;
	}

#define COMPOUND_ASSIGNMENT(cop, op) \
	template <class Expr> \
	const multivector& operator cop(const Expr &expr) { \
	    return *this = *this op expr; \
	}

	COMPOUND_ASSIGNMENT(+=, +);
	COMPOUND_ASSIGNMENT(-=, -);
	COMPOUND_ASSIGNMENT(*=, *);
	COMPOUND_ASSIGNMENT(/=, /);
	COMPOUND_ASSIGNMENT(%=, %);
	COMPOUND_ASSIGNMENT(&=, &);
	COMPOUND_ASSIGNMENT(|=, |);
	COMPOUND_ASSIGNMENT(^=, ^);
	COMPOUND_ASSIGNMENT(<<=, <<);
	COMPOUND_ASSIGNMENT(>>=, >>);

#undef COMPOUND_ASSIGNMENT

	template <typename column_t>
	const multivector& operator=(const MultiSpMV<T,column_t,N> &spmv);

	template <class Expr, typename column_t>
	const multivector& operator=(const MultiExSpMV<Expr,T,column_t,N> &xmv);

	const multivector& operator=(const MultiConv<T,N> &cnv);

	template <class Expr>
	const multivector& operator=(const MultiExConv<Expr,T,N> &cnv);

	template <class func>
	const multivector& operator=(const MultiGConv<func,T,N> &cnv);

	template <class Expr, class func>
	const multivector& operator=(const MultiExGConv<Expr,func,T,N> &cnv);
	/// @}

    private:
	std::array<vex::vector<T>,N> vec;
};

/// Copy multivector to host vector.
template <class T, uint N>
void copy(const multivector<T,N> &mv, std::vector<T> &hv) {
    for(uint i = 0; i < N; i++)
	vex::copy(mv(i).begin(), mv(i).end(), hv.begin() + i * mv.size());
}

/// Copy host vector to multivector.
template <class T, uint N>
void copy(const std::vector<T> &hv, multivector<T,N> &mv) {
    for(uint i = 0; i < N; i++)
	vex::copy(hv.begin() + i * mv.size(), hv.begin() + (i + 1) * mv.size(),
		mv(i).begin());
}

/// \cond INTERNAL

/// Multivector expression template
template <class Expr, uint N>
struct MultiExpression {
    static const bool is_multiex = true;
    static const uint dim = N;
    typedef Expr subtype;

    MultiExpression(std::array<std::unique_ptr<subtype>, dim> &ex) {
	for(uint i = 0; i < dim; i++) expr[i] = std::move(ex[i]);
    }

    const subtype& operator()(uint i) const {
	return *expr[i];
    }

    std::array<std::unique_ptr<subtype>, dim> expr;
};

template<class T, class Enable = void>
struct multiex_traits {
    static const uint dim = 0;
    typedef T subtype;
};

template<class T>
struct multiex_traits<T, typename std::enable_if<T::is_multiex>::type> {
    static const uint dim = T::dim;
    typedef typename T::subtype subtype;
};

template<class T>
struct multiex_traits<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    static const uint dim = 0;
    typedef T subtype;
};

template <class T, class Enable = void>
struct valid_multiex : std::false_type {};

template <typename T>
struct valid_multiex<T,
    typename std::enable_if<T::is_multiex>::type
    > : std::true_type {};

template <typename T>
struct valid_multiex<T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type
    > : std::true_type {};

#ifdef VEXCL_VARIADIC_TEMPLATES
template <class... Expr>
struct multiex_dim {
    static const uint dim = 0;
};

template <class Head, class... Tail>
struct multiex_dim<Head, Tail...> {
    static const uint dim =
	multiex_traits<Head>::dim > multiex_dim<Tail...>::dim ?
	multiex_traits<Head>::dim : multiex_dim<Tail...>::dim;
};
#else
template <class LHS, class RHS>
struct multiex_dim {
    static const uint dim =
	multiex_traits<LHS>::dim > multiex_traits<RHS>::dim ?
	multiex_traits<LHS>::dim : multiex_traits<RHS>::dim;
};
#endif

//---------------------------------------------------------------------------
// Arithmetic expressions
//---------------------------------------------------------------------------
#define DEFINE_BINARY_OP(kind, oper) \
template <class LHS, class RHS> \
typename std::enable_if< \
    valid_expr<LHS>::value && valid_expr<RHS>::value, \
    BinaryExpression<LHS, kind, RHS> \
    >::type \
    operator oper(const LHS &lhs, const RHS &rhs) { \
	return BinaryExpression<LHS, kind, RHS>(lhs, rhs); \
    } \
template <class LHS, class RHS> \
typename std::enable_if<compatible_multiex<LHS, RHS>::value, \
	 MultiExpression< \
	    BinaryExpression< \
		typename multiex_traits<LHS>::subtype, \
		kind, \
		typename multiex_traits<RHS>::subtype \
		>, \
	    multiex_dim<LHS, RHS>::dim \
	    >>::type \
operator oper(const LHS &lhs, const RHS &rhs) { \
    typedef BinaryExpression< \
		typename multiex_traits<LHS>::subtype, \
		kind, \
		typename multiex_traits<RHS>::subtype \
		> subtype; \
    std::array<std::unique_ptr<subtype>, multiex_dim<LHS, RHS>::dim> ex; \
    for(uint i = 0; i < multiex_dim<LHS, RHS>::dim; i++) \
	ex[i].reset(new subtype( \
		    extract_component(lhs, i), extract_component(rhs, i))); \
    return MultiExpression<subtype, multiex_dim<LHS, RHS>::dim>(ex); \
}

/// \endcond

/** \defgroup binop Binary operations
 * @{
 * You can use these in vector or multivector expressions.
 */
DEFINE_BINARY_OP(binop::Add,          + )
DEFINE_BINARY_OP(binop::Subtract,     - )
DEFINE_BINARY_OP(binop::Multiply,     * )
DEFINE_BINARY_OP(binop::Divide,       / )
DEFINE_BINARY_OP(binop::Remainder,    % )
DEFINE_BINARY_OP(binop::Greater,      > )
DEFINE_BINARY_OP(binop::Less,         < )
DEFINE_BINARY_OP(binop::GreaterEqual, >=)
DEFINE_BINARY_OP(binop::LessEqual,    <=)
DEFINE_BINARY_OP(binop::Equal,        ==)
DEFINE_BINARY_OP(binop::NotEqual,     !=)
DEFINE_BINARY_OP(binop::BitwiseAnd,   & )
DEFINE_BINARY_OP(binop::BitwiseOr,    | )
DEFINE_BINARY_OP(binop::BitwiseXor,   ^ )
DEFINE_BINARY_OP(binop::LogicalAnd,   &&)
DEFINE_BINARY_OP(binop::LogicalOr,    ||)
DEFINE_BINARY_OP(binop::RightShift,   >>)
DEFINE_BINARY_OP(binop::LeftShift,    <<)

#undef DEFINE_BINARY_OP
/** @} */
//---------------------------------------------------------------------------
// Builtin functions
//---------------------------------------------------------------------------
#ifdef VEXCL_VARIADIC_TEMPLATES

/// \cond INTERNAL

/// Builtin function call.
template <class func_name, class... Expr>
class BuiltinFunction : public expression {
    public:
	BuiltinFunction(const Expr&... expr) : expr(expr...) {}

	void preamble(std::ostream &os, std::string name) const {
	    build_preamble<0>(os, name);
	}

	std::string kernel_name() const {
	    return std::string(func_name::value()) + build_kernel_name<0>();
	}

	void kernel_prm(std::ostream &os, std::string name) const {
	    build_kernel_prm<0>(os, name);
	}

	void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	    set_kernel_args<0>(k, devnum, pos);
	}

	void kernel_expr(std::ostream &os, std::string name) const {
	    os << func_name::value() << "(";
	    build_kernel_expr<0>(os, name);
	    os << ")";
	}

	size_t part_size(uint dev) const {
	    return get_part_size<0>(dev);
	}
    private:
	std::tuple<const KernelGenerator<Expr>...> expr;

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), std::string>::type
	build_kernel_name() const {
	    return "";
	}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), std::string>::type
	build_kernel_name() const {
	    return std::get<num>(expr).kernel_name() + build_kernel_name<num + 1>();
	}

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), void>::type
	build_kernel_prm(std::ostream &os, std::string name) const {}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), void>::type
	build_kernel_prm(std::ostream &os, std::string name) const {
	    std::ostringstream cname;
	    cname << name << num + 1;
	    std::get<num>(expr).kernel_prm(os, cname.str());
	    build_kernel_prm<num + 1>(os, name);
	}

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), void>::type
	set_kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), void>::type
	set_kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	    std::get<num>(expr).kernel_args(k, devnum, pos);
	    set_kernel_args<num + 1>(k, devnum, pos);
	}

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), void>::type
	build_kernel_expr(std::ostream &os, std::string name) const {}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), void>::type
	build_kernel_expr(std::ostream &os, std::string name) const {
	    std::ostringstream cname;
	    cname << name << num + 1;
	    std::get<num>(expr).kernel_expr(os, cname.str());
	    if (num + 1 < sizeof...(Expr)) {
		os << ", ";
		build_kernel_expr<num + 1>(os, name);
	    }
	}

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), void>::type
	build_preamble(std::ostream &os, std::string name) const {}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), void>::type
	build_preamble(std::ostream &os, std::string name) const {
	    std::ostringstream cname;
	    cname << name << num + 1;
	    std::get<num>(expr).preamble(os, cname.str());
	    build_preamble<num + 1>(os, name);
	}

	//------------------------------------------------------------
	template <int num>
	typename std::enable_if<num == sizeof...(Expr), size_t>::type
	get_part_size(uint dev) const {
	    return 0;
	}

	template <int num>
	typename std::enable_if<num < sizeof...(Expr), size_t>::type
	get_part_size(uint dev) const {
	    return std::max(
		    std::get<num>(expr).part_size(dev),
		    get_part_size<num + 1>(dev)
		    );
	}
};

template <typename T>
struct Not : std::integral_constant<bool, !T::value> {};

template <typename... T>
struct All : std::true_type {};

template <typename Head, typename... Tail>
struct All<Head, Tail...>
    : std::conditional<Head::value, All<Tail...>, std::false_type>::type
{};

#define DEFINE_BUILTIN_FUNCTION(name) \
struct name##_name { \
    static const bool is_builtin = true; \
    static const bool is_userfun = false; \
    static const char* value() { \
	return #name; \
    } \
}; \
template <class... Expr> \
inline typename std::enable_if< \
  All<All<valid_expr<Expr>...>, Not<All<std::is_arithmetic<Expr>...>>>::value,\
  BuiltinFunction<name##_name, Expr...>>::type \
name(const Expr&... expr) { \
return BuiltinFunction<name##_name, Expr...>(expr...); \
} \
template <class... MultiEx> \
inline typename std::enable_if< \
  All<valid_multiex<MultiEx>..., Not<All<std::is_arithmetic<MultiEx>...>>>::value, \
    MultiExpression< \
	BuiltinFunction<name##_name, typename multiex_traits<MultiEx>::subtype...>, multiex_dim<MultiEx...>::dim \
	>>::type \
name(const MultiEx&... multiexpr) { \
    std::array< \
	std::unique_ptr< \
	    BuiltinFunction<name##_name, typename multiex_traits<MultiEx>::subtype...> \
	    >, \
	multiex_dim<MultiEx...>::dim> ex; \
    for(uint i = 0; i < multiex_dim<MultiEx...>::dim; i++) \
	ex[i].reset( \
		new BuiltinFunction<name##_name, typename multiex_traits<MultiEx>::subtype...>(extract_component(multiexpr, i)...) \
		); \
    return MultiExpression< \
	BuiltinFunction<name##_name, typename multiex_traits<MultiEx>::subtype...>, multiex_dim<MultiEx...>::dim \
	>(ex); \
}

/// \endcond

/** \defgroup builtins Builtin functions
 * @{
 * You can use these functions in vector or multivector expressions.
 */
DEFINE_BUILTIN_FUNCTION(acos)
DEFINE_BUILTIN_FUNCTION(acosh)
DEFINE_BUILTIN_FUNCTION(acospi)
DEFINE_BUILTIN_FUNCTION(asin)
DEFINE_BUILTIN_FUNCTION(asinh)
DEFINE_BUILTIN_FUNCTION(asinpi)
DEFINE_BUILTIN_FUNCTION(atan)
DEFINE_BUILTIN_FUNCTION(atan2)
DEFINE_BUILTIN_FUNCTION(atanh)
DEFINE_BUILTIN_FUNCTION(atanpi)
DEFINE_BUILTIN_FUNCTION(atan2pi)
DEFINE_BUILTIN_FUNCTION(cbrt)
DEFINE_BUILTIN_FUNCTION(ceil)
DEFINE_BUILTIN_FUNCTION(copysign)
DEFINE_BUILTIN_FUNCTION(cos)
DEFINE_BUILTIN_FUNCTION(cosh)
DEFINE_BUILTIN_FUNCTION(cospi)
DEFINE_BUILTIN_FUNCTION(erfc)
DEFINE_BUILTIN_FUNCTION(erf)
DEFINE_BUILTIN_FUNCTION(exp)
DEFINE_BUILTIN_FUNCTION(exp2)
DEFINE_BUILTIN_FUNCTION(exp10)
DEFINE_BUILTIN_FUNCTION(expm1)
DEFINE_BUILTIN_FUNCTION(fabs)
DEFINE_BUILTIN_FUNCTION(fdim)
DEFINE_BUILTIN_FUNCTION(floor)
DEFINE_BUILTIN_FUNCTION(fma)
DEFINE_BUILTIN_FUNCTION(fmax)
DEFINE_BUILTIN_FUNCTION(fmin)
DEFINE_BUILTIN_FUNCTION(fmod)
DEFINE_BUILTIN_FUNCTION(fract)
DEFINE_BUILTIN_FUNCTION(frexp)
DEFINE_BUILTIN_FUNCTION(hypot)
DEFINE_BUILTIN_FUNCTION(ilogb)
DEFINE_BUILTIN_FUNCTION(ldexp)
DEFINE_BUILTIN_FUNCTION(lgamma)
DEFINE_BUILTIN_FUNCTION(lgamma_r)
DEFINE_BUILTIN_FUNCTION(log)
DEFINE_BUILTIN_FUNCTION(log2)
DEFINE_BUILTIN_FUNCTION(log10)
DEFINE_BUILTIN_FUNCTION(log1p)
DEFINE_BUILTIN_FUNCTION(logb)
DEFINE_BUILTIN_FUNCTION(mad)
DEFINE_BUILTIN_FUNCTION(maxmag)
DEFINE_BUILTIN_FUNCTION(minmag)
DEFINE_BUILTIN_FUNCTION(modf)
DEFINE_BUILTIN_FUNCTION(nan)
DEFINE_BUILTIN_FUNCTION(nextafter)
DEFINE_BUILTIN_FUNCTION(pow)
DEFINE_BUILTIN_FUNCTION(pown)
DEFINE_BUILTIN_FUNCTION(powr)
DEFINE_BUILTIN_FUNCTION(remainder)
DEFINE_BUILTIN_FUNCTION(remquo)
DEFINE_BUILTIN_FUNCTION(rint)
DEFINE_BUILTIN_FUNCTION(rootn)
DEFINE_BUILTIN_FUNCTION(round)
DEFINE_BUILTIN_FUNCTION(rsqrt)
DEFINE_BUILTIN_FUNCTION(sin)
DEFINE_BUILTIN_FUNCTION(sincos)
DEFINE_BUILTIN_FUNCTION(sinh)
DEFINE_BUILTIN_FUNCTION(sinpi)
DEFINE_BUILTIN_FUNCTION(sqrt)
DEFINE_BUILTIN_FUNCTION(tan)
DEFINE_BUILTIN_FUNCTION(tanh)
DEFINE_BUILTIN_FUNCTION(tanpi)
DEFINE_BUILTIN_FUNCTION(tgamma)
DEFINE_BUILTIN_FUNCTION(trunc)

#undef DEFINE_BUILTIN_FUNCTION
/** @} */
#else

/// \cond INTERNAL

/// Builtin function call.
template <class func_name, class Expr>
struct BuiltinFunction : public expression {
    BuiltinFunction(const Expr &expr) : expr(expr) {}

    void preamble(std::ostream &os, std::string name) const {
	expr.preamble(os, name);
    }

    std::string kernel_name() const {
	return func_name::value() + expr.kernel_name();
    }

    void kernel_expr(std::ostream &os, std::string name) const {
	os << func_name::value() << "(";
	expr.kernel_expr(os, name);
	os << ")";
    }

    void kernel_prm(std::ostream &os, std::string name) const {
	expr.kernel_prm(os, name);
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	expr.kernel_args(k, devnum, pos);
    }

    size_t part_size(uint dev) const {
	return expr.part_size(dev);
    }

    private:
	const Expr &expr;
};

#define DEFINE_BUILTIN_FUNCTION(name) \
struct name##_name { \
    static const bool is_builtin = true; \
    static const bool is_userfun = false; \
    static const char* value() { \
	return #name; \
    } \
}; \
template <class Expr> \
typename std::enable_if<Expr::is_expr, \
BuiltinFunction<name##_name, Expr>>::type \
name(const Expr &expr) { \
return BuiltinFunction<name##_name, Expr>(expr); \
} \
template <class MultiEx> \
typename std::enable_if<MultiEx::is_multiex, \
  MultiExpression< \
    BuiltinFunction<name##_name, typename MultiEx::subtype>, MultiEx::dim \
    >>::type \
name(const MultiEx& multiexpr) { \
  std::array<std::unique_ptr< \
    BuiltinFunction<name##_name, typename MultiEx::subtype>>, MultiEx::dim \
    > ex; \
  for(uint i = 0; i < MultiEx::dim; i++) \
    ex[i].reset( \
	new BuiltinFunction<name##_name, typename MultiEx::subtype>(multiexpr(i)) \
	); \
  return MultiExpression< \
    BuiltinFunction<name##_name, typename MultiEx::subtype>, MultiEx::dim>(ex); \
}

/// \endcond

/** \defgroup builtins Builtin functions
 * @{
 * You can use these functions in vector or multivector expressions.
 */
DEFINE_BUILTIN_FUNCTION(acos)
DEFINE_BUILTIN_FUNCTION(acosh)
DEFINE_BUILTIN_FUNCTION(acospi)
DEFINE_BUILTIN_FUNCTION(asin)
DEFINE_BUILTIN_FUNCTION(asinh)
DEFINE_BUILTIN_FUNCTION(asinpi)
DEFINE_BUILTIN_FUNCTION(atan)
DEFINE_BUILTIN_FUNCTION(atanh)
DEFINE_BUILTIN_FUNCTION(atanpi)
DEFINE_BUILTIN_FUNCTION(cbrt)
DEFINE_BUILTIN_FUNCTION(ceil)
DEFINE_BUILTIN_FUNCTION(cos)
DEFINE_BUILTIN_FUNCTION(cosh)
DEFINE_BUILTIN_FUNCTION(cospi)
DEFINE_BUILTIN_FUNCTION(erfc)
DEFINE_BUILTIN_FUNCTION(erf)
DEFINE_BUILTIN_FUNCTION(exp)
DEFINE_BUILTIN_FUNCTION(exp2)
DEFINE_BUILTIN_FUNCTION(exp10)
DEFINE_BUILTIN_FUNCTION(expm1)
DEFINE_BUILTIN_FUNCTION(fabs)
DEFINE_BUILTIN_FUNCTION(floor)
DEFINE_BUILTIN_FUNCTION(ilogb)
DEFINE_BUILTIN_FUNCTION(lgamma)
DEFINE_BUILTIN_FUNCTION(log)
DEFINE_BUILTIN_FUNCTION(log2)
DEFINE_BUILTIN_FUNCTION(log10)
DEFINE_BUILTIN_FUNCTION(log1p)
DEFINE_BUILTIN_FUNCTION(logb)
DEFINE_BUILTIN_FUNCTION(nan)
DEFINE_BUILTIN_FUNCTION(rint)
DEFINE_BUILTIN_FUNCTION(rootn)
DEFINE_BUILTIN_FUNCTION(round)
DEFINE_BUILTIN_FUNCTION(rsqrt)
DEFINE_BUILTIN_FUNCTION(sin)
DEFINE_BUILTIN_FUNCTION(sinh)
DEFINE_BUILTIN_FUNCTION(sinpi)
DEFINE_BUILTIN_FUNCTION(sqrt)
DEFINE_BUILTIN_FUNCTION(tan)
DEFINE_BUILTIN_FUNCTION(tanh)
DEFINE_BUILTIN_FUNCTION(tanpi)
DEFINE_BUILTIN_FUNCTION(tgamma)
DEFINE_BUILTIN_FUNCTION(trunc)

#undef DEFINE_BUILTIN_FUNCTION
/** @} */
#endif

//---------------------------------------------------------------------------
// User-defined functions.
//---------------------------------------------------------------------------
template <class T>
struct UserFunctionDeclaration {
    static std::string get() { return ""; }
};

#ifdef VEXCL_VARIADIC_TEMPLATES
/// \cond INTERNAL

template <const char *body, class T>
struct UserFunction {};

template<const char *body, class RetType, class... ArgType>
struct UserFunction<body, RetType(ArgType...)>;

template <const char *body, class RetType, class... ArgType>
struct UserFunctionDeclaration<UserFunction<body, RetType(ArgType...)>>;

/// Custom user function expression template
template<class RetType, class... ArgType>
struct UserFunctionFamily {
    template <const char *body, class... Expr>
    class Function : public expression {
	public:
	    Function(const Expr&... expr) : expr(expr...) {}

	    void preamble(std::ostream &os, std::string name) const {
		build_preamble<0>(os, name);

		os << UserFunctionDeclaration<
		    UserFunction<body, RetType(ArgType...)>
		    >::get(name);
	    }

	    std::string kernel_name() const {
		return std::string("uf") + build_kernel_name<0>();
	    }

	    void kernel_prm(std::ostream &os, std::string name) const {
		build_kernel_prm<0>(os, name);
	    }

	    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
		set_kernel_args<0>(k, devnum, pos);
	    }

	    void kernel_expr(std::ostream &os, std::string name) const {
		os << name << "_fun(";
		build_kernel_expr<0>(os, name);
		os << ")";
	    }

	    size_t part_size(uint dev) const {
		return get_part_size<0>(dev);
	    }
	private:
	    std::tuple<const KernelGenerator<Expr>...> expr;

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), std::string>::type
	    build_kernel_name() const {
		return "";
	    }

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), std::string>::type
	    build_kernel_name() const {
		return std::get<num>(expr).kernel_name() + build_kernel_name<num + 1>();
	    }

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), void>::type
	    build_kernel_prm(std::ostream &os, std::string name) const {}

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), void>::type
	    build_kernel_prm(std::ostream &os, std::string name) const {
		std::ostringstream cname;
		cname << name << num + 1;
		std::get<num>(expr).kernel_prm(os, cname.str());
		build_kernel_prm<num + 1>(os, name);
	    }

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), void>::type
	    set_kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {}

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), void>::type
	    set_kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
		std::get<num>(expr).kernel_args(k, devnum, pos);
		set_kernel_args<num + 1>(k, devnum, pos);
	    }

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), void>::type
	    build_kernel_expr(std::ostream &os, std::string name) const {}

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), void>::type
	    build_kernel_expr(std::ostream &os, std::string name) const {
		std::ostringstream cname;
		cname << name << num + 1;
		std::get<num>(expr).kernel_expr(os, cname.str());
		if (num + 1 < sizeof...(Expr)) {
		    os << ", ";
		    build_kernel_expr<num + 1>(os, name);
		}
	    }

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), void>::type
	    build_preamble(std::ostream &os, std::string name) const {}

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), void>::type
	    build_preamble(std::ostream &os, std::string name) const {
		std::ostringstream cname;
		cname << name << num + 1;
		std::get<num>(expr).preamble(os, cname.str());
		build_preamble<num + 1>(os, name);
	    }

	    //------------------------------------------------------------
	    template <int num>
	    typename std::enable_if<num == sizeof...(Expr), size_t>::type
	    get_part_size(uint dev) const {
		return 0;
	    }

	    template <int num>
	    typename std::enable_if<num < sizeof...(Expr), size_t>::type
	    get_part_size(uint dev) const {
		return std::max(
			std::get<num>(expr).part_size(dev),
			get_part_size<num + 1>(dev)
			);
	    }

    };
};

/// \endcond

/// Custom user function
/**
 * Is used for introduction of custom functions into expressions. For example,
 * to count how many elements in x vector are greater than their counterparts
 * in y vector, the following code may be used:
 * \code
 * // Body of the function. Has to be extern const char[] in order to be used
 * // as template parameter.
 * extern const char one_greater_than_other[] = "return prm1 > prm2 ? 1 : 0;";
 *
 * size_t count_if_greater(const vex::vector<float> &x, const vex::vector<float> &y) {
 *     Reductor<size_t, SUM> sum(x.queue_list());
 *
 *     UserFunction<one_greater_than_other, size_t, float, float> greater;
 *
 *     return sum(greater(x, y));
 * }
 * \endcode
 * \param body Body of user function. Parameters are named prm1, ..., prm<n>.
 * \param RetType return type of the function.
 * \param ArgType types of function arguments.
 */
template<const char *body, class RetType, class... ArgType>
struct UserFunction<body, RetType(ArgType...)> {
    static const bool is_builtin = false; \
    static const bool is_userfun = true;

    /// Apply user function to the list of expressions.
    /**
     * Number of expressions in the list has to coincide with number of
     * ArgTypes
     */
    template <class... Expr>
    typename std::enable_if<
	sizeof...(ArgType) == sizeof...(Expr) &&
	All<All<valid_expr<Expr>...>, Not<All<std::is_arithmetic<Expr>...>>>::value,
    typename UserFunctionFamily<RetType, ArgType...>::template Function<body, Expr...>
    >::type
    operator()(const Expr&... expr) const {
	return typename UserFunctionFamily<RetType, ArgType...>::template Function<body, Expr...>(expr...);
    }

    template <class... Expr>
    typename std::enable_if<
	sizeof...(ArgType) == sizeof...(Expr) &&
	All<All<valid_multiex<Expr>...>, Not<All<std::is_arithmetic<Expr>...>>>::value,
	MultiExpression<
	    typename UserFunctionFamily<RetType, ArgType...>::template Function<body, typename multiex_traits<Expr>::subtype...>,
	    multiex_dim<Expr...>::dim
	>>::type
    operator()(const Expr&... expr) const {
	std::array<
	    std::unique_ptr<
		typename UserFunctionFamily<RetType, ArgType...>::template Function<body, typename multiex_traits<Expr>::subtype...>
		>,
	    multiex_dim<Expr...>::dim> ex;
	for(uint i = 0; i < multiex_dim<Expr...>::dim; i++)
	    ex[i].reset(
		    new typename UserFunctionFamily<RetType, ArgType...>::template Function<body, typename multiex_traits<Expr>::subtype...>(extract_component(expr, i)...)
		    );
	return MultiExpression<
		typename UserFunctionFamily<RetType, ArgType...>::template Function<body, typename multiex_traits<Expr>::subtype...>,
			 multiex_dim<Expr...>::dim >(ex);
    }

    template <class T>
    GConv<UserFunction,T> operator()(const GStencilProd<T> &s) const {
	return GConv<UserFunction, T>(s);
    }

    template <class T, uint N>
    MultiGConv<UserFunction,T,N> operator()(const MultiGStencilProd<T,N> &s) const {
	return MultiGConv<UserFunction, T, N>(s);
    }

    static const char* value() {
	return "user_fun";
    }
};

template <const char *body, class RetType, class... ArgType>
struct UserFunctionDeclaration<UserFunction<body, RetType(ArgType...)>> {
    static std::string get(const std::string &name = "user") {
	std::ostringstream decl;
	decl << type_name<RetType>() << " " << name << "_fun(";
	build_arg_list<ArgType...>(decl, 0);
	decl << "\n\t)\n{\n" << body << "\n}\n";
	return decl.str();
    }

    template <class T>
    static void build_arg_list(std::ostream &os, uint num) {
	os << "\n\t" << type_name<T>() << " prm" << num + 1;
    }

    template <class T, class... Args>
    static typename std::enable_if<sizeof...(Args), void>::type
    build_arg_list(std::ostream &os, uint num) {
	os << "\n\t" << type_name<T>() << " prm" << num + 1 << ",";
	build_arg_list<Args...>(os, num + 1);
    }
};
#endif

/// \cond INTERNAL

/// Returns device weight after simple bandwidth test
inline double device_vector_perf(
	const cl::Context &context, const cl::Device &device
	)
{
    static const size_t test_size = 1024U * 1024U;
    std::vector<cl::CommandQueue> queue(1,
	    cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE)
	    );

    // Allocate test vectors on current device and measure execution
    // time of a simple kernel.
    vex::vector<float> a(queue, test_size);
    vex::vector<float> b(queue, test_size);
    vex::vector<float> c(queue, test_size);

    b = 1;
    c = 2;

    // Skip the first run.
    a = b + c;

    // Measure the second run.
    profiler prof(queue);
    prof.tic_cl("");
    a = b + c;
    return 1.0 / prof.toc("");
}

/// \endcond

} // namespace vex

#ifdef WIN32
#  pragma warning(pop)
#endif
#endif
