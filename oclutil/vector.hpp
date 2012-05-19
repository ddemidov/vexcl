#ifndef OCLUTIL_VECTOR_HPP
#define OCLUTIL_VECTOR_HPP

/**
 * \file   vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device vector.
 */

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <CL/cl.hpp>
#include <oclutil/util.hpp>
#include <cassert>

/// OpenCL convenience utilities.
namespace clu {

template <class T> class vector;
template <class T> void copy(const clu::vector<T> &dv, T *hv);
template <class T> void copy(const T *hv, clu::vector<T> &dv);
template <class T> void copy(const clu::vector<T> &dv, std::vector<T> &hv);
template <class T> void copy(const std::vector<T> &hv, clu::vector<T> &dv);
template <class T> T sum(const clu::vector<T> &x);
template <class T> T inner_product(const clu::vector<T> &x, const clu::vector<T> &y);

template<class T> class SpMV;

/// Convenience class for work with cl::Buffer.
template<class T>
class vector {
    public:
	static constexpr bool is_expression = true;
	static bool show_kernels;

	/// \internal Proxy class.
	class element {
	    public:
		operator T() {
		    T val;
		    queue.enqueueReadBuffer(
			    buf, CL_TRUE,
			    index * sizeof(T), sizeof(T),
			    &val
			    );
		    return val;
		}

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

		cl::CommandQueue  queue;
		cl::Buffer        buf;
		uint              index;

		friend class vector;
		friend class vector::iterator;
	};

	/// \internal Proxy class.
	class const_element {
	    public:
		operator T() {
		    T val;
		    queue.enqueueReadBuffer(
			    buf, CL_TRUE,
			    index * sizeof(T), sizeof(T),
			    &val
			    );
		    return val;
		}
	    private:
		const_element(cl::CommandQueue q, cl::Buffer b, uint i)
		    : queue(q), buf(b), index(i) {}

		cl::CommandQueue  queue;
		cl::Buffer        buf;
		uint              index;

		friend class vector;
		friend class vector::iterator;
	};

	/// \internal Iterator class.
	class iterator {
	    public:
		element operator*() {
		    return element(
			    vec.queue[part], vec.buf[part],
			    pos - vec.part[part]
			    );
		}

		iterator& operator++() {
		    pos++;
		    while (pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		    return *this;
		}

		iterator operator+(ptrdiff_t d) const {
		    return iterator(vec, pos + d);
		}

		bool operator!=(const iterator &it) const {
		    return pos != it.pos;
		}

	    private:
		iterator(vector &vec, size_t pos)
		    : vec(vec), pos(pos), part(0)
		{
		    while(pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		}

		vector &vec;
		size_t  pos;
		size_t  part;

		friend class vector;

	};

	/// \internal Iterator class.
	class const_iterator {
	    public:
		const_element operator*() {
		    return const_element(
			    vec.queue[part], vec.buf[part],
			    pos - vec.part[part]
			    );
		}

		const_iterator& operator++() {
		    pos++;
		    while (pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		    return *this;
		}

		const_iterator operator+(ptrdiff_t d) const {
		    return const_iterator(vec, pos + d);
		}

		bool operator!=(const const_iterator &it) const {
		    return pos != it.pos;
		}

	    private:
		const_iterator(const vector &vec, size_t pos)
		    : vec(vec), pos(pos), part(0)
		{
		    while(pos >= vec.part[part + 1] && part < vec.nparts())
			part++;
		}

		const vector &vec;
		size_t  pos;
		size_t  part;

		friend class vector;

	};

	/// Empty constructor.
	vector() {}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue, cl_mem_flags flags,
		size_t size, const T *host = 0
	      ) : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()),
	          queue(queue), part(partition(size, queue.size())),
	          buf(queue.size()), bytes(queue.size(), 0),
		  event(queue.size())
	{
	    if (size) allocate_buffers(flags, host);
	}

	/// Copy host data to the new buffer.
	vector(const std::vector<cl::CommandQueue> &queue, cl_mem_flags flags,
		const std::vector<T> &host
	      ) : context(queue[0].getInfo<CL_QUEUE_CONTEXT>()),
	          queue(queue), part(partition(host.size(), queue.size())),
	          buf(queue.size()), bytes(queue.size(), 0),
		  event(queue.size())
	{
	    if (!host.empty()) allocate_buffers(flags, host.data());
	}

	cl::Buffer operator()(uint p = 0) const {
	    return buf[p];
	}

	const_iterator begin() const {
	    return const_iterator(*this, 0);
	}

	const_iterator end() const {
	    return const_iterator(*this, size());
	}

	iterator begin() {
	    return iterator(*this, 0);
	}

	iterator end() {
	    return iterator(*this, size());
	}

	element operator[](size_t index) {
	    uint p = 0;
	    while(index >= part[p + 1] && p < nparts()) p++;
	    return element(queue[p], buf[p], index - part[p]);
	}

	size_t size() const {
	    return part.back();
	}

	size_t nparts() const {
	    return queue.size();
	}

	size_t part_size(uint p) const {
	    return part[p + 1] - part[p];
	}

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

	void prologue(std::ostream &os, std::string name = "v") const {
	}

	template <class Expr>
	    void operator=(const Expr &expr) {
		if (!exdata<Expr>::compiled) {
		    std::ostringstream kernel;

		    std::string kernel_name = expr.kernel_name();

		    kernel <<
			"#if defined(cl_khr_fp64)\n"
			"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
			"#elif defined(cl_amd_fp64)\n"
			"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
			"#endif\n";

		    expr.prologue(kernel);

		    kernel
			<< "kernel void " << kernel_name
			<< "(\n\tunsigned int n,\n\tglobal "
			<< type_name<T>() << " *res";

		    expr.kernel_prm(kernel);

		    kernel << "\n\t)\n{\n\tunsigned int i = get_global_id(0);\n"
			"\tif (i < n) res[i] = ";

		    expr.kernel_expr(kernel);

		    kernel << ";\n}" << std::endl;

		    if (show_kernels)
			std::cout << kernel.str() << std::endl;

		    std::vector<cl::Device> device;
		    for(auto q = queue.begin(); q != queue.end(); q++)
			device.push_back(q->getInfo<CL_QUEUE_DEVICE>());

		    auto program = build_sources(context, kernel.str());

		    exdata<Expr>::kernel = cl::Kernel(program, kernel_name.c_str());
		    exdata<Expr>::compiled = true;
		}

		for(uint d = 0; d < queue.size(); d++) {
		    uint pos = 0, psize = part[d + 1] - part[d];
		    exdata<Expr>::kernel.setArg(pos++, psize);
		    exdata<Expr>::kernel.setArg(pos++, buf[d]);

		    expr.kernel_args(exdata<Expr>::kernel, d, pos);

		    queue[d].enqueueNDRangeKernel(exdata<Expr>::kernel, cl::NullRange,
			    alignup(psize, 256U), 256U
			    );
		}
	    }

	void operator=(const SpMV<T> &spmv);

    private:
	template <class Expr>
	struct exdata {
	    static bool       compiled;
	    static cl::Kernel kernel;
	};

	cl::Context                     context;
	std::vector<cl::CommandQueue>	queue;
	std::vector<size_t>             part;
	std::vector<cl::Buffer>		buf;
	std::vector<size_t>		bytes;

	mutable std::vector<cl::Event>          event;

	void allocate_buffers(cl_mem_flags flags, const T *hostptr) {
	    for(uint i = 0; i < nparts(); i++) {
		bytes[i] = (part[i + 1] - part[i]) * sizeof(T);
		buf[i] = cl::Buffer(context, flags, bytes[i]);
	    }

	    if (hostptr) write_data(hostptr);
	}

	void write_data(const T *hostptr) {
	    for(uint i = 0; i < queue.size(); i++) {
		queue[i].enqueueWriteBuffer(
			buf[i], CL_FALSE, 0, bytes[i], hostptr + part[i],
			0, &event[i]
			);
	    }

	    cl::Event::waitForEvents(event);
	}

	void read_data(T *hostptr) const {
	    for(uint i = 0; i < queue.size(); i++) {
		queue[i].enqueueReadBuffer(
			buf[i], CL_FALSE, 0, bytes[i], hostptr + part[i],
			0, &event[i]
			);
	    }

	    cl::Event::waitForEvents(event);
	}

	friend void copy<>(const clu::vector<T> &dv, T *hv);
	friend void copy<>(const T *hv, clu::vector<T> &dv);
	friend void copy<>(const clu::vector<T> &dv, std::vector<T> &hv);
	friend void copy<>(const std::vector<T> &hv, clu::vector<T> &dv);

	friend T sum<>(const clu::vector<T> &x);
	friend T inner_product<>(const clu::vector<T> &x, const clu::vector<T> &y);
};

template <class T> bool vector<T>::show_kernels = false;

template <class T> template <class Expr>
bool vector<T>::exdata<Expr>::compiled = false;

template <class T> template <class Expr>
cl::Kernel vector<T>::exdata<Expr>::kernel;

/// Copy device vector to host vector.
template <class T>
void copy(const clu::vector<T> &dv, T *hv) {
    dv.read_data(hv);
}

/// Copy host vector to device vector.
template <class T>
void copy(const T *hv, clu::vector<T> &dv) {
    dv.write_data(hv);
}

/// Copy device vector to host vector.
template <class T>
void copy(const clu::vector<T> &dv, std::vector<T> &hv) {
    dv.read_data(hv.data());
}

/// Copy host vector to device vector.
template <class T>
void copy(const std::vector<T> &hv, clu::vector<T> &dv) {
    dv.write_data(hv.data());
}

/// \internal Expression template.
template <class LHS, char OP, class RHS>
struct BinaryExpression {
    static constexpr bool is_expression = true;

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

    void prologue(std::ostream &os, std::string name = "") const {
	lhs.prologue(os, name + "l");
	rhs.prologue(os, name + "r");
    }

    size_t part_size(uint dev) const {
	return std::max(lhs.part_size(dev), rhs.part_size(dev));
    }

    const LHS &lhs;
    const RHS &rhs;
};

/// \internal Sum of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '+', RHS>
    >::type
    operator+(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'+',RHS>(lhs, rhs);
    }

/// \internal Difference of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '-', RHS>
    >::type
    operator-(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'-',RHS>(lhs, rhs);
    }

/// \internal Product of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '*', RHS>
    >::type
    operator*(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'*',RHS>(lhs, rhs);
    }

/// \internal Division of two expressions.
template <class LHS, class RHS>
typename std::enable_if<
    LHS::is_expression && RHS::is_expression,
    BinaryExpression<LHS, '/', RHS>
    >::type
    operator/(const LHS &lhs, const RHS &rhs) {
	return BinaryExpression<LHS,'/',RHS>(lhs, rhs);
    }

/// \internal Constant for use in vector expressions.
template <class T>
struct Constant {
    static constexpr bool is_expression = true;

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

    void prologue(std::ostream &os, std::string name = "c") const {
    }

    size_t part_size(uint dev) const {
	return 1;
    }

    private:
	T value;
};

/// Constant for use in vector expressions.
template <class T>
Constant<T> Const(T value) { return Constant<T>(value); }

enum UnaryFunction {
    SQRT,
    SIN,
    COS
};

/// \internal Unary expression template.
template <UnaryFunction F, class Expr>
struct UnaryExpression {
    static constexpr bool is_expression = true;

    UnaryExpression(const Expr &expr) : expr(expr) {}

    std::string kernel_name() const {
	return funstr() + expr.kernel_name();
    }

    void kernel_expr(std::ostream &os, std::string name = "") const {
	os << funstr() << "(";
	expr.kernel_expr(os, name);
	os << ")";
    }

    void kernel_prm(std::ostream &os, std::string name = "") const {
	expr.kernel_prm(os, name);
    }

    void kernel_args(cl::Kernel &k, uint devnum, uint &pos) const {
	expr.kernel_args(k, devnum, pos);
    }

    void prologue(std::ostream &os, std::string name = "") const {
	expr.prologue(os, name);
    }

    size_t part_size(uint dev) const {
	return expr.part_size(dev);
    }

    private:
	const Expr &expr;

	static std::string funstr() {
	    switch (F) {
		case SQRT:
		    return "sqrt";
		case SIN:
		    return "sin";
		case COS:
		    return "cos";
	    }
	}
};

/// Square root of argument.
template <class Expr>
typename std::enable_if<Expr::is_expression, UnaryExpression<SQRT, Expr>>::type
sqrt(const Expr &e) { return UnaryExpression<SQRT,Expr>(e); }

/// Sine of argument.
template <class Expr>
typename std::enable_if<Expr::is_expression, UnaryExpression<SIN, Expr>>::type
sin(const Expr &e) { return UnaryExpression<SIN,Expr>(e); }

/// Cosine of argument.
template <class Expr>
typename std::enable_if<Expr::is_expression, UnaryExpression<COS, Expr>>::type
cos(const Expr &e) { return UnaryExpression<COS,Expr>(e); }

} // namespace clu

#endif
