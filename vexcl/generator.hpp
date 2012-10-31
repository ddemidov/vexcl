#ifndef VEXCL_GENERATOR_HPP
#define VEXCL_GENERATOR_HPP

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
 * \file   generator.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL kernel generator.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <boost/proto/proto.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>

/// Vector expression template library for OpenCL.
namespace vex {

/// Kernel generation interface.
namespace generator {

//---------------------------------------------------------------------------
// The recorder class. Holds static output stream for kernel recording and
// static variable index (used in variable names).
//---------------------------------------------------------------------------

/// \cond INTERNAL
template <bool dummy = true>
class recorder {
    public:
	static void set(std::ostream &s) {
	    os = &s;
	}

	static std::ostream& get() {
	    return os ? *os : std::cout;
	}

	static size_t var_id() {
	    return ++index;
	}
    private:
	static size_t index;
	static std::ostream *os;
};

template <bool dummy>
size_t recorder<dummy>::index = 0;

template <bool dummy>
std::ostream *recorder<dummy>::os = 0;

inline size_t var_id() {
    return recorder<>::var_id();
}

inline std::ostream& get_recorder() {
    return recorder<>::get();
}


/// Set output stream for kernel recorder.
inline void set_recorder(std::ostream &os) {
    recorder<>::set(os);
}

//---------------------------------------------------------------------------
// Setting up boost::proto.
//---------------------------------------------------------------------------

namespace proto = boost::proto;
using proto::_;

struct variable {};

// --- The grammar ----------------------------------------------------------

struct symbolic_grammar
    : proto::or_<
	  proto::or_<
	      proto::terminal< variable >,
	      proto::and_<
	          proto::terminal< _ >,
		  proto::if_< boost::is_arithmetic< proto::_value >() >
	      >
          >,
	  boost::proto::or_<
	      boost::proto::unary_plus< symbolic_grammar >,
	      boost::proto::negate< symbolic_grammar >,
	      boost::proto::logical_not< symbolic_grammar >,
	      boost::proto::pre_inc< symbolic_grammar >,
	      boost::proto::pre_dec< symbolic_grammar >,
	      boost::proto::post_inc< symbolic_grammar >,
	      boost::proto::post_dec< symbolic_grammar >
	  >,
	  boost::proto::or_<
	      boost::proto::or_<
	          boost::proto::plus          < symbolic_grammar, symbolic_grammar >,
	          boost::proto::minus         < symbolic_grammar, symbolic_grammar >,
	          boost::proto::multiplies    < symbolic_grammar, symbolic_grammar >,
	          boost::proto::divides       < symbolic_grammar, symbolic_grammar >,
	          boost::proto::modulus       < symbolic_grammar, symbolic_grammar >,
	          boost::proto::shift_left    < symbolic_grammar, symbolic_grammar >,
	          boost::proto::shift_right   < symbolic_grammar, symbolic_grammar >
	      >,
	      boost::proto::or_<
	          boost::proto::less          < symbolic_grammar, symbolic_grammar >,
	          boost::proto::greater       < symbolic_grammar, symbolic_grammar >,
	          boost::proto::less_equal    < symbolic_grammar, symbolic_grammar >,
	          boost::proto::greater_equal < symbolic_grammar, symbolic_grammar >,
	          boost::proto::equal_to      < symbolic_grammar, symbolic_grammar >,
	          boost::proto::not_equal_to  < symbolic_grammar, symbolic_grammar >
	      >,
	      boost::proto::or_<
	          boost::proto::logical_and   < symbolic_grammar, symbolic_grammar >,
	          boost::proto::logical_or    < symbolic_grammar, symbolic_grammar >
	      >,
	      boost::proto::or_<
	          boost::proto::bitwise_and   < symbolic_grammar, symbolic_grammar >,
	          boost::proto::bitwise_or    < symbolic_grammar, symbolic_grammar >,
	          boost::proto::bitwise_xor   < symbolic_grammar, symbolic_grammar >
	      >
	  >,
	  boost::proto::function<
	      boost::proto::terminal<
	          boost::proto::convertible_to<builtin_function>
	      >,
	      boost::proto::vararg<symbolic_grammar>
          >
      >
{};

template <class Expr>
struct symbolic_expr;

struct symbolic_domain
    : proto::domain< proto::generator< symbolic_expr >, symbolic_grammar >
{};

template <class Expr>
struct symbolic_expr
    : proto::extends< Expr, symbolic_expr< Expr >, symbolic_domain >
{
    typedef proto::extends< Expr, symbolic_expr< Expr >, symbolic_domain > base_type;

    symbolic_expr(const Expr &expr = Expr()) : base_type(expr) {}
};

template <typename T>
struct symbolic;

template <typename T>
std::ostream& operator<<(std::ostream &os, const symbolic<T> &sym);

struct symbolic_context {
    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {};

#define BINARY_OPERATION(bin_tag, bin_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::bin_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, symbolic_context &ctx) const { \
	    get_recorder() << "( "; \
	    proto::eval(proto::left(expr), ctx); \
	    get_recorder() << " " #bin_op " "; \
	    proto::eval(proto::right(expr), ctx); \
	    get_recorder() << " )"; \
	} \
    }

    BINARY_OPERATION(plus,          +);
    BINARY_OPERATION(minus,         -);
    BINARY_OPERATION(multiplies,    *);
    BINARY_OPERATION(divides,       /);
    BINARY_OPERATION(modulus,       %);
    BINARY_OPERATION(shift_left,   <<);
    BINARY_OPERATION(shift_right,  >>);
    BINARY_OPERATION(less,          <);
    BINARY_OPERATION(greater,       >);
    BINARY_OPERATION(less_equal,   <=);
    BINARY_OPERATION(greater_equal,>=);
    BINARY_OPERATION(equal_to,     ==);
    BINARY_OPERATION(not_equal_to, !=);
    BINARY_OPERATION(logical_and,  &&);
    BINARY_OPERATION(logical_or,   ||);
    BINARY_OPERATION(bitwise_and,   &);
    BINARY_OPERATION(bitwise_or,    |);
    BINARY_OPERATION(bitwise_xor,   ^);

#undef BINARY_OPERATION

#define UNARY_PRE_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, symbolic_context &ctx) const { \
	    get_recorder() << "( " #the_op "( "; \
	    proto::eval(proto::child(expr), ctx); \
	    get_recorder() << " ) )"; \
	} \
    }

    UNARY_PRE_OPERATION(unary_plus,   +);
    UNARY_PRE_OPERATION(negate,       -);
    UNARY_PRE_OPERATION(logical_not,  !);
    UNARY_PRE_OPERATION(pre_inc,     ++);
    UNARY_PRE_OPERATION(pre_dec,     --);

#undef UNARY_PRE_OPERATION

#define UNARY_POST_OPERATION(the_tag, the_op) \
    template <typename Expr> \
    struct eval<Expr, proto::tag::the_tag> { \
	typedef void result_type; \
	void operator()(const Expr &expr, symbolic_context &ctx) const { \
	    get_recorder() << "( ( "; \
	    proto::eval(proto::child(expr), ctx); \
	    get_recorder() << " )" #the_op " )"; \
	} \
    }

    UNARY_POST_OPERATION(post_inc, ++);
    UNARY_POST_OPERATION(post_dec, --);

#undef UNARY_POST_OPERATION

    template <class Expr>
    struct eval<Expr, proto::tag::function> {
	typedef void result_type;

	struct display {
	    mutable int pos;
	    symbolic_context &ctx;

	    display(symbolic_context &ctx) : pos(0), ctx(ctx) {}

	    template <class Arg>
	    void operator()(const Arg &arg) const {
		if (pos++) get_recorder() << ", ";
		proto::eval(arg, ctx);
	    }
	};

	void operator()(const Expr &expr, symbolic_context &ctx) const {
	    get_recorder() << proto::value(proto::child_c<0>(expr)).name() << "( ";

	    boost::fusion::for_each(
		    boost::fusion::pop_front(expr),
		    display(ctx)
		    );

	    get_recorder() << " )";
	}
    };

    template <typename Expr>
    struct eval<Expr, proto::tag::terminal> {
	typedef void result_type;

	template <typename Term>
	void operator()(const Term &term, symbolic_context &ctx) const {
	    get_recorder() << std::scientific << std::setprecision(12)
		<< proto::value(term);
	}

	template <typename T>
	void operator()(const symbolic<T> &v, symbolic_context &ctx) const {
	    get_recorder() << v;
	}
    };
};


//---------------------------------------------------------------------------
// Builtin functions.
//---------------------------------------------------------------------------

template <class Expr>
void record(const Expr &expr) {
    symbolic_context ctx;
    proto::eval(proto::as_expr(expr), ctx);
}
/// \endcond

//---------------------------------------------------------------------------
// The symbolic class.
//---------------------------------------------------------------------------

template <typename T>
class symbolic
    : public symbolic_expr< proto::terminal< variable >::type >
{
    public:
	/// Scope/Type of the symbolic variable.
	enum scope_type {
	    LocalVar        = 0, ///< Local variable.
	    VectorParameter = 1, ///< Vector kernel parameter.
	    ScalarParameter = 2, ///< Scalar kernel parameter.
	};

	/// Constness of vector parameter.
	enum constness_type {
	    NonConst = 0,   ///< Parameter should be written back at kernel exit.
	    Const = 1	    ///< Parameter is readonly.
	};

	/// Default constructor. Results in local kernel variable.
	symbolic(scope_type scope = LocalVar, constness_type constness = NonConst)
	    : num(var_id()), scope(scope), constness(constness)
	{
	    if (scope == LocalVar) {
		get_recorder() << type_name<T>() << " " << *this << ";\n";
	    }
	}

	/// Expression constructor. Results in local variable initialized by expression.
	template <class Expr>
	explicit symbolic(const Expr &expr)
	    : num(var_id()), scope(LocalVar), constness(NonConst)
	{
	    get_recorder() << type_name<T>() << " " << *this << " = ";
	    record(expr);
	    get_recorder() << ";\n";
	}

	/// Assignment operator. Results in assignment written to recorder.
	template <class Expr>
	const symbolic& operator=(const Expr &expr) const {
	    get_recorder() << *this << " = ";
	    record(expr);
	    get_recorder() << ";\n";
	    return *this;
	}

#define COMPOUND_ASSIGNMENT(cop, op) \
	template <class Expr> \
	const symbolic& operator cop(const Expr &expr) { \
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

	size_t id() const {
	    return num;
	}

	/// Initialize local variable at kernel enter.
	std::string init() const {
	    std::ostringstream s;

	    if (scope != LocalVar) {
		s << type_name<T>() << " " << *this << " = p_" << *this;

		switch (scope) {
		    case VectorParameter:
			s << "[idx];\n";
			break;
		    case ScalarParameter:
			s << ";\n";
			break;
		    default:
			break;
		}
	    }

	    return s.str();
	}

	/// Write local variable to parameter at kernel exit.
	std::string write() const {
	    std::ostringstream s;

	    if (scope == VectorParameter && constness == NonConst)
		s << "p_" << *this << "[idx] = " << *this << ";\n";

	    return s.str();
	}

	/// Returns string for parameter declaration.
	std::string prmdecl() const {
	    std::ostringstream s;

	    if (scope == VectorParameter)
		s << "global ";

	    if (constness == Const)
		s << "const ";

	    s << type_name<T>();

	    if (scope == VectorParameter)
		s << "*";

	    s << " p_" << *this;

	    return s.str();
	}
    private:
	size_t         num;
	scope_type     scope;
	constness_type constness;
};

template <typename T>
std::ostream& operator<<(std::ostream &os, const symbolic<T> &sym) {
    return os << "var" << sym.id();
}

/// Autogenerated kernel.
template <class... Args>
class Kernel {
    public:
	/// Launches kernel with provided parameters.
	template <class... Param>
	void operator()(const Param&... param) {
	    static_assert(
		    sizeof...(Param) == sizeof...(Args),
		    "Wrong number of kernel parameters"
		    );

	    for(uint d = 0; d < queue.size(); d++) {
		if (size_t psize = prm_size(d, param...)) {
		    cl::Context context = qctx(queue[d]);
		    cl::Device  device  = qdev(queue[d]);

		    uint pos = 0;
		    krn[context()].setArg(pos++, psize);

		    set_params setprm(krn[context()], d, pos);
		    for_each(std::tie(param...), setprm);

		    size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
			alignup(psize, wgs[context()]) :
			device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * wgs[context()] * 4;

		    queue[d].enqueueNDRangeKernel(krn[context()],
			    cl::NullRange, g_size, wgs[context()]
			    );
		}
	    }
	}

    private:
	template <class... SymPrm> friend
	Kernel<SymPrm...> build_kernel(
		const std::vector<cl::CommandQueue> &queue,
		const std::string &name, const std::string& body, const SymPrm&... args
		);

	Kernel(
		const std::vector<cl::CommandQueue> &queue,
		const std::string &name, const std::string &body,
		const Args&... args
	      ) : queue(queue)
	{
	    std::ostringstream source;

	    source
		<< standard_kernel_header
		<< "kernel void " << name << "(\n"
		<< "\t" << type_name<size_t>() << " n";

	    declare_params declprm(source);
	    for_each(std::tie(args...), declprm);

	    source <<
		"\n)\n{\n\t"
		"for(size_t idx = get_global_id(0); idx < n; "
		"idx += get_global_size(0)) {\n";

	    read_params readprm(source);
	    for_each(std::tie(args...), readprm);

	    source << body;
	    
	    write_params writeprm(source);
	    for_each(std::tie(args...), writeprm);

	    source << "\t}\n}\n";

#ifdef VEXCL_SHOW_KERNELS
	    std::cout << source.str() << std::endl;
#endif

	    for(auto q = queue.begin(); q != queue.end(); q++) {
		cl::Context context = qctx(*q);
		cl::Device  device  = qdev(*q);

		auto program = build_sources(context, source.str());

		krn[context()] = cl::Kernel(program, name.c_str());
		wgs[context()] = kernel_workgroup_size(krn[context()], device);
	    }
	}

	struct declare_params {
	    std::ostream &os;

	    declare_params(std::ostream &os) : os(os) {}

	    template <class T>
	    void operator()(const T &v) const {
		os << ",\n\t" << v.prmdecl();
	    }
	};

	struct read_params {
	    std::ostream &os;

	    read_params(std::ostream &os) : os(os) {}

	    template <class T>
	    void operator()(const T &v) const {
		os << v.init();
	    }
	};

	struct write_params {
	    std::ostream &os;

	    write_params(std::ostream &os) : os(os) {}

	    template <class T>
	    void operator()(const T &v) const {
		os << v.write();
	    }
	};

	struct set_params {
	    cl::Kernel &krn;
	    uint d, &pos;

	    set_params(cl::Kernel &krn, uint d, uint &pos)
		: krn(krn), d(d), pos(pos) {};

	    template <class T>
	    void operator()(const T &v) const {
		krn.setArg(pos++, v);
	    }
	    template <class T>
	    void operator()(const vector<T> &v) const {
		krn.setArg(pos++, v(d));
	    }
	};

	std::vector<cl::CommandQueue> queue;

	std::map<cl_context, cl::Kernel> krn;
	std::map<cl_context, uint> wgs;

	size_t prm_size(uint d) const {
	    throw std::logic_error(
		    "Kernel has to have at least one vector parameter"
		    );
	}

	template <class Head, class... Tail>
	size_t prm_size(uint d, const vector<Head> &head, const Tail&... tail) const {
	    return head.part_size(d);
	}

	template <class Head, class... Tail>
	size_t prm_size(uint d, const Head &head, const Tail&... tail) const {
	    return prm_size(d, tail...);
	}
};

/// Builds kernel from recorded expression sequence and symbolic parameter list.
template <class... Args>
Kernel<Args...> build_kernel(
	const std::vector<cl::CommandQueue> &queue,
	const std::string &name, const std::string& body, const Args&... args
	)
{
    return Kernel<Args...>(queue, name, body, args...);
}

} // namespace generator;

} // namespace vex;

#endif
