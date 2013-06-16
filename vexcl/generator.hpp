#ifndef VEXCL_GENERATOR_HPP
#define VEXCL_GENERATOR_HPP

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
#include <boost/tuple/tuple.hpp>
#include <boost/fusion/adapted/boost_tuple.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/function_arity.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <boost/preprocessor/repetition.hpp>
#ifndef VEXCL_MAX_ARITY
#  define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#endif

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
    static_assert(dummy, "dummy parameter should be true");

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

struct variable {};

// --- The grammar ----------------------------------------------------------

struct symbolic_grammar
    : boost::proto::or_<
          boost::proto::or_<
              boost::proto::terminal< variable >,
              boost::proto::and_<
                  boost::proto::terminal< boost::proto::_ >,
                  boost::proto::if_< is_cl_native< boost::proto::_value >() >
              >
          >,
          BUILTIN_OPERATIONS(symbolic_grammar)
      >
{};

template <class Expr>
struct symbolic_expr;

struct symbolic_domain
    : boost::proto::domain< boost::proto::generator< symbolic_expr >, symbolic_grammar >
{
    template <typename T>
    struct as_child : proto_base_domain::as_expr<T> {};
};

template <class Expr>
struct symbolic_expr
    : boost::proto::extends< Expr, symbolic_expr< Expr >, symbolic_domain >
{
    typedef boost::proto::extends< Expr, symbolic_expr< Expr >, symbolic_domain > base_type;

    symbolic_expr(const Expr &expr = Expr()) : base_type(expr) {}
};

template <typename T>
class symbolic;

template <typename T>
std::ostream& operator<<(std::ostream &os, const symbolic<T> &sym);

struct symbolic_context {
    template <typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval {};

#define BINARY_OPERATION(bin_tag, bin_op) \
    template <typename Expr> \
    struct eval<Expr, boost::proto::tag::bin_tag> { \
        typedef void result_type; \
        void operator()(const Expr &expr, symbolic_context &ctx) const { \
            get_recorder() << "( "; \
            boost::proto::eval(boost::proto::left(expr), ctx); \
            get_recorder() << " " #bin_op " "; \
            boost::proto::eval(boost::proto::right(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
        typedef void result_type; \
        void operator()(const Expr &expr, symbolic_context &ctx) const { \
            get_recorder() << "( " #the_op "( "; \
            boost::proto::eval(boost::proto::child(expr), ctx); \
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
    struct eval<Expr, boost::proto::tag::the_tag> { \
        typedef void result_type; \
        void operator()(const Expr &expr, symbolic_context &ctx) const { \
            get_recorder() << "( ( "; \
            boost::proto::eval(boost::proto::child(expr), ctx); \
            get_recorder() << " )" #the_op " )"; \
        } \
    }

    UNARY_POST_OPERATION(post_inc, ++);
    UNARY_POST_OPERATION(post_dec, --);

#undef UNARY_POST_OPERATION

    template <class Expr>
    struct eval<Expr, boost::proto::tag::function> {
        typedef void result_type;

        struct display {
            mutable int pos;
            symbolic_context &ctx;

            display(symbolic_context &ctx) : pos(0), ctx(ctx) {}

            template <class Arg>
            void operator()(const Arg &arg) const {
                if (pos++) get_recorder() << ", ";
                boost::proto::eval(arg, ctx);
            }
        };

        void operator()(const Expr &expr, symbolic_context &ctx) const {
            get_recorder() << boost::proto::value(boost::proto::child_c<0>(expr)).name() << "( ";

            boost::fusion::for_each(
                    boost::fusion::pop_front(expr),
                    display(ctx)
                    );

            get_recorder() << " )";
        }
    };

    template <typename Expr>
    struct eval<Expr, boost::proto::tag::terminal> {
        typedef void result_type;

        template <typename Term>
        void operator()(const Term &term, symbolic_context &) const {
            get_recorder() << std::scientific << std::setprecision(12)
                << boost::proto::value(term);
        }

        template <typename T>
        void operator()(const symbolic<T> &v, symbolic_context &) const {
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
    boost::proto::eval(boost::proto::as_expr(expr), ctx);
}
/// \endcond

//---------------------------------------------------------------------------
// The symbolic class.
//---------------------------------------------------------------------------

template <typename T>
class symbolic
    : public symbolic_expr< boost::proto::terminal< variable >::type >
{
    public:
        typedef T value_type;

        /// Scope/Type of the symbolic variable.
        enum scope_type {
            LocalVar        = 0, ///< Local variable.
            VectorParameter = 1, ///< Vector kernel parameter.
            ScalarParameter = 2  ///< Scalar kernel parameter.
        };

        /// Constness of vector parameter.
        enum constness_type {
            NonConst = 0,   ///< Parameter should be written back at kernel exit.
            Const = 1       ///< Parameter is readonly.
        };

        /// Default constructor. Results in local kernel variable.
        symbolic() : num(var_id()), scope(LocalVar), constness(NonConst)
        {
            get_recorder() << "\t\t" << type_name<T>() << " " << *this << ";\n";
        }

        /// Constructor.
        explicit symbolic(scope_type scope, constness_type constness = NonConst)
            : num(var_id()), scope(scope), constness(constness)
        {
            if (scope == LocalVar) {
                get_recorder() << "\t\t" << type_name<T>() << " " << *this << ";\n";
            }
        }

        /// Expression constructor. Results in local variable initialized by expression.
        template <class Expr>
        symbolic(const Expr &expr)
            : num(var_id()), scope(LocalVar), constness(NonConst)
        {
            get_recorder() << "\t\t" << type_name<T>() << " " << *this << " = ";
            record(expr);
            get_recorder() << ";\n";
        }

        /// Assignment operator. Results in assignment written to recorder.
        const symbolic& operator=(const symbolic &c) const {
            get_recorder() << "\t\t" << *this << " = ";
            record(c);
            get_recorder() << ";\n";
            return *this;
        }

        /// Assignment operator. Results in assignment written to recorder.
        template <class Expr>
        const symbolic& operator=(const Expr &expr) const {
            get_recorder() << "\t\t" << *this << " = ";
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
                s << "\t\t" << type_name<T>() << " " << *this << " = p_" << *this;

                switch (scope) {
                    case VectorParameter:
                        s << "[idx];\n";
                        break;
                    case ScalarParameter:
                        s << ";\n";
                        break;
                    case LocalVar:
                        break;
                }
            }

            return s.str();
        }

        /// Write local variable to parameter at kernel exit.
        std::string write() const {
            std::ostringstream s;

            if (scope == VectorParameter && constness == NonConst)
                s << "\t\tp_" << *this << "[idx] = " << *this << ";\n";

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
template <size_t NP>
class Kernel {
    public:
        template <class ArgTuple>
        Kernel(
                const std::vector<cl::CommandQueue> &queue,
                const std::string &name, const std::string &body,
                const ArgTuple& args
              ) : queue(queue)
        {
            static_assert(
                    boost::tuples::length<ArgTuple>::value == NP,
                    "Wrong number of kernel parameters"
                    );

            for(auto q = queue.begin(); q != queue.end(); q++) {
                cl::Context context = qctx(*q);
                cl::Device  device  = qdev(*q);

                std::ostringstream source;

                source
                    << standard_kernel_header(device)
                    << "kernel void " << name << "(\n"
                    << "\t" << type_name<size_t>() << " n";

                boost::fusion::for_each(args, declare_params(source));

                source << "\n)\n{\n";

                if ( is_cpu(device) ) {
                    source <<
                        "\tsize_t chunk_size  = (n + get_global_size(0) - 1) / get_global_size(0);\n"
                        "\tsize_t chunk_start = get_global_id(0) * chunk_size;\n"
                        "\tsize_t chunk_end   = min(n, chunk_start + chunk_size);\n"
                        "\tfor(size_t idx = chunk_start; idx < chunk_end; ++idx) {\n";
                } else {
                    source <<
                        "\tfor(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n";
                }

                boost::fusion::for_each(args, read_params(source));

                source << body;

                boost::fusion::for_each(args, write_params(source));

                source << "\t}\n}\n";

                auto program = build_sources(context, source.str());

                krn[context()] = cl::Kernel(program, name.c_str());
                wgs[context()] = kernel_workgroup_size(krn[context()], device);
            }
        }

#ifndef BOOST_NO_VARIADIC_TEMPLATES
        /// Launches kernel with provided parameters.
        template <class... Param>
        void operator()(const Param&... param) {
            launch(boost::tie(param...));
        }
#else

#define PRINT_PARAM(z, n, data) const Param ## n &param ## n
#define FUNCALL_OPERATOR(z, n, data) \
        template < BOOST_PP_ENUM_PARAMS(n, class Param) > \
        void operator()( BOOST_PP_ENUM(n, PRINT_PARAM, ~) ) { \
            launch(boost::tie( BOOST_PP_ENUM_PARAMS(n, param) )); \
        }

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, FUNCALL_OPERATOR, ~)

#undef PRINT_PARAM
#undef FUNCALL_OPERATOR

#endif
    private:

        template <class ParamTuple>
        void launch(const ParamTuple &param) {
            static_assert(
                    boost::tuples::length<ParamTuple>::value == NP,
                    "Wrong number of kernel parameters"
                    );

            for(uint d = 0; d < queue.size(); d++) {
                if (size_t psize = boost::fusion::fold(param, 0, param_size(d))) {
                    cl::Context context = qctx(queue[d]);
                    cl::Device  device  = qdev(queue[d]);

                    uint pos = 0;
                    krn[context()].setArg(pos++, psize);

                    set_params setprm(krn[context()], d, pos);
                    boost::fusion::for_each(param, setprm);

                    size_t g_size = num_workgroups(device) * wgs[context()];

                    queue[d].enqueueNDRangeKernel(krn[context()],
                            cl::NullRange, g_size, wgs[context()]);
                }
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

        struct param_size {
            uint device;

            param_size(uint device) : device(device) {}

            typedef size_t result_type;

            template <class T>
            size_t operator()(size_t s, const T &v) const {
                return std::max(s, v.part_size(device));
            }
        };
};

/// Function body generator.
class Function {
    public:
        template <class Ret, class ArgTuple>
        Function(const std::string &body, const Ret &ret, const ArgTuple &arg)
        {
            boost::fusion::for_each(arg, read_params(source));

            source << body;

            source << "\t\treturn " << ret << ";\n";
        }

        std::string get() const {
            return source.str();
        }
    private:
        std::ostringstream source;

        struct read_params {
            std::ostream &os;
            mutable int prm_idx;

            read_params(std::ostream &os) : os(os), prm_idx(0) {}

            template <class T>
            void operator()(const T &v) const {
                os << "\t\t" << type_name<typename T::value_type>() << " "
                   << v << " = prm" << ++prm_idx << ";\n";
            }
        };
};

#ifndef BOOST_NO_VARIADIC_TEMPLATES
/// Builds kernel from recorded expression sequence and symbolic parameter list.
template <class... Args>
Kernel<sizeof...(Args)> build_kernel(
        const std::vector<cl::CommandQueue> &queue,
        const std::string &name, const std::string& body, const Args&... args
        )
{
    return Kernel<sizeof...(Args)>(queue, name, body, boost::tie(args...));
}

/// Builds function body from recorded expression and symbolic return value and parameters.
template <class Ret, class... Args>
std::string make_function(std::string body, const Ret &ret, const Args&... args) {
    return Function(body, ret, boost::tie(args...)).get();
}
#else

#define PRINT_ARG(z, n, data) const Arg ## n &arg ## n

#define BUILD_KERNEL(z, n, data) \
template < BOOST_PP_ENUM_PARAMS(n, class Arg) > \
Kernel<n> build_kernel( \
        const std::vector<cl::CommandQueue> &queue, \
        const std::string &name, const std::string& body, \
        BOOST_PP_ENUM(n, PRINT_ARG, ~) \
        ) \
{ \
    return Kernel<n>(queue, name, body, boost::tie( BOOST_PP_ENUM_PARAMS(n, arg) )); \
}
BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, BUILD_KERNEL, ~)
#undef BUILD_KERNEL

#define MAKE_FUNCTION(z, n, data) \
template <class Ret, BOOST_PP_ENUM_PARAMS(n, class Arg)> \
std::string make_function(std::string body, const Ret &ret, \
        BOOST_PP_ENUM(n, PRINT_ARG, ~)) \
{ \
    return Function(body, ret, boost::tie( BOOST_PP_ENUM_PARAMS(n, arg) )); \
}
BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, MAKE_FUNCTION, ~)
#undef MAKE_FUNCTION

#undef PRINT_ARG

#endif

/// UserFunction implementation from a generic functor
template <class Signature, class Functor>
struct FunctorAdapter : UserFunction<FunctorAdapter<Signature, Functor>, Signature>
{
    static std::string body_string;

    FunctorAdapter(Functor &&f) {
        using boost::function_types::function_arity;

        body_string = get_body(std::forward<Functor>(f),
                boost::mpl::size_t< function_arity<Signature>::value >() );
    }

    // Empty constructor. Used in UserFunction::operator(). Hopefuly the body
    // string is already constructed by the time the constructor is called.
    FunctorAdapter() {}

    static std::string body() {
        return body_string;
    }

#define PRINT_PRM(z, n, data) \
    typedef symbolic< typename boost::mpl::at<params, boost::mpl::int_<n>>::type > Prm ## n; \
    Prm ## n prm ## n(Prm ## n::ScalarParameter); \
    source << "\t\t" << type_name<typename Prm ## n::value_type>() << " " << prm ## n << " = prm" << n + 1 << ";\n";

#define BODY_GETTER(z, n, data)                                                          \
    static std::string get_body(Functor &&f, boost::mpl::size_t<n>) {                                 \
        typedef typename boost::function_types::result_type<Signature>::type     result; \
        typedef typename boost::function_types::parameter_types<Signature>::type params; \
        std::ostringstream source;                                                       \
        set_recorder(source);                                                            \
        BOOST_PP_REPEAT(n, PRINT_PRM, ~)                                                 \
        symbolic< result > ret = f( BOOST_PP_ENUM_PARAMS(n, prm) );              \
        source << "\t\treturn " << ret << ";\n";                                         \
        return source.str();                                                             \
    }

    BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, BODY_GETTER, ~)

#undef BODY_GETTER
#undef PRINT_PRM
};

template <class Signature, class Functor>
std::string FunctorAdapter<Signature, Functor>::body_string;

/// Generates user-defined function from a genric functor.
/**
 * Takes function signature as template parameter, functor as a single
 * argument.
 * Returns user-defined function ready to be used in vector expressions.
 */
template <class Signature, class Functor>
FunctorAdapter<Signature, Functor> make_function(Functor &&f) {
    return FunctorAdapter<Signature, Functor>(std::forward<Functor>(f));
}

} // namespace generator;

} // namespace vex;


// vim: et
#endif
