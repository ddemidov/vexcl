#ifndef VEXCL_SYMBOLIC_HPP
#define VEXCL_SYMBOLIC_HPP

#include <iostream>
#include <type_traits>
#include <vexcl/util.hpp>

namespace vex {

namespace generator {

template <class T, class Enable = void>
struct terminal {
    static void get(std::ostream &os, const T &v) {
	os << v;
    }
};

template <class T>
struct terminal< T, typename std::enable_if<T::is_symbolic>::type > {
    static void get(std::ostream &os, const T &v) {
	v.get_string(os);
    }
};

template <typename T>
class symbolic {
    public:
	static const bool is_symbolic = true;

	symbolic(bool need_declaration = true) : num(index++)
	{
	    if (need_declaration) {
		(*os) << type_name<T>() << " ";
		get_string(*os);
		(*os) << ";\n";
	    }
	}

	symbolic(const symbolic &s) : num(index++) {
	    (*os) << type_name<T>() << " ";
	    get_string(*os);
	    (*os) << " = ";
	    s.get_string(*os);
	    (*os) << ";\n";
	}

	void get_string(std::ostream &s) const {
	    s << "var" << num;
	}

	template <class Expr>
	const symbolic& operator=(const Expr &expr) const {
	    get_string(*os);
	    (*os) << " = ";
	    terminal<Expr>::get(*os, expr);
	    (*os) << ";\n";
	    return *this;
	}

	const symbolic& operator=(const symbolic &s) {
	    get_string(*os);
	    (*os) << " = ";
	    s.get_string(*os);
	    (*os) << ";\n";
	    return *this;
	}

	static void set_output(std::ostream &s) {
	    os = &s;
	}
    private:
	static size_t index;
	static std::ostream *os;

	size_t num;
};

template <typename T> size_t symbolic<T>::index = 0;
template <typename T> std::ostream* symbolic<T>::os = 0;

template <class LHS, binop::kind OP, class RHS>
struct symbolic_expression {
    static const bool is_symbolic = true;

    symbolic_expression(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    const LHS &lhs;
    const RHS &rhs;

    void get_string(std::ostream &s) const {
	s << "(";
	terminal<LHS>::get(s, lhs);
	s << " " << binop::traits<OP>::oper() << " ";
	terminal<RHS>::get(s, rhs);
	s << ")";
    }
};

#define DEFINE_BINARY_OP(kind, oper) \
template <class LHS, class RHS> \
symbolic_expression<LHS, kind, RHS> operator oper(const LHS &lhs, const RHS &rhs) { \
    return symbolic_expression<LHS, kind, RHS>(lhs, rhs); \
}

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

} // namespace generator;

} // namespace vex;

#endif
