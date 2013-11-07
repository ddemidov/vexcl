#ifndef VEXCL_CAST_HPP
#define VEXCL_CAST_HPP

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
 * \file   vexcl/cast.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Cast an expression to a type.
 */

#include <vexcl/operations.hpp>

namespace vex {

/// \cond INTERNAL
struct cast_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< cast_terminal >::type
    > cast_terminal_expression;

template <typename T, class Expr>
struct casted_expession : public cast_terminal_expression
{
    typedef T value_type;
    const Expr expr;

    casted_expession(const Expr &expr) : expr(expr) {}
};

/// \endcond

/// Cast an expression to a given type.
template <typename T, class Expr>
#ifdef DOXYGEN
casted_expession<T, Expr>
#else
typename std::enable_if<
    boost::proto::matches<
            typename boost::proto::result_of::as_expr< Expr >::type,
            vector_expr_grammar
    >::value,
    casted_expession<T, typename boost::proto::result_of::as_child<const Expr, vector_domain>::type
    >
#endif
>::type
cast(const Expr &expr)
{
    return casted_expession<
                T,
                typename boost::proto::result_of::as_child<const Expr, vector_domain>::type
            >(boost::proto::as_child<vector_domain>(expr));
}

/// \cond INTERNAL

namespace traits {

template <>
struct is_vector_expr_terminal< cast_terminal > : std::true_type {};

template <>
struct proto_terminal_is_value< cast_terminal > : std::true_type {};

template <typename T, class Expr>
struct terminal_preamble< casted_expession<T, Expr> > {
    static std::string get(const casted_expession<T, Expr> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        std::ostringstream s;

        detail::output_terminal_preamble termpream(s, dev, prm_name, state);
        boost::proto::eval(boost::proto::as_child(term.expr), termpream);

        return s.str();
    }
};

template <typename T, class Expr>
struct kernel_param_declaration< casted_expession<T, Expr> > {
    static std::string get(const casted_expession<T, Expr> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        std::ostringstream s;

        detail::declare_expression_parameter declare(s, dev, prm_name, state);
        detail::extract_terminals()(boost::proto::as_child(term.expr),  declare);

        return s.str();
    }
};

template <typename T, class Expr>
struct local_terminal_init< casted_expession<T, Expr> > {
    static std::string get(const casted_expession<T, Expr> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        std::ostringstream s;

        detail::output_local_preamble init_ctx(s, dev, prm_name, state);
        boost::proto::eval(boost::proto::as_child(term.expr), init_ctx);

        return s.str();
    }
};

template <typename T, class Expr>
struct partial_vector_expr< casted_expession<T, Expr> > {
    static std::string get(const casted_expession<T, Expr> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        std::ostringstream s;

        detail::vector_expr_context expr_ctx(s, dev, prm_name, state);
        boost::proto::eval(boost::proto::as_child(term.expr), expr_ctx);

        return s.str();
    }
};

template <typename T, class Expr>
struct kernel_arg_setter< casted_expession<T, Expr> > {
    static void set(const casted_expession<T, Expr> &term,
            backend::kernel &kernel, unsigned device, size_t index_offset,
            detail::kernel_generator_state_ptr state)
    {
        detail::set_expression_argument setarg(kernel, device, index_offset, state);
        detail::extract_terminals()( boost::proto::as_child(term.expr), setarg);
    }
};

template <typename T, class Expr>
struct expression_properties< casted_expession<T, Expr> > {
    static void get(const casted_expession<T, Expr> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        detail::get_expression_properties prop;
        detail::extract_terminals()(boost::proto::as_child(term.expr), prop);

        queue_list = prop.queue;
        partition  = prop.part;
        size       = prop.size;
    }
};

} // namespace traits

/// \endcond

} // namespace vex

#endif
