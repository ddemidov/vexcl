#ifndef VEXCL_TAGGED_TERMINAL_HPP
#define VEXCL_TAGGED_TERMINAL_HPP

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

THE SOFTWARE IS PROVTagED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl/tagged_terminal.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Tagged terminal wrapper.
 */

#include <vector>
#include <string>
#include <sstream>

#include <vexcl/operations.hpp>

namespace vex {

/// \cond INTERNAL
struct tagged_terminal_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< tagged_terminal_terminal >::type
    > tagged_terminal_expression;

template <size_t Tag, class Term>
struct tagged_terminal : tagged_terminal_expression
{
    const Term & term;
    tagged_terminal(const Term &term) : term(term) {}

    // Expression assignments.
#define ASSIGNMENT(cop, op) \
    template <class Expr> \
    typename std::enable_if< \
        boost::proto::matches< \
            typename boost::proto::result_of::as_expr<Expr>::type, \
            vector_expr_grammar \
        >::value, \
        const tagged_terminal& \
    >::type \
    operator cop(const Expr &expr) { \
        std::vector<cl::CommandQueue> queue; \
        std::vector<size_t> part; \
        size_t size; \
        traits::expression_properties<tagged_terminal>::get(*this, queue, part, size); \
        detail::assign_expression<op>(*this, expr, queue, part); \
        return *this; \
    }

    ASSIGNMENT(=,   assign::SET);
    ASSIGNMENT(+=,  assign::ADD);
    ASSIGNMENT(-=,  assign::SUB);
    ASSIGNMENT(*=,  assign::MUL);
    ASSIGNMENT(/=,  assign::DIV);
    ASSIGNMENT(%=,  assign::MOD);
    ASSIGNMENT(&=,  assign::AND);
    ASSIGNMENT(|=,  assign::OR);
    ASSIGNMENT(^=,  assign::XOR);
    ASSIGNMENT(<<=, assign::LSH);
    ASSIGNMENT(>>=, assign::RSH);

#undef ASSIGNMENT
};

namespace traits {

template <>
struct is_vector_expr_terminal< tagged_terminal_terminal >
    : std::true_type
{ };

template <size_t Tag, class Term>
struct terminal_preamble< tagged_terminal<Tag, Term> > {
    static std::string get(const cl::Device &device,
            int component, int position,
            detail::kernel_generator_state &state)
    {
        auto s = state.find("tagged_terminal");

        if (s == state.end()) {
            s = state.insert(std::make_pair(
                        std::string("tagged_terminal"),
                        boost::any(std::map<size_t, int>())
                        )).first;
        }

        auto &pos = boost::any_cast< std::map<size_t, int>& >(s->second);
        auto p = pos.find(Tag);

        typedef
            typename boost::remove_const<
                typename boost::remove_reference<
                    typename boost::proto::result_of::as_child<Term>::type
                >::type
            >::type TermType;

        if (p == pos.end()) {
            pos[Tag] = position;

            return terminal_preamble<TermType>::get(device, component, position, state);
        } else {
            return "";
        }
    }
};

template <size_t Tag, class Term>
struct kernel_param_declaration< tagged_terminal<Tag, Term> > {
    static std::string get(const cl::Device &device,
            int component, int position,
            detail::kernel_generator_state &state)
    {
        auto s = state.find("tagged_terminal");

        if (s == state.end()) {
            s = state.insert(std::make_pair(
                        std::string("tagged_terminal"),
                        boost::any(std::map<size_t, int>())
                        )).first;
        }

        auto &pos = boost::any_cast< std::map<size_t, int>& >(s->second);
        auto p = pos.find(Tag);

        typedef
            typename boost::remove_const<
                typename boost::remove_reference<
                    typename boost::proto::result_of::as_child<Term>::type
                >::type
            >::type TermType;

        if (p == pos.end()) {
            pos[Tag] = position;

            return kernel_param_declaration<TermType>::get(device, component, position, state);
        } else {
            return "";
        }
    }
};

template <size_t Tag, class Term>
struct partial_vector_expr< tagged_terminal<Tag, Term> > {
    static std::string get(const cl::Device &device,
            int component, int position,
            detail::kernel_generator_state &state)
    {
        auto s = state.find("tagged_terminal");

        if (s == state.end()) {
            s = state.insert(std::make_pair(
                        std::string("tagged_terminal"),
                        boost::any(std::map<size_t, int>())
                        )).first;
        }

        auto &pos = boost::any_cast< std::map<size_t, int>& >(s->second);
        auto p = pos.find(Tag);

        typedef
            typename boost::remove_const<
                typename boost::remove_reference<
                    typename boost::proto::result_of::as_child<Term>::type
                >::type
            >::type TermType;

        if (p == pos.end()) {
            pos[Tag] = position;
            return partial_vector_expr<TermType>::get(device, component, position, state);
        } else {
            return partial_vector_expr<TermType>::get(device, component, p->second, state);
        }
    }
};

template <size_t Tag, class Term>
struct kernel_arg_setter< tagged_terminal<Tag, Term> > {
    static void set(cl::Kernel &kernel, unsigned device, size_t index_offset,
            unsigned &position, const tagged_terminal<Tag, Term> &term,
            detail::kernel_generator_state &state)
    {
        auto s = state.find("tagged_terminal_arg");

        if (s == state.end()) {
            s = state.insert(std::make_pair(
                        std::string("tagged_terminal_arg"),
                        boost::any(std::map<size_t, unsigned>())
                        )).first;
        }

        auto &pos = boost::any_cast< std::map<size_t, unsigned>& >(s->second);
        auto p = pos.find(Tag);

        typedef
            typename boost::remove_const<
                typename boost::remove_reference<
                    decltype(boost::proto::as_child(term.term))
                >::type
            >::type TermType;

        if (p == pos.end()) {
            pos[Tag] = position;
            kernel_arg_setter<TermType>::set(kernel, device, index_offset, position, boost::proto::as_child(term.term), state);
        }
    }
};

template <size_t Tag, class Term>
struct expression_properties< tagged_terminal<Tag, Term> > {
    static void get(const tagged_terminal<Tag, Term> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        typedef
            typename boost::remove_const<
                typename boost::remove_reference<
                    decltype(boost::proto::as_child(term.term))
                >::type
            >::type TermType;

        expression_properties<TermType>::get(boost::proto::as_child(term.term), queue_list, partition, size);
    }
};

} // namespace traits

/// \endcond

/// Tags terminal with a unique (in a single expression) tag.
/**
 * By tagging terminals user guarantees that the terminals with same tags
 * actually refer to the same data. VexCL is able to use this information in
 * order to reduce number of kernel parameters and unnecessary global memory
 * I/O operations.
 *
 * Example:
 * \code
 * s = sum(tag<1>(x) * tag<1>(x) + tag<2>(y) * tag<2>(y));
 * \endcode
 */
template <size_t Tag, class Expr>
#ifdef DOXYGEN
tagged_terminal<Tag, Expr>
#else
typename std::enable_if<
    boost::proto::matches<
        typename boost::proto::result_of::as_expr<Expr>::type,
        boost::proto::terminal<boost::proto::_>
    >::value,
    tagged_terminal<Tag, Expr>
>::type
#endif
tag(const Expr &expr) {
    return tagged_terminal<Tag, Expr>(expr);
}

} //namespace vex
#endif
