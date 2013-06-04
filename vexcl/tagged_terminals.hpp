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

namespace vex {

struct tagged_terminal_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< tagged_terminal_terminal >::type
    > tagged_terminal_expression;

template <size_t Tag, class Term>
class tagged_terminal : public tagged_terminal_expression
{
    public:
        tagged_terminal(const Term &term) : term(term) {}
    private:
        const Term & term;
};

template <>
struct is_vector_expr_terminal< tagged_terminal_terminal >
    : std::true_type
{ };

template <size_t Tag, class Term>
struct partial_vector_expr< vector_view<T, Slice> > {
    static std::string get(int component, int position) {
        return Slice::partial_expression(component, position);
    }
};

template <size_t Tag, class Term>
struct kernel_param_declaration< tagged_terminal<Tag, Term> > {
    static std::string get(int component, int position) {
        return kernel_param_declaration<Term>::get(component, position);
    }
};

template <typename T, class Slice>
struct kernel_arg_setter< vector_view<T, Slice> > {
    static void set(cl::Kernel &kernel, uint device, size_t index_offset, uint &position, const vector_view<T, Slice> &term) {
        assert(device == 0);

        Slice::setArgs(kernel, device, index_offset, position, term);
    }
};

template <typename T, class Slice>
struct expression_properties< vector_view<T, Slice> > {
    static void get(const vector_view<T, Slice> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.base.queue_list();
        partition  = term.base.partition();
        size       = term.slice.size();

        assert(partition.size() == 2);
        partition.back() = size;
    }
};


/// Taggs terminal with a unique (in a single expression) tag.
/**
 * By tagging terminals with same tags user guarantees that the terminals
 * actually refer to the same data. VexCL should be able to use this
 * information to reduce number of kernel parameters and unnecessary global
 * memory I/O.
 *
 * Example:
 * \code
 * s = sum(tag<1>(x) * tag<1>(x) + tag<2>(y) * tag<2>(y));
 * \endcode
 */
template <size_t Tag, class Expr>
tagged_terminal<Tag Expr> tag(const Expr &tag) {
    return tagged_terminal<Tag, Expr>(expr);
}

} //namespace vex
#endif
