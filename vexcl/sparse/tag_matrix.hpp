#ifndef VEXCL_SPARSE_TAG_MATRIX_HPP
#define VEXCL_SPARSE_TAG_MATRIX_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/sparse/tag_matrix.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Tagged sparse matrix.
 */


#include <vexcl/operations.hpp>
#include <vexcl/sparse/product.hpp>

namespace vex {
namespace sparse {

template <size_t Tag, class Matrix>
struct tagged_matrix {
    const Matrix &A;

    tagged_matrix(const Matrix &A) : A(A) {}

    template <class Expr>
    friend
    typename std::enable_if<
        boost::proto::matches<
            typename boost::proto::result_of::as_expr<Expr>::type,
            vector_expr_grammar
        >::value,
        matrix_vector_product<tagged_matrix, Expr>
    >::type
    operator*(const tagged_matrix &A, const Expr &x) {
        return matrix_vector_product<tagged_matrix, Expr>(A, x);
    }

    template <class Vector>
    static void terminal_preamble(const Vector &x, backend::source_generator &src,
        const backend::command_queue &q, const std::string &prm_name,
        detail::kernel_generator_state_ptr state)
    {
        Matrix::terminal_preamble(x, src, q, prm_name.str(), state);
    }

    template <class Vector>
    static void local_terminal_init(const Vector &x, backend::source_generator &src,
        const backend::command_queue &q, const std::string &prm_name,
        detail::kernel_generator_state_ptr state)
    {
        Matrix::local_terminal_init(x, src, q, prm_name, state);
    }

    template <class Vector>
    static void kernel_param_declaration(const Vector &x, backend::source_generator &src,
        const backend::command_queue &q, const std::string &prm_name,
        detail::kernel_generator_state_ptr state)
    {
        Matrix::kernel_param_declaration(x, src, q, prm_name.str(), state);
    }

    template <class Vector>
    static void partial_vector_expr(const Vector &x, backend::source_generator &src,
        const backend::command_queue &q, const std::string &prm_name,
        detail::kernel_generator_state_ptr state)
    {
        Matrix::partial_vector_expr(x, src, q, prm_name, state);
    }

    template <class Vector>
    void kernel_arg_setter(const Vector &x,
        backend::kernel &kernel, unsigned part, size_t index_offset,
        detail::kernel_generator_state_ptr state) const
    {
        A.kernel_arg_setter(x, kernel, part, index_offset, state);
    }

    template <class Vector>
    void expression_properties(const Vector &x,
        std::vector<backend::command_queue> &queue_list,
        std::vector<size_t> &partition,
        size_t &size) const
    {
        A.expression_properties(x, queue_list, partition, size);
    }
};

} // namespace sparse
} // namespace vex

#endif
