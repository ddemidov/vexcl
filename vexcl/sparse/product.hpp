#ifndef VEXCL_SPARSE_PRODUCT_HPP
#define VEXCL_SPARSE_PRODUCT_HPP

namespace vex {
namespace sparse {

struct matrix_vector_product_terminal {};

typedef vector_expression<
        typename boost::proto::terminal< matrix_vector_product_terminal >::type
        > matrix_vector_product_expression;

template <class Matrix, class Vector>
struct matrix_vector_product : matrix_vector_product_expression
{
    const Matrix &A;
    const Vector &x;

    matrix_vector_product(const Matrix &A, const Vector &x)
        : A(A), x(x) {}
};

} // namespace sparse

namespace traits {

template <> struct is_vector_expr_terminal< sparse::matrix_vector_product_terminal >
    : std::true_type {};

template <class Matrix, class Vector>
struct terminal_is_value< sparse::matrix_vector_product<Matrix, Vector> >
    : std::true_type {};

template <class Matrix, class Vector>
struct terminal_preamble< sparse::matrix_vector_product<Matrix, Vector> > {
    static void get(backend::source_generator &src,
            const sparse::matrix_vector_product<Matrix, Vector> &term,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        term.A.terminal_preamble(term.x, src, q, prm_name, state);
    }
};

template <class Matrix, class Vector>
struct local_terminal_init< sparse::matrix_vector_product<Matrix, Vector> > {
    static void get(backend::source_generator &src,
            const sparse::matrix_vector_product<Matrix, Vector> &term,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        term.A.local_terminal_init(term.x, src, q, prm_name, state);
    }
};

template <class Matrix, class Vector>
struct kernel_param_declaration< sparse::matrix_vector_product<Matrix, Vector> > {
    static void get(backend::source_generator &src,
            const sparse::matrix_vector_product<Matrix, Vector> &term,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        term.A.kernel_param_declaration(term.x, src, q, prm_name, state);
    }
};

template <class Matrix, class Vector>
struct partial_vector_expr< sparse::matrix_vector_product<Matrix, Vector> > {
    static void get(backend::source_generator &src,
            const sparse::matrix_vector_product<Matrix, Vector> &term,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
    {
        term.A.partial_vector_expr(term.x, src, q, prm_name, state);
    }
};

template <class Matrix, class Vector>
struct kernel_arg_setter< sparse::matrix_vector_product<Matrix, Vector> > {
    static void set(const sparse::matrix_vector_product<Matrix, Vector> &term,
            backend::kernel &kernel, unsigned part, size_t index_offset,
            detail::kernel_generator_state_ptr state)
    {
        term.A.kernel_arg_setter(term.x, kernel, part, index_offset, state);
    }
};

template <class Matrix, class Vector>
struct expression_properties< sparse::matrix_vector_product<Matrix, Vector> > {
    static void get(const sparse::matrix_vector_product<Matrix, Vector> &term,
            std::vector<backend::command_queue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        term.A.expression_properties(term.x, queue_list, partition, size);
    }
};

} // namespace traits

} // namespace vex

#endif
