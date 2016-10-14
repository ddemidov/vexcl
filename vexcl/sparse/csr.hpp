#ifndef VEXCL_SPARSE_CSR_HPP
#define VEXCL_SPARSE_CSR_HPP

#include <vector>
#include <type_traits>
#include <utility>

#include <boost/range.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <vexcl/sparse/product.hpp>

namespace vex {
namespace sparse {

template <typename Val, typename Col, typename Ptr> class ell;

template <typename Val, typename Col = int, typename Ptr = Col>
class csr {
    public:
        typedef Val value_type;

        typedef Val val_type;
        typedef Col col_type;
        typedef Ptr ptr_type;

        template <class PtrRange, class ColRange, class ValRange>
        csr(
                const std::vector<backend::command_queue> &q,
                size_t nrows, size_t ncols,
                const PtrRange &ptr,
                const ColRange &col,
                const ValRange &val,
                bool fast_setup = true
           )
            : q(q[0]), n(nrows), m(ncols), nnz(boost::size(val)),
              ptr(q[0], boost::size(ptr), &ptr[0]),
              col(q[0], boost::size(col), &col[0]),
              val(q[0], boost::size(val), &val[0])
        {
            precondition(q.size() == 1,
                    "sparse::csr is only supported for single-device contexts");
        }

        // Dummy matrix; used internally to pass empty parameters to kernels.
        csr(const backend::command_queue &q)
            : q(q), n(0), m(0), nnz(0)
        {}

        template <class Expr>
        friend
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                vector_expr_grammar
            >::value,
            matrix_vector_product<csr, Expr>
        >::type
        operator*(const csr &A, const Expr &x) {
            return matrix_vector_product<csr, Expr>(A, x);
        }

        template <class Vector>
        static void terminal_preamble(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
        {
            detail::output_terminal_preamble tp(src, q, prm_name + "_x", state);
            boost::proto::eval(boost::proto::as_child(x), tp);
        }

        template <class Vector>
        static void local_terminal_init(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
        {
            typedef typename detail::return_type<Vector>::type x_type;
            typedef decltype(std::declval<Val>() * std::declval<x_type>()) res_type;

            src.new_line()
                << type_name<res_type>() << " " << prm_name << "_sum = "
                << res_type() << ";";
            src.new_line() << "if (" << prm_name << "_ptr)";
            src.open("{");
            src.new_line() << type_name<Ptr>() << " row_beg = " << prm_name << "_ptr[idx];";
            src.new_line() << type_name<Ptr>() << " row_end = " << prm_name << "_ptr[idx+1];";
            src.new_line() << "for(" << type_name<Ptr>() << " j = row_beg; j < row_end; ++j)";
            src.open("{");

            src.new_line() << type_name<Col>() << " idx = " << prm_name << "_col[j];";

            detail::output_local_preamble init_x(src, q, prm_name + "_x", state);
            boost::proto::eval(boost::proto::as_child(x), init_x);

            src.new_line() << prm_name << "_sum += " << prm_name << "_val[j] * ";

            detail::vector_expr_context expr_x(src, q, prm_name + "_x", state);
            boost::proto::eval(boost::proto::as_child(x), expr_x);

            src << ";";

            src.close("}");
            src.close("}");
        }

        template <class Vector>
        static void kernel_param_declaration(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
        {
            src.parameter< global_ptr<Ptr> >(prm_name + "_ptr");
            src.parameter< global_ptr<Col> >(prm_name + "_col");
            src.parameter< global_ptr<Val> >(prm_name + "_val");

            detail::declare_expression_parameter decl_x(src, q, prm_name + "_x", state);
            detail::extract_terminals()(boost::proto::as_child(x), decl_x);
        }

        template <class Vector>
        static void partial_vector_expr(const Vector &x, backend::source_generator &src,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
        {
            src << prm_name << "_sum";
        }

        template <class Vector>
        void kernel_arg_setter(const Vector &x,
            backend::kernel &kernel, unsigned part, size_t index_offset,
            detail::kernel_generator_state_ptr state) const
        {
            if (nnz) {
                kernel.push_arg(ptr);
                kernel.push_arg(col);
                kernel.push_arg(val);
            } else {
                kernel.push_arg(static_cast<size_t>(0));
                kernel.push_arg(static_cast<size_t>(0));
                kernel.push_arg(static_cast<size_t>(0));
            }

            detail::set_expression_argument x_args(kernel, part, index_offset, state);
            detail::extract_terminals()( boost::proto::as_child(x), x_args);
        }

        template <class Vector>
        void expression_properties(const Vector &x,
            std::vector<backend::command_queue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size) const
        {
            queue_list = std::vector<backend::command_queue>(1, q);
            partition  = std::vector<size_t>(2, 0);
            partition.back() = size = n;
        }

        size_t rows()     const { return n; }
        size_t cols()     const { return m; }
        size_t nonzeros() const { return nnz; }
    private:
        backend::command_queue q;

        size_t n, m, nnz;

        backend::device_vector<Ptr> ptr;
        backend::device_vector<Col> col;
        backend::device_vector<Val> val;

        friend class ell<val_type, col_type, ptr_type>;
};

} // namespace sparse
} // namespace vex

#endif
