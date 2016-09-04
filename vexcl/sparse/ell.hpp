#ifndef VEXCL_SPARSE_ELL_HPP
#define VEXCL_SPARSE_ELL_HPP

#include <vector>
#include <type_traits>
#include <utility>

#include <boost/foreach.hpp>
#include <boost/range.hpp>

#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/sparse/product.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <vexcl/reductor.hpp>

namespace vex {
namespace sparse {

template <typename Val, typename Col = int, typename Ptr = Col>
class ell {
    public:
        typedef Val value_type;

        template <class PtrRange, class ColRange, class ValRange>
        ell(
                const std::vector<backend::command_queue> &q,
                const PtrRange &ptr,
                const ColRange &col,
                const ValRange &val
           ) :
            q(q[0]),
            n(boost::size(ptr) - 1),
            nnz(boost::size(val)),
            ell_pitch(alignup(n, 16U))
        {
            precondition(q.size() == 1,
                    "sparse::ell is only supported for single-device contexts");

            /* 1. Get optimal ELL widths for local and remote parts. */
            // Speed of ELL relative to CSR:
            const double ell_vs_csr = 3.0;

            // Find maximum widths for local and remote parts:
            size_t max_width = 0;
            for(size_t i = 0; i < n; ++i)
                max_width = std::max<size_t>(max_width, ptr[i+1] - ptr[i]);

            // Build width distribution histogram.
            std::vector<Ptr> hist(max_width + 1, 0);
            for(size_t i = 0; i < n; ++i)
                ++hist[ptr[i+1] - ptr[i]];

            // Estimate optimal width for ELL part of the matrix.
            ell_width = max_width;
            for(size_t i = 0, rows = n; i < max_width; ++i) {
                rows -= hist[i]; // Number of rows wider than i.
                if (ell_vs_csr * rows < n) {
                    ell_width = i;
                    break;
                }
            }

            /* 2. Count nonzeros in CSR part of the matrix. */
            std::vector<Ptr> _csr_ptr(n+1);
            _csr_ptr[0] = 0;
            for(size_t i = 0; i < n; ++i) {
                size_t w = ptr[i+1] - ptr[i];
                _csr_ptr[i+1] = _csr_ptr[i] + (w > ell_width ? w - ell_width : 0);
            }
            csr_nnz = _csr_ptr[n];

            /* 3. Split the input matrix into ELL and CSR submatrices. */
            std::vector<Col> _ell_col(ell_pitch * ell_width, static_cast<Col>(-1));
            std::vector<Val> _ell_val(ell_pitch * ell_width);
            std::vector<Col> _csr_col(csr_nnz);
            std::vector<Val> _csr_val(csr_nnz);

            for(size_t i = 0; i < n; ++i) {
                size_t w = 0;
                Ptr csr_head = _csr_ptr[i];
                for(Ptr j = ptr[i], e = ptr[i+1]; j < e; ++j, ++w) {
                    Col c = col[j];
                    Val v = val[j];

                    if (w < ell_width) {
                        _ell_col[i + w * ell_pitch] = c;
                        _ell_val[i + w * ell_pitch] = v;
                    } else {
                        _csr_col[csr_head] = c;
                        _csr_val[csr_head] = v;
                        ++csr_head;
                    }
                }
            }

            if (ell_width) {
                ell_col = backend::device_vector<Col>(q[0], ell_pitch * ell_width, _ell_col.data());
                ell_val = backend::device_vector<Val>(q[0], ell_pitch * ell_width, _ell_val.data());
            }

            if (csr_nnz) {
                csr_ptr = backend::device_vector<Col>(q[0], n + 1,   _csr_ptr.data());
                csr_col = backend::device_vector<Col>(q[0], csr_nnz, _csr_col.data());
                csr_val = backend::device_vector<Val>(q[0], csr_nnz, _csr_val.data());
            }
        }

        template <class Expr>
        friend
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                vector_expr_grammar
            >::value,
            matrix_vector_product<ell, Expr>
        >::type
        operator*(const ell &A, const Expr &x) {
            return matrix_vector_product<ell, Expr>(A, x);
        }

        template <class Vector>
        void terminal_preamble(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state) const
        {
            detail::output_terminal_preamble tp(src, q, prm_name + "_x", state);
            boost::proto::eval(boost::proto::as_child(x), tp);
        }

        template <class Vector>
        void local_terminal_init(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state) const
        {
            typedef typename detail::return_type<Vector>::type x_type;
            typedef decltype(std::declval<Val>() * std::declval<x_type>()) res_type;

            src.new_line()
                << type_name<res_type>() << " " << prm_name << "_sum = "
                << res_type() << ";";
            src.open("{");

            // ELL part
            src.new_line() << "for(size_t j = 0; j < " << prm_name << "_ell_width; ++j)";
            src.open("{");
            src.new_line()
                << type_name<Col>() << " c = " << prm_name << "_ell_col[idx + j * "
                << prm_name << "_ell_pitch];";
            src.new_line() << "if (c != (" << type_name<Col>() << ")(-1))";
            src.open("{");
            src.new_line() << type_name<Val>() << " v = " << prm_name
                << "_ell_val[idx + j * " << prm_name << "_ell_pitch];";

            src.new_line() << type_name<Col>() << " idx = c;";

            {
                detail::output_local_preamble init_x(src, q, prm_name + "_x", state);
                boost::proto::eval(boost::proto::as_child(x), init_x);
            }

            src.new_line() << prm_name << "_sum += v * ";

            {
                detail::vector_expr_context expr_x(src, q, prm_name + "_x", state);
                boost::proto::eval(boost::proto::as_child(x), expr_x);
            }
            src << ";";

            src.close("} else break;");
            src.close("}");

            // CSR part
            src.new_line() << "if (" << prm_name << "_csr_ptr)";
            src.open("{");
            src.new_line() << type_name<Ptr>() << " csr_beg = " << prm_name << "_csr_ptr[idx];";
            src.new_line() << type_name<Ptr>() << " csr_end = " << prm_name << "_csr_ptr[idx+1];";
            src.new_line() << "for(" << type_name<Ptr>() << " j = csr_beg; j < csr_end; ++j)";
            src.open("{");

            src.new_line() << type_name<Col>() << " idx = " << prm_name << "_csr_col[j];";

            {
                detail::output_local_preamble init_x(src, q, prm_name + "_x", state);
                boost::proto::eval(boost::proto::as_child(x), init_x);
            }

            src.new_line() << prm_name << "_sum += " << prm_name << "_csr_val[j] * ";

            {
                detail::vector_expr_context expr_x(src, q, prm_name + "_x", state);
                boost::proto::eval(boost::proto::as_child(x), expr_x);
            }
            src << ";";

            src.close("}");
            src.close("}");
            src.close("}");
        }

        template <class Vector>
        void kernel_param_declaration(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state) const
        {
            src.parameter< size_t >(prm_name + "_ell_width");
            src.parameter< size_t >(prm_name + "_ell_pitch");

            src.parameter< global_ptr<Col> >(prm_name + "_ell_col");
            src.parameter< global_ptr<Val> >(prm_name + "_ell_val");
            src.parameter< global_ptr<Ptr> >(prm_name + "_csr_ptr");
            src.parameter< global_ptr<Col> >(prm_name + "_csr_col");
            src.parameter< global_ptr<Val> >(prm_name + "_csr_val");

            detail::declare_expression_parameter decl_x(src, q, prm_name + "_x", state);
            detail::extract_terminals()(boost::proto::as_child(x), decl_x);
        }

        template <class Vector>
        void partial_vector_expr(const Vector &x, backend::source_generator &src,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr) const
        {
            src << prm_name << "_sum";
        }

        template <class Vector>
        void kernel_arg_setter(const Vector &x,
            backend::kernel &kernel, unsigned part, size_t index_offset,
            detail::kernel_generator_state_ptr state) const
        {
            kernel.push_arg(ell_width);
            kernel.push_arg(ell_pitch);
            if (ell_width) {
                kernel.push_arg(ell_col);
                kernel.push_arg(ell_val);
            } else {
                kernel.push_arg(static_cast<size_t>(0));
                kernel.push_arg(static_cast<size_t>(0));
            }
            if (csr_nnz) {
                kernel.push_arg(csr_ptr);
                kernel.push_arg(csr_col);
                kernel.push_arg(csr_val);
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
        size_t cols()     const { return n; }
        size_t nonzeros() const { return nnz; }
    private:
        backend::command_queue q;

        size_t n, nnz, ell_width, csr_nnz, ell_pitch;

        backend::device_vector<Col> ell_col;
        backend::device_vector<Val> ell_val;

        backend::device_vector<Ptr> csr_ptr;
        backend::device_vector<Col> csr_col;
        backend::device_vector<Val> csr_val;
};

} // namespace sparse
} // namespace vex

#endif
