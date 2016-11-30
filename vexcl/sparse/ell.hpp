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

#include <vexcl/reductor.hpp>
#include <vexcl/vector_view.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/function.hpp>
#include <vexcl/eval.hpp>
#include <vexcl/vector_pointer.hpp>
#include <vexcl/scan.hpp>
#include <vexcl/sparse/csr.hpp>

namespace vex {
namespace sparse {

template <typename Val, typename Col = int, typename Ptr = Col>
class ell {
    public:
        typedef Val value_type;

        typedef Val val_type;
        typedef Col col_type;
        typedef Ptr ptr_type;

        template <class PtrRange, class ColRange, class ValRange>
        ell(
                const std::vector<backend::command_queue> &q,
                size_t nrows, size_t ncols,
                const PtrRange &ptr,
                const ColRange &col,
                const ValRange &val,
                bool fast = true
           ) :
            q(q[0]), n(nrows), m(ncols), nnz(boost::size(val)),
            ell_pitch(alignup(nrows, 16U)), csr_nnz(0)
        {
            precondition(q.size() == 1,
                    "sparse::ell is only supported for single-device contexts");

            if (fast) {
                convert(csr<Val,Col,Ptr>(q, nrows, ncols, ptr, col, val));
                return;
            }

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

            if (ell_width == 0) {
                assert(csr_nnz == nnz);

                csr_ptr = backend::device_vector<Col>(q[0], n + 1,   &ptr[0]);
                csr_col = backend::device_vector<Col>(q[0], csr_nnz, &col[0]);
                csr_val = backend::device_vector<Val>(q[0], csr_nnz, &val[0]);

                return;
            }

            // Count nonzeros in CSR part of the matrix.
            for(size_t i = ell_width + 1; i <= max_width; ++i)
                csr_nnz += hist[i] * (i - ell_width);

            /* 3. Split the input matrix into ELL and CSR submatrices. */
            std::vector<Col> _ell_col(ell_pitch * ell_width, static_cast<Col>(-1));
            std::vector<Val> _ell_val(ell_pitch * ell_width);
            std::vector<Ptr> _csr_ptr;
            std::vector<Col> _csr_col;
            std::vector<Val> _csr_val;

            if (csr_nnz) {
                _csr_ptr.resize(n + 1);
                _csr_col.resize(csr_nnz);
                _csr_val.resize(csr_nnz);

                _csr_ptr[0] = 0;
                for(size_t i = 0; i < n; ++i) {
                    size_t w = ptr[i+1] - ptr[i];
                    _csr_ptr[i+1] = _csr_ptr[i] + (w > ell_width ? w - ell_width : 0);
                }
            }


            for(size_t i = 0; i < n; ++i) {
                size_t w = 0;
                Ptr csr_head = csr_nnz ? _csr_ptr[i] : 0;
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

            ell_col = backend::device_vector<Col>(q[0], ell_pitch * ell_width, _ell_col.data());
            ell_val = backend::device_vector<Val>(q[0], ell_pitch * ell_width, _ell_val.data());

            if (csr_nnz) {
                csr_ptr = backend::device_vector<Col>(q[0], n + 1,   _csr_ptr.data());
                csr_col = backend::device_vector<Col>(q[0], csr_nnz, _csr_col.data());
                csr_val = backend::device_vector<Val>(q[0], csr_nnz, _csr_val.data());
            }
        }

        // Dummy matrix; used internally to pass empty parameters to kernels.
        ell(const backend::command_queue &q)
            : q(q), n(0), m(0), nnz(0), ell_width(0), ell_pitch(0), csr_nnz(0)
        {}

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
        static void kernel_param_declaration(const Vector &x, backend::source_generator &src,
            const backend::command_queue &q, const std::string &prm_name,
            detail::kernel_generator_state_ptr state)
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
        size_t cols()     const { return m; }
        size_t nonzeros() const { return nnz; }
    private:
        backend::command_queue q;

        size_t n, m, nnz, ell_width, ell_pitch, csr_nnz;

        backend::device_vector<Col> ell_col;
        backend::device_vector<Val> ell_val;

        backend::device_vector<Ptr> csr_ptr;
        backend::device_vector<Col> csr_col;
        backend::device_vector<Val> csr_val;

        backend::kernel& csr2ell_kernel() const {
            using namespace vex::detail;
            static kernel_cache cache;

            auto kernel = cache.find(q);
            if (kernel == cache.end()) {
                backend::source_generator src(q);

                src.begin_kernel("convert_csr2ell");
                src.begin_kernel_parameters();
                src.template parameter<size_t>("n");
                src.template parameter<size_t>("ell_width");
                src.template parameter<size_t>("ell_pitch");
                src.template parameter< global_ptr<const ptr_type> >("ptr");
                src.template parameter< global_ptr<const col_type> >("col");
                src.template parameter< global_ptr<const val_type> >("val");
                src.template parameter< global_ptr<col_type> >("ell_col");
                src.template parameter< global_ptr<val_type> >("ell_val");
                src.template parameter< global_ptr<const ptr_type> >("csr_ptr");
                src.template parameter< global_ptr<col_type> >("csr_col");
                src.template parameter< global_ptr<val_type> >("csr_val");
                src.end_kernel_parameters();
                src.grid_stride_loop().open("{");

                src.new_line() << type_name<int>() << " w = 0;";
                src.new_line() << type_name<ptr_type>() << " csr_head = 0;";
                src.new_line() << "if (csr_ptr) csr_head = csr_ptr[idx];";
                src.new_line() << "for(" << type_name<ptr_type>() << " j = ptr[idx], e = ptr[idx+1]; j < e; ++j, ++w)";
                src.open("{");
                src.new_line() << type_name<col_type>() << " c = col[j];";
                src.new_line() << type_name<val_type>() << " v = val[j];";
                src.new_line() << "if (w < ell_width) {";
                src.new_line() << "  ell_col[idx + w * ell_pitch] = c;";
                src.new_line() << "  ell_val[idx + w * ell_pitch] = v;";
                src.new_line() << "} else {";
                src.new_line() << "  csr_col[csr_head] = c;";
                src.new_line() << "  csr_val[csr_head] = v;";
                src.new_line() << "  ++csr_head;";
                src.new_line() << "}";
                src.close("}");
                //src.new_line() << "for(; w < ell_width; ++w)";
                //src.new_line() << "  ell_col[idx + w * ell_pitch] = (" << type_name<col_type>() << ")(-1);";
                src.close("}");
                src.end_kernel();

                kernel = cache.insert(q, backend::kernel(q, src.str(), "convert_csr2ell"));
            }

            return kernel->second;
        }

        void convert(const csr<val_type, col_type, ptr_type> &A) {
            /* 1. Get optimal ELL widths for local and remote parts. */
            // Speed of ELL relative to CSR:
            const double ell_vs_csr = 3.0;

            // Find maximum widths for local and remote parts:
            std::vector<backend::command_queue> ctx(1, q);
            Reductor<int, MAX> max(ctx);

            vex::vector<ptr_type> ptr(q, A.ptr);
            vex::vector<col_type> col(q, A.col);
            vex::vector<val_type> val(q, A.val);

            VEX_FUNCTION(ptr_type, row_width, (size_t, i)(const ptr_type*, ptr),
                    return ptr[i+1] - ptr[i];
                    );

            int max_width = max(row_width(element_index(0, n), raw_pointer(ptr)));

            // Build width distribution histogram.
            vex::vector<int> hist(ctx, max_width + 1);
            hist = 0;
            eval(atomic_add(&permutation(row_width(element_index(0, n), raw_pointer(ptr)))(hist), 1));

            // Estimate optimal width for ELL part of the matrix,
            // count nonzeros in CSR part of the matrix
            ell_width = max_width;
            {
                auto h = hist.map(0);

                for(int i = 0, rows = n; i < max_width; ++i) {
                    rows -= h[i]; // Number of rows wider than i.
                    if (ell_vs_csr * rows < n) {
                        ell_width = i;
                        break;
                    }
                }

                for(int i = ell_width + 1; i <= max_width; ++i)
                    csr_nnz += h[i] * (i - ell_width);
            }

            if (ell_width == 0) {
                assert(csr_nnz == nnz);

                csr_ptr = A.ptr;
                csr_col = A.col;
                csr_val = A.val;

                return;
            }

            if (csr_nnz) {
                VEX_FUNCTION(int, csr_width, (size_t, ell_width)(size_t, i)(const ptr_type*, ptr),
                        if (i == 0) return 0;
                        int w = ptr[i] - ptr[i-1];
                        return (w > ell_width) ? (w - ell_width) : 0;
                        );

                vex::vector<ptr_type> csr_w(ctx, n+1);

                csr_ptr = backend::device_vector<ptr_type>(q, n + 1);
                csr_col = backend::device_vector<col_type>(q, csr_nnz);
                csr_val = backend::device_vector<val_type>(q, csr_nnz);

                csr_w = csr_width(ell_width, element_index(), raw_pointer(ptr));
                vector<ptr_type> csr_p(q, csr_ptr);
                inclusive_scan(csr_w, csr_p);
            }


            /* 3. Split the input matrix into ELL and CSR submatrices. */
            ell_col = backend::device_vector<Col>(q, ell_pitch * ell_width);
            ell_val = backend::device_vector<Val>(q, ell_pitch * ell_width);

            vector<col_type>(q, ell_col) = -1;

            auto &convert = csr2ell_kernel();

            convert.push_arg(n);
            convert.push_arg(ell_width);
            convert.push_arg(ell_pitch);
            convert.push_arg(A.ptr);
            convert.push_arg(A.col);
            convert.push_arg(A.val);
            convert.push_arg(ell_col);
            convert.push_arg(ell_val);
            if (csr_nnz) {
                convert.push_arg(csr_ptr);
                convert.push_arg(csr_col);
                convert.push_arg(csr_val);
            } else {
                convert.push_arg(static_cast<size_t>(0));
                convert.push_arg(static_cast<size_t>(0));
                convert.push_arg(static_cast<size_t>(0));
            }
            convert(q);
        }

};

} // namespace sparse
} // namespace vex

#endif
