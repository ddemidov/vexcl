#ifndef VEXCL_SPMAT_HPP
#define VEXCL_SPMAT_HPP

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
 * \file   vexcl/spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL sparse matrix.
 */

#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <type_traits>

#include <vexcl/vector.hpp>

namespace vex {

/// Sparse matrix in hybrid ELL-CSR format.
template <typename val_t, typename col_t = size_t, typename idx_t = size_t>
class SpMat {
    public:
        typedef val_t value_type;
        typedef typename cl_scalar_of<val_t>::type scalar_type;

        /// Empty constructor.
        SpMat() : nrows(0), ncols(0), nnz(0) {}

        /// Constructor.
        /**
         * Constructs GPU representation of the matrix. Input matrix is in CSR
         * format. GPU matrix utilizes ELL format and is split equally across
         * all compute devices. When there are more than one device, secondary
         * queue can be used to perform transfer of ghost values across GPU
         * boundaries in parallel with computation kernel.
         * \param queue vector of queues. Each queue represents one
         *            compute device.
         * \param n   number of rows in the matrix.
         * \param m   number of cols in the matrix.
         * \param row row index into col and val vectors.
         * \param col column numbers of nonzero elements of the matrix.
         * \param val values of nonzero elements of the matrix.
         */
        SpMat(const std::vector<cl::CommandQueue> &queue,
              size_t n, size_t m, const idx_t *row, const col_t *col, const val_t *val
              )
            : queue(queue), part(partition(n, queue)),
              event1(queue.size(), std::vector<cl::Event>(1)),
              event2(queue.size(), std::vector<cl::Event>(1)),
              mtx(queue.size()), exc(queue.size()),
              nrows(n), ncols(m), nnz(row[n])
        {
            auto col_part = partition(m, queue);

            // Create secondary queues.
            for(auto q = queue.begin(); q != queue.end(); q++)
                squeue.push_back(cl::CommandQueue(qctx(*q), qdev(*q)));

            std::vector<std::set<col_t>> ghost_cols = setup_exchange(col_part, row, col);

            // Each device get it's own strip of the matrix.
#ifdef _OPENMP
#  pragma omp parallel for schedule(static,1)
#endif
            for(int d = 0; d < static_cast<int>(queue.size()); d++) {
                if (part[d + 1] > part[d]) {
                    cl::Device device = qdev(queue[d]);

                    if ( is_cpu(device) )
                        mtx[d].reset(
                                new SpMatCSR(queue[d], row, col, val,
                                    part[d], part[d+1], col_part[d], col_part[d+1],
                                    ghost_cols[d])
                                );
                    else
                        mtx[d].reset(
                                new SpMatHELL(queue[d], row, col, val,
                                    part[d], part[d+1], col_part[d], col_part[d+1],
                                    ghost_cols[d])
                                );
                }
            }
        }


        /// Matrix-vector multiplication.
        /**
         * Matrix vector multiplication (\f$y = \alpha Ax\f$ or \f$y += \alpha
         * Ax\f$) is performed in parallel on all registered compute devices.
         * Ghost values of x are transfered across GPU boundaries as needed.
         * \param x      input vector.
         * \param y      output vector.
         * \param alpha  coefficient in front of matrix-vector product
         * \param append if set, matrix-vector product is appended to y.
         *               Otherwise, y is replaced with matrix-vector product.
         */
        void mul(const vex::vector<val_t> &x, vex::vector<val_t> &y,
                 scalar_type alpha = 1, bool append = false) const
        {
            using namespace detail;

            static kernel_cache cache;

            if (rx.size()) {
                // Transfer remote parts of the input vector.
                for(unsigned d = 0; d < queue.size(); d++) {
                    cl::Context context = qctx(queue[d]);
                    cl::Device  device  = qdev(queue[d]);

                    auto gather = cache.find(context());

                    if (gather == cache.end()) {
                        std::ostringstream source;

                        source << backend::standard_kernel_header(device) <<
                            "typedef " << type_name<val_t>() << " val_t;\n"
                            "kernel void gather_vals_to_send(\n"
                            "    " << type_name<size_t>() << " n,\n"
                            "    global const val_t *vals,\n"
                            "    global const " << type_name<col_t>() << " *cols_to_send,\n"
                            "    global val_t *vals_to_send\n"
                            "    )\n"
                            "{\n"
                            "    for(size_t i = get_global_id(0); i < n; i += get_global_size(0))\n"
                            "        vals_to_send[i] = vals[cols_to_send[i]];\n"
                            "}\n";

                        backend::kernel krn(queue[d], source.str(), "gather_vals_to_send");
                        gather = cache.insert(std::make_pair(context(), krn)).first;
                    }

                    if (size_t ncols = cidx[d + 1] - cidx[d]) {
                        gather->second.push_arg(ncols);
                        gather->second.push_arg(x(d));
                        gather->second.push_arg(exc[d].cols_to_send);
                        gather->second.push_arg(exc[d].vals_to_send);

                        gather->second(queue[d]);

                        squeue[d].enqueueReadBuffer(exc[d].vals_to_send, CL_FALSE,
                                0, ncols * sizeof(val_t), &rx[cidx[d]], &event1[d], &event2[d][0]
                                );
                    }
                }
            }

            // Compute contribution from local part of the matrix.
            for(unsigned d = 0; d < queue.size(); d++)
                if (mtx[d]) mtx[d]->mul_local(x(d), y(d), alpha, append);

            // Compute contribution from remote part of the matrix.
            if (rx.size()) {
                for(unsigned d = 0; d < queue.size(); d++)
                    if (cidx[d + 1] > cidx[d]) event2[d][0].wait();

                for(unsigned d = 0; d < queue.size(); d++) {
                    cl::Context context = qctx(queue[d]);

                    if (exc[d].cols_to_recv.size()) {
                        for(size_t i = 0; i < exc[d].cols_to_recv.size(); i++)
                            exc[d].vals_to_recv[i] = rx[exc[d].cols_to_recv[i]];

                        squeue[d].enqueueWriteBuffer(
                                exc[d].rx, CL_FALSE, 0, bytes(exc[d].vals_to_recv),
                                exc[d].vals_to_recv.data(), 0, &event2[d][0]
                                );

                        mtx[d]->mul_remote(exc[d].rx, y(d), alpha, event2[d]);
                    }
                }
            }
        }

        /// Number of rows.
        size_t rows() const { return nrows; }
        /// Number of columns.
        size_t cols() const { return ncols; }
        /// Number of non-zero entries.
        size_t nonzeros() const { return nnz;   }

        static std::string inline_preamble(
                const cl::Device &device, const std::string &prm_name,
                detail::kernel_generator_state_ptr)
        {
            if (is_cpu(device))
                return SpMatCSR::inline_preamble(prm_name);
            else
                return SpMatHELL::inline_preamble(prm_name);
        }

        static std::string inline_expression(
                const cl::Device &device, const std::string &prm_name,
                detail::kernel_generator_state_ptr)
        {
            if (is_cpu(device))
                return SpMatCSR::inline_expression(prm_name);
            else
                return SpMatHELL::inline_expression(prm_name);
        }

        static std::string inline_parameters(
                const cl::Device &device, const std::string &prm_name,
                detail::kernel_generator_state_ptr)
        {
            if (is_cpu(device))
                return SpMatCSR::inline_parameters(prm_name);
            else
                return SpMatHELL::inline_parameters(prm_name);
        }

        static void inline_arguments(backend::kernel &kernel, unsigned device,
                size_t /*index_offset*/, const SpMat &A, const vector<val_t> &x,
                detail::kernel_generator_state_ptr)
        {
            A.mtx[device]->setArgs(kernel, device, x);
        }
    private:
        template <typename T>
        static inline size_t bytes(const std::vector<T> &v) {
            return v.size() * sizeof(T);
        }

        struct sparse_matrix {
            virtual void mul_local(
                    const cl::Buffer &x, const cl::Buffer &y,
                    scalar_type alpha, bool append
                    ) const = 0;

            virtual void mul_remote(
                    const cl::Buffer &x, const cl::Buffer &y,
                    scalar_type alpha, const std::vector<cl::Event> &event
                    ) const = 0;

            virtual void setArgs(backend::kernel &kernel, unsigned device, const vector<val_t> &x) const = 0;

            virtual ~sparse_matrix() {}
        };

#include <vexcl/spmat/hybrid_ell.inl>
#include <vexcl/spmat/csr.inl>

        struct exdata {
            std::vector<col_t> cols_to_recv;
            mutable std::vector<val_t> vals_to_recv;

            cl::Buffer cols_to_send;
            cl::Buffer vals_to_send;
            mutable cl::Buffer rx;
        };

        const std::vector<cl::CommandQueue> queue;
        std::vector<cl::CommandQueue>       squeue;
        const std::vector<size_t>           part;

        mutable std::vector<std::vector<cl::Event>> event1;
        mutable std::vector<std::vector<cl::Event>> event2;

        std::vector< std::unique_ptr<sparse_matrix> > mtx;

        std::vector<exdata> exc;
        std::vector<size_t> cidx;
        mutable std::vector<val_t> rx;

        size_t nrows;
        size_t ncols;
        size_t nnz;

        std::vector<std::set<col_t>> setup_exchange(
                const std::vector<size_t> &col_part,
                const idx_t *row, const col_t *col
                )
        {
            auto is_local = [col_part](size_t c, int device) {
                return c >= col_part[device] && c < col_part[device + 1];
            };

            std::vector<std::set<col_t>> ghost_cols(queue.size());

            if (queue.size() <= 1) return ghost_cols;

            // Build sets of ghost points.
#ifdef _OPENMP
#  pragma omp parallel for schedule(static,1)
#endif
            for(int d = 0; d < static_cast<int>(queue.size()); d++) {
                for(size_t i = part[d]; i < part[d + 1]; i++) {
                    for(idx_t j = row[i]; j < row[i + 1]; j++) {
                        if (!is_local(col[j], d)) {
                            ghost_cols[d].insert(col[j]);
                        }
                    }
                }
            }

            // Complete set of points to be exchanged between devices.
            std::vector<col_t> cols_to_send;
            {
                std::set<col_t> cols_to_send_s;
                for(unsigned d = 0; d < queue.size(); d++)
                    cols_to_send_s.insert(ghost_cols[d].begin(), ghost_cols[d].end());

                cols_to_send.insert(cols_to_send.begin(), cols_to_send_s.begin(), cols_to_send_s.end());
            }

            // Build local structures to facilitate exchange.
            if (cols_to_send.size()) {
#ifdef _OPENMP
#  pragma omp parallel for schedule(static,1)
#endif
                for(int d = 0; d < static_cast<int>(queue.size()); d++) {
                    if (size_t rcols = ghost_cols[d].size()) {
                        exc[d].cols_to_recv.resize(rcols);
                        exc[d].vals_to_recv.resize(rcols);

                        exc[d].rx = cl::Buffer(qctx(queue[d]), CL_MEM_READ_ONLY, rcols * sizeof(val_t));

                        for(size_t i = 0, j = 0; i < cols_to_send.size(); i++)
                            if (ghost_cols[d].count(cols_to_send[i]))
                                exc[d].cols_to_recv[j++] = static_cast<col_t>(i);
                    }
                }

                rx.resize(cols_to_send.size());
                cidx.resize(queue.size() + 1);

                {
                    auto beg = cols_to_send.begin();
                    auto end = cols_to_send.end();
                    for(unsigned d = 0; d <= queue.size(); d++) {
                        cidx[d] = std::lower_bound(beg, end, static_cast<col_t>(col_part[d]))
                                - cols_to_send.begin();
                        beg = cols_to_send.begin() + cidx[d];
                    }
                }

                for(unsigned d = 0; d < queue.size(); d++) {
                    if (size_t ncols = cidx[d + 1] - cidx[d]) {
                        cl::Context context = qctx(queue[d]);

                        exc[d].cols_to_send = cl::Buffer(
                                context, CL_MEM_READ_ONLY, ncols * sizeof(col_t));

                        exc[d].vals_to_send = cl::Buffer(
                                context, CL_MEM_READ_WRITE, ncols * sizeof(val_t));

                        for(size_t i = cidx[d]; i < cidx[d + 1]; i++)
                            cols_to_send[i] -= static_cast<col_t>(col_part[d]);

                        queue[d].enqueueWriteBuffer(
                                exc[d].cols_to_send, CL_TRUE, 0, ncols * sizeof(col_t),
                                &cols_to_send[cidx[d]]);
                    }
                }
            }

            return ghost_cols;
        }
};

/// \cond INTERNAL

template <typename val_t, typename col_t, typename idx_t>
struct spmv
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    typedef val_t                      value_type;
    typedef SpMat<val_t, col_t, idx_t> mat;
    typedef vector<val_t>              vec;

    const mat &A;
    const vec &x;

    typename cl_scalar_of<val_t>::type scale;

    spmv(const mat &A, const vec &x) : A(A), x(x), scale(1) {}

    template<bool negate, bool append>
    void apply(vec &y) const {
        A.mul(x, y, negate ? -scale : scale, append);
    }
};

template <typename val_t, typename col_t, typename idx_t>
spmv< val_t, col_t, idx_t > operator*(const SpMat<val_t, col_t, idx_t> &A, const vector<val_t> &x)
{
    return spmv<val_t, col_t, idx_t>(A, x);
}

namespace traits {

template <typename val_t, typename col_t, typename idx_t>
struct is_scalable< spmv<val_t, col_t, idx_t> > : std::true_type {};

} // namespace traits

#ifdef VEXCL_MULTIVECTOR_HPP

template <typename val_t, typename col_t, typename idx_t, class MV>
struct multispmv
    : multivector_expression<
        boost::proto::terminal< additive_multivector_transform >::type
        >
{
    typedef val_t                      value_type;
    typedef SpMat<val_t, col_t, idx_t> mat;

    const mat &A;
    const MV  &x;

    typename cl_scalar_of<val_t>::type scale;

    multispmv(const mat &A, const MV &x) : A(A), x(x), scale(1) {}

    template <bool negate, bool append, class W>
    typename std::enable_if<
        std::is_base_of<multivector_terminal_expression, W>::value
        && std::is_same<val_t, typename W::sub_value_type>::value
        && traits::number_of_components<MV>::value == traits::number_of_components<W>::value,
        void
    >::type
    apply(W &y) const {
        for(size_t i = 0; i < traits::number_of_components<MV>::value; i++)
            A.mul(x(i), y(i), negate ? -scale : scale, append);
    }
};

template <typename val_t, typename col_t, typename idx_t, class MV>
typename std::enable_if<
    std::is_base_of<multivector_terminal_expression, MV>::value &&
    std::is_same<val_t, typename MV::sub_value_type>::value,
    multispmv< val_t, col_t, idx_t, MV >
>::type
operator*(const SpMat<val_t, col_t, idx_t> &A, const MV &x) {
    return multispmv< val_t, col_t, idx_t, MV >(A, x);
}

namespace traits {

template <typename val_t, typename col_t, typename idx_t, class MV>
struct is_scalable< multispmv<val_t, col_t, idx_t, MV> > : std::true_type {};

} // namespace traits

#endif

/// \endcond

/// Weights device wrt to spmv performance.
/**
 * Launches the following kernel on each device:
 * \code
 * y = A * x;
 * \endcode
 * where x and y are vectors, and A is matrix for 3D Poisson problem in square
 * domain. Each device gets portion of the vector proportional to the
 * performance of this operation.
 */
inline double device_spmv_perf(const cl::CommandQueue &q) {
    static const size_t test_size = 64U;

    std::vector<cl::CommandQueue> queue(1, q);

    // Construct matrix for 3D Poisson problem in cubic domain.
    const size_t n   = test_size;
    const float  h2i = (n - 1.0f) * (n - 1.0f);

    std::vector<size_t> row;
    std::vector<size_t> col;
    std::vector<float>  val;

    row.reserve(n * n * n + 1);
    col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
    val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

    row.push_back(0);
    for(size_t k = 0, idx = 0; k < n; k++) {
        for(size_t j = 0; j < n; j++) {
            for(size_t i = 0; i < n; i++, idx++) {
                if (
                        i == 0 || i == (n - 1) ||
                        j == 0 || j == (n - 1) ||
                        k == 0 || k == (n - 1)
                   )
                {
                    col.push_back(idx);
                    val.push_back(1);
                    row.push_back(row.back() + 1);
                } else {
                    col.push_back(idx - n * n);
                    val.push_back(-h2i);

                    col.push_back(idx - n);
                    val.push_back(-h2i);

                    col.push_back(idx - 1);
                    val.push_back(-h2i);

                    col.push_back(idx);
                    val.push_back(6 * h2i);

                    col.push_back(idx + 1);
                    val.push_back(-h2i);

                    col.push_back(idx + n);
                    val.push_back(-h2i);

                    col.push_back(idx + n * n);
                    val.push_back(-h2i);

                    row.push_back(row.back() + 7);
                }
            }
        }
    }

    // Create device vectors and copy of the matrix.
    size_t n3 = n * n * n;
    vex::SpMat<float>  A(queue, n3, n3, row.data(), col.data(), val.data());
    vex::vector<float> x(queue, n3);
    vex::vector<float> y(queue, n3);

    // Warming run.
    x = 1;
    A.mul(x, y);

    // Measure performance.
    profiler<> prof(queue);
    prof.tic_cl("");
    A.mul(x, y);
    double time = prof.toc("");
    return 1.0 / time;
}

} // namespace vex

#include <vexcl/spmat/ccsr.hpp>
#include <vexcl/spmat/inline_spmv.hpp>

#endif
