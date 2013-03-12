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

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290 4800)
#  define NOMINMAX
#endif

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vexcl/vector.hpp>

namespace vex {


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
inline double device_spmv_perf(const cl::CommandQueue&);

/// \cond INTERNAL

/// Return size of std::vector in bytes.
template <typename T>
size_t bytes(const std::vector<T> &x) {
    return x.size() * sizeof(T);
}

struct matrix_terminal {};

template <class M, class V>
struct spmv
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    typedef typename M::value_type value_type;

    const M &A;
    const V &x;

    value_type scale;

    spmv(const M &m, const V &v) : A(m), x(v), scale(1) {}

    template<bool negate, bool append>
    void apply(V &y) const {
        A.mul(x, y, negate ? -scale : scale, append);
    }
};

template <class M, class V>
typename std::enable_if<
    std::is_base_of<matrix_terminal, M>::value &&
    std::is_base_of<vector_terminal_expression, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type>::value,
    spmv< M, V >
>::type
operator*(const M &A, const V &x) {
    return spmv< M, V >(A, x);
}

template <class M, class V>
struct is_scalable< spmv<M, V> > : std::true_type {};

#ifdef VEXCL_MULTIVECTOR_HPP

template <class M, class V>
struct multispmv
    : multivector_expression<
        boost::proto::terminal< additive_multivector_transform >::type
        >
{
    typedef typename M::value_type value_type;

    const M &A;
    const V &x;

    value_type scale;

    multispmv(const M &m, const V &v) : A(m), x(v), scale(1) {}

    template <bool negate, bool append, class W>
    typename std::enable_if<
        std::is_base_of<multivector_terminal_expression, W>::value
#ifndef WIN32
        && std::is_same<value_type, typename W::value_type::value_type>::value
#endif
        && number_of_components<V>::value == number_of_components<W>::value,
        void
    >::type
    apply(W &y) const {
        for(size_t i = 0; i < number_of_components<V>::value; i++)
            A.mul(x(i), y(i), negate ? -scale : scale, append);
    }
};

template <class M, class V>
typename std::enable_if<
    std::is_base_of<matrix_terminal,      M>::value &&
    std::is_base_of<multivector_terminal_expression, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type::value_type>::value,
    multispmv< M, V >
>::type
operator*(const M &A, const V &x) {
    return multispmv< M, V >(A, x);
}

template <class M, class V>
struct is_scalable< multispmv<M, V> > : std::true_type {};

#endif

/// \endcond

/// Sparse matrix in hybrid ELL-CSR format.
template <typename real, typename column_t = size_t, typename idx_t = size_t>
class SpMat : matrix_terminal {
    public:
        typedef real value_type;

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
              size_t n, size_t m, const idx_t *row, const column_t *col, const real *val
              );

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
        void mul(const vex::vector<real> &x, vex::vector<real> &y,
                 real alpha = 1, bool append = false) const;

        /// Number of rows.
        size_t rows() const { return nrows; }
        /// Number of columns.
        size_t cols() const { return ncols; }
        /// Number of non-zero entries.
        size_t nonzeros() const { return nnz;   }
    private:
        struct sparse_matrix {
            virtual void mul_local(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, bool append
                    ) const = 0;

            virtual void mul_remote(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, const std::vector<cl::Event> &event
                    ) const = 0;

            virtual ~sparse_matrix() {}
        };

        struct SpMatELL : sparse_matrix {
            static const column_t ncol = -1;

            SpMatELL(
                    const cl::CommandQueue &queue,
                    size_t beg, size_t end, column_t xbeg, column_t xend,
                    const idx_t *row, const column_t *col, const real *val,
                    const std::set<column_t> &remote_cols
                    );

            void prepare_kernels(const cl::Context &context) const;

            void mul_local(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, bool append
                    ) const;

            void mul_remote(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, const std::vector<cl::Event> &event
                    ) const;

            const cl::CommandQueue &queue;

            size_t n, pitch;

            struct {
                uint w;
                cl::Buffer col;
                cl::Buffer val;
            } loc_ell, rem_ell;

            struct {
                size_t n;
                cl::Buffer idx;
                cl::Buffer row;
                cl::Buffer col;
                cl::Buffer val;
            } loc_csr, rem_csr;

            static std::map<cl_context, bool>       compiled;
            static std::map<cl_context, cl::Kernel> zero;
            static std::map<cl_context, cl::Kernel> spmv_set;
            static std::map<cl_context, cl::Kernel> spmv_add;
            static std::map<cl_context, cl::Kernel> csr_add;
            static std::map<cl_context, uint>       wgsize;
        };

        struct SpMatCSR : public sparse_matrix {
            SpMatCSR(
                    const cl::CommandQueue &queue,
                    size_t beg, size_t end, column_t xbeg, column_t xend,
                    const idx_t *row, const column_t *col, const real *val,
                    const std::set<column_t> &remote_cols
                    );

            void prepare_kernels(const cl::Context &context) const;

            void mul_local(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, bool append
                    ) const;

            void mul_remote(
                    const cl::Buffer &x, const cl::Buffer &y,
                    real alpha, const std::vector<cl::Event> &event
                    ) const;

            const cl::CommandQueue &queue;

            size_t n;

            bool has_loc;
            bool has_rem;

            struct {
                cl::Buffer row;
                cl::Buffer col;
                cl::Buffer val;
            } loc, rem;

            static std::map<cl_context, bool>       compiled;
            static std::map<cl_context, cl::Kernel> zero;
            static std::map<cl_context, cl::Kernel> spmv_set;
            static std::map<cl_context, cl::Kernel> spmv_add;
            static std::map<cl_context, uint>       wgsize;
        };

        struct exdata {
            std::vector<column_t> cols_to_recv;
            mutable std::vector<real> vals_to_recv;

            cl::Buffer cols_to_send;
            cl::Buffer vals_to_send;
            mutable cl::Buffer rx;
        };

        const std::vector<cl::CommandQueue> queue;
        std::vector<cl::CommandQueue>       squeue;
        const std::vector<size_t>           part;

        mutable std::vector<std::vector<cl::Event>> event1;
        mutable std::vector<std::vector<cl::Event>> event2;

        std::vector<std::unique_ptr<sparse_matrix>> mtx;

        std::vector<exdata> exc;
        std::vector<size_t> cidx;
        mutable std::vector<real> rx;

        size_t nrows;
        size_t ncols;
        size_t nnz;


        static std::map<cl_context, bool>       compiled;
        static std::map<cl_context, cl::Kernel> gather_vals_to_send;
        static std::map<cl_context, uint>       wgsize;

        std::vector<std::set<column_t>> setup_exchange(
                size_t n, const std::vector<size_t> &xpart,
                const idx_t *row, const column_t *col, const real *val
                );
};

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, bool> SpMat<real,column_t,idx_t>::compiled;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::gather_vals_to_send;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, uint> SpMat<real,column_t,idx_t>::wgsize;

template <typename real, typename column_t, typename idx_t>
SpMat<real,column_t,idx_t>::SpMat(
        const std::vector<cl::CommandQueue> &queue,
        size_t n, size_t m, const idx_t *row, const column_t *col, const real *val
        )
    : queue(queue), part(partition(n, queue)),
      event1(queue.size(), std::vector<cl::Event>(1)),
      event2(queue.size(), std::vector<cl::Event>(1)),
      mtx(queue.size()), exc(queue.size()),
      nrows(n), ncols(m), nnz(row[n])
{
    auto xpart = partition(m, queue);

    for(auto q = queue.begin(); q != queue.end(); q++) {
        cl::Context context = qctx(*q);
        cl::Device  device  = qdev(*q);

        // Compile kernels.
        if (!compiled[context()]) {
            std::ostringstream source;

            source << standard_kernel_header <<
                "typedef " << type_name<real>() << " real;\n"
                "kernel void gather_vals_to_send(\n"
                "    " << type_name<size_t>() << " n,\n"
                "    global const real *vals,\n"
                "    global const " << type_name<column_t>() << " *cols_to_send,\n"
                "    global real *vals_to_send\n"
                "    )\n"
                "{\n"
                "    size_t i = get_global_id(0);\n"
                "    if (i < n) vals_to_send[i] = vals[cols_to_send[i]];\n"
                "}\n";

            auto program = build_sources(context, source.str());

            gather_vals_to_send[context()] = cl::Kernel(program, "gather_vals_to_send");

            wgsize[context()] = kernel_workgroup_size(
                    gather_vals_to_send[context()], device
                    );

            compiled[context()] = true;
        }

        // Create secondary queues.
        squeue.push_back(cl::CommandQueue(context, device));
    }

    std::vector<std::set<column_t>> remote_cols = setup_exchange(n, xpart, row, col, val);

    // Each device get it's own strip of the matrix.
#pragma omp parallel for schedule(static,1)
    for(int d = 0; d < static_cast<int>(queue.size()); d++) {
        if (part[d + 1] > part[d]) {
            cl::Device device = qdev(queue[d]);

            if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
                mtx[d].reset(
                        new SpMatCSR(queue[d],
                            part[d], part[d + 1],
                            xpart[d], xpart[d + 1],
                            row, col, val, remote_cols[d])
                        );
            else
                mtx[d].reset(
                        new SpMatELL(queue[d],
                            part[d], part[d + 1],
                            xpart[d], xpart[d + 1],
                            row, col, val, remote_cols[d])
                        );
        }
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::mul(const vex::vector<real> &x, vex::vector<real> &y,
        real alpha, bool append) const
{
    if (rx.size()) {
        // Transfer remote parts of the input vector.
        for(uint d = 0; d < queue.size(); d++) {
            cl::Context context = qctx(queue[d]);

            if (size_t ncols = cidx[d + 1] - cidx[d]) {
                size_t g_size = alignup(ncols, wgsize[context()]);

                uint pos = 0;
                gather_vals_to_send[context()].setArg(pos++, ncols);
                gather_vals_to_send[context()].setArg(pos++, x(d));
                gather_vals_to_send[context()].setArg(pos++, exc[d].cols_to_send);
                gather_vals_to_send[context()].setArg(pos++, exc[d].vals_to_send);

                queue[d].enqueueNDRangeKernel(gather_vals_to_send[context()],
                        cl::NullRange, g_size, wgsize[context()], 0, &event1[d][0]);

                squeue[d].enqueueReadBuffer(exc[d].vals_to_send, CL_FALSE,
                        0, ncols * sizeof(real), &rx[cidx[d]], &event1[d], &event2[d][0]
                        );
            }
        }
    }

    // Compute contribution from local part of the matrix.
    for(uint d = 0; d < queue.size(); d++)
        if (mtx[d]) mtx[d]->mul_local(x(d), y(d), alpha, append);

    // Compute contribution from remote part of the matrix.
    if (rx.size()) {
        for(uint d = 0; d < queue.size(); d++)
            if (cidx[d + 1] > cidx[d]) event2[d][0].wait();

        for(uint d = 0; d < queue.size(); d++) {
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

template <typename real, typename column_t, typename idx_t>
std::vector<std::set<column_t>> SpMat<real,column_t,idx_t>::setup_exchange(
        size_t, const std::vector<size_t> &xpart,
        const idx_t *row, const column_t *col, const real *
        )
{
    std::vector<std::set<column_t>> remote_cols(queue.size());

    if (queue.size() <= 1) return remote_cols;

    // Build sets of ghost points.
#pragma omp parallel for schedule(static,1)
    for(int d = 0; d < static_cast<int>(queue.size()); d++) {
        for(size_t i = part[d]; i < part[d + 1]; i++) {
            for(idx_t j = row[i]; j < row[i + 1]; j++) {
                if (col[j] < static_cast<column_t>(xpart[d]) || col[j] >= static_cast<column_t>(xpart[d + 1])) {
                    remote_cols[d].insert(col[j]);
                }
            }
        }
    }

    // Complete set of points to be exchanged between devices.
    std::vector<column_t> cols_to_send;
    {
        std::set<column_t> cols_to_send_s;
        for(uint d = 0; d < queue.size(); d++)
            cols_to_send_s.insert(remote_cols[d].begin(), remote_cols[d].end());

        cols_to_send.insert(cols_to_send.begin(), cols_to_send_s.begin(), cols_to_send_s.end());
    }

    // Build local structures to facilitate exchange.
    if (cols_to_send.size()) {
#pragma omp parallel for schedule(static,1)
        for(int d = 0; d < static_cast<int>(queue.size()); d++) {
            if (size_t rcols = remote_cols[d].size()) {
                exc[d].cols_to_recv.resize(rcols);
                exc[d].vals_to_recv.resize(rcols);

                exc[d].rx = cl::Buffer(qctx(queue[d]), CL_MEM_READ_ONLY, rcols * sizeof(real));

                for(size_t i = 0, j = 0; i < cols_to_send.size(); i++)
                    if (remote_cols[d].count(cols_to_send[i])) exc[d].cols_to_recv[j++] = i;
            }
        }

        rx.resize(cols_to_send.size());
        cidx.resize(queue.size() + 1);

        {
            auto beg = cols_to_send.begin();
            auto end = cols_to_send.end();
            for(uint d = 0; d <= queue.size(); d++) {
                cidx[d] = std::lower_bound(beg, end, xpart[d]) - cols_to_send.begin();
                beg = cols_to_send.begin() + cidx[d];
            }
        }

        for(uint d = 0; d < queue.size(); d++) {
            if (size_t ncols = cidx[d + 1] - cidx[d]) {
                cl::Context context = qctx(queue[d]);

                exc[d].cols_to_send = cl::Buffer(
                        context, CL_MEM_READ_ONLY, ncols * sizeof(column_t));

                exc[d].vals_to_send = cl::Buffer(
                        context, CL_MEM_READ_WRITE, ncols * sizeof(real));

                for(size_t i = cidx[d]; i < cidx[d + 1]; i++)
                    cols_to_send[i] -= xpart[d];

                queue[d].enqueueWriteBuffer(
                        exc[d].cols_to_send, CL_TRUE, 0, ncols * sizeof(column_t),
                        &cols_to_send[cidx[d]]);
            }
        }
    }

    return remote_cols;
}

//---------------------------------------------------------------------------
// SpMat::SpMatELL
//---------------------------------------------------------------------------
template <typename real, typename column_t, typename idx_t>
const column_t SpMat<real,column_t,idx_t>::SpMatELL::ncol;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, bool> SpMat<real,column_t,idx_t>::SpMatELL::compiled;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatELL::zero;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatELL::spmv_set;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatELL::spmv_add;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatELL::csr_add;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, uint> SpMat<real,column_t,idx_t>::SpMatELL::wgsize;

template <typename real, typename column_t, typename idx_t>
SpMat<real,column_t,idx_t>::SpMatELL::SpMatELL(
        const cl::CommandQueue &queue,
        size_t beg, size_t end, column_t xbeg, column_t xend,
        const idx_t *row, const column_t *col, const real *val,
        const std::set<column_t> &remote_cols
        )
    : queue(queue), n(end - beg), pitch(alignup(n, 16U))
{
    cl::Context context = qctx(queue);

    prepare_kernels(context);

    // Get optimal ELL widths for local and remote parts.
    {
        // Speed of ELL relative to CSR (e.g. 2.0 -> ELL is twice as fast):
        static const double ell_vs_csr = 3.0;

        // Get max widths for remote and local parts.
        loc_ell.w = rem_ell.w = 0;
        for(size_t i = beg; i < end; i++) {
            uint w = 0;
            for(idx_t j = row[i]; j < row[i + 1]; j++)
                if (col[j] >= xbeg && col[j] < xend) w++;

            loc_ell.w = std::max(loc_ell.w, w);
            rem_ell.w = std::max<uint>(rem_ell.w, row[i + 1] - row[i] - w);
        }

        // Build histograms for width distribution.
        std::vector<size_t> loc_hist(loc_ell.w + 1, 0);
        std::vector<size_t> rem_hist(rem_ell.w + 1, 0);

        for(size_t i = beg; i < end; i++) {
            uint w = 0;
            for(idx_t j = row[i]; j < row[i + 1]; j++)
                if (col[j] >= xbeg && col[j] < xend) w++;

            loc_hist[w]++;
            rem_hist[row[i + 1] - row[i] - w]++;
        }

        // Find optimal width for local part.
        {
            for(size_t i = 0, nrows = end - beg, rows = nrows; i < loc_ell.w; i++) {
                rows -= loc_hist[i]; // Number of rows wider than i.
                if (ell_vs_csr * rows < nrows) {
                    loc_ell.w = i;
                    break;
                }
            }
        }

        // Find optimal width for remote part.
        {
            for(size_t i = 0, nrows = end - beg, rows = nrows; i < rem_ell.w; i++) {
                rows -= rem_hist[i]; // Number of rows wider than i.
                if (ell_vs_csr * rows < nrows) {
                    rem_ell.w = i;
                    break;
                }
            }
        }
    }

    // Count nonzeros in COO parts of the matrix.
    loc_csr.n = rem_csr.n = 0;
    size_t loc_nnz = 0, rem_nnz = 0;
    for(size_t i = beg; i < end; i++) {
        uint w = 0;
        for(idx_t j = row[i]; j < row[i + 1]; j++)
            if (col[j] >= xbeg && col[j] < xend) w++;

        if (w > loc_ell.w) {
            loc_csr.n++;
            loc_nnz += w - loc_ell.w;
        }
        if (row[i + 1] - row[i] - w > rem_ell.w) {
            rem_csr.n++;
            rem_nnz += row[i + 1] - row[i] - w - rem_ell.w;
        }
    }


    // Renumber columns.
    std::unordered_map<column_t,column_t> r2l(2 * remote_cols.size());
    for(auto c = remote_cols.begin(); c != remote_cols.end(); c++) {
        size_t idx = r2l.size();
        r2l[*c] = idx;
    }

    // Prepare ELL and COO formats for transfer to devices.
    std::vector<column_t> lell_col(pitch * loc_ell.w, ncol);
    std::vector<real>     lell_val(pitch * loc_ell.w, 0);
    std::vector<column_t> rell_col(pitch * rem_ell.w, ncol);
    std::vector<real>     rell_val(pitch * rem_ell.w, 0);

    std::vector<idx_t>    lcsr_idx;
    std::vector<column_t> lcsr_row;
    std::vector<column_t> lcsr_col;
    std::vector<real>     lcsr_val;

    lcsr_idx.reserve(loc_csr.n + 1);
    lcsr_row.reserve(loc_csr.n);
    lcsr_col.reserve(loc_nnz);
    lcsr_val.reserve(loc_nnz);

    std::vector<idx_t>    rcsr_idx;
    std::vector<column_t> rcsr_row;
    std::vector<column_t> rcsr_col;
    std::vector<real>     rcsr_val;

    rcsr_idx.reserve(rem_csr.n + 1);
    rcsr_row.reserve(rem_csr.n);
    rcsr_col.reserve(rem_nnz);
    rcsr_val.reserve(rem_nnz);

    lcsr_idx.push_back(0);
    rcsr_idx.push_back(0);

    for(size_t i = beg, k = 0; i < end; i++, k++) {
        size_t lc = 0, rc = 0;
        for(idx_t j = row[i]; j < row[i + 1]; j++) {
            if (col[j] >= xbeg && col[j] < xend) {
                if (lc < loc_ell.w) {
                    lell_col[k + pitch * lc] = col[j] - xbeg;
                    lell_val[k + pitch * lc] = val[j];
                    lc++;
                } else {
                    lcsr_col.push_back(col[j] - xbeg);
                    lcsr_val.push_back(val[j]);
                }
            } else {
                assert(r2l.count(col[j]));
                if (rc < rem_ell.w) {
                    rell_col[k + pitch * rc] = r2l[col[j]];
                    rell_val[k + pitch * rc] = val[j];
                    rc++;
                } else {
                    rcsr_col.push_back(r2l[col[j]]);
                    rcsr_val.push_back(val[j]);
                }
            }
        }
        if (lcsr_col.size() > static_cast<size_t>(lcsr_idx.back())) {
            lcsr_row.push_back(i - beg);
            lcsr_idx.push_back(lcsr_col.size());
        }
        if (rcsr_col.size() > static_cast<size_t>(rcsr_idx.back())) {
            rcsr_row.push_back(i - beg);
            rcsr_idx.push_back(rcsr_col.size());
        }
    }

    cl::Event event;

    // Copy local part to the device.
    if (loc_ell.w) {
        loc_ell.col = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lell_col));
        loc_ell.val = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lell_val));

        queue.enqueueWriteBuffer(loc_ell.col, CL_FALSE, 0,
                bytes(lell_col), lell_col.data());

        queue.enqueueWriteBuffer(loc_ell.val, CL_FALSE, 0,
                bytes(lell_val), lell_val.data(), 0, &event);
    }

    if (loc_csr.n) {
        loc_csr.idx = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lcsr_idx));
        loc_csr.row = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lcsr_row));
        loc_csr.col = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lcsr_col));
        loc_csr.val = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(lcsr_val));

        queue.enqueueWriteBuffer(loc_csr.idx, CL_FALSE, 0,
                bytes(lcsr_idx), lcsr_idx.data());

        queue.enqueueWriteBuffer(loc_csr.row, CL_FALSE, 0,
                bytes(lcsr_row), lcsr_row.data());

        queue.enqueueWriteBuffer(loc_csr.col, CL_FALSE, 0,
                bytes(lcsr_col), lcsr_col.data());

        queue.enqueueWriteBuffer(loc_csr.val, CL_FALSE, 0,
                bytes(lcsr_val), lcsr_val.data(), 0, &event);
    }

    // Copy remote part to the device.
    if (rem_ell.w) {
        rem_ell.col = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rell_col));
        rem_ell.val = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rell_val));

        queue.enqueueWriteBuffer(rem_ell.col, CL_FALSE, 0,
                bytes(rell_col), rell_col.data());

        queue.enqueueWriteBuffer(rem_ell.val, CL_FALSE, 0,
                bytes(rell_val), rell_val.data(), 0, &event);
    }

    if (rem_csr.n) {
        rem_csr.idx = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rcsr_idx));
        rem_csr.row = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rcsr_row));
        rem_csr.col = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rcsr_col));
        rem_csr.val = cl::Buffer(context, CL_MEM_READ_ONLY, bytes(rcsr_val));

        queue.enqueueWriteBuffer(rem_csr.idx, CL_FALSE, 0,
                bytes(rcsr_idx), rcsr_idx.data());

        queue.enqueueWriteBuffer(rem_csr.row, CL_FALSE, 0,
                bytes(rcsr_row), rcsr_row.data());

        queue.enqueueWriteBuffer(rem_csr.col, CL_FALSE, 0,
                bytes(rcsr_col), rcsr_col.data());

        queue.enqueueWriteBuffer(rem_csr.val, CL_FALSE, 0,
                bytes(rcsr_val), rcsr_val.data(), 0, &event);
    }

    // Wait for data to be copied before it gets deallocated.
    if (loc_ell.w || loc_csr.n || rem_ell.w || rem_csr.n) event.wait();
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatELL::prepare_kernels(const cl::Context &context) const {
    if (!compiled[context()]) {
        std::ostringstream source;

        source << standard_kernel_header <<
            "typedef " << type_name<real>() << " real;\n"
            "#define NCOL ((" << type_name<column_t>() << ")(-1))\n"
            "kernel void zero(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global real *y\n"
            "    )\n"
            "{\n"
            "    size_t grid_size = get_global_size(0);\n"
            "    for (size_t row = get_global_id(0); row < n; row += grid_size)\n"
            "        y[row] = 0;\n"
            "}\n"
            "kernel void spmv_set(\n"
            "    " << type_name<size_t>() << " n, uint w, " << type_name<size_t>() << " pitch,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t grid_size = get_global_size(0);\n"
            "    for (size_t row = get_global_id(0); row < n; row += grid_size) {\n"
            "        real sum = 0;\n"
            "        for(size_t j = 0; j < w; j++) {\n"
            "            " << type_name<column_t>() << " c = col[row + j * pitch];\n"
            "            if (c != NCOL) sum += val[row + j * pitch] * x[c];\n"
            "        }\n"
            "        y[row] = alpha * sum;\n"
            "    }\n"
            "}\n"
            "kernel void spmv_add(\n"
            "    " << type_name<size_t>() << " n, uint w, " << type_name<size_t>() << " pitch,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t grid_size = get_global_size(0);\n"
            "    for(size_t row = get_global_id(0); row < n; row += grid_size) {\n"
            "        real sum = 0;\n"
            "        for(size_t j = 0; j < w; j++) {\n"
            "            " << type_name<column_t>() << " c = col[row + j * pitch];\n"
            "            if (c != NCOL) sum += val[row + j * pitch] * x[c];\n"
            "        }\n"
            "        y[row] += alpha * sum;\n"
            "    }\n"
            "}\n"
            "kernel void csr_add(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global const " << type_name<idx_t>() << " *idx,\n"
            "    global const " << type_name<column_t>() << " *row,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t grid_size = get_global_size(0);\n"
            "    for (size_t i = get_global_id(0); i < n; i += grid_size) {\n"
            "        real sum = 0;\n"
            "        size_t beg = idx[i];\n"
            "        size_t end = idx[i + 1];\n"
            "        for(size_t j = beg; j < end; j++)\n"
            "            sum += val[j] * x[col[j]];\n"
            "        y[row[i]] += alpha * sum;\n"
            "    }\n"
            "}\n";

        auto program = build_sources(context, source.str());

        zero[context()]     = cl::Kernel(program, "zero");
        spmv_set[context()] = cl::Kernel(program, "spmv_set");
        spmv_add[context()] = cl::Kernel(program, "spmv_add");
        csr_add[context()]  = cl::Kernel(program, "csr_add");

        std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

        wgsize[context()] = std::min(
                kernel_workgroup_size(spmv_set[context()], device[0]),
                kernel_workgroup_size(spmv_add[context()], device[0])
                );

        wgsize[context()] = std::min<uint>(wgsize[context()],
                kernel_workgroup_size(csr_add[context()], device[0])
                );

        compiled[context()] = true;
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatELL::mul_local(
        const cl::Buffer &x, const cl::Buffer &y,
        real alpha, bool append
        ) const
{
    cl::Context context = qctx(queue);
    cl::Device  device  = qdev(queue);

    size_t g_size = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
        * wgsize[context()] * 4;

    if (loc_ell.w) {
        if (append) {
            uint pos = 0;
            spmv_add[context()].setArg(pos++, n);
            spmv_add[context()].setArg(pos++, loc_ell.w);
            spmv_add[context()].setArg(pos++, pitch);
            spmv_add[context()].setArg(pos++, loc_ell.col);
            spmv_add[context()].setArg(pos++, loc_ell.val);
            spmv_add[context()].setArg(pos++, x);
            spmv_add[context()].setArg(pos++, y);
            spmv_add[context()].setArg(pos++, alpha);

            queue.enqueueNDRangeKernel(spmv_add[context()],
                    cl::NullRange, g_size, wgsize[context()]);
        } else {
            uint pos = 0;
            spmv_set[context()].setArg(pos++, n);
            spmv_set[context()].setArg(pos++, loc_ell.w);
            spmv_set[context()].setArg(pos++, pitch);
            spmv_set[context()].setArg(pos++, loc_ell.col);
            spmv_set[context()].setArg(pos++, loc_ell.val);
            spmv_set[context()].setArg(pos++, x);
            spmv_set[context()].setArg(pos++, y);
            spmv_set[context()].setArg(pos++, alpha);

            queue.enqueueNDRangeKernel(spmv_set[context()],
                    cl::NullRange, g_size, wgsize[context()]);
        }
    } else if (!append) {
        uint pos = 0;
        zero[context()].setArg(pos++, n);
        zero[context()].setArg(pos++, y);

        queue.enqueueNDRangeKernel(zero[context()],
                cl::NullRange, g_size, wgsize[context()]);
    }

    if (loc_csr.n) {
        uint pos = 0;
        csr_add[context()].setArg(pos++, loc_csr.n);
        csr_add[context()].setArg(pos++, loc_csr.idx);
        csr_add[context()].setArg(pos++, loc_csr.row);
        csr_add[context()].setArg(pos++, loc_csr.col);
        csr_add[context()].setArg(pos++, loc_csr.val);
        csr_add[context()].setArg(pos++, x);
        csr_add[context()].setArg(pos++, y);
        csr_add[context()].setArg(pos++, alpha);

        queue.enqueueNDRangeKernel(csr_add[context()],
                cl::NullRange, g_size, wgsize[context()]);
    }
}
template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatELL::mul_remote(
        const cl::Buffer &x, const cl::Buffer &y,
        real alpha, const std::vector<cl::Event> &event
        ) const
{
    cl::Context context = qctx(queue);
    cl::Device  device  = qdev(queue);

    size_t g_size = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
        * wgsize[context()] * 4;

    if (rem_ell.w) {
        uint pos = 0;
        spmv_add[context()].setArg(pos++, n);
        spmv_add[context()].setArg(pos++, rem_ell.w);
        spmv_add[context()].setArg(pos++, pitch);
        spmv_add[context()].setArg(pos++, rem_ell.col);
        spmv_add[context()].setArg(pos++, rem_ell.val);
        spmv_add[context()].setArg(pos++, x);
        spmv_add[context()].setArg(pos++, y);
        spmv_add[context()].setArg(pos++, alpha);

        queue.enqueueNDRangeKernel(spmv_add[context()],
                cl::NullRange, g_size, wgsize[context()], &event
                );
    }

    if (rem_csr.n) {
        uint pos = 0;
        csr_add[context()].setArg(pos++, rem_csr.n);
        csr_add[context()].setArg(pos++, rem_csr.idx);
        csr_add[context()].setArg(pos++, rem_csr.row);
        csr_add[context()].setArg(pos++, rem_csr.col);
        csr_add[context()].setArg(pos++, rem_csr.val);
        csr_add[context()].setArg(pos++, x);
        csr_add[context()].setArg(pos++, y);
        csr_add[context()].setArg(pos++, alpha);

        queue.enqueueNDRangeKernel(csr_add[context()],
                cl::NullRange, g_size, wgsize[context()], &event);
    }
}

//---------------------------------------------------------------------------
// SpMat::SpMatCSR
//---------------------------------------------------------------------------
template <typename real, typename column_t, typename idx_t>
std::map<cl_context, bool> SpMat<real,column_t,idx_t>::SpMatCSR::compiled;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatCSR::zero;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatCSR::spmv_set;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMat<real,column_t,idx_t>::SpMatCSR::spmv_add;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, uint> SpMat<real,column_t,idx_t>::SpMatCSR::wgsize;

template <typename real, typename column_t, typename idx_t>
SpMat<real,column_t,idx_t>::SpMatCSR::SpMatCSR(
        const cl::CommandQueue &queue,
        size_t beg, size_t end, column_t xbeg, column_t xend,
        const idx_t *row, const column_t *col, const real *val,
        const std::set<column_t> &remote_cols
        )
    : queue(queue), n(end - beg), has_loc(false), has_rem(false)
{
    cl::Context context = qctx(queue);

    prepare_kernels(context);

    if (beg == 0 && remote_cols.empty()) {
        if (row[n]) {
            loc.row = cl::Buffer(
                    context, CL_MEM_READ_ONLY, (n + 1) * sizeof(idx_t));

            loc.col = cl::Buffer(
                    context, CL_MEM_READ_ONLY, row[n] * sizeof(column_t));

            loc.val = cl::Buffer(
                    context, CL_MEM_READ_ONLY, row[n] * sizeof(real));

            queue.enqueueWriteBuffer(
                    loc.row, CL_FALSE, 0, (n + 1) * sizeof(idx_t), row);

            queue.enqueueWriteBuffer(
                    loc.col, CL_FALSE, 0, row[n] * sizeof(column_t), col);

            queue.enqueueWriteBuffer(
                    loc.val, CL_TRUE, 0, row[n] * sizeof(real), val);
        }

        has_loc = row[n];
        has_rem = false;
    } else {
        std::vector<idx_t>    lrow;
        std::vector<column_t> lcol;
        std::vector<real>     lval;

        std::vector<idx_t>    rrow;
        std::vector<column_t> rcol;
        std::vector<real>     rval;

        lrow.reserve(end - beg + 1);
        lrow.push_back(0);

        lcol.reserve(row[end] - row[beg]);
        lval.reserve(row[end] - row[beg]);

        if (!remote_cols.empty()) {
            rrow.reserve(end - beg + 1);
            rrow.push_back(0);

            rcol.reserve(row[end] - row[beg]);
            rval.reserve(row[end] - row[beg]);
        }

        // Renumber columns.
        std::unordered_map<column_t,column_t> r2l(2 * remote_cols.size());
        for(auto c = remote_cols.begin(); c != remote_cols.end(); c++) {
            size_t idx = r2l.size();
            r2l[*c] = idx;
        }

        for(size_t i = beg; i < end; i++) {
            for(idx_t j = row[i]; j < row[i + 1]; j++) {
                if (col[j] >= xbeg && col[j] < xend) {
                    lcol.push_back(col[j] - xbeg);
                    lval.push_back(val[j]);
                } else {
                    assert(r2l.count(col[j]));
                    rcol.push_back(r2l[col[j]]);
                    rval.push_back(val[j]);
                }
            }

            lrow.push_back(lcol.size());
            rrow.push_back(rcol.size());
        }

        cl::Event event;

        // Copy local part to the device.
        if (lrow.back()) {
            loc.row = cl::Buffer(
                    context, CL_MEM_READ_ONLY, lrow.size() * sizeof(idx_t));

            queue.enqueueWriteBuffer(
                    loc.row, CL_FALSE, 0, lrow.size() * sizeof(idx_t), lrow.data());

            loc.col = cl::Buffer(
                    context, CL_MEM_READ_ONLY, lcol.size() * sizeof(column_t));

            loc.val = cl::Buffer(
                    context, CL_MEM_READ_ONLY, lval.size() * sizeof(real));

            queue.enqueueWriteBuffer(
                    loc.col, CL_FALSE, 0, lcol.size() * sizeof(column_t), lcol.data());

            queue.enqueueWriteBuffer(
                    loc.val, CL_FALSE, 0, lval.size() * sizeof(real), lval.data(),
                    0, &event);
        }

        // Copy remote part to the device.
        if (!remote_cols.empty()) {
            rem.row = cl::Buffer(
                    context, CL_MEM_READ_ONLY, rrow.size() * sizeof(idx_t));

            rem.col = cl::Buffer(
                    context, CL_MEM_READ_ONLY, rcol.size() * sizeof(column_t));

            rem.val = cl::Buffer(
                    context, CL_MEM_READ_ONLY, rval.size() * sizeof(real));

            queue.enqueueWriteBuffer(
                    rem.row, CL_FALSE, 0, rrow.size() * sizeof(idx_t), rrow.data());

            queue.enqueueWriteBuffer(
                    rem.col, CL_FALSE, 0, rcol.size() * sizeof(column_t), rcol.data());

            queue.enqueueWriteBuffer(
                    rem.val, CL_FALSE, 0, rval.size() * sizeof(real), rval.data(),
                    0, &event);
        }

        if (lrow.back() || !remote_cols.empty()) event.wait();

        has_loc = lrow.back();
        has_rem = !remote_cols.empty();
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatCSR::prepare_kernels(const cl::Context &context) const {
    if (!compiled[context()]) {
        std::ostringstream source;

        source << standard_kernel_header <<
            "typedef " << type_name<real>() << " real;\n"
            "kernel void zero(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global real *y\n"
            "    )\n"
            "{\n"
            "    size_t i = get_global_id(0);\n"
            "    if (i < n) y[i] = 0;\n"
            "}\n"
            "kernel void spmv_set(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global const " << type_name<idx_t>() << " *row,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t i = get_global_id(0);\n"
            "    if (i < n) {\n"
            "        real sum = 0;\n"
            "        size_t beg = row[i];\n"
            "        size_t end = row[i + 1];\n"
            "        for(size_t j = beg; j < end; j++)\n"
            "            sum += val[j] * x[col[j]];\n"
            "        y[i] = alpha * sum;\n"
            "    }\n"
            "}\n"
            "kernel void spmv_add(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global const " << type_name<idx_t>() << " *row,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t i = get_global_id(0);\n"
            "    if (i < n) {\n"
            "        real sum = 0;\n"
            "        size_t beg = row[i];\n"
            "        size_t end = row[i + 1];\n"
            "        for(size_t j = beg; j < end; j++)\n"
            "            sum += val[j] * x[col[j]];\n"
            "        y[i] += alpha * sum;\n"
            "    }\n"
            "}\n";

        auto program = build_sources(context, source.str());

        zero[context()]     = cl::Kernel(program, "zero");
        spmv_set[context()] = cl::Kernel(program, "spmv_set");
        spmv_add[context()] = cl::Kernel(program, "spmv_add");

        std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

        wgsize[context()] = std::min(
                kernel_workgroup_size(spmv_set[context()], device[0]),
                kernel_workgroup_size(spmv_add[context()], device[0])
                );

        compiled[context()] = true;
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatCSR::mul_local(
        const cl::Buffer &x, const cl::Buffer &y,
        real alpha, bool append
        ) const
{
    cl::Context context = qctx(queue);

    if (has_loc) {
        if (append) {
            uint pos = 0;
            spmv_add[context()].setArg(pos++, n);
            spmv_add[context()].setArg(pos++, loc.row);
            spmv_add[context()].setArg(pos++, loc.col);
            spmv_add[context()].setArg(pos++, loc.val);
            spmv_add[context()].setArg(pos++, x);
            spmv_add[context()].setArg(pos++, y);
            spmv_add[context()].setArg(pos++, alpha);

            queue.enqueueNDRangeKernel(spmv_add[context()],
                    cl::NullRange, n, cl::NullRange);
        } else {
            uint pos = 0;
            spmv_set[context()].setArg(pos++, n);
            spmv_set[context()].setArg(pos++, loc.row);
            spmv_set[context()].setArg(pos++, loc.col);
            spmv_set[context()].setArg(pos++, loc.val);
            spmv_set[context()].setArg(pos++, x);
            spmv_set[context()].setArg(pos++, y);
            spmv_set[context()].setArg(pos++, alpha);

            queue.enqueueNDRangeKernel(spmv_set[context()],
                    cl::NullRange, n, cl::NullRange);
        }
    } else if (!append) {
        uint pos = 0;
        zero[context()].setArg(pos++, n);
        zero[context()].setArg(pos++, y);

        queue.enqueueNDRangeKernel(zero[context()],
                cl::NullRange, n, cl::NullRange);
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMat<real,column_t,idx_t>::SpMatCSR::mul_remote(
        const cl::Buffer &x, const cl::Buffer &y,
        real alpha, const std::vector<cl::Event> &event
        ) const
{
    if (!has_rem) return;

    cl::Context context = qctx(queue);

    uint pos = 0;
    spmv_add[context()].setArg(pos++, n);
    spmv_add[context()].setArg(pos++, rem.row);
    spmv_add[context()].setArg(pos++, rem.col);
    spmv_add[context()].setArg(pos++, rem.val);
    spmv_add[context()].setArg(pos++, x);
    spmv_add[context()].setArg(pos++, y);
    spmv_add[context()].setArg(pos++, alpha);

    queue.enqueueNDRangeKernel(spmv_add[context()],
            cl::NullRange, n, cl::NullRange, &event
            );
}

/// Sparse matrix in CCSR format.
/**
 * Compressed CSR format. row, col, and val arrays contain unique rows of the
 * matrix. Column numbers in col array are relative to diagonal. idx array
 * contains index into row vector, corresponding to each row of the matrix. So
 * that matrix-vector multiplication may be performed as follows:
 * \code
 * for(uint i = 0; i < n; i++) {
 *     real sum = 0;
 *     for(uint j = row[idx[i]]; j < row[idx[i] + 1]; j++)
 *         sum += val[j] * x[i + col[j]];
 *     y[i] = sum;
 * }
 * \endcode
 * This format does not support multi-device computation, so it accepts single
 * queue at initialization. Vectors x and y should also be single-queued and
 * reside on the same device with matrix.
 */
template <typename real, typename column_t = ptrdiff_t, typename idx_t = size_t>
class SpMatCCSR : matrix_terminal {
    public:
        typedef real value_type;

        /// Constructor for CCSR format.
        /**
         * Constructs GPU representation of the CCSR matrix.
         * \param queue single queue.
         * \param n     number of rows in the matrix.
         * \param m     number of unique rows in the matrix.
         * \param idx   index into row vector.
         * \param row   row index into col and val vectors.
         * \param col   column positions of nonzero elements wrt to diagonal.
         * \param val   values of nonzero elements of the matrix.
         */
        SpMatCCSR(const cl::CommandQueue &queue,
                size_t n, size_t m, const size_t *idx, const idx_t *row,
                const column_t *col, const real *val
                );

        /// Matrix-vector multiplication.
        /**
         * Matrix vector multiplication (y = alpha Ax or y += alpha Ax).
         * Vectors x and y should also be single-queued and reside on the same
         * device with matrix.
         * \param x      input vector.
         * \param y      output vector.
         * \param alpha  coefficient in front of matrix-vector product
         * \param append if set, matrix-vector product is appended to y.
         *               Otherwise, y is replaced with matrix-vector product.
         */
        void mul(const vex::vector<real> &x, vex::vector<real> &y,
                real alpha = 1, bool append = false) const;
    private:
        void prepare_kernels(const cl::Context &context) const;

        void mul_local(
                const cl::Buffer &x, const cl::Buffer &y,
                real alpha, bool append
                ) const;

        const cl::CommandQueue &queue;

        size_t n;

        struct {
            cl::Buffer idx;
            cl::Buffer row;
            cl::Buffer col;
            cl::Buffer val;
        } mtx;

        static std::map<cl_context, bool>       compiled;
        static std::map<cl_context, cl::Kernel> spmv_set;
        static std::map<cl_context, cl::Kernel> spmv_add;
        static std::map<cl_context, uint>       wgsize;
};

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, bool> SpMatCCSR<real,column_t,idx_t>::compiled;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMatCCSR<real,column_t,idx_t>::spmv_set;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, cl::Kernel> SpMatCCSR<real,column_t,idx_t>::spmv_add;

template <typename real, typename column_t, typename idx_t>
std::map<cl_context, uint> SpMatCCSR<real,column_t,idx_t>::wgsize;

template <typename real, typename column_t, typename idx_t>
SpMatCCSR<real,column_t,idx_t>::SpMatCCSR(
        const cl::CommandQueue &queue, size_t n, size_t m, const size_t *idx,
        const idx_t *row, const column_t *col, const real *val
        )
    : queue(queue), n(n)
{
    static_assert(std::is_signed<column_t>::value,
            "Column type for CCSR format has to be signed."
            );

    cl::Context context = qctx(queue);

    prepare_kernels(context);

    mtx.idx = cl::Buffer(context, CL_MEM_READ_ONLY, n * sizeof(idx_t));
    mtx.row = cl::Buffer(context, CL_MEM_READ_ONLY, (m + 1) * sizeof(idx_t));
    mtx.col = cl::Buffer(context, CL_MEM_READ_ONLY, row[m] * sizeof(column_t));
    mtx.val = cl::Buffer(context, CL_MEM_READ_ONLY, row[m] * sizeof(real));

    queue.enqueueWriteBuffer(mtx.idx, CL_FALSE, 0, n * sizeof(idx_t), idx);
    queue.enqueueWriteBuffer(mtx.row, CL_FALSE, 0, (m + 1) * sizeof(idx_t), row);
    queue.enqueueWriteBuffer(mtx.col, CL_FALSE, 0, row[m] * sizeof(column_t), col);
    queue.enqueueWriteBuffer(mtx.val, CL_TRUE,  0, row[m] * sizeof(real), val);
}

template <typename real, typename column_t, typename idx_t>
void SpMatCCSR<real,column_t,idx_t>::prepare_kernels(const cl::Context &context) const {
    if (!compiled[context()]) {
        std::ostringstream source;

        source << standard_kernel_header <<
            "typedef " << type_name<real>() << " real;\n"
            "kernel void spmv_set(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global const " << type_name<idx_t>() << " *idx,\n"
            "    global const " << type_name<idx_t>() << " *row,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t i = get_global_id(0);\n"
            "    if (i < n) {\n"
            "        real sum = 0;\n"
            "        size_t pos = idx[i];\n"
            "        size_t beg = row[pos];\n"
            "        size_t end = row[pos + 1];\n"
            "        for(size_t j = beg; j < end; j++)\n"
            "            sum += val[j] * x[i + col[j]];\n"
            "        y[i] = alpha * sum;\n"
            "    }\n"
            "}\n"
            "kernel void spmv_add(\n"
            "    " << type_name<size_t>() << " n,\n"
            "    global const " << type_name<idx_t>() << " *idx,\n"
            "    global const " << type_name<idx_t>() << " *row,\n"
            "    global const " << type_name<column_t>() << " *col,\n"
            "    global const real *val,\n"
            "    global const real *x,\n"
            "    global real *y,\n"
            "    real alpha\n"
            "    )\n"
            "{\n"
            "    size_t i = get_global_id(0);\n"
            "    if (i < n) {\n"
            "        real sum = 0;\n"
            "        size_t pos = idx[i];\n"
            "        size_t beg = row[pos];\n"
            "        size_t end = row[pos + 1];\n"
            "        for(size_t j = beg; j < end; j++)\n"
            "            sum += val[j] * x[i + col[j]];\n"
            "        y[i] += alpha * sum;\n"
            "    }\n"
            "}\n";

        auto program = build_sources(context, source.str());

        spmv_set[context()] = cl::Kernel(program, "spmv_set");
        spmv_add[context()] = cl::Kernel(program, "spmv_add");

        std::vector<cl::Device> device = context.getInfo<CL_CONTEXT_DEVICES>();

        wgsize[context()] = std::min(
                kernel_workgroup_size(spmv_set[context()], device[0]),
                kernel_workgroup_size(spmv_add[context()], device[0])
                );

        compiled[context()] = true;
    }
}

template <typename real, typename column_t, typename idx_t>
void SpMatCCSR<real,column_t,idx_t>::mul(
        const vex::vector<real> &x, vex::vector<real> &y,
        real alpha, bool append
        ) const
{
    cl::Context context = qctx(queue);

    if (append) {
        uint pos = 0;
        spmv_add[context()].setArg(pos++, n);
        spmv_add[context()].setArg(pos++, mtx.idx);
        spmv_add[context()].setArg(pos++, mtx.row);
        spmv_add[context()].setArg(pos++, mtx.col);
        spmv_add[context()].setArg(pos++, mtx.val);
        spmv_add[context()].setArg(pos++, x());
        spmv_add[context()].setArg(pos++, y());
        spmv_add[context()].setArg(pos++, alpha);

        queue.enqueueNDRangeKernel(spmv_add[context()],
                cl::NullRange, n, cl::NullRange);
    } else {
        uint pos = 0;
        spmv_set[context()].setArg(pos++, n);
        spmv_set[context()].setArg(pos++, mtx.idx);
        spmv_set[context()].setArg(pos++, mtx.row);
        spmv_set[context()].setArg(pos++, mtx.col);
        spmv_set[context()].setArg(pos++, mtx.val);
        spmv_set[context()].setArg(pos++, x());
        spmv_set[context()].setArg(pos++, y());
        spmv_set[context()].setArg(pos++, alpha);

        queue.enqueueNDRangeKernel(spmv_set[context()],
                cl::NullRange, n, cl::NullRange);
    }
}

/// Returns device weight after spmv test
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
    profiler prof(queue);
    prof.tic_cl("");
    A.mul(x, y);
    double time = prof.toc("");
    return 1.0 / time;
}

} // namespace vex

#ifdef WIN32
#  pragma warning(pop)
#endif

// vim: et
#endif
