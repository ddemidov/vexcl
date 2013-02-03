#ifndef VEXCL_MPI_SPMAT_HPP
#define VEXCL_MPI_SPMAT_HPP

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
 * \file   mpi/spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper for vex::SpMat.
 */

#include <vector>
#include <unordered_map>
#include <algorithm>

#include <mpi.h>

#include <vexcl/spmat.hpp>
#include <vexcl/gather.hpp>
#include <vexcl/mpi/util.hpp>

namespace vex {
namespace mpi {

/// \cond INTERNAL
struct mpi_matrix_terminal {};

template <class M, class V>
struct mpi_spmv
    : mpi_vector_expression< boost::proto::terminal< mpi_additive_vector_transform >::type >
{
    const M &A;
    const V &x;

    mpi_spmv(const M &m, const V &v) : A(m), x(v) {}

    template <bool negate, bool append, class W>
    typename std::enable_if<
        std::is_base_of<mpi_vector_terminal_expression, W>::value &&
        std::is_same<typename M::value_type, typename W::value_type>::value,
        void
    >::type
    apply(W &y) const {
        A.mul(x, y, negate ? -1 : 1, append);
    }
};

template <class M, class V>
typename std::enable_if<
    std::is_base_of<mpi_matrix_terminal, M>::value &&
    std::is_base_of<mpi_vector_terminal_expression, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type>::value,
    mpi_spmv< M, V >
>::type
operator*(const M &A, const V &x) {
    return mpi_spmv< M, V >(A, x);
}

#ifdef VEXCL_MPI_MULTIVECTOR_HPP

template <class M, class V>
struct mpi_multispmv
    : mpi_multivector_expression<
        boost::proto::terminal< mpi_additive_multivector_transform >::type
        >
{
    const M &A;
    const V &x;

    mpi_multispmv(const M &m, const V &v) : A(m), x(v) {}

    template <bool negate, bool append, class W>
    typename std::enable_if<
        std::is_base_of<mpi_multivector_terminal_expression, W>::value
#ifndef WIN32
	&& std::is_same<typename M::value_type, typename W::value_type::value_type>::value
#endif
	&& number_of_components<V>::value == number_of_components<W>::value
	, void
    >::type
    apply(W &y) const {
        for(size_t i = 0; i < number_of_components<V>::value; i++) {
            auto dst = y(i);
            A.mul(x(i), dst, negate ? -1 : 1, append);
        }
    }
};

template <class M, class V>
typename std::enable_if<
    std::is_base_of<mpi_matrix_terminal,                 M>::value &&
    std::is_base_of<mpi_multivector_terminal_expression, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type::value_type>::value,
    mpi_multispmv< M, V >
>::type
operator*(const M &A, const V &x) {
    return mpi_multispmv< M, V >(A, x);
}

#endif

/// \endcond

/// MPI wrapper for vex::SpMat class template.
template <typename real, typename column_t = size_t, typename idx_t = size_t>
class SpMat : public mpi_matrix_terminal {
    public:
        typedef real value_type;

        /// Empty constructor.
        SpMat() {}

        /// Constructor.
        /**
         * Constructs local part of distributed matrix. Each process holds
         * continuous strip of the matrix rows.
         * \param comm  MPI communicator.
         * \param queue vector of command queues.
         * \param n     Number of rows in the local part of the matrix.
         * \param m     Local size of a vector that the matrix will be
         *              multiplied by. Should be equal to n if the matrix is
         *              square.
         * \param row   Index into col and val vectors.
         * \param col   Global column numbers of the local non-zero entries.
         * \param val   Values of the local non-zero entries.
         */
        SpMat(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                size_t n, size_t m,
                const idx_t *row, const column_t *col, const real *val
             ) : mpi(comm)
        {
            // Split into local and remote parts; renumber columns.
            std::vector<idx_t> loc_row(n + 1, 0);
            std::vector<idx_t> rem_row(n + 1, 0);

            std::vector<column_t> loc_col;
            std::vector<column_t> rem_col;

            std::vector<real> loc_val;
            std::vector<real> rem_val;

            auto col_part = mpi.restore_partitioning(m);
            auto part_beg = col_part[mpi.rank];
            auto part_end = col_part[mpi.rank + 1];

            std::set<column_t> remote_cols;

            for(size_t i = 0; i < n; ++i) {
                for(size_t j = row[i], e = row[i + 1]; j < e; ++j) {
                    column_t c = col[j];

                    if (static_cast<column_t>(part_beg) <= c && c < static_cast<column_t>(part_end)) {
                        ++loc_row[i + 1];
                    } else {
                        remote_cols.insert(c);
                        ++rem_row[i + 1];
                    }
                }
            }

            std::partial_sum(loc_row.begin(), loc_row.end(), loc_row.begin());
            std::partial_sum(rem_row.begin(), rem_row.end(), rem_row.begin());

            loc_col.reserve(loc_row.back());
            loc_val.reserve(loc_row.back());

            rem_col.reserve(rem_row.back());
            rem_val.reserve(rem_row.back());

            std::unordered_map<column_t,column_t> r2l(2 * remote_cols.size());
            {
                size_t idx = 0;
                for(auto c = remote_cols.begin(); c != remote_cols.end(); c++) {
                    r2l[*c] = idx++;
                }
            }

            for(size_t i = 0; i < n; ++i) {
                for(size_t j = row[i], e = row[i + 1]; j < e; ++j) {
                    column_t c = col[j];
                    real     v = val[j];

                    if (static_cast<column_t>(part_beg) <= c && c < static_cast<column_t>(part_end)) {
                        loc_col.push_back(c - part_beg);
                        loc_val.push_back(v);
                    } else {
                        rem_col.push_back(r2l[c]);
                        rem_val.push_back(v);
                    }
                }
            }

            if (loc_row.back()) {
                loc_mtx.reset(
                        new vex::SpMat<real, column_t, idx_t>(
                            queue, n, m,
                            loc_row.data(), loc_col.data(), loc_val.data())
                        );
            }

            if (rem_row.back()) {
                rem_x.resize(queue, remote_cols.size());

                rem_mtx.reset(
                        new vex::SpMat<real, column_t, idx_t>(
                            queue, n, remote_cols.size(),
                            rem_row.data(), rem_col.data(), rem_val.data())
                        );
            }

            exc.reset(new exchange<real>(mpi.comm, queue, col_part, remote_cols));
        }

        /// Matrix-vector multiplication.
        /**
         * Matrix vector multiplication (\f$y = \alpha Ax\f$ or \f$y += \alpha
         * Ax\f$) is performed in parallel on all registered compute devices.
         * Ghost values of x are transfered across MPI processes as needed.
         * \param x      input vector.
         * \param y      output vector.
         * \param alpha  coefficient in front of matrix-vector product
         * \param append if set, matrix-vector product is appended to y.
         *               Otherwise, y is replaced with matrix-vector product.
         */
        template <class V, class W>
        typename std::enable_if<
            std::is_base_of<mpi_vector_terminal_expression, V>::value &&
            std::is_same<real, typename V::value_type>::value &&
            std::is_base_of<mpi_vector_terminal_expression, W>::value &&
            std::is_same<real, typename W::value_type>::value,
            void
        >::type
        mul(const V &x, W &y, real alpha = 1, bool append = false) const {
            if (rem_mtx) {
                exc->start(x.data());
            }

            if (loc_mtx)
                loc_mtx->mul(x.data(), y.data(), alpha, append);
            else if (!append)
                y.data() = 0;

            if (rem_mtx) {
                exc->finish(rem_x);
                rem_mtx->mul(rem_x, y.data(), alpha, true);
            }
        }
    private:
        comm_data mpi;

        std::unique_ptr< vex::SpMat<real, column_t, idx_t> > loc_mtx;
        std::unique_ptr< vex::SpMat<real, column_t, idx_t> > rem_mtx;

        mutable vex::vector<real> rem_x;

        std::unique_ptr< exchange<real> > exc;
};

} // namespace mpi
} // namespace vex

#endif
