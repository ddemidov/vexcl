#ifndef VEXCL_MPI_SPMAT_HPP
#define VEXCL_MPI_SPMAT_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   vexcl/mpi/spmat.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper around vex::SpMat.
 */

#include <vector>
#include <unordered_map>
#include <algorithm>

#include <mpi.h>

#include <vexcl/spmat.hpp>
#include <vexcl/mpi/util.hpp>

namespace vex {
namespace mpi {

template <typename real, typename column_t, typename idx_t>
class SpMat {
    public:
        SpMat() {}

        SpMat(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                size_t n, size_t m,
                const idx_t *row, const column_t *col, const real *val
             )
            : mpi(comm),
              row_part(restore_partitioning(mpi, n)),
              col_part(restore_partitioning(mpi, m))
        {
            // Split into local and remote parts; renumber columns.
            std::vector<idx_t> loc_row(n + 1, 0);
            std::vector<idx_t> rem_row(n + 1, 0);

            std::vector<column_t> loc_col;
            std::vector<column_t> rem_col;

            std::vector<real> loc_val;
            std::vector<real> rem_val;

            column_t part_beg = col_part[mpi.rank];
            column_t part_end = col_part[mpi.rank + 1];

            std::set<column_t> remote_cols;

            for(size_t i = 0; i < n; ++i) {
                for(size_t j = row[i], e = row[i + 1]; j < e; ++j) {
                    column_t c = col[j];

                    if (part_beg <= c && c < part_end) {
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

                    if (part_beg <= c && c < part_end) {
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
                        new vex::SpMat( queue, n, m,
                            loc_row.data(), loc_col.data(), loc_val.data())
                        );
            }

            if (rem_row.back()) {
                rem_x.resize(queue, remote_cols.size());

                rem_mtx.reset(
                        new vex::SpMat( queue, n, remote_cols.size(),
                            rem_row.data(), rem_col.data(), rem_val.data())
                        );
            }

            // TODO: setup exchange
        }

        void mul(const vex::mpi::vector<real> &x, vex::mpi::vector<real> &y,
                real alpha = 1, bool append = false) const
        {
            if (rem_mtx) {
                // TODO: start exchange
            }

            if (loc_mtx)
                loc_mtx->mul(x.data(), y.data(), alpha, append);
            else if (!append)
                y.data() = 0;

            if (rem_mtx) {
                // TODO: finish exchange
                rem_mtx->mul(rem_x.data(), y.data, alpha, true);
            }
        }
    private:
        comm_data mpi;
        std::vector<size_t> row_part;
        std::vector<size_t> col_part;

        std::unique_ptr< vex::SpMat<real, column_t, idx_t> > loc_mtx;
        std::unique_ptr< vex::SpMat<real, column_t, idx_t> > rem_mtx;

        vex::vector<real> rem_x;
};

} // namespace mpi
} // namespace vex

#endif
