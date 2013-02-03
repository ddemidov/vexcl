#ifndef VEXCL_MPI_UTIL_HPP
#define VEXCL_MPI_UTIL_HPP

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
 * \file   mpi/util.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI related utilities.
 */

#include <mpi.h>
#include <vector>
#include <set>
#include <algorithm>

#ifndef VEXCL_MPI_TAG_START
#   define VEXCL_MPI_TAG_START 1000
#endif

namespace vex {
namespace mpi {

/// Convert C++ type to MPI_Datatype
template <typename T>
inline MPI_Datatype mpi_type() {
    throw std::logic_error("Unsupported type");
};

#define DEFINE_MPI_TYPE(ctype,mpitype) \
template<> inline MPI_Datatype mpi_type<ctype>() { return mpitype; }

DEFINE_MPI_TYPE(int,      MPI_INT);
DEFINE_MPI_TYPE(unsigned, MPI_UNSIGNED);
DEFINE_MPI_TYPE(float,    MPI_FLOAT);
DEFINE_MPI_TYPE(double,   MPI_DOUBLE);

#if (__WORDSIZE == 64) || defined(_WIN64)
DEFINE_MPI_TYPE(size_t,    MPI_UNSIGNED_LONG_LONG);
DEFINE_MPI_TYPE(ptrdiff_t, MPI_LONG_LONG_INT);
#endif

#undef DEFINE_MPI_TYPE

/// Convenience wrapper for MPI communicator.
struct comm_data {
    MPI_Comm comm;
    int      rank;
    int      size;

    /// Empty constructor.
    comm_data() {}

    /// Constructor.
    comm_data(MPI_Comm comm) : comm(comm) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    /// Cooolective condition checking.
    /**
     * Will throw on every process if condition fails on any of the processes.
     */
    void precondition(bool cond, const char *msg) const {
        bool glob;

        if (!cond) {
            std::cerr << "Condition failed at process "
                << rank << ": " << msg << std::endl;
        }

        MPI_Allreduce(&cond, &glob, 1, MPI_BYTE, MPI_LAND, comm);

        if (!glob) throw std::runtime_error(msg);
    }

    /// Returns global partitioning given local size of a vector.
    std::vector<size_t> restore_partitioning(size_t local_size) const {
        std::vector<size_t> part(size + 1);

        part[0] = 0;

        MPI_Allgather(
                &local_size,     1, mpi_type<size_t>(),
                part.data() + 1, 1, mpi_type<size_t>(),
                comm
                );

        std::partial_sum(part.begin(), part.end(), part.begin());

        return part;
    }

};

/// Unique MPI tag generator.
template <int START>
int get_mpi_tag() {
    static int tag = START;
    return ++tag;
}

/// Ghost points exchanger.
template <typename value_t>
class exchange {
    public:
        /// Constructor.
        /**
         * \param comm  MPI communicator.
         * \param queue Vector of command queues.
         * \param part  Global partitioning of the vector owning ghost points.
         * \param remote_cols Global numbers of remote points that are required
         *                    by the local process.
         */
        template <typename column_t>
        exchange(
                MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                const std::vector<size_t> &part,
                const std::set<column_t>  &remote_cols
                )
            : mpi(comm)
        {
            static const int tagExcCols = get_mpi_tag<VEXCL_MPI_TAG_START>();

            column_owner owner(part);

            recv.idx.resize(mpi.size + 1, 0);

            recv.req.resize(mpi.size, MPI_REQUEST_NULL);
            send.req.resize(mpi.size, MPI_REQUEST_NULL);

            // Count columns that we receive.
            for(auto c = remote_cols.begin(); c != remote_cols.end(); ++c)
                ++recv.idx[owner(*c) + 1];

            // Exchange the counts.
            std::vector<size_t> comm_matrix(mpi.size * mpi.size);
            MPI_Allgather(&recv.idx[1], mpi.size, mpi_type<size_t>(),
                    comm_matrix.data(), mpi.size, mpi_type<size_t>(),
                    mpi.comm);

            std::partial_sum(recv.idx.begin(), recv.idx.end(), recv.idx.begin());
            recv.val.resize(recv.idx.back());

            send.idx.reserve(mpi.size + 1);
            send.idx.push_back(0);
            for(auto i = 0; i < mpi.size; ++i)
                send.idx.push_back(send.idx.back()
                        + comm_matrix[mpi.size * i + mpi.rank]);

            send.col.resize(send.idx.back());
            send.val.resize(send.idx.back());

            // Ready to exchange exact column numbers.
            std::vector<size_t> rcols(remote_cols.begin(), remote_cols.end());

            // Start receiving columns they need.
            for(int i = 0; i < mpi.size; ++i)
                if (int n = send.idx[i + 1] - send.idx[i])
                    MPI_Irecv(&send.col[send.idx[i]], n, mpi_type<size_t>(),
                            i, tagExcCols, mpi.comm, &send.req[i]);

            // Start sending columns we need to them.
            for(int i = 0; i < mpi.size; ++i)
                if (int n = recv.idx[i + 1] - recv.idx[i])
                    MPI_Isend(&rcols[recv.idx[i]], n, mpi_type<size_t>(),
                            i, tagExcCols, mpi.comm, &recv.req[i]);

            MPI_Waitall(mpi.size, send.req.data(), MPI_STATUSES_IGNORE);

            // Renumber columns to send.
            for(auto c = send.col.begin(); c != send.col.end(); ++c)
                *c -= part[mpi.rank];

            get.reset(new gather<value_t>(
                        queue, part[mpi.rank + 1] - part[mpi.rank], send.col
                        ));

            MPI_Waitall(mpi.size, recv.req.data(), MPI_STATUSES_IGNORE);
        }

        /// Number of remote points that are required by the local process.
        size_t remote_size() const {
            return recv.idx.back();
        }

        /// Starts asyncronous exchange of ghost points.
        void start(const vex::vector<value_t> &local_data) {
            static const int tagExcVals = get_mpi_tag<VEXCL_MPI_TAG_START>();

            (*get)(local_data, send.val);

            for(int i = 0; i < mpi.size; ++i)
                if (int n = recv.idx[i + 1] - recv.idx[i])
                    MPI_Irecv(&recv.val[recv.idx[i]], n, mpi_type<value_t>(),
                            i, tagExcVals, mpi.comm, &recv.req[i]);

            for(int i = 0; i < mpi.size; ++i)
                if (int n = send.idx[i + 1] - send.idx[i])
                    MPI_Isend(&send.val[send.idx[i]], n, mpi_type<value_t>(),
                            i, tagExcVals, mpi.comm, &send.req[i]);
        }

        /// Blocks until asyncronous exchange of ghost points is done.
        void finish(vex::vector<value_t> &remote_data) {
            MPI_Waitall(mpi.size, recv.req.data(), MPI_STATUSES_IGNORE);

            vex::copy(recv.val, remote_data);

            MPI_Waitall(mpi.size, send.req.data(), MPI_STATUSES_IGNORE);
        }
    private:
        comm_data mpi;

        struct {
            std::vector<size_t>      idx;
            std::vector<value_t>     val;
            std::vector<MPI_Request> req;
        } recv;

        struct {
            std::vector<size_t>      idx;
            std::vector<size_t>      col;
            std::vector<value_t>     val;
            std::vector<MPI_Request> req;
        } send;

        std::unique_ptr< gather<value_t> > get;
};

} // namespace mpi
} // namespace vex

#endif
