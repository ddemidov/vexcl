#ifndef VEXCL_MPI_UTIL_HPP
#define VEXCL_MPI_UTIL_HPP

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
 * \file   vexcl/mpi/util.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI related utilities.
 */

#include <mpi.h>

namespace vex {
namespace mpi {

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
DEFINE_MPI_TYPE(size_t,    MPI_UINT64_T);
DEFINE_MPI_TYPE(ptrdiff_t, MPI_INT64_T);
#endif

#undef DEFINE_MPI_TYPE

struct comm_data {
    MPI_Comm comm;
    int      rank;
    int      size;

    comm_data() {}

    comm_data(MPI_Comm comm) : comm(comm) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }
};

inline void precondition(MPI_Comm comm, bool cond, const char *msg) {
    bool glob;

    if (!cond) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        std::cerr << "Condition failed at process " << rank << ": " << msg << std::endl;
    }

    MPI_Allreduce(&cond, &glob, 1, MPI_C_BOOL, MPI_LAND, comm);

    if (!glob) throw std::runtime_error(msg);
}

} // namespace mpi
} // namespace vex

#endif
