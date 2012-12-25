#ifndef VEXCL_MPI_VECTOR_HPP
#define VEXCL_MPI_VECTOR_HPP

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
 * \file   vexcl/mpi/vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper for vex::vector.
 */

#include <mpi.h>
#include <vexcl/vector.hpp>
#include <vexcl/mpi/util.hpp>
#include <vexcl/mpi/operations.hpp>

namespace vex {
/// MPI wrappers for VexCL types.
namespace mpi {

template <typename T>
class vector 
    : public mpi_vector_expression<
        typename boost::proto::terminal< mpi_vector_terminal >::type
      >
{
    public:
        typedef T      value_type;
        typedef size_t size_type;

        vector() {}

        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue, size_t n)
            : mpi(comm), part(mpi.size + 1), local_data(queue, n)
        {
            part[0] = 0;

            MPI_Allgather(
                    &n,              1, mpi_type<size_t>(),
                    part.data() + 1, 1, mpi_type<size_t>(),
                    mpi.comm);

            std::partial_sum(part.begin(), part.end(), part.begin());
        }
        
        size_t size() const {
            return part.back();
        }

        size_t local_size() const {
            return part[mpi.rank + 1] - part[mpi.rank];
        }

        vex::vector<T>& data() {
            return local_data;
        }

        const vex::vector<T>& data() const {
            return local_data;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_vector_expr_grammar
            >::value,
            const vector&
        >::type
        operator=(const Expr &expr) {
            local_data = extract_local_expression()(boost::proto::as_child(expr));
            return *this;
        }
    private:
        comm_data mpi;

        std::vector<size_t> part;
        vex::vector<T> local_data;
};


} // namespace mpi
} // namespace vex

#endif
