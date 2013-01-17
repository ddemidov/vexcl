#ifndef VEXCL_MPI_REDUCE_HPP
#define VEXCL_MPI_REDUCE_HPP

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
 * \file   mpi/reduce.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper for vex::Reductor.
 */

#include <mpi.h>
#include <vexcl/reduce.hpp>
#include <vexcl/mpi/util.hpp>
#include <vexcl/mpi/operations.hpp>

namespace vex {
namespace mpi {

/// \cond INTERNAL

template <class RDC>
inline MPI_Op mpi_reduce_op() {
    throw std::logic_error("Unsupported reduce operation");
}

#define DEFINE_REDUCE_OP(rdc, op) \
template<> inline MPI_Op mpi_reduce_op<rdc>() { return op; };

DEFINE_REDUCE_OP(vex::SUM, MPI_SUM);
DEFINE_REDUCE_OP(vex::MAX, MPI_MAX);
DEFINE_REDUCE_OP(vex::MIN, MPI_MIN);

#undef DEFINE_REDUCE_OP

/// \endcond

/// MPI wrapper for vex::Reductor class template.
template <typename T, class RDC>
class Reductor {
    public:
        /// Constructor.
        Reductor(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue)
            : mpi(comm), reduce(queue)
        {}

        /// Compute and return reduction of the input expression.
        /**
         * The input expression may be as simple as a single vector, although
         * expressions of arbitrary complexity may be reduced.
         */
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<Expr, mpi_vector_expr_grammar>::value, T
        >::type
        operator()(const Expr &expr) const {
            T local = reduce(extract_local_expression()(boost::proto::as_child(expr)));
            T global;

            MPI_Allreduce(&local, &global, 1,
                    mpi_type<T>(), mpi_reduce_op<RDC>(), mpi.comm);

            return global;
        }

#ifdef VEXCL_MPI_MULTIVECTOR_HPP
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<Expr, mpi_multivector_expr_grammar>::value,
            std::array<T, boost::result_of<mutltiex_dimension(Expr)>::type::value>
        >::type
        operator()(const Expr &expr) const {
            std::array<T, boost::result_of<mutltiex_dimension(Expr)>::type::value>
                local = reduce(extract_local_expression()(boost::proto::as_child(expr))),
                global;

            MPI_Allreduce(local.data(), global.data(), boost::result_of<mutltiex_dimension(Expr)>::type::value,
                    mpi_type<T>(), mpi_reduce_op<RDC>(), mpi.comm);

            return global;
        }

#endif
    private:
        comm_data mpi;
        vex::Reductor<T,RDC> reduce;
};

} // namespace mpi
} // namespace vex


#endif
