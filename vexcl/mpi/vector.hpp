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
#include <memory>
#include <vexcl/vector.hpp>
#include <vexcl/mpi/util.hpp>
#include <vexcl/mpi/operations.hpp>

namespace vex {
/// MPI wrappers for VexCL types.
namespace mpi {

template <typename T, bool own>
struct mpi_vector_storage {};

template <typename T>
struct mpi_vector_storage<T, true> {
    typedef std::unique_ptr< vex::vector<T> > type;
};

template <typename T>
struct mpi_vector_storage<T, false> {
    typedef vex::vector<T>* type;
};

template <typename T, bool own>
class vector 
    : public mpi_vector_expression<
        typename boost::proto::terminal< mpi_vector_terminal >::type
      >
{
    public:
        typedef T      value_type;
        typedef size_t size_type;

        vector() {}

        vector(const vector &v) : mpi(v.mpi), part(v.part) {
            copy_local_data<own>(v);
        }

        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue, size_t n)
            : mpi(comm), part(mpi.size + 1),
              local_data(new vex::vector<T>(queue, n))
        {
            static_assert(own, "Wrong constructor for non-owning vector");

            part[0] = 0;

            MPI_Allgather(
                    &n,              1, mpi_type<size_t>(),
                    part.data() + 1, 1, mpi_type<size_t>(),
                    mpi.comm);

            std::partial_sum(part.begin(), part.end(), part.begin());
        }

        vector(MPI_Comm comm, vex::vector<T> &v)
            : mpi(comm), part(mpi.size + 1), local_data(&v)
        {
            static_assert(!own, "Wrong constructor for owning vector");

            size_t n = v.size();

            part[0] = 0;

            MPI_Allgather(
                    &n,              1, mpi_type<size_t>(),
                    part.data() + 1, 1, mpi_type<size_t>(),
                    mpi.comm);

            std::partial_sum(part.begin(), part.end(), part.begin());
        }

        void resize(const vector &v) {
            mpi  = v.mpi;
            part = v.part;
            local_data->resize(v.local_data);
        }
        
        size_t size() const {
            return part.empty() ? 0 : part.back();
        }

        size_t local_size() const {
            return part.empty() ? 0 : part[mpi.rank + 1] - part[mpi.rank];
        }

        vex::vector<T>& data() {
            return *local_data;
        }

        const vex::vector<T>& data() const {
            return *local_data;
        }

        const vector& operator=(const vector &v) {
            data() = v.data();
            return *this;
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
            (*local_data) = extract_local_expression()(boost::proto::as_child(expr));
            return *this;
        }

        MPI_Comm comm() const {
            return mpi.comm;
        }
    private:
        comm_data mpi;
        std::vector<size_t> part; // TODO: is this really necessary?
        typename mpi_vector_storage<T, own>::type local_data;

        template <bool own_data>
        typename std::enable_if<own_data, void>::type
        copy_local_data(const vector &v) {
            local_data.reset(new vex::vector<T>(v.data()));
        }

        template <bool own_data>
        typename std::enable_if<!own_data, void>::type
        copy_local_data(const vector &v) {
            local_data = const_cast<vex::vector<T>*>(&(v.data()));
        }
};


} // namespace mpi
} // namespace vex

namespace boost { namespace fusion { namespace traits {

template <class T, bool own>
struct is_sequence< vex::mpi::vector<T, own> > : std::false_type
{};

} } }

#endif
