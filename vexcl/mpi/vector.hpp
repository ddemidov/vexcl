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

        vector() : l_size(0) {}

        vector(const vector &v) : mpi(v.mpi), l_size(v.l_size) {
            copy_local_data<own>(v);
        }

        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue, size_t n)
            : mpi(comm), l_size(n), local_data(new vex::vector<T>(queue, n))
        {
            static_assert(own, "Wrong constructor for non-owning vector");
        }

        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                std::vector<T> &host
              )
            : mpi(comm), l_size(host.size()), local_data(new vex::vector<T>(queue, host))
        {
            static_assert(own, "Wrong constructor for non-owning vector");
        }

        vector(MPI_Comm comm, vex::vector<T> &v)
            : mpi(comm), l_size(v.size()), local_data(&v)
        {
            static_assert(!own, "Wrong constructor for owning vector");
        }

        void resize(const vector &v) {
            mpi = v.mpi;
            local_data.reset(v.local_data
                    ? new vex::vector<value_type>(v.data())
                    : 0);
        }

        size_t global_size() const {
            size_t g_size;
            MPI_Allreduce(const_cast<size_t*>(&l_size), &g_size, 1,
                    mpi_type<size_t>(), MPI_SUM, mpi.comm);
            return g_size;
        }

        size_t local_size() const {
            return l_size;
        }

        vex::vector<T>& data() {
            return *local_data;
        }

        const vex::vector<T>& data() const {
            return *local_data;
        }

        const typename vex::vector<T>::element operator[](size_t index) const {
            return (*local_data)[index];
        }

        /// Access element.
        typename vex::vector<T>::element operator[](size_t index) {
            return (*local_data)[index];
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
        size_t l_size;
        typename mpi_vector_storage<T, own>::type local_data;

        template <bool own_data>
        typename std::enable_if<own_data, void>::type
        copy_local_data(const vector &v) {
            if (v.local_data)
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
