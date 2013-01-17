#ifndef VEXCL_MPI_VECTOR_HPP
#define VEXCL_MPI_VECTOR_HPP

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
 * \file   mpi/vector.hpp
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

/// \cond INTERNAL
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

typedef mpi_vector_expression<
    typename boost::proto::terminal< mpi_vector_terminal >::type
    > mpi_vector_terminal_expression;

/// \endcond

/// MPI wrapper for vex::vector class template.
template <typename T, bool own>
class vector : public mpi_vector_terminal_expression {
    public:
        typedef T      value_type;
        typedef size_t size_type;
        typedef vex::vector<T> base_type;

        /// Empty constructor.
        vector() : l_size(0) {}

        /// Copy constructor.
        vector(const vector &v) : mpi(v.mpi), l_size(v.l_size) {
            copy_local_data<own>(v);
        }

        /// Constructor.
        /**
         * \param comm  MPI communicator.
         * \param queue vector of command queues.
         * \param n     Size of local part of the vector.
         * \param host  Host vector that holds local data to be copied to
         *              compute device(s). May be NULL.
         * \param flags cl::Buffer creation flags.
         */
        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                size_t n, const T *host = 0, cl_mem_flags flags = CL_MEM_READ_WRITE
              )
            : mpi(comm), l_size(n),
              local_data(new vex::vector<T>(queue, n, host, flags))
        {
            static_assert(own, "Wrong constructor for non-owning vector");
        }

        /// Constructor.
        /**
         * \param comm  MPI communicator.
         * \param queue vector of command queues.
         * \param host  Host vector that holds local data to be copied to
         *              compute device(s). May be NULL.
         * \param flags cl::Buffer creation flags.
         */
        vector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                std::vector<T> &host, cl_mem_flags flags = CL_MEM_READ_WRITE
              )
            : mpi(comm), l_size(host.size()), local_data(new vex::vector<T>(queue, host, flags))
        {
            static_assert(own, "Wrong constructor for non-owning vector");
        }

        /// Constructs non-owning multivector.
        /**
         * This constructor is called from vex::mpi::multivector::operator()
         * and should not be used by hand. Copies reference to device vector.
         */
        vector(MPI_Comm comm, base_type &v)
            : mpi(comm), l_size(v.size()), local_data(&v)
        {
            static_assert(!own, "Wrong constructor for owning vector");
        }

        /// Resize vector.
        void resize(const vector &v) {
            mpi = v.mpi;
            local_data.reset(v.local_data
                    ? new vex::vector<value_type>(v.data())
                    : 0);
        }

        /// Resize vector.
        void resize(const std::vector<cl::CommandQueue> &queue,
                size_t size, const T *host = 0,
                cl_mem_flags flags = CL_MEM_READ_WRITE
                )
        {
            local_data.reset(new base_type(queue, size, host, flags));
        }

        /// Resize vector.
        void resize(const std::vector<cl::CommandQueue> &queue,
                const std::vector<T> &host,
                cl_mem_flags flags = CL_MEM_READ_WRITE
              )
        {
            local_data.reset(new base_type(queue, host, flags));
        }

        /// Global size of the vector.
        /**
         * \note Involves collective MPI operation.
         */
        size_t global_size() const {
            size_t g_size;
            MPI_Allreduce(const_cast<size_t*>(&l_size), &g_size, 1,
                    mpi_type<size_t>(), MPI_SUM, mpi.comm);
            return g_size;
        }

        /// Local size of the vector.
        size_t local_size() const {
            return l_size;
        }

        /// Reference to the local vex::vector instance.
        base_type& data() {
            return *local_data;
        }

        /// Reference to the local vex::vector instance.
        const base_type& data() const {
            return *local_data;
        }

        /// Access local element.
        /**
         * \param index Local number of the element.
         */
        const typename base_type::element operator[](size_t index) const {
            return (*local_data)[index];
        }

        /// Access local element.
        /**
         * \param index Local number of the element.
         */
        typename base_type::element operator[](size_t index) {
            return (*local_data)[index];
        }

        /// Copies data from source vector.
        const vector& operator=(const vector &v) {
            data() = v.data();
            return *this;
        }

        /// MPI communicator used by the vector.
        MPI_Comm comm() const {
            return mpi.comm;
        }

        /** \name Expression assignments.
         * @{
         */
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

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator=(const Expr &expr) {
            apply_additive_transform</*append=*/false>(
                    *this, simplify_additive_transform()( expr )
                    );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_vector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator=(const Expr &expr) {
            *this = mpi_extract_vector_expressions()( expr );

            apply_additive_transform</*append=*/true>(
                    *this, simplify_additive_transform()(
                            mpi_extract_additive_vector_transforms()( expr )
                        )
                    );

            return *this;
        }

#define COMPOUND_ASSIGNMENT(cop, op) \
        template <class Expr> \
        const vector& operator cop(const Expr &expr) { \
            return *this = *this op expr; \
        }

        COMPOUND_ASSIGNMENT(+=, +);
        COMPOUND_ASSIGNMENT(-=, -);
        COMPOUND_ASSIGNMENT(*=, *);
        COMPOUND_ASSIGNMENT(/=, /);
        COMPOUND_ASSIGNMENT(%=, %);
        COMPOUND_ASSIGNMENT(&=, &);
        COMPOUND_ASSIGNMENT(|=, |);
        COMPOUND_ASSIGNMENT(^=, ^);
        COMPOUND_ASSIGNMENT(<<=, <<);
        COMPOUND_ASSIGNMENT(>>=, >>);

#undef COMPOUND_ASSIGNMENT
        /** @} */

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
