#ifndef VEXCL_MPI_MULTIVECTOR_HPP
#define VEXCL_MPI_MULTIVECTOR_HPP

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
 * \file   mpi/multivector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper for vex::multivector.
 */

#include <mpi.h>
#include <vexcl/multivector.hpp>
#include <vexcl/mpi/operations.hpp>

#include <boost/preprocessor/repetition.hpp>
#ifndef VEXCL_MAX_ARITY
#  define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#endif

namespace vex {

/// \cond INTERNAL

template <typename T, size_t N, bool own>
struct number_of_components< vex::mpi::multivector<T, N, own> >
    : boost::mpl::size_t<N>
{};

/// \endcond

namespace mpi {

/// \cond INTERNAL

typedef mpi_multivector_expression<
    typename boost::proto::terminal< mpi_multivector_terminal >::type
    > mpi_multivector_terminal_expression;

/// \endcond

/// MPI wrapper for vex::multivector class template
template <typename T, size_t N, bool own>
class multivector : public mpi_multivector_terminal_expression {
    public:
        typedef vex::multivector<T,N,own>      base_type;
        typedef typename base_type::value_type value_type;

        /// Empty constructor.
        multivector() : l_size(0) {}

        /// Constructor.
        /**
         * If host pointer is not NULL, it is copied to the underlying vector
         * components of the multivector.
         * \param comm  MPI communicator.
         * \param queue queue list to be shared between all components.
         * \param n     Local size of each component.
         * \param host  Host vector that holds local data to be copied to
         *              the components. Size of host vector should be divisible
         *              by N. Components of the created multivector will have
         *              size equal to host.size() / N. The data will be
         *              partitioned equally between all components.
         * \param flags cl::Buffer creation flags.
         */
        multivector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue,
                size_t n, const T *host = 0,
                cl_mem_flags flags = CL_MEM_READ_WRITE
                )
            : mpi(comm), l_size(n), local_data(queue, n, host, flags)
        {
            static_assert(own, "Wrong constructor for non-owning multivector");
        }

        /// Constructs non-owning multivector.
        /**
         * This constructor is called from vex::tie and should not be used by
         * hand. Copies references to component vectors.
         */
        multivector(MPI_Comm comm, const vex::multivector<T, N, false> &mv)
            : mpi(comm), l_size(mv.size()), local_data(mv)
        {
            static_assert(!own, "Wrong constructor for owning multivector");
        }

        /// Resize multivector.
        /**
         * \param queue queue list to be shared between all components.
         * \param size  Local size of each component.
         */
        void resize(const std::vector<cl::CommandQueue> &queue, size_t size) {
            local_data.resize(queue, size);
        }

        /// Resize multivector.
        void resize(const multivector &v) {
            mpi = v.mpi;
            local_data.resize(v.local_data.queue_list(), v.local_size());
        }

        /// Global size of the multivector.
        /**
         * \note Involves collective MPI operation.
         */
        size_t global_size() const {
            size_t g_size;
            MPI_Allreduce(const_cast<size_t*>(&l_size), &g_size, 1,
                    mpi_type<size_t>(), MPI_SUM, mpi.comm);
            return g_size;
        }

        /// Local size of the multivector.
        size_t local_size() const {
            return l_size;
        }

        /// Reference to the local vex::multivector instance.
        base_type& data() {
            return local_data;
        }

        /// Reference to the local vex::multivector instance.
        const base_type& data() const {
            return local_data;
        }

        /// Component of the multivector.
        const vex::mpi::vector<T,false> operator()(uint i) const {
            return vex::mpi::vector<T,false>(mpi.comm,
                    const_cast<vex::vector<T>&>(local_data(i))
                    );
        }

        /// Component of the multivector.
        vex::mpi::vector<T,false> operator()(uint i) {
            return vex::mpi::vector<T,false>(mpi.comm, local_data(i));
        }

        /** \name Expression assignments.
         * @{
         * All operations are delegated to components of the multivector.
         */
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_multivector_expr_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr &expr) {
            local_data = extract_local_expression()(boost::proto::as_child(expr));
            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_additive_multivector_transform_grammar
            >::value,
            const multivector&
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
                mpi_multivector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                mpi_additive_multivector_transform_grammar
            >::value,
            const multivector&
        >::type
        operator=(const Expr &expr) {
            *this = mpi_extract_multivector_expressions()( expr );

            apply_additive_transform</*append=*/true>(
                    *this, simplify_additive_transform()(
                            mpi_extract_additive_vector_transforms()( expr )
                        )
                    );

            return *this;
        }

#define COMPOUND_ASSIGNMENT(cop, op) \
        template <class Expr> \
        const multivector& operator cop(const Expr &expr) { \
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

#define PRINT_EXTRACT(z, n, data) extract_local_expression()(boost::proto::as_child(std::get< n >(expr)))
#define TUPLE_ASSIGNMENT(z, n, data) \
        template < BOOST_PP_ENUM_PARAMS(n, class Expr) > \
        const multivector& operator=(const std::tuple< BOOST_PP_ENUM_PARAMS(n, Expr) > &expr) { \
            local_data = std::make_tuple( \
                    BOOST_PP_ENUM(n, PRINT_EXTRACT, ~) \
                    ); \
            return *this; \
        }

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TUPLE_ASSIGNMENT, ~)

#undef PRINT_EXTRACT
#undef TUPLE_ASSIGNMENT
        /** @} */
    private:
        comm_data mpi;
        size_t l_size;
        base_type local_data;
};

} // namespace mpi

#define PRINT_ARG(z, n, unused) vex::mpi::vector<T, own ## n> &v ## n
#define PRINT_TIE_ARG(z, n, unused) v ## n.data()
#define TIE_FUNCTION(z, n, unused) \
template< typename T, BOOST_PP_ENUM_PARAMS(n, bool own) > \
mpi::multivector<T, n, false> tie( BOOST_PP_ENUM(n, PRINT_ARG, ~) ) { \
    return mpi::multivector<T, n, false>( v0.comm(), \
            vex::tie( BOOST_PP_ENUM(n, PRINT_TIE_ARG, ~) )); \
}

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, TIE_FUNCTION, ~)

#undef PRINT_ARG
#undef PRINT_TIE_ARG
#undef TIE_FUNCTION

} // namespace vex

namespace boost { namespace fusion { namespace traits {

template <class T, size_t N, bool own>
struct is_sequence< vex::mpi::multivector<T, N, own> > : std::false_type
{};

} } }

#endif
