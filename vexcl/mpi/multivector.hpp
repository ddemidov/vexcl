#ifndef VEXCL_MPI_MULTIVECTOR_HPP
#define VEXCL_MPI_MULTIVECTOR_HPP

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
 * \file   vexcl/mpi/multivector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  MPI wrapper around vex::multivector.
 */

#include <mpi.h>
#include <vexcl/multivector.hpp>
#include <vexcl/mpi/operations.hpp>

#include <boost/preprocessor/repetition.hpp>
#ifndef VEXCL_MAX_ARITY
#  define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#endif

namespace vex {

template <typename T, size_t N, bool own>
struct number_of_components< vex::mpi::multivector<T, N, own> >
    : boost::mpl::size_t<N>
{};

namespace mpi {

template <typename T, size_t N, bool own>
class multivector
    : public mpi_multivector_expression<
        typename boost::proto::terminal< mpi_multivector_terminal >::type
      >
{
    public:
        typedef typename vex::multivector<T,N,own>::value_type value_type;

        multivector() {}

        multivector(MPI_Comm comm, const std::vector<cl::CommandQueue> &queue, size_t n)
            : mpi(comm), part(mpi.restore_partitioning(n)), local_data(queue, n)
        {
            static_assert(own, "Wrong constructor for non-owning multivector");
        }

        multivector(MPI_Comm comm, const vex::multivector<T, N, false> &mv)
            : mpi(comm), part(mpi.restore_partitioning(mv.size())), local_data(mv)
        {
            static_assert(!own, "Wrong constructor for owning multivector");
        }

        void resize(const multivector &v) {
            mpi  = v.mpi;
            part = v.part;
            local_data.resize(v.local_data.queue_list(), v.local_size());
        }

        size_t size() const {
            return part.empty() ? 0 : part.back();
        }

        size_t local_size() const {
            return part.empty() ? 0 : part[mpi.rank + 1] - part[mpi.rank];
        }

        vex::multivector<T,N,own>& data() {
            return local_data;
        }

        const vex::multivector<T,N,own>& data() const {
            return local_data;
        }

        const vex::mpi::vector<T,false> operator()(uint i) const {
            return vex::mpi::vector<T,false>(mpi.comm,
                    const_cast<vex::vector<T>&>(local_data(i))
                    );
        }

        vex::mpi::vector<T,false> operator()(uint i) {
            return vex::mpi::vector<T,false>(mpi.comm, local_data(i));
        }

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

    private:
        comm_data mpi;
        std::vector<size_t> part; // TODO: is this really necessary?
        vex::multivector<T, N, own> local_data;
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
