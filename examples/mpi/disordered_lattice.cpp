#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <mpi.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/mpi.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

/* Resizing of vex::mpi types for odeint */
namespace boost { namespace numeric { namespace odeint {

// vex::mpi::vector
template< typename T, bool own >
struct is_resizeable< vex::mpi::vector< T, own > > : boost::true_type { };

template< typename T, bool own >
struct resize_impl< vex::mpi::vector< T, own > , vex::mpi::vector< T, own > >
{
    static void resize( vex::mpi::vector< T, own > &x1 , const vex::mpi::vector< T, own > &x2 )
    {
        x1.resize( x2 );
    }
};

template< typename T, bool own >
struct same_size_impl< vex::mpi::vector< T, own > , vex::mpi::vector< T, own > >
{
    static bool same_size( const vex::mpi::vector< T, own > &x1 , const vex::mpi::vector< T, own > &x2 )
    {
        return x1.local_size() == x2.local_size();
    }
};

} } }

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::mpi::vector< value_type > state_type;

class ham_lattice {
    public:
        ham_lattice(
                vex::mpi::comm_data &mpi, vex::Context &ctx,
                size_t n1, size_t n2, value_type beta, value_type K
                ) : beta(beta)
        {
            const size_t n = n1 * n2;

            size_t chunk_size = (n + mpi.size - 1) / mpi.size;
            size_t chunk_start = mpi.rank * chunk_size;
            size_t chunk_end   = std::min(n, chunk_start + chunk_size);
            chunk_size = chunk_end - chunk_start;

            srand48(mpi.rank);

            auto part = mpi.restore_partitioning(chunk_size);

            std::vector<size_t>     row;
            std::vector<size_t>     col;
            std::vector<value_type> val;

            row.reserve(chunk_size + 1);
            col.reserve(5 * chunk_size);
            val.reserve(5 * chunk_size);

            row.push_back( 0 );
            index_modulus index(n);

            for(int idx = part[mpi.rank]; static_cast<size_t>(idx) < part[mpi.rank + 1]; ++idx) {
                int is[5] = { idx , index( idx + 1 ) , index( idx - 1 ) , index( idx - n2 ) , index( idx + n2 ) };

                std::sort( is , is + 5 );

                for( int k = 0 ; k < 5 ; ++k ) {
                    col.push_back( is[k] );
                    if( is[k] == idx )
                        val.push_back( -drand48() - 4.0 * K );
                    else
                        val.push_back( K );
                }

                row.push_back( col.size() );
            }

            A.reset(new vex::mpi::SpMat<value_type>(
                        mpi.comm, ctx, chunk_size, chunk_size,
                        row.data(), col.data(), val.data()
                        ));
        }

        void operator()(const state_type &q, state_type &dp) const {
            dp = (*A) * q - beta * q * q * q;
        }

    private:
        value_type beta;
        std::unique_ptr< vex::mpi::SpMat<double> > A;

        struct index_modulus {
            int N;

            index_modulus(int n) : N(n) {}

            inline int operator()(int idx) const {
                if( idx <  0 ) return idx + N;
                if( idx >= N ) return idx - N;
                return idx;
            }
        };
};


int main( int argc , char *argv[] ) {
    MPI_Init(&argc, &argv);
    vex::mpi::comm_data mpi(MPI_COMM_WORLD);

    const size_t n1 = argc > 1 ? atoi(argv[1]) : 64;
    const size_t n2 = n1;
    const size_t n  = n1 * n2;

    size_t chunk_size = (n + mpi.size - 1) / mpi.size;
    size_t chunk_start = mpi.rank * chunk_size;
    size_t chunk_end   = std::min(n, chunk_start + chunk_size);
    chunk_size = chunk_end - chunk_start;

    const value_type K     = 0.1;
    const value_type beta  = 0.01;
    const value_type t_max = 100.0;
    const value_type dt    = 0.01;

    try {
        vex::Context ctx( vex::Filter::Exclusive(
                    vex::Filter::Env && vex::Filter::Count(1) ) );

        mpi.precondition(!ctx.empty(), "No OpenCL devices found");

        for(int i = 0; i < mpi.size; ++i) {
            if (i == mpi.rank)
                std::cout << mpi.rank << ": " << ctx.device(0) << std::endl;

            MPI_Barrier(mpi.comm);
        }

        std::pair<state_type, state_type> X(
                state_type( mpi.comm, ctx, chunk_size ),
                state_type( mpi.comm, ctx, chunk_size )
                );

        X.first  = 0.0;
        X.second = 0.0;

        auto part = mpi.restore_partitioning(chunk_size);
        size_t center = (n + n2) / 2;
        if (part[mpi.rank] <= center && center < part[mpi.rank + 1])
            X.first[center - part[mpi.rank]] = 1.0;

        odeint::symplectic_rkn_sb3a_mclachlan<
            state_type, state_type, value_type, state_type, state_type, value_type,
            odeint::vector_space_algebra , odeint::default_operations
            > stepper;

        ham_lattice sys( mpi, ctx, n1, n2, beta, K );

        odeint::integrate_const( stepper , std::ref( sys ) , X , value_type(0.0) , t_max , dt );

        std::cout << mpi.rank << ": " << X.first[0] << "\t" << X.second[0] << std::endl;

    } catch(const cl::Error &e) {
	std::cout << e << std::endl;
    } catch(const std::exception &e) {
	std::cout << e.what() << std::endl;
    }

    MPI_Finalize();
}
