#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <mpi.h>

#include <vexcl/vexcl.hpp>
#include <vexcl/mpi/mpi.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

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
        return x1.size() == x2.size();
    }
};

// vex::mpi::multivector
template< typename T , size_t N, bool own >
struct is_resizeable< vex::mpi::multivector< T , N , own > > : boost::true_type { };

template< typename T , size_t N, bool own >
struct resize_impl< vex::mpi::multivector< T , N , own > , vex::mpi::multivector< T , N , own > >
{
    static void resize( vex::mpi::multivector< T , N , own > &x1 , const vex::mpi::multivector< T , N , own > &x2 )
    {
        x1.resize( x2 );
    }
};

template< typename T , size_t N, bool own >
struct same_size_impl< vex::mpi::multivector< T , N , own > , vex::mpi::multivector< T , N , own > >
{
    static bool same_size( const vex::mpi::multivector< T , N , own > &x1 , const vex::mpi::multivector< T , N , own > &x2 )
    {
        return x1.size() == x2.size();
    }
};

} } }

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::mpi::vector< value_type >    vector_type;
typedef vex::mpi::multivector< value_type, 2 > state_type;

struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    oscillator(value_type omega, value_type amp, value_type offset, value_type omega_d)
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d )
    {
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
        value_type eps = m_offset + m_amp * cos( m_omega_d * t );

        dxdt.data()(0) = eps * x.data()(0) + m_omega * x.data()(1);
        dxdt.data()(1) = eps * x.data()(1) - m_omega * x.data()(0);
    }
};


size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    int mpi_rank;
    int mpi_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    srand48(mpi_rank);

    try {
        n = argc > 1 ? atoi(argv[1]) : 1024;
        using namespace std;

        vex::Context ctx( vex::Filter::Exclusive(
                    vex::Filter::Env && vex::Filter::Count(1) ) );

        if (!ctx.size())
            throw std::runtime_error("No OpenCL device found");

        for(int i = 0; i < mpi_size; ++i) {
            if (i == mpi_rank)
                std::cout << mpi_rank << ": "
                          << ctx.device(0).getInfo<CL_DEVICE_NAME>()
                          << std::endl;

            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::vector<value_type> x( 2 * n );
        std::generate( x.begin() , x.end() , drand48 );

        state_type X(MPI_COMM_WORLD, ctx.queue(), n);

        vex::copy( x.begin() , x.begin() + n, X.data()(0).begin() );
        vex::copy( x.begin() + n, x.end() , X.data()(1).begin() );


        odeint::runge_kutta4<
            state_type , value_type , state_type , value_type ,
                       odeint::vector_space_algebra , odeint::default_operations
                           > stepper;

        odeint::integrate_const( stepper , oscillator( 1.0 , 0.2 , 0.0 , 1.2 )
                , X , value_type(0.0) , t_max , dt );

        cout << mpi_rank << ": " << X.data()(0)[0] << endl;

    } catch(const cl::Error &e) {
	std::cout << e << std::endl;
    } catch(const std::exception &e) {
	std::cout << e.what() << std::endl;
    }

    MPI_Finalize();
}
