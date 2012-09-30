#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <vexcl/generator/symbolic.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::generator::symbolic<value_type> state_type;

namespace boost { namespace numeric { namespace odeint {

template<>
struct is_resizeable< state_type > : boost::true_type { };

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 , const state_type &x2 )
    {
    }
};

template<>
struct same_size_impl< state_type , state_type >
{
    static bool same_size( const state_type &x1 , const state_type &x2 )
    {
        return true;
    }
};

} } }

void sys_func(const state_type &x, state_type &dxdt, value_type t) {
    dxdt = x;
}

int main() {
    state_type::set_output(std::cout);

    state_type x(false);

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra, odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper, sys_func, x , 0.0 , 0.1 , 0.1 );
}
