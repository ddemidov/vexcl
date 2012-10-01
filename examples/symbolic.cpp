#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>
#include <vexcl/generator.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl_resize.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::generator::symbolic<value_type> symbolic_state;
typedef vex::vector<value_type> real_state;

template <class state_type>
void sys_func(const state_type &x, state_type &dxdt, value_type t) {
    dxdt = 0.042 * x;
}

int main() {
    const size_t n  = 1024 * 1024;
    const double dt = 0.01;
    const double t_max = 100.0;

    vex::Context ctx( vex::Filter::Env,
	    CL_QUEUE_PROFILING_ENABLE
	    );
    std::cout << ctx;

    // Custom kernel body will be recorded here:
    std::ostringstream body;
    vex::generator::set_recorder(body);

    // This state type is used for kernel recording.
    symbolic_state sym_x(symbolic_state::Parameter);

    // Construct arbitrary stepper with symbolic state type...
    odeint::runge_kutta4<
	    symbolic_state , value_type , symbolic_state , value_type ,
	    odeint::vector_space_algebra, odeint::default_operations
	    > sym_stepper;

    // ... record one step to a kernel body, ...
    sym_stepper.do_step(sys_func<symbolic_state>, sym_x, 0, dt);

    // ... and construct custom kernel:
    auto kernel = vex::generator::build_kernel(ctx.queue(), "test", body.str(),
	    sym_x);

    // Construct and init real state vector:
    real_state x(ctx.queue(), n);
    x = 1.0;

    vex::profiler prof(ctx.queue());

    // Do integration loop:
    prof.tic_cl("Custom");
    for(value_type t = 0; t < t_max; t += dt)
	kernel(x);
    prof.toc("Custom");

    // Show result:
    std::cout << "Custom kernel: " << x[0] << std::endl;

    //------------------------------------------------------------
    // Compare result with normal odeint solution.
    odeint::runge_kutta4<
	    real_state , value_type , real_state , value_type ,
	    odeint::vector_space_algebra, odeint::default_operations
	    > stepper;

    x = 1.0;
    prof.tic_cl("odeint");
    for(value_type t = 0; t < t_max; t += dt)
	stepper.do_step(sys_func<real_state>, x, t, dt);
    prof.toc("odeint");

    std::cout << "odeint: " << x[0] << std::endl;

    std::cout << prof;
}
