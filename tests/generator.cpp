#define BOOST_TEST_MODULE KernelGenerator
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

using namespace vex;

template <class state_type>
void sys_func(const state_type &x, state_type &dx, double dt) {
    dx = dt * sin(x);
}

template <class state_type, class SysFunction>
void runge_kutta_4(SysFunction sys, state_type &x, double dt) {
    state_type xtmp, k1, k2, k3, k4;

    sys(x, k1, dt);

    xtmp = x + 0.5 * k1;
    sys(xtmp, k2, dt);

    xtmp = x + 0.5 * k2;
    sys(xtmp, k3, dt);

    xtmp = x + k3;
    sys(xtmp, k4, dt);

    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

BOOST_AUTO_TEST_CASE(kernel_generator)
{
    const int n = 1024;

    std::ostringstream body;
    vex::generator::set_recorder(body);

    typedef vex::generator::symbolic<double> sym_state;

    double dt = 0.01;
    sym_state sym_x(sym_state::VectorParameter);

    // Record expression sequience.
    runge_kutta_4(sys_func<sym_state>, sym_x, dt);

    // Build kernel.
    auto kernel = vex::generator::build_kernel(
            ctx, "rk4_stepper", body.str(), sym_x);

    std::vector<double> x = random_vector(n);
    vex::vector<double> X(ctx, x);

    for(int i = 0; i < 100; i++) kernel(X);

    check_sample(X, [&](size_t idx, double a) {
            double s = x[idx];
            for(int i = 0; i < 100; i++)
                runge_kutta_4(sys_func<double>, s, dt);

            BOOST_CHECK_CLOSE(a, s, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()
