#define BOOST_TEST_MODULE KernelGenerator
#include <boost/test/unit_test.hpp>
#include <boost/phoenix/phoenix.hpp>
#include "context_setup.hpp"

#define N (1024 * 1024)
#define M 1024

template <class state_type>
state_type sys_func(const state_type &x) {
    return sin(x);
}

template <class state_type, class SysFunction>
void runge_kutta_4(SysFunction sys, state_type &x, double dt) {
    state_type k1 = dt * sys(x);
    state_type k2 = dt * sys(x + 0.5 * k1);
    state_type k3 = dt * sys(x + 0.5 * k2);
    state_type k4 = dt * sys(x + k3);

    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

BOOST_AUTO_TEST_CASE(kernel_generator)
{
    typedef vex::symbolic<double> sym_state;

    const size_t n  = N;
    const double dt = 0.01;

    std::ostringstream body;
    vex::generator::set_recorder(body);

    sym_state sym_x(sym_state::VectorParameter);

    // Record expression sequence.
    runge_kutta_4(sys_func<sym_state>, sym_x, dt);

    // Build kernel.
    auto kernel = vex::generator::build_kernel(
            ctx, "rk4_stepper", body.str(), sym_x);

    vex::vector<double> X(ctx, n);
    X = 1;

    kernel(X);

    vex::stopwatch<> watch;
    for(int i = 0; i < M; i++) kernel(X);

    std::cout << X[0] << " in " << watch.toc() << " seconds" << std::endl;
}

/*
An alternative variant, which does not use the generator facility.
Intermediate subexpression are captured with help of 'auto' keyword, and
are combined into larger expression.

This is not as effective as generated kernel, because same input vector
(here 'x') is passed as several different parameters. This specific example
takes about twice as long to execute as the above variant.

Nevertheless, this may be more convenient in some cases.
*/
BOOST_AUTO_TEST_CASE(lazy_evaluation)
{
    const size_t n  = N;
    const double dt = 0.01;

    auto rk4 = [](vex::vector<double> &x, double dt) {
        using vex::tag;

        auto X  = tag<1>(x);
        auto DT = tag<2>(dt);

        double _two = 2;
        double _half = 0.5;

        auto two  = tag<3>(2);
        auto half = tag<4>(0.5);


        auto k1 = DT * sin(X);
        auto k2 = DT * sin(X + half * k1);
        auto k3 = DT * sin(X + half * k2);
        auto k4 = DT * sin(X + k3);

        X += (k1 + two * k2 + two * k3 + k4) / 6;
    };

    vex::vector<double> X(ctx, n);
    X = 1;

    rk4(X, dt);

    vex::stopwatch<> watch;
    for(int i = 0; i < M; i++) rk4(X, dt);

    std::cout << X[0] << " in " << watch.toc() << " seconds" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
