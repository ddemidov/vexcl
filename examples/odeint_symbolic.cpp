#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>

#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/multivector.hpp>
#include <vexcl/generator.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/profiler.hpp>

#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

namespace odeint = boost::numeric::odeint;

typedef vex::symbolic<double> sym_vector;
typedef boost::array<sym_vector, 3> sym_state;

//---------------------------------------------------------------------------
struct lorenz_system {
    const sym_vector &R;
    double sigma, b;

    lorenz_system(const sym_vector &R, double sigma = 10.0, double b = 8.0 / 3.0)
        : R(R), sigma(sigma), b(b) { }

    void operator()(const sym_state &x, sym_state &dxdt, double t) const {
        dxdt[0] = sigma * ( x[1] - x[0] );
        dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
        dxdt[2] = -b * x[2] + x[0] * x[1];
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
    const double dt = 0.01;
    const double t_max = 0.1; // 10.0 in the paper
    const double Rmin = 0.1;
    const double Rmax = 50.0;
    const double dR = (Rmax - Rmin) / (n - 1);

    vex::Context ctx( vex::Filter::Any );
    std::cout << ctx << std::endl;

    // Custom kernel body will be recorded here
    std::ostringstream body;
    vex::generator::set_recorder(body);

    // State types that would become kernel parameters
    sym_state  sym_S = {{
        sym_vector(sym_vector::VectorParameter),
            sym_vector(sym_vector::VectorParameter),
            sym_vector(sym_vector::VectorParameter)
    }};

    sym_vector sym_R(sym_vector::VectorParameter, sym_vector::Const);

    // Stepper type
    odeint::runge_kutta4_classic<
        sym_state, double, sym_state, double,
        odeint::range_algebra, odeint::default_operations
            > stepper;

    // Record single RK4 step
    lorenz_system sys(sym_R);
    stepper.do_step(sys, sym_S, 0, dt);

    // Generate the kernel from the recorded sequence
    auto kernel = vex::generator::build_kernel(ctx, "lorenz",
            body.str(), sym_S[0], sym_S[1], sym_S[2], sym_R);

    // Real state initialization
    vex::vector<double> X(ctx, n), Y(ctx, n), Z(ctx, n), R(ctx, n);
    X = Y = Z = 10.0;
    // R = Rmin + dR * vex::element_index();
    for(size_t i = 0; i < n; ++i) R[i] = Rmin + dR * i;


    // Integration loop
    vex::profiler<> prof(ctx);
    prof.tic_cl("Solving ODEs");
    for(double t = 0; t < t_max; t += dt)
        kernel(X, Y, Z, R);
    prof.toc("Solving ODEs");

    std::cout << "X[0] = " << X[0] << std::endl;
    std::cout << prof << std::endl;
}

