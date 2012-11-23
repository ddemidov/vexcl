#include <iostream>
#include <vector>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>
#include <viennacl/linalg/cg.hpp>

#include "genproblem.hpp"

typedef double real;
int main() {
    try {
        // Prepare problem (1D Poisson equation).
        size_t n = 1024;

        std::vector<size_t> row;
        std::vector<size_t> col;
        std::vector<real>   val;
        std::vector<real>   rhs;

        genproblem(n, row, col, val, rhs);

        // Move data to GPUs.
        vex::Context ctx(
                vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
                vex::Filter::DoublePrecision
                );

        std::cout << ctx << std::endl;

        vex::SpMat <real> A(ctx.queue(), n, n, row.data(), col.data(), val.data());
        vex::vector<real> f(ctx.queue(), rhs);

        // Solve problem with ViennaCL's CG implementation.
        vex::vector<real> x = viennacl::linalg::solve(
                A, f, viennacl::linalg::cg_tag(1e-8, n)
                );

        // Test for convergence.
        f -= A * x;

        vex::Reductor<real, vex::MAX> max(ctx.queue());
        std::cout << "res = " << max(fabs(f)) << std::endl;

    } catch(const cl::Error &err) {
        std::cerr << "OpenCL Error: " << err << std::endl;
    } catch(const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
    }
}

// vim: et
