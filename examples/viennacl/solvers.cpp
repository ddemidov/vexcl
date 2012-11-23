#include <iostream>
#include <vector>

#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>

#include "genproblem.hpp"

typedef double real;

template <class Tag>
void do_solve(const vex::SpMat<real> &A, vex::vector<real> f, const Tag &tag)
{
    vex::vector<real> x = viennacl::linalg::solve(A, f, tag);

    std::cout << "  Iterations: " << tag.iters() << std::endl
              << "  Error:      " << tag.error() << std::endl;

    // Test for convergence.
    f -= A * x;

    static vex::Reductor<real, vex::MAX> max(vex::current_context().queue());
    std::cout << "  max(residual) = " << max(fabs(f)) << std::endl;
}

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

        // Solve problem with ViennaCL's solvers:
        std::cout << "CG" << std::endl;
        do_solve(A, f, viennacl::linalg::cg_tag(1e-8, n));

        std::cout << "BiCGStab" << std::endl;
        do_solve(A, f, viennacl::linalg::bicgstab_tag(1e-8, n));

    } catch(const cl::Error &err) {
        std::cerr << "OpenCL Error: " << err << std::endl;
    } catch(const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
    }
}

// vim: et
