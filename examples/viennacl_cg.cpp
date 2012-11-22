#include <iostream>
#include <vector>
#include <tuple>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>
#include <viennacl/linalg/cg.hpp>

typedef double real;

int main() {
    try {
        size_t n = 1024;
        real h = 1.0 / (n - 1);

        // Prepare problem (1D Poisson equation).
        std::vector<size_t> row;
        std::vector<size_t> col;
        std::vector<real>   val;
        std::vector<real>   rhs;

        row.reserve(n + 1);
        col.reserve(2 + (n - 2) * 3);
        val.reserve(2 + (n - 2) * 3);
        rhs.reserve(n);

        row.push_back(0);
        for(size_t i = 0; i < n; i++) {
            if (i == 0 || i == n-1) {
                col.push_back(i);
                val.push_back(1);
                rhs.push_back(0);
                row.push_back(row.back() + 1);
            } else {
                col.push_back(i-1);
                val.push_back(-1/(h*h));

                col.push_back(i);
                val.push_back(2/(h*h));

                col.push_back(i+1);
                val.push_back(-1/(h*h));

                rhs.push_back(2);
                row.push_back(row.back() + 3);
            }
        }

        // Move data to GPUs.
        vex::Context ctx(
                vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
                vex::Filter::DoublePrecision
                );

        vex::StaticContext<>::set(ctx);

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
