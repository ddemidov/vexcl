#include <iostream>
#include <vector>
#include <tuple>
#include <vexcl/vexcl.hpp>

using namespace vex;

typedef double real;
// Solve system of linear equations A u = f with conjugate gradients method.
// Input matrix is represented in CSR format (parameters row, col, and val).
void cg_gpu(
        const std::vector<size_t> &row, // Indices to col and val vectors.
        const std::vector<size_t> &col, // Column numbers of non-zero elements.
        const std::vector<real>   &val, // Values of non-zero elements.
        const std::vector<real>   &rhs, // Right-hand side.
        std::vector<real> &x            // In: initial approximation; out: result.
        )
{
    const size_t n = x.size();

    // Init OpenCL
    vex::Context ctx(Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision);

    if (!ctx.size()) throw std::logic_error("No compute devices available");

    // Move data to GPU(s)
    vex::SpMat <real> A(ctx.queue(), n, n, row.data(), col.data(), val.data());
    vex::vector<real> f(ctx.queue(), rhs, CL_MEM_READ_ONLY);
    vex::vector<real> u(ctx.queue(), x);
    vex::vector<real> r(ctx.queue(), n);
    vex::vector<real> p(ctx.queue(), n);
    vex::vector<real> q(ctx.queue(), n);

    Reductor<real,MAX> max(ctx.queue());
    Reductor<real,SUM> sum(ctx.queue());

    // Solve equation Au = f with conjugate gradients method.
    real rho1 = 0, rho2 = 1;
    r = f - A * u;

    for(uint iter = 0; max(fabs(r)) > 1e-8 && iter < n; iter++) {
        rho1 = sum(r * r);

        if (iter == 0) {
            p = r;
        } else {
            real beta = rho1 / rho2;
            p = r + beta * p;
        }

        q = A * p;

        real alpha = rho1 / sum(p * q);

        u += alpha * p;
        r -= alpha * q;

        rho2 = rho1;
    }

    // Get result to host.
    copy(u, x);
}

int main() {
    size_t n = 1024;
    real h = 1.0 / (n - 1);

    try {
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

        std::vector<real> x(n, 0);

        // Solve problem.
        cg_gpu(row, col, val, rhs, x);

        // Compute actual residual.
        double res = 0;
        for(size_t i = 0; i < n; i++) {
            double y = i * h;
            res = std::max(res, fabs(x[i] - y * (1 - y)));
        }

        std::cout << "res = " << res << std::endl;
    } catch (const cl::Error &e) {
        std::cerr << "Error: " << e << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// vim: et
