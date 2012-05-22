#include <iostream>
#include <vector>
#include <tuple>
#include <oclutil/oclutil.hpp>

using namespace clu;

typedef double real;
// Solve system of linear equations A u = f with conjugate gradients method.
// Input matrix is represented in CSR format (parameters row, col, and val).
void cg_gpu(
	const std::vector<uint> &row,	// Indices to col and val vectors.
	const std::vector<uint> &col,	// Column numbers of non-zero elements.
	const std::vector<real> &val,	// Values of non-zero elements.
	const std::vector<real> &rhs,	// Right-hand side.
	std::vector<real> &x		// In: initial approximation; out: result.
	)
{
    // Init OpenCL
    cl::Context context;
    std::vector<cl::CommandQueue> queue;

    std::tie(context, queue) = queue_list(
	Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision()
	);

    // Move data to GPU(s)
    uint n = x.size();
    clu::SpMat<real>  A(queue, queue, n, row.data(), col.data(), val.data());
    clu::vector<real> f(queue, CL_MEM_READ_ONLY,  rhs);
    clu::vector<real> u(queue, CL_MEM_READ_WRITE, x);
    clu::vector<real> r(queue, CL_MEM_READ_WRITE, n);
    clu::vector<real> p(queue, CL_MEM_READ_WRITE, n);
    clu::vector<real> q(queue, CL_MEM_READ_WRITE, n);

    Reductor<real,MAX> max(queue);

    // Solve equation Au = f with conjugate gradients method.
    real rho1, rho2;
    r = f - A * u;

    for(uint iter = 0; max(Abs(r)) > 1e-8 && iter < n; iter++) {
	rho1 = inner_product(r, r);

	if (iter == 0) {
	    p = r;
	} else {
	    real beta = rho1 / rho2;
	    p = r + Const(beta) * p;
	}

	q = A * p;

	real alpha = rho1 / inner_product(p, q);

	u += Const(alpha) * p;
	r -= Const(alpha) * q;

	rho2 = rho1;
    }

    // Get result to host.
    copy(u, x);
}

int main() {
    uint n = 1024;
    real h = 1.0 / (n - 1);

    // Prepare problem (1D Poisson equation).
    std::vector<uint> row;
    std::vector<uint> col;
    std::vector<real> val;
    std::vector<real> rhs;

    row.reserve(n + 1);
    col.reserve(2 + (n - 2) * 3);
    val.reserve(2 + (n - 2) * 3);
    rhs.reserve(n);

    row.push_back(0);
    for(uint i = 0; i < n; i++) {
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
    for(uint i = 0; i < n; i++) {
	double y = i * h;
	res = std::max(res, fabs(x[i] - y * (1 - y)));
    }

    std::cout << "res = " << res << std::endl;
}
