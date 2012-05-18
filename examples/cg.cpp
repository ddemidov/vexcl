#include <iostream>
#include <vector>
#include <oclutil/oclutil.hpp>

using namespace clu;

typedef double real;

void cg_gpu(
	const std::vector<uint> &row,
	const std::vector<uint> &col,
	const std::vector<real> &val,
	const std::vector<real> &rhs,
	std::vector<real> &x
	)
{
    // Init OpenCL
    cl::Context      context;
    std::vector<cl::CommandQueue> queue;

    std::tie(context, queue) = queue_list(
	    Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision(),
	    true);

    uint n = x.size();

    // Move data to GPU(s)
    clu::SpMat<real>  A(queue, n, row.data(), col.data(), val.data());
    clu::vector<real> f(queue, CL_MEM_READ_ONLY,  rhs);
    clu::vector<real> u(queue, CL_MEM_READ_WRITE, x);
    clu::vector<real> r(queue, CL_MEM_READ_WRITE, n);
    clu::vector<real> p(queue, CL_MEM_READ_WRITE, n);
    clu::vector<real> q(queue, CL_MEM_READ_WRITE, n);


    // Solve equation Ax = f with conjugate gradients method.
    real rho1, rho2;
    q = A * u;
    r = f - q;

    rho1 = inner_product(r, r);

    for(uint iter = 0; rho1 > 1e-8 && iter < n; iter++) {
	if (iter == 0) {
	    p = Const(1) * r;
	} else {
	    real beta = rho1 / rho2;
	    p = r + Const(beta) * p;
	}

	q = A * p;

	real alpha = rho1 / inner_product(p, q);

	u = u + Const(alpha) * p;
	r = r - Const(alpha) * q;

	rho2 = rho1;
	rho1 = inner_product(r, r);
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
