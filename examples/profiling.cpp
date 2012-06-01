#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <vexcl/vexcl.hpp>

using namespace vex;

typedef double real;

#define BENCHMARK_VECTOR
#define BENCHMARK_REDUCTOR
#define BENCHMARK_SPMAT
#define BENCHMARK_CPU

//---------------------------------------------------------------------------
std::pair<double,double> benchmark_vector(
	std::vector<cl::CommandQueue> &queue, profiler &prof
	)
{
    const uint N = 1024 * 1024;
    const uint M = 1024;
    double time_elapsed;

    std::vector<real> A(N, 0);
    std::vector<real> B(N);
    std::vector<real> C(N);
    std::vector<real> D(N);

    std::generate(B.begin(), B.end(), [](){ return (double)rand() / RAND_MAX; });
    std::generate(C.begin(), C.end(), [](){ return (double)rand() / RAND_MAX; });
    std::generate(D.begin(), D.end(), [](){ return (double)rand() / RAND_MAX; });

    vex::vector<real> a(queue, CL_MEM_READ_WRITE, A);
    vex::vector<real> b(queue, CL_MEM_READ_WRITE, B);
    vex::vector<real> c(queue, CL_MEM_READ_WRITE, C);
    vex::vector<real> d(queue, CL_MEM_READ_WRITE, D);

    a += b + c * d;
    a = Const(0);

    prof.tic_cl("OpenCL");
    for(uint i = 0; i < M; i++)
	a += b + c * d;
    time_elapsed = prof.toc("OpenCL");

    double gflops = (3.0 * N * M) / time_elapsed / 1e9;
    double bwidth = (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9;

    std::cout
	<< "Vector arithmetic\n"
	<< "  OpenCL"
	<< "\n    GFLOPS:    " << gflops
	<< "\n    Bandwidth: " << bwidth
	<< std::endl;

#ifdef BENCHMARK_CPU
    prof.tic_cpu("C++");
    for(uint i = 0; i < M; i++)
	for(uint j = 0; j < N; j++)
	    A[j] += B[j] + C[j] * D[j];
    time_elapsed = prof.toc("C++");

    {
	double gflops = (3.0 * N * M) / time_elapsed / 1e9;
	double bwidth = (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9;

	std::cout
	    << "  C++"
	    << "\n    GFLOPS:    " << gflops
	    << "\n    Bandwidth: " << bwidth
	    << std::endl;
    }

    vex::copy(A, b);
    Reductor<real,SUM> sum(queue);

    a -= b;
    std::cout << "  res = " << sum(a * a)
	      << std::endl << std::endl;
#endif

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
std::pair<double, double> benchmark_reductor(
	std::vector<cl::CommandQueue> &queue, profiler &prof
	)
{
    const size_t N = 16 * 1024 * 1024;
    const size_t M = 1024 / 16;
    double time_elapsed;

    std::vector<real> A(N);
    std::vector<real> B(N);

    std::generate(A.begin(), A.end(), [](){ return (double)rand() / RAND_MAX; });
    std::generate(B.begin(), B.end(), [](){ return (double)rand() / RAND_MAX; });

    vex::vector<real> a(queue, CL_MEM_READ_WRITE, A);
    vex::vector<real> b(queue, CL_MEM_READ_WRITE, B);

    Reductor<real,SUM> sum(queue);

    double sum_cl = sum(a * b);
    sum_cl = 0;

    prof.tic_cl("OpenCL");
    for(uint i = 0; i < M; i++)
	sum_cl += sum(a * b);
    time_elapsed = prof.toc("OpenCL");

    double gflops = 2.0 * N * M / time_elapsed / 1e9;
    double bwidth = 2.0 * N * M * sizeof(real) / time_elapsed / 1e9;

    std::cout
	<< "Reduction\n"
	<< "  OpenCL"
	<< "\n    GFLOPS:    " << gflops
	<< "\n    Bandwidth: " << bwidth
	<< std::endl;

#ifdef BENCHMARK_CPU
    double sum_cpp = 0;
    prof.tic_cpu("C++");
    for(size_t i = 0; i < M; i++)
	sum_cpp += std::inner_product(A.begin(), A.end(), B.begin(), 0.0);
    time_elapsed = prof.toc("C++");

    {
	double gflops = 2.0 * N * M / time_elapsed / 1e9;
	double bwidth = 2.0 * N * M * sizeof(real) / time_elapsed / 1e9;

	std::cout
	    << "  C++"
	    << "\n    GFLOPS:    " << gflops
	    << "\n    Bandwidth: " << bwidth
	    << std::endl;
    }

    std::cout << "  res = " << fabs(sum_cl - sum_cpp)
	      << std::endl << std::endl;
#endif

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
std::pair<double,double> benchmark_spmv(
	std::vector<cl::CommandQueue> &queue, profiler &prof
	)
{
    // Construct matrix for 3D Poisson problem in cubic domain.
    const uint n = 128;
    const uint N = n * n * n;
    const uint M = 1024;

    double time_elapsed;

    const real h   = 1.0 / (n - 1);
    const real h2i = (n - 1) * (n - 1);

    std::vector<uint> row;
    std::vector<uint> col;
    std::vector<real> val;
    std::vector<real> X(n * n * n, 1e-2);
    std::vector<real> Y(n * n * n, 0);

    row.reserve(n * n * n + 1);
    col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
    val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

    row.push_back(0);
    for(uint k = 0, idx = 0; k < n; k++) {
	real z = k * h;
	for(uint j = 0; j < n; j++) {
	    real y = j * h;
	    for(uint i = 0; i < n; i++, idx++) {
		real x = i * h;
		if (
			i == 0 || i == (n - 1) ||
			j == 0 || j == (n - 1) ||
			k == 0 || k == (n - 1)
		   )
		{
		    col.push_back(idx);
		    val.push_back(1);
		    row.push_back(row.back() + 1);
		} else {
		    col.push_back(idx - n * n);
		    val.push_back(-h2i);

		    col.push_back(idx - n);
		    val.push_back(-h2i);

		    col.push_back(idx - 1);
		    val.push_back(-h2i);

		    col.push_back(idx);
		    val.push_back(6 * h2i);

		    col.push_back(idx + 1);
		    val.push_back(-h2i);

		    col.push_back(idx + n);
		    val.push_back(-h2i);

		    col.push_back(idx + n * n);
		    val.push_back(-h2i);

		    row.push_back(row.back() + 7);
		}
	    }
	}
    }

    uint nnz = row.back();

    // Transfer data to compute devices.
    vex::SpMat<real>  A(queue, n * n * n, row.data(), col.data(), val.data());
    vex::vector<real> x(queue, CL_MEM_READ_WRITE, X);
    vex::vector<real> y(queue, CL_MEM_READ_WRITE, Y);

    // Get timings.
    y += A * x;
    y = Const(0);

    prof.tic_cl("OpenCL");
    for(uint i = 0; i < M; i++)
	y += A * x;
    time_elapsed = prof.toc("OpenCL");

    double gflops = 15.0 * N * M / time_elapsed / 1e9;
    double bwidth = M * (7 * N * (2 * sizeof(real) + sizeof(uint)) + 2 * N * sizeof(real)) / time_elapsed / 1e9;

    std::cout
	<< "SpMV\n"
	<< "  OpenCL"
	<< "\n    GFLOPS:    " << gflops
	<< "\n    Bandwidth: " << bwidth
	<< std::endl;

#ifdef BENCHMARK_CPU
    prof.tic_cpu("C++");
    for(size_t k = 0; k < M; k++)
	for(uint i = 0; i < N; i++) {
	    real s = 0;
	    for(uint j = row[i]; j < row[i + 1]; j++)
		s += val[j] * X[col[j]];
	    Y[i] += s;
	}
    time_elapsed = prof.toc("C++");

    {
	double gflops = (2.0 * nnz + N) * M / time_elapsed / 1e9;
	double bwidth = M * (nnz * (2 * sizeof(real) + sizeof(uint)) + 4 * N * sizeof(real)) / time_elapsed / 1e9;

	std::cout
	    << "  C++"
	    << "\n    GFLOPS:    " << gflops
	    << "\n    Bandwidth: " << bwidth
	    << std::endl;
    }

    copy(Y, x);

    y -= x;

    Reductor<real,SUM> sum(queue);

    std::cout << "  res = " << sum(y * y) << std::endl << std::endl;
#endif

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
int main() {
    try {
	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	std::tie(context, queue) = queue_list(
		Filter::DoublePrecision() && Filter::Env(),
		CL_QUEUE_PROFILING_ENABLE);

	std::cout << queue << std::endl;

	std::ofstream log("profiling.dat", std::ios::app);

	log << queue.size() << " ";

	double gflops, bwidth;

	profiler prof(queue);

#ifdef BENCHMARK_VECTOR
	prof.tic_cpu("Vector arithmetic");
	std::tie(gflops, bwidth) = benchmark_vector(queue, prof);
	prof.toc("Vector arithmetic");

	log << gflops << " " << bwidth << " ";
#endif

#ifdef BENCHMARK_REDUCTOR
	prof.tic_cpu("Reduction");
	std::tie(gflops, bwidth) = benchmark_reductor(queue, prof);
	prof.toc("Reduction");

	log << gflops << " " << bwidth << " ";
#endif

#ifdef BENCHMARK_SPMAT
	prof.tic_cpu("SpMV");
	std::tie(gflops, bwidth) = benchmark_spmv(queue, prof);
	prof.toc("SpMV");

	log << gflops << " " << bwidth << std::endl;
#endif

	std::cout << prof << std::endl;
    } catch (const cl::Error &e) {
	std::cerr << e << std::endl;
	return 1;
    }
}
