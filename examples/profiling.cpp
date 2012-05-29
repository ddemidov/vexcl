#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

using namespace vex;

typedef double real;

void benchmark_vector(std::vector<cl::CommandQueue> &queue, profiler &prof) {
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

    std::cout << "Vector arithmetic" << std::endl;
    std::cout << "  OpenCL\n    GFLOPS:    "
	      << (3.0 * N * M) / time_elapsed / 1e9 << std::endl;

    std::cout << "    Bandwidth: "
	      << (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9
	      << std::endl;

    prof.tic_cpu("C++");
    for(uint i = 0; i < M; i++)
	for(uint j = 0; j < N; j++)
	    A[j] += B[j] + C[j] * D[j];
    time_elapsed = prof.toc("C++");

    std::cout << "  C++\n    GFLOPS:    "
	      << (3.0 * N * M) / time_elapsed / 1e9 << std::endl;

    std::cout << "    Bandwidth: "
	      << (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9
	      << std::endl;

    vex::copy(A, b);
    Reductor<real,SUM> sum(queue);

    a -= b;
    std::cout << "  res = " << sum(a * a) << std::endl;
}

int main() {
    try {
	const char *platform = getenv("OCL_PLATFORM");
	const char *device   = getenv("OCL_DEVICE");

	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	std::tie(context, queue) = queue_list(
		Filter::Platform(platform ? platform : "") &&
		Filter::Name(device ? device : ""),
		CL_QUEUE_PROFILING_ENABLE
		);

	std::cout << queue << std::endl;

	profiler prof(queue);

	prof.tic_cpu("Vector arithmetic");
	benchmark_vector(queue, prof);
	prof.toc("Vector arithmetic");

	std::cout << prof << std::endl;
    } catch (const cl::Error &e) {
	std::cerr << e << std::endl;
	return 1;
    }
}
