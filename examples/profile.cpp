#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

using namespace vex;

typedef double real;

int main() {
    try {
	const uint N = 1024 * 1024;
	const uint M = 1024;

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

	vex::vector<real> a(queue, CL_MEM_READ_WRITE, N);
	vex::vector<real> b(queue, CL_MEM_READ_WRITE, N);
	vex::vector<real> c(queue, CL_MEM_READ_WRITE, N);
	vex::vector<real> d(queue, CL_MEM_READ_WRITE, N);

	std::vector<real> A(N);
	std::vector<real> B(N, 1);
	std::vector<real> C(N, 2);
	std::vector<real> D(N, 3);

	b = Const(1);
	c = Const(2);
	d = Const(3);

	a = b + c * d;

	profiler prof(queue);

	prof.tic_gpu("OpenCL");

	for(uint i = 0; i < M; i++)
	    a = b + c * d;

	{
	real time_elapsed = prof.toc("OpenCL");

	std::cout << "OpenCL\n  GFLOPS:           "
	          << (2.0 * N * M) / time_elapsed / 1e9 << std::endl;

	std::cout << "  Memory bandwidth: "
	          << (4.0 * N * M * sizeof(real)) / time_elapsed / 1e9
		  << std::endl;
	}

	prof.tic_cpu("C++");
	for(uint i = 0; i < M; i++)
	    for(uint j = 0; j < N; j++)
		A[j] = B[j] + C[j] * D[j];
	{
	real time_elapsed = prof.toc("C++");

	std::cout << "C++\n  GFLOPS:           "
	          << (2.0 * N * M) / time_elapsed / 1e9 << std::endl;

	std::cout << "  Memory bandwidth: "
	          << (4.0 * N * M * sizeof(real)) / time_elapsed / 1e9
		  << std::endl;
	}

	std::cout << prof << std::endl;
    } catch (const cl::Error &e) {
	std::cerr << e << std::endl;
	return 1;
    }
}
