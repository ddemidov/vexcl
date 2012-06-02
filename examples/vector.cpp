#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <algorithm>
#include <vexcl/vexcl.hpp>

#ifdef WIN32
#  pragma warning (disable : 4244)
#endif

using namespace vex;

int main() {
    srand(time(0));

    try {
	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	// Select every device supporting double precision.
	std::tie(context, queue) = queue_list(Filter::DoublePrecision(), true);

	if (queue.empty()) {
	    std::cerr << "No OpenCL devices found." << std::endl;
	    return 1;
	}

	std::cout << queue << std::endl;

	// Allocate host vector, fill it with random values.
	const uint N = 1024 * 1024;
	std::vector<double> host_vec(N);
	std::generate(host_vec.begin(), host_vec.end(), []() {
		return double(rand()) / RAND_MAX;
		});

	// Allocate device vector initialized with host vector data.
	// Device vector will be partitioned between selected devices.
	vex::vector<double> x(queue, host_vec);

	// Allocate uninitialized device vectors.
	vex::vector<double> y(queue, N);
	vex::vector<double> z(queue, N);

	// Appropriate kernels are compiled (once) and called automagically:
	// Fill device vector with constant value...
	y = 42;
	// ... or with some expression:
	z = Sqrt(2 * x) + Cos(y);

	Reductor<double,SUM> sum(queue);

	std::cout << "y * y = " << sum(y * y) << std::endl;

	// Check results at random location.
	uint pos = rand() % N;

	// You can read or write device vector elements with [] notation (very
	// ineffective but sometimes convenient).
	std::cout << y[pos] << std::endl;
	std::cout << "res = " << z[pos] - sqrt(2 * x[pos]) - cos(y[pos]) << std::endl;

	// Or you can read the entire vector to host:
	copy(z, host_vec);

	vex::copy(host_vec.begin(), host_vec.end(), z.begin());
	vex::copy(z.begin(), z.end(), host_vec.data());
    } catch (const cl::Error &e) {
	std::cout << "OpenCL error: " << e << std::endl;
    }
}
