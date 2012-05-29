#include <iostream>
#include <vector>
#include <tuple>

#define VEXCL_SMART_PARTITION
#include <vexcl/vexcl.hpp>

using namespace vex;

typedef double real;

int main() {
    try {
	const uint N = 1024 * 1024;

	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	std::tie(context, queue) = queue_list(Filter::All(),
		CL_QUEUE_PROFILING_ENABLE
		);
	std::cout << queue << std::endl;

	auto part = partition(1024, queue);

	for(auto p = part.begin(); p != part.end(); p++)
	    std::cout << *p << " ";
	std::cout << std::endl;
    } catch (const cl::Error &e) {
	std::cerr << e << std::endl;
	return 1;
    }
}
