#include <iostream>
#include <oclutil/oclutil.hpp>
using namespace clu;
int main() {
    cl::Context context;
    std::vector<cl::CommandQueue> queue;
    // Select no more than 2 NVIDIA GPUs:
    std::tie(context, queue) = queue_list(
	    [](const cl::Device &d) {
	    return d.getInfo<CL_DEVICE_VENDOR>() == "NVIDIA Corporation";
	    } && Filter::Count(2), true
	    );
}

