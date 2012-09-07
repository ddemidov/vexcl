#include <iostream>
#include <vexcl/vexcl.hpp>
using namespace vex;

int main() {
    std::cout << "OpenCL devices:" << std::endl << std::endl;
    auto dev = device_list(Filter::All);
    for (auto d = dev.begin(); d != dev.end(); d++) {
	std::cout << "  " << d->getInfo<CL_DEVICE_NAME>() << std::endl
		  << "    CL_DEVICE_VENDOR_ID           = " << d->getInfo<CL_DEVICE_VENDOR_ID>() << std::endl
		  << "    CL_PLATFORM_NAME              = " << cl::Platform(d->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>() << std::endl
		  << "    CL_DEVICE_OPENCL_C_VERSION    = " << d->getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl
		  << "    CL_DEVICE_MAX_COMPUTE_UNITS   = " << d->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
	          << "    CL_DEVICE_HOST_UNIFIED_MEMORY = " << d->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>() << std::endl
		  << "    CL_DEVICE_GLOBAL_MEM_SIZE     = " << d->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl
		  << "    CL_DEVICE_LOCAL_MEM_SIZE      = " << d->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl
		  << "    CL_DEVICE_MAX_CLOCK_FREQUENCY = " << d->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl
		  << std::endl;
    }
}
