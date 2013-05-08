#include <iostream>
#include <vexcl/devlist.hpp>
using namespace vex;


int main() {
    std::cout << "OpenCL devices:" << std::endl << std::endl;
    auto dev = device_list(Filter::All);
    for (auto d = dev.begin(); d != dev.end(); d++) {
        std::cout << "  " << d->getInfo<CL_DEVICE_NAME>() << std::endl
                  << "    " << std::left << std::setw(32)
                  << "CL_PLATFORM_NAME" << " = "
                  << cl::Platform(d->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>()
                  << std::endl;

#define SHOW_DEVPROP(name) \
        std::cout << "    " << std::left << std::setw(32) << #name << " = " \
                  << d->getInfo< name >() << std::endl

        SHOW_DEVPROP(CL_DEVICE_VENDOR);
        SHOW_DEVPROP(CL_DEVICE_MAX_COMPUTE_UNITS);
        SHOW_DEVPROP(CL_DEVICE_HOST_UNIFIED_MEMORY);
        SHOW_DEVPROP(CL_DEVICE_GLOBAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_LOCAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        SHOW_DEVPROP(CL_DEVICE_MAX_CLOCK_FREQUENCY);

#undef SHOW_DEVPROP

        std::cout << std::endl;
    }
}

// vim: et
