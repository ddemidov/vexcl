oclutil
=======

oclutil is header-only template library created for ease of C++ based OpenCL
development.

Selection of compute devices
----------------------------

You can select any number of available compute devices, which satisfy provided
filters. Filter is a functor returning bool and acting on a cl::Device
parameter. Several standard filters are provided, such as device type or name
filter, double precision support etc. Filters can be combined with logical
operators. In the example below all available NVIDIA GPUs are selected:

```C++
#include <iostream>
#include <oclutil/devlist.hpp>

using namespace std;

int main() {
    auto device = device_list(
        Filter::Type(CL_DEVICE_TYPE_GPU) && [](const cl::Device &d) {
            return d.getInfo<CL_DEVICE_VENDOR>() == "NVIDIA Corporation";
        });

    std::cout << device.size() << " GPUs found:" << std::endl;
    for(auto &d : device)
        std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << std::endl;
}

```
