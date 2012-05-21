#include <iostream>
#include <oclutil/oclutil.hpp>
using namespace clu;
int main() {
    auto device = device_list(
        Filter::Name("Radeon") && Filter::DoublePrecision()
        );
    std::cout << device.size() << " GPUs found:" << std::endl;
    for(auto d = device.begin(); d != device.end(); d++)
        std::cout << "\t" << d->getInfo<CL_DEVICE_NAME>() << std::endl;
}
