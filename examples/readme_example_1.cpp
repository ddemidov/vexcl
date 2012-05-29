#include <iostream>
#include <vexcl/vexcl.hpp>
using namespace vex;
int main() {
    auto device = device_list(
        Filter::Name("Radeon") && Filter::DoublePrecision()
        );
    std::cout << device << std::endl;
}
