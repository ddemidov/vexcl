#include <iostream>
#include <oclutil/oclutil.hpp>
using namespace clu;

int main() {
    std::cout << "OpenCL devices:" << std::endl;
    auto dev = device_list(Filter::All(), true);
}
