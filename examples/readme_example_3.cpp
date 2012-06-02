#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>
#include <vexcl/vexcl.hpp>
using namespace vex;

int main() {
    const uint n = 1 << 20;
    std::vector<double> x(n);
    std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

    std::vector<cl::Context>      context;
    std::vector<cl::CommandQueue> queue;
    std::tie(context, queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision());

    vex::vector<double> X(queue, CL_MEM_READ_ONLY,  x);
    vex::vector<double> Y(queue, CL_MEM_READ_WRITE, n);
    vex::vector<double> Z(queue, CL_MEM_READ_WRITE, n);
    Y = 42;
    Z = Sqrt(2 * X) + Cos(Y);
    copy(Z, x);
    assert(x[42] == Z[42]);
    Reductor<double,SUM> sum(queue);
    Reductor<double,MAX> max(queue);
    std::cout << max(Abs(X) - 0.5) << std::endl;
    std::cout << sum(Sqrt(2 * X) + Cos(Y)) << std::endl;
}
