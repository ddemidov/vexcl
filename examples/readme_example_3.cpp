#include <iostream>
#include <vector>
#include <cassert>
#include <oclutil/oclutil.hpp>
using namespace clu;
int main() {
    const uint n = 1 << 20;
    std::vector<double> x(n);
    std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

    cl::Context context;
    std::vector<cl::CommandQueue> queue;
    std::tie(context, queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU));

    clu::vector<double> X(queue, CL_MEM_READ_ONLY,  x);
    clu::vector<double> Y(queue, CL_MEM_READ_WRITE, n);
    clu::vector<double> Z(queue, CL_MEM_READ_WRITE, n);
    Y = Const(42);
    Z = Sqrt(Const(2) * X) + Cos(Y);
    copy(Z, x);
    assert(x[42] == Z[42]);
    Reductor<double> sum(queue);
    std::cout << sum(Z) << std::endl;
    std::cout << sum(Sqrt(Const(2) * X) + Cos(Y)) << std::endl;
}
