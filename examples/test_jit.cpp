#include <iostream>
#include <string>
#include <vector>

#include <vexcl/devlist.hpp>
#include <vexcl/backend.hpp>

int main() {
    using namespace vex;

    Context ctx(Filter::Any);
    std::cout << ctx << std::endl;

    backend::source_generator src(ctx.queue(0));

    src.begin_kernel("sum");
    src.begin_kernel_parameters();
    src.parameter<size_t>("n");
    src.parameter< global_ptr<double> >("x");
    src.parameter< global_ptr<double> >("y");
    src.end_kernel_parameters();
    src.grid_stride_loop().open("{");
    src.new_line() << "y[idx] += x[idx];";
    src.close("}").end_kernel();

    backend::kernel sum(ctx.queue(0), src.str(), "sum");

    const size_t n = 1024 * 1024;
    std::vector<double> x(n, 1.0), y(n, 1.0);

    sum(ctx.queue(0), n, x.data(), y.data());

    for(size_t i = 0; i < n; ++i) {
        std::cout << y[i] << " ";
        if (i == 2) {
            std::cout << "... ";
            i = n - 4;
        }
    }
    std::cout << std::endl;
}
