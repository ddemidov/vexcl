#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>

int main() {
    const size_t n = 1024;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    vex::vector<double> x(ctx, n);
    vex::vector<double> y(ctx, n);
    x = 1;
    y = 0;

    vex::scan(x, y, /*exclusive*/true);
    std::cout << y[n - 1] << std::endl;

    vex::scan(x, y, /*inclusive*/false);
    std::cout << y[n - 1] << std::endl;
}
