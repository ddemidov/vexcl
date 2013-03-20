#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>

int main() {
    const size_t n = 1 << 10;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    vex::vector<float> x(ctx, n);
    vex::vector<float> y(ctx, n);
    x = 1;
    y = 0;

    vex::exclusive_scan(x, y);
    std::cout << y[n-1] << std::endl;

    vex::inclusive_scan(x, y);
    std::cout << y[n-1] << std::endl;
}

