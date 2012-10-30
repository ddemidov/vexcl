#include <iostream>
#include <cmath>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/devlist.hpp>
#include <vexcl/reduce_proto.hpp>

extern const char greater_body[] = "return prm1 < prm2;";
vex::UserFunction<greater_body, bool(double, double)> greater;

int main() {
    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    const size_t n = 1024 * 1024;

    vex::vector<double> a(ctx.queue(), n);
    vex::vector<double> b(ctx.queue(), n);
    vex::vector<double> c(ctx.queue(), n);

    a = 1;
    b = 2;
    c = sin(M_PI/4 * a) + b;

    std::cout << c[42] << std::endl;

    vex::Reductor<double, vex::SUM> sum(ctx.queue());
    vex::Reductor<double, vex::MAX> max(ctx.queue());

    std::cout << sum(a) / n << std::endl;
    std::cout << max(sin(M_PI/4 * a) + b) << std::endl;
}
