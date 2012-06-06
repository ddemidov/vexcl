#include <iostream>
#include <vector>
#include <tuple>
#include <vexcl/vexcl.hpp>

using namespace vex;

int main() {
    vex::Context ctx(Filter::Env);

    const size_t n = 1024;

    multivector<float, 4> x(ctx.queue(), n);
    multivector<float, 4> y(ctx.queue(), n);
    multivector<float, 4> z(ctx.queue(), n);

    for(uint i = 0; i < 4; i++) {
	y[i] = 1;
	z[i] = 2;
    }

    x = 2 * y + z;

    Reductor<float,MIN> min(ctx.queue());
    Reductor<float,MAX> max(ctx.queue());

    for(uint i = 0; i < 4; i++)
	std::cout << i << " " << min(x[i]) << " " << max(x[i]) << std::endl;
}

