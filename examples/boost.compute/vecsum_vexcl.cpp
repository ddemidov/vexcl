#include <vexcl/vexcl.hpp>

int main() {
    const size_t n = 1 << 10;
    const size_t m = 1 << 12;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    vex::vector<float> x(ctx, n);
    vex::vector<float> y(ctx, n);
    x = 0;
    y = 1;

    for(size_t i = 0; i < m; ++i) {
        x = x + y;
    }

    ctx.queue(0).finish();
}
