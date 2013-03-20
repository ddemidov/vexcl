#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>

int main() {
    const size_t n = 1 << 20;
    const size_t m = 1 << 10;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    vex::profiler prof(ctx);

    vex::vector<float> x(ctx, n);
    vex::vector<float> y(ctx, n);
    x = 0;
    y = 1;

    prof.tic_cl("vexcl");
    for(size_t i = 0; i < m; ++i) {
        x = x + y;
    }
    prof.toc("vexcl");

    boost::compute::command_queue q( ctx.queue(0)() );

    auto x_begin = vex::compute::begin(x, 0);
    auto x_end   = vex::compute::end  (x, 0);
    auto y_begin = vex::compute::begin(y, 0);

    prof.tic_cl("compute");
    for(size_t i = 0; i < m; ++i) {
        boost::compute::transform(
                x_begin, x_end, y_begin, x_begin, boost::compute::plus<float>(), q
                );
    }
    prof.toc("compute");

    std::cout << prof << std::endl;
}
