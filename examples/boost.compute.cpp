#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>

int main() {
    const size_t n = 1024;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    vex::vector<float> x(ctx, n);
    vex::vector<float> y(ctx, n);
    x = 1;
    y = 0;

    vex::Reductor<float, vex::SUM> sum(ctx);

    // This works:
    float s = 0;
    for(size_t d = 0; d < ctx.size(); ++d) {
        boost::compute::command_queue q( ctx.queue(d)() );

        s += boost::compute::accumulate(
                vex::compute::begin(x, d), vex::compute::end(x, d), 0.0f, q
                );
    }
    std::cout << sum(x) << " == " << s << std::endl;

    // And these don't:
    vex::scan(x, y, /*exclusive*/true);
    std::cout << y[n-1] << std::endl;

    vex::scan(x, y, /*inclusive*/false);
    std::cout << y[n-1] << std::endl;
}
