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

    std::cout << x[0] << std::endl;
    x = 0;

    boost::compute::command_queue q( ctx.queue(0)() );

    boost::compute::buffer xbuf( x(0)() );
    boost::compute::buffer ybuf( y(0)() );

    auto x_begin = boost::compute::make_buffer_iterator<float>(xbuf, 0);
    auto x_end   = boost::compute::make_buffer_iterator<float>(xbuf, n);
    auto y_begin = boost::compute::make_buffer_iterator<float>(ybuf, 0);

    prof.tic_cl("compute");
    for(size_t i = 0; i < m; ++i) {
        boost::compute::transform(
                x_begin, x_end, y_begin, x_begin, boost::compute::plus<float>(), q
                );
    }
    prof.toc("compute");

    std::cout << x[0] << std::endl;

    std::cout << prof << std::endl;
}
