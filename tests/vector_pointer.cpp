#define BOOST_TEST_MODULE VectorPointer
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_pointer.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <vexcl/temporary.hpp>
#include <vexcl/constants.hpp>
#include <vexcl/function.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(nbody)
{
    const size_t n = 1024;

    std::vector<vex::command_queue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    VEX_FUNCTION(double, nbody, (size_t, n)(size_t, j)(double*, x),
            double sum = 0;
            for(size_t i = 0; i < n; ++i)
                if (i != j) sum += x[i];
                    return sum;
            );

    y = nbody(n, vex::element_index(), vex::raw_pointer(x));

    check_sample(y, [&](size_t idx, double v) {
            double sum = 0;
            for(size_t i = 0; i < n; ++i)
                if (i != idx) sum += X[i];
            BOOST_CHECK_CLOSE(v, sum, 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(manual_stencil)
{
    const size_t n = 1024;

    std::vector<vex::command_queue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    VEX_CONSTANT(nil, 0);
    VEX_CONSTANT(one, 1);
    VEX_CONSTANT(two, 2);

    auto N = vex::tag<1>( x.size() );
    auto p = vex::raw_pointer(x);

    auto i     = vex::make_temp<1>( vex::element_index() );
    auto left  = vex::make_temp<2>( if_else(i > nil(), i - one(), i ) );
    auto right = vex::make_temp<3>( if_else(i + one() < N, i + one(), i ) );

    // Use pointer arithmetics
    y = *(p + i) * two() - *(p + left) - *(p + right);

    check_sample(y, [&](size_t idx, double v) {
            double xc = X[idx];
            double xl = X[idx > 0 ? idx - 1 : idx];
            double xr = X[idx + 1 < n ? idx + 1 : idx];
            BOOST_CHECK_CLOSE(v, 2 * xc - xr - xl, 1e-8);
            });

    // Same thing with index operators
    y = p[i] * two() - p[left] - p[right];

    check_sample(y, [&](size_t idx, double v) {
            double xc = X[idx];
            double xl = X[idx > 0 ? idx - 1 : idx];
            double xr = X[idx + 1 < n ? idx + 1 : idx];
            BOOST_CHECK_CLOSE(v, 2 * xc - xr - xl, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()

