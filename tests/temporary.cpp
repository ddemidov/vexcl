#define BOOST_TEST_MODULE TemporaryTerminal
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/temporary.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(temporary)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, random_vector<double>(n));
    vex::vector<double> y(ctx, n);

    VEX_FUNCTION(sqr, double(double), "return prm1 * prm1;");

    auto s = vex::make_temp<double,1>( sqr(x) + 25 );
    y = s * (x + s);

    check_sample(y, [&](size_t idx, double v) {
            double X = x[idx];
            BOOST_CHECK_CLOSE(v, X * X * (X + X * X), 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(nested_temporary)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, random_vector<double>(n));
    vex::vector<double> y(ctx, n);

    auto t1 = vex::make_temp<double, 1>( log(x) );
    auto t2 = vex::make_temp<double, 2>( t1 + sin(x) );

    y = t1 * t2;

    check_sample(y, [&](size_t idx, double v) {
            double X = x[idx];
            BOOST_CHECK_CLOSE(v, log(X) * (log(X) + sin(X)), 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()

