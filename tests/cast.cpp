#define BOOST_TEST_MODULE VectorArithmetics
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/cast.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(casted_expession)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);

    x = vex::cast<double>(5);

    check_sample(x, [](size_t, double a) { BOOST_CHECK_EQUAL(a, 5); });
}

BOOST_AUTO_TEST_SUITE_END()

