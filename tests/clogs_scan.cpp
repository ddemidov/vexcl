#define BOOST_TEST_MODULE ClogsScan
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/external/clogs.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(clogs_scan)
{
    const size_t n = 1024;

    std::vector<cl_int> x = random_vector<cl_int>(n);

    vex::vector<cl_int> X(ctx, x);
    vex::vector<cl_int> Y(ctx, n);

    vex::exclusive_scan(X, Y);

    std::partial_sum(x.begin(), x.end(), x.begin());
    std::rotate(x.begin(), x.end() - 1, x.end());
    x[0] = 0.0;

    check_sample(Y, [&](size_t idx, cl_int a) {
            BOOST_CHECK_EQUAL(a, x[idx]);
            });
}

BOOST_AUTO_TEST_SUITE_END()
