#define BOOST_TEST_MODULE BoostComputeScan
#include <boost/test/unit_test.hpp>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(boost_compute_scan)
{
    const size_t n = 1024;

    std::vector<double> x = random_vector(n);

    vex::vector<double> X(ctx, x);
    vex::vector<double> Y(ctx, n);

    vex::exclusive_scan(X, Y);

    std::partial_sum(x.begin(), x.end(), x.begin());

    check_sample(Y, [&](size_t idx, double a) {
            BOOST_CHECK_CLOSE(a, idx ? x[idx - 1] : 0.0, 1e-8);
            });

    vex::inclusive_scan(X, Y);

    check_sample(Y, [&](size_t idx, double a) {
            BOOST_CHECK_CLOSE(a, x[idx], 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()
