#define BOOST_TEST_MODULE ClogsScan
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/external/clogs.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(clogs_scan_scalar)
{
    const size_t n = 1024;

    std::vector<cl_int> x = random_vector<cl_int>(n);

    vex::vector<cl_int> X(ctx, x);
    vex::vector<cl_int> Y(ctx, n);

    vex::exclusive_scan(X, Y);

    std::partial_sum(x.begin(), x.end(), x.begin());
    std::rotate(x.begin(), x.end() - 1, x.end());
    x[0] = 0;

    check_sample(Y, [&](size_t idx, cl_int a) {
        BOOST_CHECK_EQUAL(a, x[idx]);
    });
}

// Test with a vector type to ensure that the type inference works
BOOST_AUTO_TEST_CASE(clogs_scan_vector)
{
    const size_t n = 1234;

    std::vector<cl_uint4> x = random_vector<cl_uint4>(n);

    vex::vector<cl_uint4> X(ctx, x);
    vex::vector<cl_uint4> Y(ctx, n);

    vex::exclusive_scan(X, Y);

    std::partial_sum(x.begin(), x.end(), x.begin());
    std::rotate(x.begin(), x.end() - 1, x.end());
    x[0] = cl_uint4{};

    check_sample(Y, [&](size_t idx, cl_uint4 a) {
        for (int i = 0; i < 4; i++)
            BOOST_CHECK_EQUAL(a.s[i], x[idx].s[i]);
    });
}

BOOST_AUTO_TEST_SUITE_END()
