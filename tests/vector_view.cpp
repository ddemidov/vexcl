#define BOOST_TEST_MODULE VectorView
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(vector_view)
{
    const size_t N = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> x = random_vector<double>(2 * N);
    vex::vector<double> X(queue, x);
    vex::vector<double> Y(queue, N);

    cl_ulong size   = N;
    cl_long  stride = 2;

    vex::gslice<1> slice(0, &size, &stride);

    Y = slice(X);

    check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK(v == x[idx * 2]); });
}

BOOST_AUTO_TEST_SUITE_END()
