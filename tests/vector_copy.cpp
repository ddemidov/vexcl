#define BOOST_TEST_MODULE VectorCopy
#include <boost/test/unit_test.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(iterate_over_vector)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);
    x = 42;

    BOOST_CHECK_EQUAL(42, *std::min_element(x.begin(), x.end()));
}

BOOST_AUTO_TEST_CASE(element_access)
{
    const size_t N = 1024;
    vex::vector<double> x(ctx, N);

    for(size_t i = 0; i < N; i++)
        x[i] = 42;

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(static_cast<double>(x[idx]), 42);
    }
}

BOOST_AUTO_TEST_CASE(copy_to_std_vector)
{
    const size_t N = 1024;
    vex::vector<double> X(ctx, N);
    std::vector<double> x(N);

    X = 42;
    copy(X, x);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(x[idx], 42);
    }

    X = 67;
    vex::copy(X.begin(), X.end(), x.begin());

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(x[idx], 67);
    }
}

BOOST_AUTO_TEST_CASE(copy_from_std_vector)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, N);

    copy(x, X);
    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(X[idx], x[idx]);
    }

    std::fill(x.begin(), x.end(), 67);
    vex::copy(x.begin(), x.end(), X.begin());
    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(X[idx], 67);
    }
}

BOOST_AUTO_TEST_SUITE_END()

