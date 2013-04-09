#define BOOST_TEST_MODULE VectorConstruct
#include <boost/test/unit_test.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(empty)
{
    vex::vector<double> x;

    BOOST_CHECK_EQUAL(0u, x.size());
    BOOST_CHECK_EQUAL(0, x.end() - x.begin());
}

BOOST_AUTO_TEST_CASE(size)
{
    const size_t N = 1024;
    vex::vector<double> x(ctx, N);

    BOOST_CHECK_EQUAL(x.size(), N);
    BOOST_CHECK_EQUAL(x.end() - x.begin(), static_cast<ptrdiff_t>(N));
}

BOOST_AUTO_TEST_CASE(std_vector)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, x);

    BOOST_CHECK_EQUAL(X.size(), x.size());

    std::vector<double> y(N);
    copy(X, y);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(x[idx], y[idx]);
    }
}

BOOST_AUTO_TEST_CASE(host_pointer)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, N, x.data());

    BOOST_CHECK_EQUAL(X.size(), x.size());

    std::vector<double> y(N);
    copy(X, y);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(x[idx], y[idx]);
    }
}

BOOST_AUTO_TEST_CASE(copy_constructor)
{
    const size_t N = 1024;

    vex::vector<double> x1;
    vex::vector<double> x2(x1);

    BOOST_CHECK_EQUAL(x1.size(), 0U);
    BOOST_CHECK_EQUAL(x1.size(), x2.size());

    vex::vector<double> y1(ctx, random_vector(N));
    vex::vector<double> y2(y1);

    BOOST_CHECK_EQUAL(y1.size(), N);
    BOOST_CHECK_EQUAL(y1.size(), y2.size());

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(
                static_cast<double>(y1[idx]),
                static_cast<double>(y2[idx])
                );
    }
}

BOOST_AUTO_TEST_CASE(move_constructor)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);
    x = 42;

    vex::vector<double> y = std::move(x);

    BOOST_CHECK_EQUAL(x.size(), 0U);
    BOOST_CHECK_EQUAL(y.size(), N);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(static_cast<double>(y[idx]), 42);
    }
}

BOOST_AUTO_TEST_CASE(move_assign)
{
    const size_t N = 1024;
    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, x);

    vex::vector<double> Y;
    Y = std::move(X);

    BOOST_CHECK_EQUAL(Y.size(), x.size());

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(static_cast<double>(Y[idx]), x[idx]);
    }
}

BOOST_AUTO_TEST_CASE(vector_swap)
{
    const size_t N = 1024;
    const size_t M = 512;

    vex::vector<double> x(ctx, N);
    vex::vector<double> y(ctx, M);

    x = 42;
    y = 67;

    swap(x, y);

    BOOST_CHECK_EQUAL(y.size(), N);
    BOOST_CHECK_EQUAL(x.size(), M);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(static_cast<double>(y[idx]), 42);
    }

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % M;
        BOOST_CHECK_EQUAL(static_cast<double>(x[idx]), 67);
    }
}

BOOST_AUTO_TEST_CASE(vector_resize_to_std_vector)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector(N);
    vex::vector<double> X;

    X.resize(ctx, x);
    BOOST_CHECK_EQUAL(X.size(), x.size());

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(static_cast<double>(X[idx]), x[idx]);
    }
}

BOOST_AUTO_TEST_CASE(vector_resize_to_vex_vector)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);
    x = 42;

    vex::vector<double> y;
    y.resize(x);

    BOOST_CHECK_EQUAL(y.size(), x.size());

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(
                static_cast<double>(x[idx]),
                static_cast<double>(y[idx])
                );
    }
}

BOOST_AUTO_TEST_CASE(stl_container_of_vex_vector)
{
    const size_t N = 1024;
    const size_t M = 16;

    std::vector< vex::vector<unsigned> > x;

    for(size_t i = 0; i < M; ++i) {
        x.push_back( vex::vector<unsigned>(ctx, N) );
        x.back() = i;
    }

    for(size_t j = 0; j < M; ++j) {
        for(size_t i = 0; i < 100; ++i) {
            size_t idx = rand() % N;
            BOOST_CHECK_EQUAL(static_cast<double>(x[j][idx]), j);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
