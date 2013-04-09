#define BOOST_TEST_MODULE VectorConstruct
#include <boost/test/unit_test.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(assign_expression)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);
    vex::vector<double> y(ctx, N);
    vex::vector<double> z(ctx, N);

    y = 42;
    z = 67;
    x = 5 * sin(y) + z;

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(
                true,
                fabs(x[idx] - (5 * sin(42.0) + 67)) < 1e-12
                );
    }
}

BOOST_AUTO_TEST_CASE(reduce_expression)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, x);

    vex::Reductor<double,vex::SUM> sum(ctx);
    vex::Reductor<double,vex::MIN> min(ctx);
    vex::Reductor<double,vex::MAX> max(ctx);

    BOOST_CHECK_EQUAL(
            true,
            fabs(sum(X) - std::accumulate(x.begin(), x.end(), 0.0)) < 1e-6
            );

    BOOST_CHECK_EQUAL(
            true,
            fabs(min(X) - *std::min_element(x.begin(), x.end())) < 1e-6
            );

    BOOST_CHECK_EQUAL(
            true,
            fabs(max(X) - *std::max_element(x.begin(), x.end())) < 1e-6
            );

    BOOST_CHECK_EQUAL(
            true,
            max(fabs(X - X)) < 1e-12
            );
}

BOOST_AUTO_TEST_CASE(builtin_functions)
{
    const size_t N = 1024;
    std::vector<double> x = random_vector(N);
    vex::vector<double> X(ctx, x);
    vex::vector<double> Y(ctx, N);

    Y = pow(sin(X), 2.0) + pow(cos(X), 2.0);

    for(size_t i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(true, fabs(Y[idx] - 1) < 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(user_defined_functions)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);
    vex::vector<double> y(ctx, N);

    x = 1;
    y = 2;

    VEX_FUNCTION(greater, size_t(double, double), "return prm1 > prm2;");

    vex::Reductor<size_t,vex::SUM> sum(ctx);

    BOOST_CHECK_EQUAL( sum( greater(x, y) ), 0U);
    BOOST_CHECK_EQUAL( sum( greater(y, x) ), N);
}

BOOST_AUTO_TEST_CASE(user_defined_functions_same_signature)
{
    const size_t N = 1024;
    vex::vector<double> x(ctx, N);

    x = 1;

    VEX_FUNCTION(times2, double(double), "return prm1 * 2;");
    VEX_FUNCTION(times4, double(double), "return prm1 * 4;");

    vex::Reductor<size_t,vex::SUM> sum(ctx);

    BOOST_CHECK_EQUAL( sum( times2(x) ), 2 * N );
    BOOST_CHECK_EQUAL( sum( times4(x) ), 4 * N );
}

BOOST_AUTO_TEST_CASE(element_index)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);

    x = sin(0.5 * vex::element_index());

    for(int i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(true, fabs(x[idx] - sin(0.5 * idx)) < 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(vector_values)
{
    const size_t N = 16 * 1024;

    VEX_FUNCTION(make_int4, cl_int4(int), "return (int4)(prm1, prm1, prm1, prm1);");

    cl_int4 c = {{1, 2, 3, 4}};

    vex::vector<cl_int4> X(ctx, N);
    X = c * (make_int4(5 + vex::element_index()));

    for(int i = 0; i < 100; ++i) {
        size_t idx = rand() % N;

        cl_int4 v = X[idx];

        for(int j = 0; j < 4; ++j)
            BOOST_CHECK_EQUAL(v.s[j] - c.s[j] * (5 + static_cast<int>(idx)), 0);
    }
}

BOOST_AUTO_TEST_CASE(nested_functions)
{
    const size_t N = 1024;

    VEX_FUNCTION(f, int(int), "return 2 * prm1;");
    VEX_FUNCTION(g, int(int), "return 3 * prm1;");

    vex::vector<int> data(ctx, N);

    data = 1;
    data = f(f(data));

    for(int i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(data[idx], 4);
    }

    data = 1;
    data = g(f(data));

    for(int i = 0; i < 100; ++i) {
        size_t idx = rand() % N;
        BOOST_CHECK_EQUAL(data[idx], 6);
    }
}

BOOST_AUTO_TEST_SUITE_END()
