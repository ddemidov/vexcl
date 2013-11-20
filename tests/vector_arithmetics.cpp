#define BOOST_TEST_MODULE VectorArithmetics
#include <boost/test/unit_test.hpp>
#include <vexcl/constants.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reductor.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <boost/math/constants/constants.hpp>
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

    check_sample(x, [](size_t, double a) {
            BOOST_CHECK_CLOSE(a, 5 * sin(42.0) + 67, 1e-12);
            });
}

BOOST_AUTO_TEST_CASE(compound_assignment)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, n);

    x = 0;
    x += 1;

    check_sample(x, [](size_t, double a) { BOOST_CHECK(a == 1); });

    x -= 2;

    check_sample(x, [](size_t, double a) { BOOST_CHECK(a == -1); });
}

BOOST_AUTO_TEST_CASE(reduce_expression)
{
    const size_t N = 1024;

    std::vector<double> x = random_vector<double>(N);
    vex::vector<double> X(ctx, x);

    vex::Reductor<double,vex::SUM> sum(ctx);
    vex::Reductor<double,vex::MIN> min(ctx);
    vex::Reductor<double,vex::MAX> max(ctx);

    BOOST_CHECK_CLOSE(sum(X), std::accumulate(x.begin(), x.end(), 0.0), 1e-6);

    BOOST_CHECK_CLOSE(min(X), *std::min_element(x.begin(), x.end()), 1e-6);
    BOOST_CHECK_CLOSE(max(X), *std::max_element(x.begin(), x.end()), 1e-6);

    BOOST_CHECK_SMALL(max(fabs(X - X)), 1e-12);
}

BOOST_AUTO_TEST_CASE(static_reductor)
{
    const size_t N = 1024;

    vex::vector<double> X(ctx, random_vector<double>(N));

    const vex::Reductor<double,vex::SUM> sum(ctx);
    const vex::Reductor<double,vex::SUM> &static_sum = vex::get_reductor<double, vex::SUM>(ctx);

    BOOST_CHECK_CLOSE(sum(X), static_sum(X), 1e-6);
}

BOOST_AUTO_TEST_CASE(builtin_functions)
{
    const size_t N = 1024;
    std::vector<double> x = random_vector<double>(N);
    vex::vector<double> X(ctx, x);
    vex::vector<double> Y(ctx, N);

    Y = pow(sin(X), 2.0) + pow(cos(X), 2.0);

    check_sample(Y, [](size_t, double a) { BOOST_CHECK_CLOSE(a, 1, 1e-8); });
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

    BOOST_CHECK( sum( greater(x, y) ) == 0U);
    BOOST_CHECK( sum( greater(y, x) ) == N);
}

BOOST_AUTO_TEST_CASE(user_defined_functions_same_signature)
{
    const size_t N = 1024;
    vex::vector<double> x(ctx, N);

    x = 1;

    VEX_FUNCTION(times2, double(double), "return prm1 * 2;");
    VEX_FUNCTION(times4, double(double), "return prm1 * 4;");

    vex::Reductor<size_t,vex::SUM> sum(ctx);

    BOOST_CHECK( sum( times2(x) ) == 2 * N );
    BOOST_CHECK( sum( times4(x) ) == 4 * N );
}

BOOST_AUTO_TEST_CASE(element_index)
{
    const size_t N = 1024;

    vex::vector<double> x(ctx, N);

    x = sin(0.5 * vex::element_index());

    check_sample(x, [](size_t idx, double a) { BOOST_CHECK_CLOSE(a, sin(0.5 * idx), 1e-6); });
}

#ifdef VEXCL_BACKEND_OPENCL
BOOST_AUTO_TEST_CASE(vector_values)
{
    const size_t N = 1024;

    VEX_FUNCTION(make_int4, cl_int4(int), "return (int4)(prm1, prm1, prm1, prm1);");

    cl_int4 c = {{1, 2, 3, 4}};

    vex::vector<cl_int4> X(ctx, N);
    X = c * (make_int4(5 + vex::element_index()));

    check_sample(X, [c](size_t idx, cl_int4 v) {
            for(int j = 0; j < 4; ++j)
                BOOST_CHECK(v.s[j] == c.s[j] * (5 + static_cast<int>(idx)));
            });
}
#endif

BOOST_AUTO_TEST_CASE(nested_functions)
{
    const size_t N = 1024;

    VEX_FUNCTION(f, int(int), "return 2 * prm1;");
    VEX_FUNCTION(g, int(int), "return 3 * prm1;");

    vex::vector<int> x(ctx, N);

    x = 1;
    x = f(f(x));
    check_sample(x, [](size_t, int a) { BOOST_CHECK(a == 4); });

    x = 1;
    x = g(f(x));
    check_sample(x, [](size_t, int a) { BOOST_CHECK(a == 6); });
}

BOOST_AUTO_TEST_CASE(custom_header)
{
    const size_t n = 1024;

    vex::vector<int> x(ctx, n);

    vex::push_program_header(ctx, "#define THE_ANSWER 42\n");

    VEX_FUNCTION(answer, int(int), "return prm1 * THE_ANSWER;");

    x = answer(1);

    check_sample(x, [](size_t, int a) {
            BOOST_CHECK(a == 42);
            });

    vex::pop_program_header(ctx);
}

BOOST_AUTO_TEST_CASE(function_with_preamble)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, random_vector<double>(n));
    vex::vector<double> y(ctx, n);

#ifdef VEXCL_BACKEND_OPENCL
    VEX_FUNCTION_WITH_PREAMBLE(one, double(double),
            "double sin2(double x) { return pow(sin(x), 2.0); }\n"
            "double cos2(double x) { return pow(cos(x), 2.0); }\n",
            "return sin2(prm1) + cos2(prm1);"
            );
#else
    VEX_FUNCTION_WITH_PREAMBLE(one, double(double),
            "__device__ double sin2(double x) { return pow(sin(x), 2.0); }\n"
            "__device__ double cos2(double x) { return pow(cos(x), 2.0); }\n",
            "return sin2(prm1) + cos2(prm1);"
            );
#endif

    y = one(x);

    check_sample(y, [](size_t, double a) {
            BOOST_CHECK_CLOSE(a, 1.0, 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(ternary_operator)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, random_vector<double>(n));
    vex::vector<double> y(ctx, n);

    y = if_else(x > 0.5, sin(x), cos(x));

    check_sample(x, y, [&](size_t, double X, double Y) {
            BOOST_CHECK_CLOSE(Y, X > 0.5 ? sin(X) : cos(X), 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(assign_to_ternary_operator)
{
    const size_t n = 32;

    vex::vector<double> x(ctx, random_vector<double>(n));
    vex::vector<double> y(ctx, n);
    vex::vector<double> z(ctx, n);

    y = 0;
    z = 0;

    vex::tie( *if_else(x < 0.5, &y, &z) ) = 42;

    check_sample(x, y, z, [&](size_t, double X, double Y, double Z) {
            BOOST_CHECK_EQUAL(X < 0.5 ? Y : Z, 42);
            BOOST_CHECK_EQUAL(X < 0.5 ? Z : Y, 0);
            });
}

BOOST_AUTO_TEST_CASE(combine_expressions)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, n);

    auto alpha  = vex::constants::two_pi() * vex::tag<1>(vex::element_index());
    auto sine   = sin(alpha);
    auto cosine = cos(alpha);

    x = pow(sine, 2.0) + pow(cosine, 2.0);

    check_sample(x, [](size_t, double v) { BOOST_CHECK_CLOSE(v, 1.0, 1e-8); });
}

BOOST_AUTO_TEST_CASE(constants)
{
    const size_t n = 1024;

    vex::vector<double> x(ctx, n);

    x = std::integral_constant<int, 42>();
    check_sample(x, [](size_t, double v) { BOOST_CHECK_EQUAL(v, 42); });

    x = vex::constants::pi();
    check_sample(x, [](size_t, double v) { BOOST_CHECK_CLOSE(v, boost::math::constants::pi<double>(), 1e-8); });
}

#if (VEXCL_CHECK_SIZES > 0)
BOOST_AUTO_TEST_CASE(expression_size_check)
{
    vex::vector<int> x(ctx, 16);
    vex::vector<int> y(ctx, 32);

    BOOST_CHECK_THROW(x = y, std::runtime_error);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
