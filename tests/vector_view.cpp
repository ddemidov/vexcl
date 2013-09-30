#define BOOST_TEST_MODULE VectorView
#include <valarray>
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/temporary.hpp>
#include <vexcl/reductor.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(vector_view_1d)
{
    const size_t N = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> x = random_vector<double>(2 * N);
    vex::vector<double> X(queue, x);
    vex::vector<double> Y(queue, N);

    size_t size   = N;
    size_t stride = 2;

    vex::gslice<1> slice(0, &size, &stride);

    Y = slice(X);

    check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK(v == x[idx * 2]); });
}

BOOST_AUTO_TEST_CASE(vector_view_2)
{
    const size_t N = 32;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::valarray<double> x(N * N);
    std::iota(&x[0], &x[N * N], 0);

    // Select every even point from sub-block [(2,4) - (10,10)]:
    size_t start    = 2 * N + 4;
    size_t size[]   = {5, 4};
    size_t stride[] = {2 * N, 2};

    std::gslice std_slice(start, std::valarray<size_t>(size, 2), std::valarray<size_t>(stride, 2));

    std::valarray<double> y = x[std_slice];

    vex::vector<double> X(queue, N * N, &x[0]);
    vex::vector<double> Y(queue, size[0] * size[1]);

    vex::gslice<2> vex_slice(start, size, stride);

    Y = vex_slice(X);

    check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, y[idx]); });
}

BOOST_AUTO_TEST_CASE(vector_slicer_2d)
{
    using vex::range;
    using vex::_;

    const size_t N = 32;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::valarray<double> x(N * N);
    std::iota(&x[0], &x[N * N], 0);

    // Select every even point from sub-block [(2,4) - (10,10)]:
    size_t start    = 2 * N + 4;
    size_t size[]   = {5, 4};
    size_t stride[] = {2 * N, 2};

    std::gslice std_slice(start, std::valarray<size_t>(size, 2), std::valarray<size_t>(stride, 2));
    std::valarray<double> y = x[std_slice];

    vex::vector<double> X(queue, N * N, &x[0]);
    vex::vector<double> Y(queue, size[0] * size[1]);
    vex::vector<double> Z(queue, N);

    size_t dim[2] = {N, N};

    vex::slicer<2> slicer(dim);



    auto slice = slicer[range(2, 2, 11)][range(4, 2, 11)];

    Y = slice(X);

    check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, y[idx]); });



    Z = slicer[5](X); // Put fith row of X into Z;

    check_sample(Z, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[5 * N + idx]); });



    Z = slicer[_][5](X); // Puth fith column of X into Z;

    check_sample(Z, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[5 + N * idx]); });
}

BOOST_AUTO_TEST_CASE(vector_permutation)
{
    const size_t N = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> x = random_vector<double>(N);
    vex::vector<double> X(queue, x);
    vex::vector<double> Y(queue, N);

    {
        vex::vector<size_t> I(queue, N);
        I = N - 1 - vex::element_index();
        auto reverse = vex::permutation(I);

        Y = reverse(X);
        check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[N - 1 - idx]); });
    }

    Y = 0;

    {
        auto reverse = vex::permutation(N - 1 - vex::element_index());

        Y = reverse(X);
        check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[N - 1 - idx]); });

        Y = 0;

        reverse(Y) = X;
        check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[N - 1 - idx]); });
    }

    Y = 0;

    {
        auto i = vex::make_temp<1>( vex::element_index() + 1);
        Y = vex::permutation( N - i )(X);

        check_sample(Y, [&](size_t idx, double v) { BOOST_CHECK_EQUAL(v, x[N - 1 - idx]); });
    }
}

BOOST_AUTO_TEST_CASE(reduce_slice)
{
    const size_t N = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    vex::vector<int> X(queue, N);
    vex::Reductor<int, vex::SUM> sum(queue);
    vex::slicer<1> slice(&N);

    X = 1;

    BOOST_CHECK_EQUAL(static_cast<int>(N / 2), sum( slice[vex::range(0, 2, N)](X) ));
}

BOOST_AUTO_TEST_CASE(assign_to_view) {
    using vex::range;
    using vex::_;

    const size_t m = 32;
    const size_t n = m * m;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    vex::vector<int> x(queue, n);
    vex::slicer<1> slicer1(vex::extents[n]);
    vex::slicer<2> slicer2(vex::extents[m][m]);

    x = 1;

    slicer1[range(1, 2, n)](x) = 2;

    check_sample(x, [&](size_t idx, int v) {
            BOOST_CHECK_EQUAL(v, static_cast<int>(idx % 2 + 1));
            });

    for(size_t i = 0; i < m; ++i)
        slicer2[_][i](x) = i;

    check_sample(x, [&](size_t idx, int v) {
            BOOST_CHECK_EQUAL(v, static_cast<int>(idx % m));
            });

    vex::tie(slicer2[1](x), slicer2[2](x)) = std::make_tuple(1, 2);

    for(size_t i = 0; i < m; ++i) {
        BOOST_CHECK_EQUAL(x[1 * m + i], 1);
        BOOST_CHECK_EQUAL(x[2 * m + i], 2);
    }

    vex::vector<size_t> I(queue, m);

    I = vex::element_index() * m;

    auto first_col = vex::permutation(I);

    first_col(x) = 42;

    for(size_t i = 0; i < m; ++i)
        BOOST_CHECK_EQUAL(x[i * m], 42);
}

BOOST_AUTO_TEST_CASE(slice_reductor_single_dim)
{
    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    using vex::extents;
    using vex::_;

    vex::vector<int> x(queue, 32 * 32);
    vex::vector<int> y(queue, 32);

    vex::slicer<2> slice(extents[32][32]);

    int isum = 0;
    int i2sum = 0;
    for(int i = 0; i < 32; ++i) {
        slice[i](x) = i;
        isum += i;
        i2sum += i * i;
    }

    y = vex::reduce<vex::SUM>(slice[_][_](x), 1);

    for(int i = 0; i < 32; ++i)
        BOOST_CHECK_EQUAL(y[i], i * 32);

    y = vex::reduce<vex::SUM>(slice[_][_](x), 0);

    for(size_t i = 0; i < 32; ++i)
        BOOST_CHECK_EQUAL(y[i], isum);

    auto t = vex::make_temp<1>(x);
    y = vex::reduce<vex::SUM>(slice[_][_], t * t, 1);

    for(int i = 0; i < 32; ++i)
        BOOST_CHECK_EQUAL(y[i], i * i * 32);

    y = vex::reduce<vex::SUM>(slice[_][_], t * t, 0);

    for(size_t i = 0; i < 32; ++i)
        BOOST_CHECK_EQUAL(y[i], i2sum);
}

BOOST_AUTO_TEST_CASE(slice_reductor_multi_dim)
{
    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    using vex::extents;
    using vex::_;

    vex::vector<int> x(queue, 32 * 32 * 32);
    vex::vector<int> y(queue, 32);

    vex::slicer<3> slice(extents[32][32][32]);

    x = 1;

    auto test = [&](size_t d1, size_t d2) {
        std::array<size_t,2> dim = {{d1, d2}};

        y = vex::reduce<vex::SUM>(slice[_][_][_](x), dim);

        check_sample(y, [&](size_t, int sum) { BOOST_CHECK_EQUAL(sum, 1024); });
    };

    test(0, 1);
    test(1, 2);
    test(0, 2);
}

BOOST_AUTO_TEST_CASE(nested_reduce)
{
    using vex::extents;
    using vex::_;

    const size_t n = 32;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n * n * n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    vex::slicer<2> s2(extents[n][n]);
    vex::slicer<3> s3(extents[n][n][n]);

    y = vex::reduce<vex::MAX>( s2[_], vex::reduce<vex::SUM>(s3[_], sin(x), 2), 1 );

    check_sample(y, [&](size_t k, double Y) {
            double ms = -std::numeric_limits<double>::max();
            for(size_t j = 0, idx = k * n * n; j < n; ++j) {
                double sum = 0;
                for(size_t i = 0; i < n; ++i, ++idx) {
                    sum += sin(X[idx]);
                }
                ms = std::max(ms, sum);
            }
            BOOST_CHECK_CLOSE(ms, Y, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()
