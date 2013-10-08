#define BOOST_TEST_MODULE VectorPointer
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_pointer.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <vexcl/temporary.hpp>
#include <vexcl/constants.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(nbody)
{
    const size_t n = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    VEX_FUNCTION(nbody, double(size_t, size_t, double*),
            "double sum = 0;\n"
            "for(size_t i = 0; i < prm1; ++i)\n"
            "    if (i != prm2) sum += prm3[i];\n"
            "return sum;"
            );

    y = nbody(n, vex::element_index(), vex::raw_pointer(x));

    check_sample(y, [&](size_t idx, double v) {
            double sum = 0;
            for(size_t i = 0; i < n; ++i)
                if (i != idx) sum += X[i];
            BOOST_CHECK_CLOSE(v, sum, 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(manual_stencil)
{
    const size_t n = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    auto p = vex::tag<1>( vex::raw_pointer(x) );
    auto N = vex::tag<3>( n );

    VEX_CONSTANT(nil, 0);
    VEX_CONSTANT(one, 1);
    VEX_CONSTANT(two, 2);

    auto i   = vex::make_temp<0>( vex::element_index() );
    auto x_l = vex::make_temp<1>( if_else(i > nil(), *(p + i - one()), *(p + i) ) );
    auto x_c = vex::make_temp<2>( *(p + i) );
    auto x_r = vex::make_temp<3>( if_else(i + one() < N, *(p + i + one()), *(p + i) ) );

    y = (two() * x_c - x_l - x_r);

    check_sample(y, [&](size_t idx, double v) {
            double xc = X[idx];
            double xl = X[idx > 0 ? idx - 1 : idx];
            double xr = X[idx + 1 < n ? idx + 1 : idx];
            BOOST_CHECK_CLOSE(v, 2 * xc - xr - xl, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()

