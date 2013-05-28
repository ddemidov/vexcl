#define BOOST_TEST_MODULE VectorView
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(vector_view)
{
    const size_t N = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    vex::vector<int> x(ctx, N * 2);
    vex::vector<int> y(ctx, N);

    x = vex::element_index();

    size_t    size   = N;
    ptrdiff_t stride = 2;

    vex::gslice<1> slice(0, &size, &stride);
    vex::vector_view< int, vex::gslice<1> > view(x, slice);

    // TODO: y = slice(x) would probably be of more use.
    y = view;

    check_sample(y, [](size_t idx, int v) { BOOST_CHECK(v == idx * 2); });
}

BOOST_AUTO_TEST_SUITE_END()
