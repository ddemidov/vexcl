#define BOOST_TEST_MODULE TensorDot
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>
#include <vexcl/tensordot.hpp>
#include <vexcl/random.hpp>
#include <vexcl/reductor.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/function.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(axes_pairs) {
    auto ax = vex::axes_pairs(1, 2, 3, 4, 5, 6);

    BOOST_CHECK_EQUAL(ax[0][0], 1);
    BOOST_CHECK_EQUAL(ax[0][1], 2);
    BOOST_CHECK_EQUAL(ax[1][0], 3);
    BOOST_CHECK_EQUAL(ax[1][1], 4);
    BOOST_CHECK_EQUAL(ax[2][0], 5);
    BOOST_CHECK_EQUAL(ax[2][1], 6);
}

BOOST_AUTO_TEST_CASE(tensordot) {
    using vex::_;
    using vex::extents;

    std::vector<vex::command_queue> queue(1, ctx.queue(0));

    const size_t N = 32;

    vex::vector<double> a(queue, N * N);
    vex::vector<double> b(queue, N * N);

    vex::Random<double> rnd;
    vex::Reductor<double, vex::SUM> sum(queue);

    a = rnd(vex::element_index(), 0);
    b = rnd(vex::element_index(), 1);

    vex::slicer<2> dim(extents[N][N]);

    BOOST_CHECK_EQUAL(
            sum(
                fabs(
                    vex::reduce<vex::SUM>(
                        vex::extents[N][N][N],
                        vex::reshape(a, extents[N][N][N], extents[0][1]) *
                        vex::reshape(b, extents[N][N][N], extents[1][2]),
                        1) -
                    vex::tensordot(
                        dim[_](a), dim[_](b), vex::axes_pairs(1, 0)
                        )
                    )
               ),
            0.0);
}

BOOST_AUTO_TEST_SUITE_END()
