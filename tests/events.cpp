#define BOOST_TEST_MODULE Let
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reductor.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(let_vector_expr)
{
    const size_t n = 1024;

    std::vector<vex::command_queue> q1(1, ctx.queue(0));
    std::vector<vex::command_queue> q2(1, vex::backend::duplicate_queue(ctx.queue(0)));

    vex::vector<int> x(q1, n);
    vex::vector<int> y(q2, n);

    vex::Reductor<size_t> count(q2);

    x = 1;
    q1[0].finish();

    x = 2;

    vex::backend::enqueue_barrier(q2[0], {vex::backend::enqueue_marker(q1[0])});

    y = x;

    BOOST_CHECK_EQUAL(count(y != 2), 0);
}

BOOST_AUTO_TEST_SUITE_END()

