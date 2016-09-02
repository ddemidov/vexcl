#define BOOST_TEST_MODULE SparseMatrices
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/sparse/csr.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(csr)
{
    std::vector<int>    ptr = {0, 2, 4};
    std::vector<int>    col = {0, 1, 0, 1};
    std::vector<double> val = {1, 1, 1, 1};
    std::vector<double> x   = {1, 1};

    std::vector<vex::command_queue> q(1, ctx.queue(0));

    vex::sparse::csr<double> A(q, ptr, col, val);

    vex::vector<double> X(q, x);
    vex::vector<double> Y(q, 2);

    Y = 0.5 * (A * (X + 1));
    std::cout << Y << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
