#define BOOST_TEST_MODULE SparseMatrices
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/sparse/csr.hpp>
#include <vexcl/sparse/ell.hpp>
#include "context_setup.hpp"
#include "random_matrix.hpp"

BOOST_AUTO_TEST_CASE(csr)
{
    const size_t n = 1024;

    std::vector<vex::command_queue> q(1, ctx.queue(0));

    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;

    random_matrix(n, n, 16, row, col, val);

    std::vector<double> x = random_vector<double>(n);

    vex::sparse::csr<double> A(q, n, n, row, col, val);
    vex::vector<double> X(q, x);
    vex::vector<double> Y(q, n);

    Y = A * X;

    check_sample(Y, [&](size_t idx, double a) {
            double sum = 0;
            for(size_t j = row[idx]; j < row[idx + 1]; j++)
                sum += val[j] * x[col[j]];

            BOOST_CHECK_CLOSE(a, sum, 1e-8);
            });
}

BOOST_AUTO_TEST_CASE(ell)
{
    const size_t n = 1024;

    std::vector<vex::command_queue> q(1, ctx.queue(0));

    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;

    random_matrix(n, n, 16, row, col, val);

    std::vector<double> x = random_vector<double>(n);

    vex::sparse::ell<double> A(q, n, n, row, col, val);
    vex::vector<double> X(q, x);
    vex::vector<double> Y(q, n);

    Y = A * X;

    check_sample(Y, [&](size_t idx, double a) {
            double sum = 0;
            for(size_t j = row[idx]; j < row[idx + 1]; j++)
                sum += val[j] * x[col[j]];

            BOOST_CHECK_CLOSE(a, sum, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()
