#define BOOST_TEST_MODULE ReduceByKey
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reduce_by_key.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(rbk)
{
    const int n = 1024 * 1024;

    std::vector<int>    x = random_vector<int>   (n);
    std::vector<double> y = random_vector<double>(n);

    std::sort(x.begin(), x.end());

    std::vector<vex::backend::command_queue> queue(1, ctx.queue(0));

    vex::vector<int>    ikeys(queue, x);
    vex::vector<double> ivals(queue, y);

    vex::vector<int>    okeys;
    vex::vector<double> ovals;

    int num_keys = vex::reduce_by_key(ikeys, ivals, okeys, ovals);

    std::vector<int> ux = x;
    ux.erase( std::unique(ux.begin(), ux.end()), ux.end() );

    BOOST_CHECK_EQUAL(ux.size(),    num_keys);
    BOOST_CHECK_EQUAL(okeys.size(), num_keys);
    BOOST_CHECK_EQUAL(ovals.size(), num_keys);

    check_sample(okeys, ovals, [&](size_t, int key, double dev_sum) {
        double host_sum = std::accumulate(
                y.begin() + (std::lower_bound(x.begin(), x.end(), key) - x.begin()),
                y.begin() + (std::upper_bound(x.begin(), x.end(), key) - x.begin()),
                0.0);
        BOOST_CHECK_CLOSE(dev_sum, host_sum, 1e-8);
        });
}

BOOST_AUTO_TEST_SUITE_END()
