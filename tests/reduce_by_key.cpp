#define BOOST_TEST_MODULE ReduceByKey
#include <algorithm>
#include <numeric>
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

struct comp {
    const cl_int  *k1;
    const cl_long *k2;

    comp(const cl_int * k1, const cl_long *k2) : k1(k1), k2(k2) {}

    template <class Tuple>
    bool operator()(size_t i, Tuple t) const {
        return std::make_tuple(k1[i], k2[i]) < t;
    }

    template <class Tuple>
    bool operator()(Tuple t, size_t i) const {
        return t < std::make_tuple(k1[i], k2[i]);
    }
};

BOOST_AUTO_TEST_CASE(rbk_tuple)
{
    const int n = 1000 * 1000;

    std::vector<cl_int>  k1(n);
    std::vector<cl_long> k2(n);

    {
        std::vector<cl_int>  k1s = random_vector<cl_int> (n);
        std::vector<cl_long> k2s = random_vector<cl_long>(n);

        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
                return std::make_tuple(k1s[i], k2s[i]) < std::make_tuple(k1s[j], k2s[j]);
                });

        for(int i = 0; i < n; ++i) {
            k1[i] = k1s[idx[i]];
            k2[i] = k2s[idx[i]];
        }
    }

    std::vector<double> y = random_vector<double>(n);

    std::vector<vex::backend::command_queue> queue(1, ctx.queue(0));

    vex::vector<cl_int>  ikey1(queue, k1);
    vex::vector<cl_long> ikey2(queue, k2);
    vex::vector<double>  ivals(queue, y);

    vex::vector<cl_int>  okey1;
    vex::vector<cl_long> okey2;
    vex::vector<double>  ovals;

    VEX_FUNCTION(equal, bool(cl_int, cl_long, cl_int, cl_long),
            "return (prm1 == prm3) && (prm2 == prm4);"
            );

    VEX_FUNCTION(plus, double(double, double),
            "return prm1 + prm2;"
            );

    int num_keys = vex::reduce_by_key(
            boost::fusion::vector_tie(ikey1, ikey2), ivals,
            boost::fusion::vector_tie(okey1, okey2), ovals,
            equal, plus
            );

    size_t unique_keys = 1;

    for(int i = 1; i < n; ++i) {
        if (std::make_tuple(k1[i-1], k2[i-1]) != std::make_tuple(k1[i], k2[i]))
            ++unique_keys;
    }

    BOOST_CHECK_EQUAL(unique_keys,  num_keys);
    BOOST_CHECK_EQUAL(okey1.size(), num_keys);
    BOOST_CHECK_EQUAL(okey2.size(), num_keys);
    BOOST_CHECK_EQUAL(ovals.size(), num_keys);

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    check_sample(okey1, okey2, ovals, [&](size_t, cl_int key1, cl_long key2, double dev_sum) {
        auto r = std::equal_range(idx.begin(), idx.end(),
            std::make_tuple(key1, key2), comp(k1.data(), k2.data()));

        double host_sum = std::accumulate(
                ivals.begin() + std::distance(idx.begin(), r.first),
                ivals.begin() + std::distance(idx.begin(), r.second),
                0.0);

        BOOST_CHECK_CLOSE(dev_sum, host_sum, 1e-8);
        });
}

BOOST_AUTO_TEST_SUITE_END()
