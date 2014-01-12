#define BOOST_TEST_MODULE Sort
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/sort.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(sort_keys)
{
    const size_t n = 1000 * 1000;

    std::vector<float> k = random_vector<float>(n);
    vex::vector<float> keys(ctx, k);

    vex::sort(keys);
    vex::copy(keys, k);

    BOOST_CHECK( std::is_sorted(k.begin(), k.end()) );
}

BOOST_AUTO_TEST_CASE(sort_keys_vals)
{
    const size_t n = 1000 * 1000;

    std::vector<int  > k = random_vector<int  >(n);
    vex::vector<int  > keys(ctx, k);
    vex::vector<float> vals(ctx, random_vector<float>(n));

    vex::sort_by_key(keys, vals);
    vex::copy(keys, k);

    BOOST_CHECK( std::is_sorted(k.begin(), k.end()) );

    struct even_first_t {
        typedef bool result_type;

        VEX_FUNCTION(device, bool(int, int),
            VEX_STRINGIZE_SOURCE(
                char bit1 = 1 & prm1;
                char bit2 = 1 & prm2;
                if (bit1 == bit2) return prm1 < prm2;
                return bit1 < bit2;
                )
            );

        result_type operator()(int a, int b) const {
            char bit1 = 1 & a;
            char bit2 = 1 & b;
            if (bit1 == bit2) return a < b;
            return bit1 < bit2;
        }

        even_first_t() {}
    } even_first;

    vex::sort_by_key(keys, vals, even_first);
    vex::copy(keys, k);

    BOOST_CHECK(std::is_sorted(k.begin(), k.end(), even_first));
}

BOOST_AUTO_TEST_CASE(sort_keys_tuple)
{
    const size_t n = 1000 * 1000;

    std::vector<int>   k1 = random_vector<int>  (n);
    std::vector<float> k2 = random_vector<float>(n);

    vex::vector<int>   keys1(ctx, k1);
    vex::vector<float> keys2(ctx, k2);

    struct less_t {
        typedef bool result_type;

        VEX_FUNCTION(device, bool(int, float, int, float),
                "return (prm1 == prm3) ? (prm2 < prm4) : (prm1 < prm3);"
                );

        result_type operator()(int a1, float a2, int b1, float b2) const {
            return (a1 == b1) ? (a2 < b2) : (a1 < b1);
        }

        less_t() {}
    } less;

    vex::sort(boost::fusion::vector_tie(keys1, keys2), less );
    vex::copy(keys1, k1);
    vex::copy(keys2, k2);

    BOOST_CHECK( std::is_sorted(
                boost::counting_iterator<size_t>(0),
                boost::counting_iterator<size_t>(n),
                [&](size_t i, size_t j) {
                    return std::make_tuple(k1[i], k2[i]) < std::make_tuple(k1[j], k2[j]);
                } ) );
}

BOOST_AUTO_TEST_CASE(sort_keys_vals_tuple)
{
    const size_t n = 1000 * 1000;

    std::vector<int>     k1 = random_vector<int>    (n);
    std::vector<float>   k2 = random_vector<float>  (n);
    std::vector<cl_long> v1 = random_vector<cl_long>(n);
    std::vector<short>   v2 = random_vector<short>  (n);

    vex::vector<int>     keys1(ctx, k1);
    vex::vector<float>   keys2(ctx, k2);
    vex::vector<cl_long> vals1(ctx, v1);
    vex::vector<short>   vals2(ctx, v2);

    struct less_t {
        typedef bool result_type;

        VEX_FUNCTION(device, bool(int, float, int, float),
                "return (prm1 == prm3) ? (prm2 < prm4) : (prm1 < prm3);"
                );

        result_type operator()(int a1, float a2, int b1, float b2) const {
            return (a1 == b1) ? (a2 < b2) : (a1 < b1);
        }

        less_t() {}
    } less;

    vex::sort_by_key(boost::fusion::vector_tie(keys1, keys2), boost::fusion::vector_tie(vals1, vals2), less );

    vex::copy(keys1, k1);
    vex::copy(keys2, k2);

    BOOST_CHECK( std::is_sorted(
                boost::counting_iterator<size_t>(0),
                boost::counting_iterator<size_t>(n),
                [&](size_t i, size_t j) {
                    return std::make_tuple(k1[i], k2[i]) < std::make_tuple(k1[j], k2[j]);
                } ) );
}

BOOST_AUTO_TEST_SUITE_END()
