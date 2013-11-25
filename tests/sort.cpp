#define BOOST_TEST_MODULE Sort
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/sort.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(sort_pairs)
{
    const size_t n = 1000 * 1000;

    std::vector<vex::backend::command_queue> queue(1, ctx.queue(0));

    std::vector<int  > k = random_vector<int  >(n);
    vex::vector<int  > keys(queue, k);
    vex::vector<float> vals(queue, random_vector<float>(n));

    try {
        vex::sort_by_key(keys, vals);
        vex::copy(keys, k);

        BOOST_CHECK( std::is_sorted(k.begin(), k.end()) );

        VEX_FUNCTION(even_first, bool(int, int),
                "char bit1 = 1 & prm1;\n"
                "char bit2 = 1 & prm2;\n"
                "if (bit1 == bit2) return prm1 < prm2;\n"
                "return bit1 < bit2;\n"
                );

        vex::sort_by_key(keys, vals, even_first);
        vex::copy(keys, k);

        BOOST_CHECK(std::is_sorted(k.begin(), k.end(), [](int a, int b) {
                    char bit1 = 1 & a;
                    char bit2 = 1 & b;
                    if (bit1 == bit2) return a < b;
                    return bit1 < bit2;
                    }));
    } catch(const vex::error &e) {
        std::cout << e << std::endl;
        throw;
    }
}

BOOST_AUTO_TEST_CASE(sort_keys)
{
    const size_t n = 1000 * 1000;

    std::vector<vex::backend::command_queue> queue(1, ctx.queue(0));

    std::vector<int  > k = random_vector<int  >(n);
    vex::vector<int  > keys(queue, k);

    try {
        vex::sort(keys);
        vex::copy(keys, k);

        BOOST_CHECK( std::is_sorted(k.begin(), k.end()) );

        VEX_FUNCTION(even_first, bool(int, int),
                "char bit1 = 1 & prm1;\n"
                "char bit2 = 1 & prm2;\n"
                "if (bit1 == bit2) return prm1 < prm2;\n"
                "return bit1 < bit2;\n"
                );

        vex::sort(keys, even_first);
        vex::copy(keys, k);

        BOOST_CHECK(std::is_sorted(k.begin(), k.end(), [](int a, int b) {
                    char bit1 = 1 & a;
                    char bit2 = 1 & b;
                    if (bit1 == bit2) return a < b;
                    return bit1 < bit2;
                    }));
    } catch(const vex::error &e) {
        std::cout << e << std::endl;
        throw;
    }
}

BOOST_AUTO_TEST_SUITE_END()
