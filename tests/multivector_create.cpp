#define BOOST_TEST_MODULE MultivectorCreate
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"


BOOST_AUTO_TEST_CASE(empty_constructor)
{
    vex::multivector<double,3> m;

    BOOST_CHECK(0U == m.size());
    BOOST_CHECK(m.end() - m.begin() == 0);

    for(int i = 0; i < 3; ++i)
        BOOST_CHECK(0U == m(i).size());
}

BOOST_AUTO_TEST_CASE(copy_constructor)
{
    typedef std::array<double, 3> elem_t;

    const size_t n = 1024;
    vex::multivector<double,3> m(ctx, n);

    m(0) = 1;
    m(1) = 2;
    m(2) = 3;

    vex::multivector<double,3> c(m);

    BOOST_CHECK(c.size() == m.size());

    check_sample(c, m, [](size_t, elem_t a, elem_t b) { BOOST_CHECK(a == b); });
}

BOOST_AUTO_TEST_CASE(element_access)
{
    const size_t n = 1024;
    const size_t m = 4;

    typedef std::array<double, m> elem_t;

    std::vector<double> host = random_vector(n * m);
    vex::multivector<double, m> x(ctx, n);

    copy(host, x);

    for(size_t i = 0; i < n; ++i) {
        elem_t val = x[i];

        for(size_t j = 0; j < m; ++j) {
            BOOST_CHECK(val[j] == host[j * n + i]);
            val[j] = 0;
        }

        x[i] = val;
    }

    copy(x, host);

    BOOST_CHECK(0 == *std::min_element(host.begin(), host.end()));
    BOOST_CHECK(0 == *std::max_element(host.begin(), host.end()));
}

BOOST_AUTO_TEST_SUITE_END()
