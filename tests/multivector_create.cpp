#define BOOST_TEST_MODULE MultivectorCreate
#include <boost/test/unit_test.hpp>
#include <vexcl/multivector.hpp>
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

    std::vector<double> host = random_vector<double>(n * m);
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

BOOST_AUTO_TEST_CASE(stl_container_of_multivector)
{
    const size_t N = 1024;
    const size_t D = 2;
    const size_t M = 16 + generator<size_t>::get() ;

    std::vector< vex::multivector<unsigned, D> > x;

    std::vector< std::array<cl_mem, D> > bufs;

    for(size_t i = 0; i < M; ++i) {
        x.push_back( vex::multivector<unsigned, D>(ctx, N) );
        x.back() = i;
        std::array<cl_mem, D> b = {{x.back()(0)(0)(), x.back()(1)(0)() }};
        bufs.push_back(b);
    }

    for(size_t i = 0; i < M; ++i) {
        BOOST_CHECK_EQUAL(bufs[i][0], x[i](0)(0)());
        BOOST_CHECK_EQUAL(bufs[i][1], x[i](1)(0)());
    }
}

BOOST_AUTO_TEST_SUITE_END()
