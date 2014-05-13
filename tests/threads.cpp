#define BOOST_TEST_MODULE Sort
#include <future>
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reductor.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(threads)
{
    const size_t n = 1024 * 1024;

    auto run = [&](unsigned device) -> cl_long {
        std::vector<vex::backend::command_queue> q(1, ctx.queue(device));
        vex::vector<int> x(q, n);
        x = 1;
        vex::Reductor<cl_long, vex::SUM> sum(q);
        return sum(x);
    };

    std::vector< std::future<cl_long> > results;
    for(unsigned d = 0; d < ctx.size(); ++d)
        results.push_back( std::async(std::launch::async, run, d) );

    cl_long sum = 0;
    for(unsigned d = 0; d < ctx.size(); ++d)
        sum += results[d].get();

    BOOST_CHECK_EQUAL(sum, n * ctx.size());
}

BOOST_AUTO_TEST_SUITE_END()
