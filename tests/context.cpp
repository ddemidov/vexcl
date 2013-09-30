#define BOOST_TEST_MODULE VexContext
#include <boost/test/unit_test.hpp>
#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>

void local_context() {
    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env) );
    std::cout << ctx << std::endl;

    BOOST_CHECK( !ctx.empty() );

    const size_t n = 1024;

    vex::vector<int> x(ctx, n);
    x = 0;

    BOOST_CHECK(x.size() == n);
    BOOST_CHECK(x[0] == 0);

    vex::purge_kernel_caches(ctx);
}

BOOST_AUTO_TEST_CASE(create_destroy)
{
    local_context();
    local_context();
}
