#ifndef CONTEXT_SETUP_HPP
#define CONTEXT_SETUP_HPP

#include <vexcl/vexcl.hpp>

struct ContextSetup {
    ContextSetup() :
        context( vex::Filter::DoublePrecision && vex::Filter::Env )
    {
        unsigned seed = static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl;

        srand(seed);
        srand48(seed);

        std::cout << context << std::endl;
    }

    vex::Context context;
};

struct ContextReference {
    ContextReference() :
        ctx( vex::current_context() )
    {}

    const vex::Context &ctx;
};

std::vector<double> random_vector(size_t n) {
    std::vector<double> x(n);
    std::generate(x.begin(), x.end(), drand48);
    return x;
}

BOOST_GLOBAL_FIXTURE( ContextSetup )
BOOST_FIXTURE_TEST_SUITE(cr, ContextReference)

BOOST_AUTO_TEST_CASE(context_ready)
{
    BOOST_CHECK_EQUAL(false, ctx.empty());
}

#endif
