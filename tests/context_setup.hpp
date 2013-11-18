#ifndef CONTEXT_SETUP_HPP
#define CONTEXT_SETUP_HPP

#include <vexcl/devlist.hpp>
#include "random_vector.hpp"

struct ContextSetup {
    ContextSetup() :
        context( vex::Filter::DoublePrecision && vex::Filter::Env )
    {
        unsigned seed = static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl;

        srand(seed);

        std::cout << context << std::endl;
    }

    vex::Context context;
};

struct ContextReference {
    ContextReference() :
        ctx( vex::current_context() )
    {
#ifdef VEXCL_BACKEND_OPENCL
        amd_is_present = std::any_of(ctx.queue().begin(), ctx.queue().end(),
            [](const cl::CommandQueue &q) {
                return vex::Filter::Platform("AMD")(q.getInfo<CL_QUEUE_DEVICE>());
            });
#else
        amd_is_present = false;
#endif
    }

    void amd_workaround() const {
#ifdef VEXCL_BACKEND_OPENCL
        // There is a bug in AMD OpenCL that requires to call finish() on
        // command queues after some kernels launch:
        // http://devgurus.amd.com/message/1295503#1295503
        if (amd_is_present) {
            std::for_each(ctx.queue().begin(), ctx.queue().end(), [](const cl::CommandQueue &q) {
                q.finish();
            });
        }
#endif
    }

    const vex::Context &ctx;
    bool amd_is_present;
};

#define SAMPLE_SIZE 32

template<class V, class F>
void check_sample(const V &v, F f) {
    for(size_t i = 0; i < SAMPLE_SIZE; ++i) {
        size_t idx = rand() % v.size();
        f(idx, v[idx]);
    }
}

template<class V1, class V2, class F>
void check_sample(const V1 &v1, const V2 &v2, F f) {
    BOOST_REQUIRE(v1.size() == v2.size());
    for(size_t i = 0; i < SAMPLE_SIZE; ++i) {
        size_t idx = rand() % v1.size();
        f(idx, v1[idx], v2[idx]);
    }
}

template<class V1, class V2, class V3, class F>
void check_sample(const V1 &v1, const V2 &v2, const V3 &v3, F f) {
    BOOST_REQUIRE(v1.size() == v2.size());
    BOOST_REQUIRE(v1.size() == v3.size());

    for(size_t i = 0; i < SAMPLE_SIZE; ++i) {
        size_t idx = rand() % v1.size();
        f(idx, v1[idx], v2[idx], v3[idx]);
    }
}

BOOST_GLOBAL_FIXTURE( ContextSetup )
BOOST_FIXTURE_TEST_SUITE(cr, ContextReference)

BOOST_AUTO_TEST_CASE(context_ready)
{
    BOOST_REQUIRE(ctx);
}

#endif
