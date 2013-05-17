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

template <typename T, class Enable = void>
struct generator {};

template<typename T>
struct generator<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{
    static T get() {
        static std::default_random_engine rng( std::rand() );
        static std::uniform_real_distribution<T> rnd((T)0, (T)1);
        return rnd(rng);
    }
};

template<typename T>
struct generator<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    static T get() {
        static std::default_random_engine rng( std::rand() );
        static std::uniform_int_distribution<T> rnd(0, 100);
        return rnd(rng);
    }
};

template<>
struct generator<cl_double2>
{
    static cl_double2 get() {
        static std::default_random_engine rng( std::rand() );
        static std::uniform_real_distribution<double> rnd(0, 100);

        cl_double2 r = {{rnd(rng), rnd(rng)}};
        return r;
    }
};

template<class T>
std::vector<T> random_vector(size_t n) {
    std::vector<T> x(n);
    std::generate(x.begin(), x.end(), generator<T>::get);
    return x;
}

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
