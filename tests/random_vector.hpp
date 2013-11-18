#ifndef TESTS_RANDOM_VECTOR_HPP
#define TESTS_RANDOM_VECTOR_HPP

#include <random>
#include <vector>

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
        cl_double2 r = {{::generator<double>::get(), ::generator<double>::get()}};
        return r;
    }
};

template<class T>
std::vector<T> random_vector(size_t n) {
    std::vector<T> x(n);
    std::generate(x.begin(), x.end(), ::generator<T>::get);
    return x;
}



#endif
