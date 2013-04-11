#define BOOST_TEST_MODULE FastFourierTransform
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(transform_expression)
{
    const size_t N = 1024;
    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    vex::vector<cl_float> data(queue, N);
    vex::FFT<cl_float> fft(queue, N);

    // should compile
    data += fft(data * data) * 5;
}

BOOST_AUTO_TEST_CASE(check_correctness)
{
    const size_t N = 1024;
    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    vex::vector<cl_float>  in  (queue, N);
    vex::vector<cl_float2> out (queue, N);
    vex::vector<cl_float>  back(queue, N);

    vex::Random<cl_float> rnd;

    in = rnd(vex::element_index(), std::rand());

    vex::FFT<cl_float,  cl_float2> fft (queue, N);
    vex::FFT<cl_float2, cl_float > ifft(queue, N, vex::inverse);

    out  = fft (in );
    back = ifft(out);

    vex::Reductor<cl_float, vex::SUM> sum(queue);

    BOOST_CHECK(std::sqrt(sum(pow(in - back, 2.0f)) / N) < 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()
