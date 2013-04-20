#define BOOST_TEST_MODULE FastFourierTransform
#include <boost/test/unit_test.hpp>
#ifdef HAVE_FFTW
#  include <fftw3.h>
#endif
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
    vex::FFT<cl_float2, cl_float > ifft(queue, N, vex::fft::inverse);

    out  = fft (in );
    back = ifft(out);

    vex::Reductor<cl_float, vex::SUM> sum(queue);

    BOOST_CHECK(std::sqrt(sum(pow(in - back, 2.0f)) / N) < 1e-3);
}

#ifdef HAVE_FFTW

void test(const vex::Context &ctx, std::vector<size_t> ns) {
    std::cout << ns[0];
    for(size_t i = 1; i < ns.size(); i++) {
        std::cout << 'x' << ns[i];
    }
    std::cout << std::endl;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    size_t n = std::accumulate(ns.begin(), ns.end(), 1UL, std::multiplies<size_t>());

    // random data.
    std::vector<double> inp_h = random_vector<double>(2 * n);

    // test
    vex::vector<cl_double2> inp (queue, n, reinterpret_cast<cl_double2*>(inp_h.data()));
    vex::vector<cl_double2> out (queue, n);
    vex::vector<cl_double2> back(queue, n);

    vex::FFT<cl_double2> fft (queue, ns);
    vex::FFT<cl_double2> ifft(queue, ns, vex::fft::inverse);

    out  = fft (inp);
    back = ifft(out);

    // reference.
    std::vector<double> ref_h(2 * n);
    {
        std::vector<int> nsi(ns.begin(), ns.end());
        fftw_plan p1 = fftw_plan_dft(nsi.size(), nsi.data(),
                reinterpret_cast<fftw_complex*>(inp_h.data()),
                reinterpret_cast<fftw_complex*>(ref_h.data()),
                FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p1);
        fftw_destroy_plan(p1);
    }

    vex::vector<cl_double2> ref (queue, n, reinterpret_cast<cl_double2*>(ref_h.data()));

    auto rms = [&](const vex::vector<cl_double2> &a, const vex::vector<cl_double2> &b) {
        static vex::Reductor<double, vex::SUM> sum(queue);
        return std::sqrt(sum(dot(a - b, a - b))) / std::sqrt(sum(dot(b, b)));
    };

    BOOST_CHECK_SMALL(rms(out,  ref), 1e-8);
    BOOST_CHECK_SMALL(rms(back, inp), 1e-8);
}

// random dimension, mostly 1.
size_t random_dim(double p, double s) {
    static std::default_random_engine rng( std::rand() );
    static std::uniform_real_distribution<double> rnd(0.0, 1.0);

    return 1 + static_cast<size_t>( s * std::pow(rnd(rng), p) );
}

BOOST_AUTO_TEST_CASE(test_dimensions)
{
    const size_t max = vex::is_cpu(ctx.device(0)) ? 1 << 10 : 1 << 20;

    vex::fft::planner p;

    for(size_t i = 0; i < 32; ++i) {
        // random number of dimensions, mostly 1.
        size_t dims = random_dim(3, 5);

        // random size.
        std::vector<size_t> n;
        size_t d_max = std::pow(max, 1.0 / dims);
        size_t total = 1;
        for(size_t d = 0 ; d < dims ; d++) {
            size_t sz = random_dim(dims == 1 ? 3 : 1, d_max);

            if(rand() % 3 != 0)
                sz = p.best_size(sz);

            n.push_back(sz);
            total *= sz;
        }

        // run
        if(total <= max) test(ctx, n);
    }
}

#endif

BOOST_AUTO_TEST_SUITE_END()
