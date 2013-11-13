#define BOOST_TEST_MODULE FastFourierTransform
#include <boost/test/unit_test.hpp>
#ifdef HAVE_FFTW
#  include <fftw3.h>
#endif
#include <vexcl/vector.hpp>
#include <vexcl/fft.hpp>
#include <vexcl/random.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/reductor.hpp>
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

void test(const vex::Context &ctx, std::vector<size_t> ns, size_t batch) {
    std::cout << "FFT(C2C) size=" << ns[0];
    for(size_t i = 1; i < ns.size(); i++) std::cout << 'x' << ns[i];
    std::cout << " batch=" << batch << std::endl;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    size_t n1 = std::accumulate(ns.begin(), ns.end(), 1UL, std::multiplies<size_t>());
    size_t n = n1 * batch;

    // random data.
    std::vector<cl_double2> inp_h = random_vector<cl_double2>(n);

    // test
    vex::vector<cl_double2> inp (queue, inp_h);
    vex::vector<cl_double2> out (queue, n);
    vex::vector<cl_double2> back(queue, n);

    std::vector<size_t> ns_(ns.begin(), ns.end());
    std::vector<vex::fft::direction> dirs (ns.size(), vex::fft::forward);
    std::vector<vex::fft::direction> idirs(ns.size(), vex::fft::inverse);
    if(batch != 1) {
        ns_.insert(ns_.begin(), batch);
        dirs.insert(dirs.begin(), vex::fft::none);
        idirs.insert(idirs.begin(), vex::fft::none);
    }
    vex::FFT<cl_double2> fft (queue, ns_, dirs);
    vex::FFT<cl_double2> ifft(queue, ns_, idirs);

    out  = fft (inp);
    back = ifft(out);

    // reference.
    std::vector<cl_double2> ref_h(n);
    std::vector<int> nsi(ns.begin(), ns.end());
    for(size_t i = 0 ; i < batch ; i++) {
        fftw_plan p1 = fftw_plan_dft(nsi.size(), nsi.data(),
            reinterpret_cast<fftw_complex*>(inp_h.data() + i * n1),
            reinterpret_cast<fftw_complex*>(ref_h.data() + i * n1),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p1);
        fftw_destroy_plan(p1);
    }
    vex::vector<cl_double2> ref(queue, n, ref_h.data());

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
    // TODO: POCL fails this test.
    if (vex::Filter::Platform("Portable Computing Language")(ctx.device(0)))
        return;

    const size_t max = vex::is_cpu(ctx.queue(0)) ? 1 << 10 : 1 << 20;

    vex::fft::planner p;

    for(size_t i = 0; i < 32; ++i) {
        // random number of dimensions, mostly 1.
        size_t dims = random_dim(3, 5);
        size_t batch = random_dim(5, 100);

        // random size.
        std::vector<size_t> n;
        size_t d_max = std::pow(max, 1.0 / dims);
        size_t total = batch;
        for(size_t d = 0 ; d < dims ; d++) {
            size_t sz = random_dim(dims == 1 ? 3 : 1, d_max);

            if(rand() % 3 != 0)
                sz = p.best_size(sz);

            n.push_back(sz);
            total *= sz;
        }

        // run
        if(total <= max) test(ctx, n, batch);
    }
}

#endif

BOOST_AUTO_TEST_SUITE_END()
