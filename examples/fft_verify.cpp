#include <vexcl/vexcl.hpp>
#include <random>
#include <fftw3.h>

using namespace vex;

float hsum(cl_float2 a) { return a.s[0] + a.s[1]; }


#include <random>

std::vector<cl_float2> random_vec(size_t n, float range) {
    std::vector<cl_float2> data(n);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-range, range);
    for(size_t i = 0 ; i < n ; i++)
        data[i] = cl_float2{{dist(gen), dist(gen)}};
    return data;
}



bool test(Context &ctx, std::vector<size_t> ns) {
    std::ostringstream name;
    for(size_t i = 0 ; i < ns.size() ; i++) {
        if(i > 0) name << 'x';
        name << ns[i];
    }
    char fc = std::cout.fill('.');
    std::cout << name.str() << ": " << std::setw(62 - name.str().size()) << "." << std::flush;
    std::cout.fill(fc);

    size_t n = 1;
    for(size_t i = 0 ; i < ns.size() ; i++) n *= ns[i];

    const float range = 1000;

    // random data.
    std::vector<cl_float2> input_h = random_vec(n, range);

    // reference.
    std::vector<cl_float2> ref_h(n);
    std::vector<int> ns_(ns.begin(), ns.end());
    fftwf_plan p1 = fftwf_plan_dft(ns_.size(), ns_.data(),
        reinterpret_cast<fftwf_complex *>(&input_h[0]),
        reinterpret_cast<fftwf_complex *>(&ref_h[0]),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p1);
    fftwf_destroy_plan(p1);

    // test
    vector<cl_float2> input(ctx.queue(), input_h);
    vector<cl_float2> output(ctx.queue(), n);
    vector<cl_float2> ref(ctx.queue(), ref_h);
    vector<cl_float2> back(ctx.queue(), n);

    Reductor<cl_float2, SUM> sum(ctx.queue());
    #define rms(e) (100 * std::sqrt(hsum(sum(pow(e, 2))) / n) / range)

    try {
        FFT<cl_float2>  fft(ctx.queue(), ns);
        output = fft(input);

        bool rc = true;
        const float rms_fft = rms(output - ref);
        rc &= rms_fft < 0.1;

        FFT<cl_float2> ifft(ctx.queue(), ns, inverse);
        back = ifft(output);
        const float rms_inv = rms(input - back);
        rc &= rms_inv < 1e-4;

        std::cout << (rc ? " success." : " failed.") << '\n';

        std::cout << "  fftw-clfft      " << rms_fft << "%" << '\n';
        std::cout << "  x-ifft(fft(x))  " << rms_inv << "%" << '\n';

        std::cout << fft.plan << '\n';

        return rc;

    } catch(cl::Error e) {
        std::cerr << "FFT error " << ": " << e << std::endl;
        throw;
    }
}


double skew_rand(double p) {
    return std::pow(1.0 * rand() / RAND_MAX, p);
}

int main() {
    Context ctx(Filter::Env && Filter::Count(1), CL_QUEUE_PROFILING_ENABLE);
    std::cout << ctx << std::endl;

    bool rc = true;

    const size_t max = 1 << 20;

    fft::default_planner p;
    for(size_t i = 0 ; i < 20 ; i++) {
        // random number of dimensions, mostly 1.
        size_t dims = 1 + size_t(skew_rand(3) * 5);
        // random size.
        std::vector<size_t> n;
        size_t d_max = std::pow(max, 1.0 / dims);
        size_t total = 1;
        for(size_t d = 0 ; d < dims ; d++) {
            size_t sz = 1 + size_t(skew_rand(dims == 1 ? 3 : 1) * d_max);
            sz = p.best_size(sz);
            n.push_back(sz);
            total *= sz;
        }
        // run
        if(total <= max)
            rc &= test(ctx, n);
    }


    return rc ? EXIT_SUCCESS : EXIT_FAILURE;
}

