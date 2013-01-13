#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
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



void test(Context &ctx, std::vector<size_t> ns) {
    size_t n = 1;
    for(size_t i = 0 ; i < ns.size() ; i++) n *= ns[i];

    const float range = 1000;

    // random data.
    std::vector<cl_float2> input_h = random_vec(n, range);

    // reference.
    std::vector<cl_float2> ref_h(n);
    int *ns_ = new int[ns.size()];
    std::copy(ns.begin(), ns.end(), ns_);
    fftwf_plan p1 = fftwf_plan_dft(ns.size(), ns_,
        reinterpret_cast<fftwf_complex *>(&input_h[0]),
        reinterpret_cast<fftwf_complex *>(&ref_h[0]),
        FFTW_FORWARD, FFTW_ESTIMATE);
    delete [] ns_;
    fftwf_execute(p1);
    fftwf_destroy_plan(p1);

    // test
    vector<cl_float2> input(ctx.queue(), input_h);
    vector<cl_float2> output(ctx.queue(), n);
    vector<cl_float2> ref(ctx.queue(), ref_h);
    vector<cl_float2> back(ctx.queue(), n);

    FFT<cl_float2>  fft(ctx.queue(), ns);
    output = fft(input);

    FFT<cl_float2> ifft(ctx.queue(), ns, inverse);
    back = ifft(output);

    Reductor<cl_float2, SUM> sum(ctx.queue());
    #define rms(e) (100 * std::sqrt(hsum(sum(pow(e, 2))) / n) / range)

    std::cerr << "n=[";
    for(size_t i = 0 ; i < ns.size() ; i++)
        std::cerr << ns[i] << ' ';
    std::cerr << "]        rms" << std::endl;
    std::cerr << "fftw-clfft      " << rms(output - ref) << "%" << std::endl;
    std::cerr << "x-ifft(fft(x))  " << rms(input - back) << "%" << std::endl;


    if(false && n < 16) std::cerr
        << "input   " << input << '\n'
        << "fftw    " << ref << '\n'
        << "clfft   " << output << '\n'
        << "inverse " << back << std::endl;
}

int main() {
    Context ctx(Filter::Position(0), CL_QUEUE_PROFILING_ENABLE);
    test(ctx, {16 * 16 * 2});
    test(ctx, {16 * 16 * 4});
    test(ctx, {16 * 16 * 8});
    test(ctx, {16 * 16 * 16});
    test(ctx, {4, 4});
    test(ctx, {16 * 4, 16 * 4});
    test(ctx, {16 * 16 * 2, 16 * 16 * 4});
    test(ctx, {2, 4});
    test(ctx, {2 * 16, 4 * 16, 8 * 16});
    test(ctx, {2 * 16, 4 * 16, 8 * 16, 2});
    return 0;
}

