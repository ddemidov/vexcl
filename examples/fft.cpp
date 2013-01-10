#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
#include <random>

using namespace vex;

//#define INPLACE

void test(Context &ctx) {
    std::cout << ctx;

    const size_t w = 1024, h = 1024, n = w * h;
    std::vector<cl_float2> a_data(n);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-1, 1);
    for(size_t i = 0 ; i < n ; i++)
        a_data[i] = cl_float2{{dist(gen), dist(gen)}};

    vector<cl_float2> a(ctx.queue(), a_data);
    vector<cl_float2> b(ctx.queue(), n);
    vector<cl_float2> c(ctx.queue(), n);

    Reductor<cl_float2, SUM> sum(ctx.queue());
    FFT<cl_float2>  fft(ctx.queue(), {w, h});
    FFT<cl_float2> ifft(ctx.queue(), {w, h}, inverse);

    // bakes FFTs on first run
    b = fft(a);
    b = ifft(b);

    // Run some
    profiler prof(ctx.queue());
    prof.tic_cl("Run");
    for(size_t i = 0 ; i < 10 ; i++) {
        b = fft(a);
        b = ifft(b);
    }
    prof.toc("Run");

    // Average error
    c = fabs(a - b) / n;
    std::cerr << "n=" << n << ".   abs(a - ifft(fft(a))) / n = " << sum(c);

    std::cerr << prof << std::endl;
}

int main() {
    for(size_t i = 0 ; ; i++) {
        // FFT only works on one device, comparing them individually.
        Context ctx(Filter::Position(i), CL_QUEUE_PROFILING_ENABLE);
        if(!ctx) break;
        test(ctx);
    }

    return 0;
}
