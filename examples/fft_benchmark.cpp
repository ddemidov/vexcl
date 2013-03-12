#include <iostream>
#include <iomanip>
#include <random>

#include <vexcl/vexcl.hpp>

using namespace vex;

const size_t runs = 1000;


#ifdef USE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>

void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess)
        throw std::runtime_error(msg);
}

void check(cufftResult status, const char *msg) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(msg);
}

double test_cufft(cl_float2 *data, size_t n, size_t m) {
    size_t dataSize = sizeof(cufftComplex) * n * m;

    cufftHandle plan;
    if(m == 1)
        check(cufftPlan1d(&plan, (int)n, CUFFT_C2C, 1), "cufftPlan1d");
    else
        check(cufftPlan2d(&plan, (int)n, (int)m, CUFFT_C2C), "cufftPlan2d");

    cufftComplex *inData;
    check(cudaMalloc((void **)(&inData), dataSize), "cudaMalloc");

    cufftComplex *outData;
    check(cudaMalloc((void **)(&outData), dataSize), "cudaMalloc");

    // Send X to device
    check(cudaMemcpy(inData, data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy");

    profiler prof;
    prof.tic_cpu("Run");
    for(size_t i = 0 ; i < runs ; i++)
        cufftExecC2C(plan, inData, outData, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    double t = prof.toc("Run");

    cufftDestroy(plan);
    cudaFree(inData);
    cudaFree(outData);
    return t;
}
#else
double test_cufft(cl_float2 *, size_t, size_t) {
    return -1;
}
#endif


#ifdef USE_FFTW
#  ifdef _OPENMP
#    include <omp.h>
#  endif
#  include <fftw3.h>

double test_fftw(cl_float2 *data, size_t n, size_t m) {
    int sz[2] = {(int)n, (int)m};
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(
        fftwf_malloc(sizeof(fftwf_complex) * n * m));
    fftwf_plan p1 = fftwf_plan_dft(m == 1 ? 1 : 2, sz,
        reinterpret_cast<fftwf_complex *>(data),
        out, FFTW_FORWARD, FFTW_MEASURE);

    profiler prof;
    prof.tic_cpu("Run");
    for(size_t i = 0 ; i < runs ; i++)
        fftwf_execute(p1);
    double t = prof.toc("Run");

    fftwf_destroy_plan(p1);
    fftwf_free(out);
    return t;
}
#else
double test_fftw(cl_float2 *, size_t, size_t) {
    return -1;
}
#endif


double test(Context &ctx, cl_float2 *data, size_t n, size_t m) {
    vector<cl_float2> a(ctx, n * m, data);
    vector<cl_float2> b(ctx, n * m);
    std::vector<size_t> sz; sz.push_back(n); if(m > 1) sz.push_back(m);
    FFT<cl_float2> fft(ctx, sz);

    // Run some
    profiler prof;
    prof.tic_cl("Run");
    for(size_t i = 0 ; i < runs ; i++)
        b = fft(a);
    double t = prof.toc("Run");

#ifdef FFT_PROFILE
    std::cerr << fft.plan.profile;
#else
    std::cerr << fft.plan;
#endif
    return t;
}

void info(double time, size_t size, size_t dim) {
    // FFT is O(n log n)
    double ops = dim == 1
        ? size * std::log(static_cast<double>(size)) // O(n log n)
        : 2.0 * size * size * std::log(static_cast<double>(size)); // O(n log n)[1D fft] * n[rows] * 2[transposed]
    std::cout << '\t';
    if(time < 0) std::cout << '-';
    else std::cout << std::scientific << (ops / time);
    std::cout << std::flush;
}

int main() {
#if defined(_OPENMP) && defined(USE_FFTW)
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
    Context ctx(Filter::Env && Filter::Count(1));
    std::cerr << ctx << std::endl;

    // sizes to test
    std::vector<size_t> ns;
    const size_t max_len = 1 << 20;
    vex::fft::prime_generator prime;
    for(size_t n = 2, k = prime() ; n <= max_len ; n *= 2) {
        ns.push_back(n);
        while(k < n) k = prime();
        if(k <= max_len) ns.push_back(k);
    }

    // random data
#ifdef USE_FFTW
    cl_float2 *data = reinterpret_cast<cl_float2 *>(
        fftwf_malloc(sizeof(cl_float2) * max_len));
#else
    cl_float2 *data = new cl_float2[max_len];
#endif
    std::minstd_rand gen;
    std::uniform_real_distribution<float> dist(-1000, 1000);
    for(size_t i = 0 ; i < 2 * max_len ; i++)
        reinterpret_cast<float*>(data)[i] = dist(gen);

    std::cout << "# prints `n log n / time` for n = 2^k\n";

    // 1D
    std::cout << "#n\tfftw^1\tclfft^1\tcufft^1" << std::endl;
    for(auto n = ns.begin() ; n != ns.end() ; n++) {
        std::cout << *n;
        info(test_fftw(data, *n, 1), *n, 1);
        info(test(ctx, data, *n, 1), *n, 1);
        info(test_cufft(data, *n, 1), *n, 1);
        std::cout << std::endl;
    }
    std::cout << "\n\n";

    // 2D
    std::cout << "#n\tfftw^2\tclfft^2\tcufft^2" << std::endl;
    for(auto n = ns.begin() ; n != ns.end() ; n++)
        if(*n * *n <= max_len) {
            std::cout << *n;
            info(test_fftw(data, *n, *n), *n, 2);
            info(test(ctx, data, *n, *n), *n, 2);
            info(test_cufft(data, *n, *n), *n, 2);
            std::cout << std::endl;
        }

    return 0;
}
