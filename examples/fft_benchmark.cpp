#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
#include <random>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <fftw3.h>
using namespace vex;


const float range = 1000;
const size_t runs = 1000;





#ifdef USE_CUDA

#include <cufft.h>
#include <cuda_runtime.h>

// Run complex to complex X[2*N] to Y[2*N].
void test_cufft(cl_float2 *data, size_t n, size_t m) {
    size_t dataSize = sizeof(cufftComplex) * n * m;
    cudaError_t status;
    cufftResult fftStatus;

    cufftHandle plan;
    if(m == 1)
        fftStatus = cufftPlan1d(&plan, (int)n, CUFFT_C2C, 1);
    else
        fftStatus = cufftPlan2d(&plan, (int)n, (int)m, CUFFT_C2C);
    assert(fftStatus == CUFFT_SUCCESS);

    cufftComplex *inData;
    status = cudaMalloc((void **)(&inData), dataSize);
    assert(status == cudaSuccess);

    cufftComplex *outData;
    status = cudaMalloc((void **)(&outData), dataSize);
    assert(status == cudaSuccess);

    // Send X to device
    status = cudaMemcpy(inData, data, dataSize, cudaMemcpyHostToDevice);
    assert(status == cudaSuccess);

    cudaDeviceSynchronize();
    profiler prof({});
    prof.tic_cpu("Run");
    for(size_t i = 0 ; i < runs ; i++)
        cufftExecC2C(plan, inData, outData, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    std::cout << '\t' << prof.toc("Run") << std::flush;

    cufftDestroy(plan);
    cudaFree(inData);
    cudaFree(outData);
}
#endif


void test_fftw(cl_float2 *data, size_t n, size_t m) {
    int sz[2] = {(int)n, (int)m};
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(
        fftwf_malloc(sizeof(fftwf_complex) * n * m));
    fftwf_plan p1 = fftwf_plan_dft(m == 1 ? 1 : 2, sz,
        reinterpret_cast<fftwf_complex *>(data),
        out, FFTW_FORWARD, FFTW_MEASURE);

    profiler prof({});
    prof.tic_cpu("Run");
    for(size_t i = 0 ; i < runs ; i++)
        fftwf_execute(p1);
    std::cout << '\t' << prof.toc("Run") << std::flush;

    fftwf_destroy_plan(p1);
    fftwf_free(out);
}


void test(Context &ctx, cl_float2 *data, size_t n, size_t m) {
    vector<cl_float2> a(ctx.queue(), n * m, data);
    vector<cl_float2> b(ctx.queue(), n * m);
    std::vector<size_t> sz; sz.push_back(n); if(m > 1) sz.push_back(m);
    FFT<cl_float2> fft(ctx.queue(), sz);
    ctx.queue()[0].finish();

    // Run some
    profiler prof({});
    prof.tic_cpu("Run");
    for(size_t i = 0 ; i < runs ; i++)
        b = fft(a);
    ctx.queue()[0].finish();
    std::cout << '\t' << prof.toc("Run") << std::flush;
}

int main() {
#ifdef USE_OPENMP
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
    Context ctx(Filter::Count(1), CL_QUEUE_PROFILING_ENABLE);

    // random data
    const size_t k_max = 9, n_max = 1 << k_max;
    const size_t max_len = n_max * n_max;
    cl_float2 *data = reinterpret_cast<cl_float2 *>(
        fftwf_malloc(sizeof(cl_float2) * max_len));
    std::minstd_rand gen;
    std::uniform_real_distribution<float> dist(-range, range);
    for(size_t i = 0 ; i < max_len ; i++)
        reinterpret_cast<float*>(data)[i] = dist(gen);

    // 1D
    std::cout << "#k\tt(fftw1)\tt(clfft1)\tt(cufft1)" << std::endl;
    for(size_t k = 1 ; k < k_max * 2 ; k++) {
        size_t n = 1 << k;
        std::cout << k;
        test_fftw(data, n, 1);
        test(ctx, data, n, 1);
        #ifdef USE_CUDA
        test_cufft(data, n, 1);
        #endif
        std::cout << std::endl;
    }
    std::cout << "\n\n";

    // 2D
    std::cout << "#k\tt(fftw2)\tt(clfft2)\t(cufft2)" << std::endl;
    for(size_t k = 1 ; k < k_max ; k++) {
        size_t n = 1 << k;
        std::cout << k;
        test_fftw(data, n, n);
        test(ctx, data, n, n);
        #ifdef USE_CUDA
        test_cufft(data, n, n);
        #endif
        std::cout << std::endl;
    }

    return 0;
}
