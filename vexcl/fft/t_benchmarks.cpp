// OpenCL FFT benchmarks
// EB Mar 2011

#ifdef WIN32
#define USE_FFTW 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#if USE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#endif
#if USE_MKL
#include <mkl_dfti.h>
#endif
#if USE_FFTW
#include <fftw3.h>
#endif
#include "CLFFT.h"
#include "TestFunctions.h"

#define BENCHMARK_IO 0

#if USE_MKL
// Run complex to complex X[2*N] to Y[2*N]. Return total time (s).
double runMKL(size_t n,const float * x,float * y,double maxBenchmarkTime)
{
  DFTI_DESCRIPTOR_HANDLE h;
  DftiCreateDescriptor(&h,DFTI_SINGLE,DFTI_COMPLEX,1,n);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiCommitDescriptor(h);

  const int nops = 2;
  double t = getRealTime();
  for (int op = 0;op < nops;op++)
    {
      DftiComputeForward(h,(void *)x,y);
    }
  t = (getRealTime() - t)/(double)nops;

  DftiFreeDescriptor(&h);
  return t;
}
#endif

#if USE_FFTW
// Run complex to complex X[2*N] to Y[2*N]. Return total time (s).
double runFFTW(size_t n,const float * x,float * y,double maxBenchmarkTime)
{
  fftwf_plan p1 = fftwf_plan_dft_1d((int)n,(fftwf_complex *)x,(fftwf_complex *)y,
                                    FFTW_FORWARD,FFTW_ESTIMATE);
  double totalIT = 0;
  double t0 = getRealTime();
  double t1;
  for (int nit=1;nit<=1024;nit<<=1)
  {
    for (int it = 0;it < nit;it++)
      {
        fftwf_execute(p1);
      }
    totalIT += nit;
    t1 = getRealTime();
    if (t1 - t0 >= maxBenchmarkTime) break;
  }
  fftwf_destroy_plan(p1);
  return (t1 - t0)/totalIT;
}

bool benchmarkFFTW(size_t maxLog2N,double maxBenchmarkTime) // float only
{
  float * x = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  rand(2 << maxLog2N, x);
  printf("FFTW(float)");
  for (size_t log2n = 8; log2n <= maxLog2N; log2n++)
  {
    size_t n = 1 << log2n;
    double t = runFFTW(n, x, y, maxBenchmarkTime);
    double flop = 5 * (double) log2n * (double) n;
    double perf = flop / t;
    printf("\t%.2f", t * 1.0e3);
  }
  printf("\n");
  free(x);
  free(y);
  return true;
}

#endif

#if USE_CUDA
// Run complex to complex X[2*N] to Y[2*N].
double runCUFFT(size_t n,const float * x,float * y, double maxBenchmarkTime)
{
  cufftHandle plan;
  cufftComplex * inData = 0;
  cufftComplex * outData = 0;
  size_t dataSize = sizeof(cufftComplex) * n;
  cudaError_t status;
  cufftResult fftStatus;

  fftStatus = cufftPlan1d(&plan,(int)n,CUFFT_C2C,1); // 1 is BATCH size
  assert(fftStatus == CUFFT_SUCCESS);
  status = cudaMalloc((void **)(&inData),dataSize);
  assert(status == cudaSuccess);
  status = cudaMalloc((void **)(&outData),dataSize);
  assert(status == cudaSuccess);

  // Send X to device
  status = cudaMemcpy(inData,x,dataSize,cudaMemcpyHostToDevice);
  assert(status == cudaSuccess);

  double totalIT = 0;
  double t0 = getRealTime();
  double t1;
  for (int nit = 1; nit <= 1024; nit++)
  {
    for (int it = 0; it < nit; it++)
    {
      // Run the FFT
#if BENCHMARK_IO
      cudaMemcpy(inData,x,dataSize,cudaMemcpyHostToDevice);
#endif
      cufftExecC2C(plan, inData, outData, CUFFT_FORWARD);
#if BENCHMARK_IO
      cudaMemcpy(y,outData,dataSize,cudaMemcpyDeviceToHost);
#endif
    }
    cudaDeviceSynchronize();
    t1 = getRealTime();
    totalIT += nit;
    if (t1 - t0 >= maxBenchmarkTime) break;
  } // nit loop
  double t = (t1 - t0)/totalIT;

  // Get Y from device
  status = cudaMemcpy(y,outData,dataSize,cudaMemcpyDeviceToHost);
  assert(status == cudaSuccess);
  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(inData);
  cudaFree(outData);

  return t;
}

bool benchmarkCUFFT(size_t maxLog2N,double maxBenchmarkTime) // float only
{
  float * x = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  rand(2 << maxLog2N, x);
  printf("CUFFT(float)");
  for (size_t log2n = 8; log2n <= maxLog2N; log2n++)
  {
    size_t n = 1 << log2n;
    double t = runCUFFT(n, x, y, maxBenchmarkTime);
    double flop = 5 * (double) log2n * (double) n;
    double perf = flop / t;
    printf("\t%.2f", t * 1.0e3);
  }
  printf("\n");
  free(x);
  free(y);
  return true;
}

#endif // CUDA

clfft::Event simpleForward1D(clfft::Context * clfft,int device,size_t n,cl_mem in,cl_mem out,clfft::EventVector deps)
{
  cl_mem b[2];
  int current = 0;
  size_t p = (size_t)1;
  clfft::Event e;
  size_t bufferSize = n * clfft->getRealTypeSize();
  b[current] = in;
  b[1-current] = out;

  while (p<n)
  {
    size_t radix = (size_t)2;
    if ( (p<<4) <= n ) radix = 16;
    else if ( (p<<3) <= n ) radix = 8;
    else if ( (p<<2) <= n) radix = 4;
    else radix = 2;
    e = clfft->enqueueRadixRKernel(device,n,1,p,radix,clfft::FORWARD_DIRECTION,b[current],b[1-current],256,deps);
    if (!CLFFT_CHECK_EVENT(e)) return e;
    deps = clfft::EventVector(e);
    p *= radix;
    current = 1 - current;
  }
  if (current != 1)
  {
    e = clfft->enqueueCopy(device,b[current],b[1-current],0,0,bufferSize,deps);
  }
  return e;
}

double runCLFFT(clfft::Context * clfft,size_t n,void * x,void * y, double maxBenchmarkTime)
{
  int realType = clfft->getRealType();
  size_t realSize = (realType == clfft::FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  size_t bufferSize = realSize * n * 2;
  cl_int status;
  cl_mem bIn = 0;
  cl_mem bOut = 0;
  bool ok = true;
  double t0,t1,t,totalIT;
  int deviceID = 0;
  t = -1;
  clfft::Event e;

  bIn = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
  bOut = clCreateBuffer(clfft->getOpenCLContext(),CL_MEM_READ_WRITE,bufferSize,0,&status);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

  e = clfft->enqueueWrite(deviceID,bIn,true,0,bufferSize,x,clfft::EventVector()); // blocking
  if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }

  t0 = getRealTime();
  t1 = 0;
  totalIT = 0;
  for (int nit = 1; nit <= 1024; nit <<= 1)
  {
    for (int it = 0; it < nit; it++)
    {
      e = simpleForward1D(clfft,deviceID,n,bIn,bOut,e);
      if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }
      status = clfft->enqueueBarrier(deviceID);
      if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
    }
    status = clfft->finish(deviceID);
    if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

    totalIT += nit;
    t1 = getRealTime();
    if (t1 - t0 >= maxBenchmarkTime) break; // Run 3s max test
  } // nit loop
  t = (t1 - t0) / totalIT; // s per FFT

  e = clfft->enqueueRead(deviceID,bOut,true,0,bufferSize,y,e); // blocking
  if (!CLFFT_CHECK_EVENT(e)) { ok = false; goto END; }

END:

  if (bIn != 0) clReleaseMemObject(bIn);
  if (bOut != 0) clReleaseMemObject(bOut);

  if (!ok) return -1; // Error
  return t;
}

bool benchmarkCLFFT(size_t maxLog2N,clfft::RealType realType,double maxBenchmarkTime)
{
  std::string msg;
  clfft::Context * clfft = 0;
  cl_context context = 0;
  bool ok = true;
  size_t realSize;
  size_t maxBufferSize;
  size_t maxN;
  void * x = 0;
  void * y = 0;

  context = createGPUContext();
  if (context == 0)
  {
    fprintf(stderr,"Could not create OpenCL context\n");
    ok = false; goto END;
  }
  clfft = clfft::Context::create(context,realType,msg);
  if (clfft == 0)
  {
    fprintf(stderr,"Creation failed:\n%s\n",msg.c_str());
    ok = false; goto END;
  }
  clReleaseContext(context); // clfft still references the context

  realSize = (realType == clfft::FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  maxN = (size_t)1 << maxLog2N;
  maxBufferSize = realSize * (size_t)2 * maxN;
  x = malloc(maxBufferSize);
  y = malloc(maxBufferSize);
  if (realType == clfft::FLOAT_REAL_TYPE) rand<float>((size_t)2*maxN,(float *)x);
  else rand<double>(2*maxN,(double *)x);

  printf("CLFFT(%s)",(realType==clfft::FLOAT_REAL_TYPE)?"float":"double");
  for (size_t log2n=8;log2n <= maxLog2N;log2n++)
  {
    size_t n = (size_t)1 << log2n;
    double t = runCLFFT(clfft,n,x,y,maxBenchmarkTime);
    double flop = 5 * (double)log2n * (double)n;
    double perf = flop / t; // flop/s per FFT
    printf("\t%.2f",t*1.0e3);
  } // log2n loop
  printf("\n");

END:
  if (x != 0) free(x);
  if (y != 0) free(y);
  delete clfft;
  return ok;
}

bool runBenchmarks(double maxBenchmarkTime)
{
  bool ok = true;

  // BENCHMARKS
  const size_t maxLog2N = 24;
  printf("n");
  for (size_t log2n=8;log2n <= maxLog2N;log2n++)
    printf("\t%ld",log2n);
  printf("\n");

  ok &= benchmarkCLFFT(maxLog2N,clfft::FLOAT_REAL_TYPE,maxBenchmarkTime);
#if USE_AMD_FFT
  ok &= benchmarkAMDFFT(maxLog2N,maxBenchmarkTime);
#endif
#if USE_CUDA
  ok &= benchmarkCUFFT(maxLog2N,maxBenchmarkTime);
#endif
#if USE_FFTW
  ok &= benchmarkFFTW(maxLog2N,maxBenchmarkTime);
#endif

  return ok;
}

int main()
{
  bool ok = true;
  srand(0);

  ok &= runBenchmarks(0.5);

#ifdef WIN32
  getchar();
#endif
}
