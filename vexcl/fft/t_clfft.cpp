// OpenCL FFT test
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

#if USE_MKL
// Run complex to complex X[2*N] to Y[2*N].
void runMKL(size_t n,const float * x,float * y)
{
  DFTI_DESCRIPTOR_HANDLE h;
  DftiCreateDescriptor(&h,DFTI_SINGLE,DFTI_COMPLEX,1,n);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiCommitDescriptor(h);
  DftiComputeForward(h,(void *)x,(void *)y);
  DftiFreeDescriptor(&h);
}
#endif // MKL

#if USE_FFTW
// Run complex to complex X[2*N] to Y[2*N].
void runFFTW(size_t n,const float * x,float * y)
{
  fftwf_plan p1 = fftwf_plan_dft_1d((int)n,(fftwf_complex *)x,(fftwf_complex *)y,
                                    FFTW_FORWARD,FFTW_ESTIMATE);
  fftwf_execute(p1);
  fftwf_destroy_plan(p1);
}
#endif // FFTW

void runCLFFT(clfft::Context * clfft,size_t n,const float * x,float * y)
{
  size_t bufferSize = 2*n*sizeof(float);
  size_t p = (size_t)1;
  const int device = 0;
  int current = 0; // Current buffer index
  cl_mem b[2] = { 0 };
  clfft::Event e;
  b[0] = clfft->createComplexBuffer(CL_MEM_READ_WRITE,n,0);
  b[1] = clfft->createComplexBuffer(CL_MEM_READ_WRITE,n,0);
  if (b[0] == 0 || b[1] == 0) goto END;
  e = clfft->enqueueWrite(device,b[current],false,0,bufferSize,x,clfft::EventVector());
  if (!CLFFT_CHECK_EVENT(e)) goto END;

  while (p<n)
  {
    size_t radix = (size_t)2;
    if ( (p<<4) <= n ) radix = 16;
    else if ( (p<<3) <= n ) radix = 8;
    else if ( (p<<2) <= n) radix = 4;
    else radix = 2;
    e = clfft->enqueueRadixRKernel(device,n,1,p,radix,clfft::FORWARD_DIRECTION,b[current],b[1-current],256,e);
    if (!CLFFT_CHECK_EVENT(e)) goto END;
    p *= radix;
    current = 1 - current;
  }

  e = clfft->enqueueRead(device,b[current],true,0,bufferSize,y,e);
  if (!CLFFT_CHECK_EVENT(e)) goto END;

 END:
  // Cleanup
  if (b[0] != 0) clReleaseMemObject(b[0]);
  if (b[1] != 0) clReleaseMemObject(b[1]);
}

bool runTests()
{
  std::string msg;
  clfft::Context * clfft = 0;
  cl_context context = 0;

  printf("Running tests...\n");

  context = createGPUContext();
  if (context == 0)
  {
    fprintf(stderr,"Could not create OpenCL context\n");
    return false;
  }

  bool ok = true;
  const size_t maxLog2N = 12;
  float * x = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y1 = (float *) malloc((2 * sizeof(float)) << maxLog2N);
  float * y2 = (float *) malloc((2 * sizeof(float)) << maxLog2N);

  rand(2 << maxLog2N,x);

  for (size_t log2N=1;log2N <= maxLog2N && ok;log2N++)
  {
    size_t n = (size_t)1<<log2N;
    printf("N=%d...\n",(int)n);
    clfft = clfft::Context::create(context,clfft::FLOAT_REAL_TYPE,msg);
    if (clfft == 0)
    {
      fprintf(stderr,"Creation failed:\n%s\n",msg.c_str());
      return false;
    }

    runCLFFT(clfft,n,x,y1);
#if USE_FFTW
    runFFTW(n,x,y2);
#elif USE_MKL
    runMKL(n,x,y2);
#endif

    double e = rmse<float>(2*n,y1,y2);
    if (e < 1.0e-3) continue; // OK

    // FAILED
    ok = false;
    printf("Input array:\n");
    dumpComplexArray(n,x);
    printf("Outputs:\n");
    dumpComplexArray(n,y1,y2);

    delete clfft;
  }

  if (!ok) printf("TESTS FAILED\n");

  free(x);
  free(y1);
  free(y2);
  clReleaseContext(context);

  return ok;
}

int main()
{
  bool ok = true;
  srand(0);

  ok &= runTests();

  printf("%s\n",(ok)?"OK":"FAILED!");
#ifdef WIN32
  printf("Press a key.\n");
  getchar();
#endif
}
