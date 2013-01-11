// OpenCL FFT kernels
// Copyright (c) 2011, Eric Bainville

#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// double
typedef double real_t;
typedef double2 real2_t;
#define FFT_PI       3.14159265358979323846
// cos,sin Pi/4
#define FFT_SQRT_1_2 0.70710678118654752440
// cos,sin Pi/8
#define FFT_C8       0.92387953251128675613
#define FFT_S8       0.38268343236508977173

#else

// float
typedef float real_t;
typedef float2 real2_t;
#define FFT_PI       3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f
// cos,sin Pi/8
#define FFT_C8       0.923879532511f
#define FFT_S8       0.382683432365f

#endif

// Set to 1 to use MAD
#define USE_MAD 1

// Return A*B
real2_t mul(real2_t a,real2_t b)
{
#if USE_MAD
  return (real2_t)(mad(a.x,b.x,-a.y*b.y),mad(a.x,b.y,a.y*b.x)); // mad
#else
  return (real2_t)(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); // no mad
#endif
}

// Return A*conj(B)
real2_t mul_conj(real2_t a,real2_t b)
{
#if USE_MAD
  return (real2_t)(mad(a.x,b.x,a.y*b.y),mad(a.x,-b.y,a.y*b.x)); // mad
#else
  return (real2_t)(a.x*b.x+a.y*b.y,-a.x*b.y+a.y*b.x); // no mad
#endif
}

// twiddle_p_q returns A*exp(-PI*i*P/Q)
real2_t twiddle_1_1(real2_t a)
{
  // A * (-1)
  return -a;
}
real2_t twiddle_1_2(real2_t a)
{
  // A * (-I)
  return (real2_t)(a.y,-a.x);
}
real2_t twiddle_1_4(real2_t a)
{
  // A * (1-I)*sqrt(1/2)
  return (real2_t)(FFT_SQRT_1_2*(a.x+a.y),FFT_SQRT_1_2*(-a.x+a.y));
}
real2_t twiddle_3_4(real2_t a)
{
  // A * (-1-I)*sqrt(1/2)
  return (real2_t)(FFT_SQRT_1_2*(-a.x+a.y),FFT_SQRT_1_2*(-a.x-a.y));
}
real2_t twiddle_1_8(real2_t a)
{
  return mul(a,(real2_t)(FFT_C8,-FFT_S8));
}
real2_t twiddle_3_8(real2_t a)
{
  return mul(a,(real2_t)(FFT_S8,-FFT_C8));
}
real2_t twiddle_5_8(real2_t a)
{
  return mul(a,(real2_t)(-FFT_S8,-FFT_C8));
}
real2_t twiddle_7_8(real2_t a)
{
  return mul(a,(real2_t)(-FFT_C8,-FFT_S8));
}

// Return A * exp(K*ALPHA*i)
real2_t twiddle(real2_t a,int k,real_t alpha)
{
  real_t cs,sn;
  sn = sincos((real_t)k*alpha,&cs);
  return mul(a,(real2_t)(cs,sn));
}

// Macros operating on complex numbers stored as 2-vectors.

// A *= B. Arguments must be variables.
#define MUL(a,b) { real2_t tmp=a; a.x=tmp.x*b.x-tmp.y*b.y; a.y=tmp.x*b.y+tmp.y*b.x; }

// A *= EXP(K*ALPHA)
#define TWIDDLE(a,k,alpha) { real_t cs,sn; sn = sincos((real_t)(k)*(alpha),&cs); real2_t twiddle = (real2_t)(cs,sn); MUL(a,twiddle); }

// In-place DFT-2, output is (a,b). Arguments must be variables.
#define DFT2(a,b) { real2_t tmp = a - b; a += b; b = tmp; }

// In-place DFT-4, output is (a,c,b,d). Arguments must be variables.
#define DFT4(a,b,c,d) { DFT2(a,c); DFT2(b,d); d=twiddle_1_2(d); DFT2(a,b); DFT2(c,d); }

// DFT8, output is (a,aa,c,cc,b,bb,d,dd)
#define DFT8(a,b,c,d,aa,bb,cc,dd) { \
  DFT2(a,aa); DFT2(b,bb); DFT2(c,cc); DFT2(d,dd); \
  bb=twiddle_1_4(bb); cc=twiddle_1_2(cc); dd=twiddle_3_4(dd); \
  DFT4(a,b,c,d); DFT4(aa,bb,cc,dd); \
}

// DFT16, output permutation is 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
#define DFT16(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7) { \
  DFT2(a0,b0); DFT2(a1,b1); DFT2(a2,b2); DFT2(a3,b3); \
  DFT2(a4,b4); DFT2(a5,b5); DFT2(a6,b6); DFT2(a7,b7); \
  b1=twiddle_1_8(b1); b2=twiddle_1_4(b2); b3=twiddle_3_8(b3); \
  b4=twiddle_1_2(b4); b5=twiddle_5_8(b5); b6=twiddle_3_4(b6); b7=twiddle_7_8(b7); \
  DFT8(a0,a1,a2,a3,a4,a5,a6,a7); \
  DFT8(b0,b1,b2,b3,b4,b5,b6,b7); \
}

/* Compute T x DFT-2.
   T is the number of threads.
   N = 2*T is the size of input vectors.
   X[N], Y[N]
   P is the length of input sub-sequences: 1,2,4,...,T.
   Each DFT-2 has input (X[I],X[I+T]), I=0..T-1,
   and output Y[J],Y|J+P], J = I with one 0 bit inserted at postion P. */
__kernel void fftRadix2Kernel(__global const real2_t * x,__global real2_t * y,int p)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0); // thread index
  int k = i&(p-1); // index in input sequence, in 0..P-1
  int j = ((i-k)<<1) + k; // output index
  int batchOffset = get_global_id(1) * (t << 1); // offset to input vector in batch operation (2D thread space) 
  real_t alpha = -FFT_PI*(real_t)k/(real_t)p;
  
  // Read and twiddle input
  x += i + batchOffset;
  real2_t u0 = x[0];
  real2_t u1 = twiddle(x[t],1,alpha);

  // In-place DFT-2
  DFT2(u0,u1);

  // Write output
  y += j + batchOffset;
  y[0] = u0;
  y[p] = u1;
}

/* Compute T x DFT-4.
   T is the number of threads.
   N = 4*T is the size of input vectors.
   X[N], Y[N]
   P is the length of input sub-sequences: 1,4,16,...,T.
   Each DFT-4 has input (X[I],X[I+T],X[I+2*T],X[I+3*T]), I=0..T-1,
   and output (Y[J],Y|J+P],Y[J+2*P],Y[J+3*P], J = I with two 0 bits inserted at postion P. */
__kernel void fftRadix4Kernel(__global const real2_t * x,__global real2_t * y,int p)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0); // thread index
  int k = i&(p-1); // index in input sequence, in 0..P-1
  int j = ((i-k)<<2) + k; // output index
  int batchOffset = get_global_id(1) * (t << 2); // offset to input vector in batch operation (2D thread space) 
  real_t alpha = -FFT_PI*(real_t)k/(real_t)(2*p);

  // Read and twiddle input
  x += i + batchOffset;
  real2_t u0 = x[0];
  real2_t u1 = twiddle(x[t],1,alpha);
  real2_t u2 = twiddle(x[2*t],2,alpha);
  real2_t u3 = twiddle(x[3*t],3,alpha);

  // In-place DFT-4
  DFT4(u0,u1,u2,u3);

  // Shuffle and write output
  y += j + batchOffset;
  y[0]   = u0;
  y[p]   = u2;
  y[2*p] = u1;
  y[3*p] = u3;
}

/* Compute T x DFT-8.
   T is the number of threads.
   N = 8*T is the size of input vectors.
   X[N], Y[N]
   P is the length of input sub-sequences: 1,8,64,...,T.
   Each DFT-8 has input (X[I+0*T],...,X[I+7*T]), I=0..T-1,
   and output (Y[J+0*P],...,Y[J+7*P], J = I with three 0 bits inserted at postion P. */
__kernel void fftRadix8Kernel(__global const real2_t * x,__global real2_t * y,int p)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0); // thread index
  int k = i&(p-1); // index in input sequence, in 0..P-1
  int j = ((i-k)<<3) + k; // output index
  int batchOffset = get_global_id(1) * (t << 3); // offset to input vector in batch operation (2D thread space) 
  real_t alpha = -FFT_PI*(real_t)k/(real_t)(4*p);

  // Read and twiddle input
  x += i + batchOffset;
  real2_t u0 = x[0];
  real2_t u1 = twiddle(x[  t],1,alpha);
  real2_t u2 = twiddle(x[2*t],2,alpha);
  real2_t u3 = twiddle(x[3*t],3,alpha);
  real2_t u4 = twiddle(x[4*t],4,alpha);
  real2_t u5 = twiddle(x[5*t],5,alpha);
  real2_t u6 = twiddle(x[6*t],6,alpha);
  real2_t u7 = twiddle(x[7*t],7,alpha);

  // In-place DFT-8
  DFT8(u0,u1,u2,u3,u4,u5,u6,u7);

  // Shuffle and write output
  y += j + batchOffset;
  y[  0] = u0;
  y[  p] = u4;
  y[2*p] = u2;
  y[3*p] = u6;
  y[4*p] = u1;
  y[5*p] = u5;
  y[6*p] = u3;
  y[7*p] = u7;
}

/* Compute T x DFT-16.
   T is the number of threads.
   N = 16*T is the size of input vectors.
   X[N], Y[N]
   P is the length of input sub-sequences: 1,16,256,...,T.
   Each DFT-16 has input (X[I+0*T],...,X[I+15*T]), I=0..T-1,
   and output (Y[J+0*P],...,Y[J+15*P], J = I with four 0 bits inserted at postion P. */
__kernel void fftRadix16Kernel(__global const real2_t * x,__global real2_t * y,int p)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0); // thread index
  int k = i&(p-1); // index in input sequence, in 0..P-1
  int j = ((i-k)<<4) + k; // output index
  int batchOffset = get_global_id(1) * (t << 4); // offset to input vector in batch operation (2D thread space) 
  real_t alpha = -FFT_PI*(real_t)k/(real_t)(8*p);

  // Read and twiddle input
  x += i + batchOffset;
  real2_t u0 = x[0];
  real2_t u1 = twiddle(x[  t],1,alpha);
  real2_t u2 = twiddle(x[2*t],2,alpha);
  real2_t u3 = twiddle(x[3*t],3,alpha);
  real2_t u4 = twiddle(x[4*t],4,alpha);
  real2_t u5 = twiddle(x[5*t],5,alpha);
  real2_t u6 = twiddle(x[6*t],6,alpha);
  real2_t u7 = twiddle(x[7*t],7,alpha);
  real2_t u8 = twiddle(x[8*t],8,alpha);
  real2_t u9 = twiddle(x[9*t],9,alpha);
  real2_t u10 = twiddle(x[10*t],10,alpha);
  real2_t u11 = twiddle(x[11*t],11,alpha);
  real2_t u12 = twiddle(x[12*t],12,alpha);
  real2_t u13 = twiddle(x[13*t],13,alpha);
  real2_t u14 = twiddle(x[14*t],14,alpha);
  real2_t u15 = twiddle(x[15*t],15,alpha);

  // In-place DFT-16
  DFT16(u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15);

  // Shuffle and write output
  y += j + batchOffset;
  y[  0] = u0;
  y[  p] = u8;
  y[2*p] = u4;
  y[3*p] = u12;
  y[4*p] = u2;
  y[5*p] = u10;
  y[6*p] = u6;
  y[7*p] = u14;
  y[8*p] = u1;
  y[9*p] = u9;
  y[10*p] = u5;
  y[11*p] = u13;
  y[12*p] = u3;
  y[13*p] = u11;
  y[14*p] = u7;
  y[15*p] = u15;
}

__kernel void unpackReal1DKernel(__global const real_t * in,__global real2_t * out,int p)
{
  int i = get_global_id(0);
  out[i] = (real2_t)((i<p)?in[i]:(real_t)0,(real_t)0);
}

__kernel void unpackComplex1DKernel(__global const real2_t * in,__global real2_t * out,int p)
{
  int i = get_global_id(0);
  out[i] = (i<p)?in[i]:(real2_t)(0);
}
