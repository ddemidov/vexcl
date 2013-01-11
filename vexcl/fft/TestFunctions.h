// Test functions
// EB Mar 2011

#ifndef TestFunctions_h
#define TestFunctions_h

#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

// Return wallclock time in seconds. (origin is arbitrary)
double getRealTime();

// Return random number in [0,1[
double rnd();

// Create an OpenCL context including all GPU devices on the first platform
// providing GPU devices.  Return a valid OpenCL context on success, and 0 otherwise.
cl_context createGPUContext();

// Initialize X[N] with random values in [-1,+1]
template <typename REAL> void rand(size_t n,REAL * x)
{
  for (size_t i=0;i<n;i++) x[i] = (REAL)(2.0*rnd()-1.0);
}

// Return RMSE between X[N] and Y[N]
template <typename REAL> double rmse(size_t n,const REAL * x,const REAL * y)
{
  double s = 0;
  for (size_t i=0;i<n;i++)
    {
      double d = (REAL)x[i] - (REAL)y[i];
      s += d*d;
    }
  return sqrt(s/(double)n);
}

// Dump X[2*N] as complex array
template <typename REAL> void dumpComplexArray(size_t n,const REAL * x)
{
  for (size_t i=0;i<n;i++)
    {
      printf("  %2d: %10f,%10f\n",(int)i,(double)x[2*i],(double)x[2*i+1]);
    }
}

// Dump X[2*N] and X_REF[2*N] as complex arrays, and show differences
template <typename REAL> void dumpComplexArray(size_t n,const REAL * x,const REAL * x_ref)
{
  for (size_t i=0;i<n;i++)
    {
      double d0 = (REAL)x_ref[2*i] - (REAL)x[2*i];
      double d1 = (REAL)x_ref[2*i+1] - (REAL)x[2*i+1];
      double h = hypot(d0,d1);
      printf("  %2d: %10f,%10f   -- %10f,%10f  %s\n",(int)i,(REAL)x[2*i],(REAL)x[2*i+1],(REAL)x_ref[2*i],(REAL)x_ref[2*i+1],(h>1.0e-4)?"****":"");
    }
  // Check permutation
  printf("Permutation:");
  for (size_t i=0;i<n;i++)
  {
    int k = -1;
    for (size_t j=0;j<n;j++)
    {
      double d0 = x_ref[2*i] - x[2*j];
      double d1 = x_ref[2*i+1] - x[2*j+1];
      double h = sqrt(d0*d0+d1*d1);
      if (h<1.0e-4) { k = (int)j; break; }
    }
    printf("%c%d",(i==0)?' ':',',k);
  }
  printf("\n");
}

#endif // #ifndef TestFunctions_h
