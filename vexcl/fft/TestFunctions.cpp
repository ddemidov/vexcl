#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#ifdef Linux
#include <sys/time.h>
#endif

#include "TestFunctions.h"

#ifdef WIN32
double getRealTime()
{
  LARGE_INTEGER freq,value;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&value);
  return (double)value.QuadPart/(double)freq.QuadPart;
}
#endif

#ifdef Linux
double getRealTime()
{
  struct timeval tv;
  gettimeofday(&tv,0);
  return (double)tv.tv_sec + 1.0e-6*(double)tv.tv_usec;
}
#endif

// Return random double in 0..1
double rnd()
{
  double s = 0;
#if 0
  const double k = 1.0/(1.0+RAND_MAX);
  s = k * (s + (double)rand());
  s = k * (s + (double)rand());
#else
  s = rand() & 15;  // to easily test partial sums
#endif
  return s;
}

// Create an OpenCL context including all GPU devices on the first platform
// providing GPU devices.  Return a valid OpenCL context on success, and 0 otherwise.
cl_context createGPUContext()
{
  const int MAX_PLATFORMS = 8;
  const int MAX_DEVICES = 16;
  cl_platform_id platform[MAX_PLATFORMS];
  cl_device_id device[MAX_DEVICES];
  cl_uint nPlatforms = 0;
  cl_uint nDevices = 0;
  cl_context context = 0;
  cl_int status = clGetPlatformIDs(MAX_PLATFORMS,platform,&nPlatforms);
  if (status < 0 || nPlatforms == 0) return 0; // No platform

  for (cl_uint p=0;p<nPlatforms;p++)
  {
    nDevices = 0;
    status = clGetDeviceIDs(platform[p],CL_DEVICE_TYPE_GPU,MAX_DEVICES,device,&nDevices);
    if (status < 0 || nDevices == 0) continue; // Failed for this platform

    // Try to create a context using all devices
    cl_context_properties props[5];
    int index = 0;
    props[index++] = CL_CONTEXT_PLATFORM;
    props[index++] = (cl_context_properties)platform[p];
    props[index++] = 0;
    context = clCreateContext(props,nDevices,device,0,0,0);
    if (context != 0) return context; // OK
  }

  return 0; // Failed
}
