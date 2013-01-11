// OpenCL FFT library
// Copyright 2011, Eric Bainville

#include "CLFFT.h"

extern const char * CLFFTKernelCode;

// Return log2(n) if N is a power of 2, and -1 otherwise.
inline int log2(size_t n)
{
  int k = 0;
  if (n <= 0) return -1; // Not a power of 2
  while (n != (size_t)1)
  {
    if (n&(size_t)1) return -1; // Has at least 2 bits set
    n >>= (size_t)1;
    k++;
  }
  return k;
}

clfft::Context * clfft::Context::create(cl_context context,RealType realType,std::string & errorMsg)
{
  errorMsg.clear();
  // Check args
  if (context == 0) { errorMsg.assign("Invalid OpenCL context"); return 0; } // Invalid context
  if (realType < 0 || realType >= NB_CLFFT_REAL_TYPES) { errorMsg.assign("Invalid data type"); return 0; } // Invalid type

  bool ok = true;
  cl_int status;
  const int MAX_DEVICES = 16;
  cl_device_id device[MAX_DEVICES];
  size_t sz;
  int nDevices;
  clfft::Context * result = 0;
  const int MAX_OPTIONS = 1024;
  char options[MAX_OPTIONS];
  
  // Alloc result
  result = new clfft::Context();
  if (result == 0) return 0; // Alloc failed
  result->mRealType = realType;
  result->mRealTypeSize = (realType == FLOAT_REAL_TYPE)?sizeof(float):sizeof(double);
  result->mContext = context;
  status = clRetainContext(context);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

  // Get context devices
  status = clGetContextInfo(context,CL_CONTEXT_DEVICES,MAX_DEVICES*sizeof(device[0]),device,&sz);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
  nDevices = (int) ( sz / sizeof(device[0]) );

  // Create and build program
  result->mProgram = clCreateProgramWithSource(context,1,&CLFFTKernelCode,0,&status);
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }
  snprintf(options,MAX_OPTIONS," -cl-fast-relaxed-math -D CONFIG_USE_DOUBLE=%d",
    (realType == DOUBLE_REAL_TYPE)?1:0
  );
  status = clBuildProgram(result->mProgram,nDevices,device,options,0,0);
  // Collect build errors in msg string
  if (status == CL_BUILD_PROGRAM_FAILURE)
  {
    for (int d=0;d<nDevices;d++)
    {
      std::string e;
      cl_int s2;
      s2 = clGetProgramBuildInfo(result->mProgram,device[d],CL_PROGRAM_BUILD_LOG,0,0,&sz);
      if (!CLFFT_CHECK_STATUS(s2) || sz == 0) continue;
      e.resize(sz+1,' ');
      s2 = clGetProgramBuildInfo(result->mProgram,device[d],CL_PROGRAM_BUILD_LOG,sz,&(e[0]),&sz);
      if (!CLFFT_CHECK_STATUS(s2) || sz == 0) continue;
      e.resize(sz);
      while (sz > 0 && !isascii(e[sz-1])) sz--; // Eliminate garbage at the end of the string
      char aux[200];
      snprintf(aux,200,"Build error for device %d:\n",(int)d);
      errorMsg.append(aux);
      errorMsg.append(e);
    }
  }
  if (!CLFFT_CHECK_STATUS(status)) { ok = false; goto END; }

  // Create kernels
  result->mKernel.resize(NB_KERNELS,0);
  result->mKernelIndex.resize(NB_KERNELS,0);
  for (int i=0;i<NB_KERNELS;i++)
  {
    const char * kname = 0;
    switch (i)
    {
    case UNPACK_REAL1D_KERNEL: kname = "unpackReal1DKernel"; break;
    case UNPACK_COMPLEX1D_KERNEL: kname = "unpackComplex1DKernel"; break;
    case FFT_RADIX2_KERNEL: kname = "fftRadix2Kernel"; break;
    case FFT_RADIX4_KERNEL: kname = "fftRadix4Kernel"; break;
    case FFT_RADIX8_KERNEL: kname = "fftRadix8Kernel"; break;
    case FFT_RADIX16_KERNEL: kname = "fftRadix16Kernel"; break;
    default: break;
    }
    if (kname == 0) { ok = false; break; } // Failed
    result->mKernel[i] = clCreateKernel(result->mProgram,kname,&status);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("Kernel creation failed: "); errorMsg.append(kname); ok = false; goto END; }
  }

  // Create command queues
  result->mQueue.resize(nDevices,0);
  result->mDevice.resize(nDevices,0);
  for (int d=0;d<nDevices;d++)
  {
    cl_command_queue_properties props = 0;
    result->mDevice[d] = device[d];
    result->mQueue[d] = clCreateCommandQueue(context,device[d],props,&status);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("Command queue creation failed"); ok = false; goto END; }
  }

  // Keep max device workgroup size
  result->mMaxWorkGroupSize.resize(nDevices,(size_t)0);
  for (int d=0;d<nDevices;d++)
  {
    status = clGetDeviceInfo(result->mDevice[d],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&(result->mMaxWorkGroupSize[d]),0);
    if (!CLFFT_CHECK_STATUS(status)) { errorMsg.append("MAX_WORK_GROUP_SIZE query failed"); ok = false; goto END; }
  }

 END:
  if (!ok) { delete result; return 0; } // Error
  return result;
}

clfft::Context::Context() : mContext(0), mProgram(0)
{
}

clfft::Context::~Context()
{
  for (std::vector<cl_command_queue>::iterator it = mQueue.begin(); it != mQueue.end(); it++) if (*it != 0) { clReleaseCommandQueue(*it); }
  for (std::vector<cl_kernel>::iterator it = mKernel.begin(); it != mKernel.end(); it++) if (*it != 0) { clReleaseKernel(*it); }
  mQueue.clear();
  mKernel.clear();
  mDevice.clear();
  if (mProgram != 0) clReleaseProgram(mProgram);
  if (mContext != 0) clReleaseContext(mContext);
}

int clfft::Context::getNDevices() const
{
  return (int)mDevice.size();
}

int clfft::Context::getRealType() const
{
  return mRealType;
}

size_t clfft::Context::getRealTypeSize() const
{
  return mRealTypeSize;
}

cl_context clfft::Context::getOpenCLContext() const
{
  return mContext;
}

cl_int clfft::Context::finish(int device)
{
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_DEVICE;
  return clFinish(mQueue[device]);
}

cl_int clfft::Context::enqueueBarrier(int device)
{
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_DEVICE;
  return clEnqueueBarrier(mQueue[device]);
}

cl_mem clfft::Context::createComplexBuffer(cl_mem_flags flags,size_t n,void * hostPtr)
{
  cl_int status;
  cl_mem m = clCreateBuffer(mContext,flags,2 * n * mRealTypeSize,hostPtr,&status);
  if (!CLFFT_CHECK_STATUS(status)) return 0;
  return m;
}

clfft::Event clfft::Context::enqueueRead(int device,cl_mem buffer,cl_bool blocking_read,size_t offset,size_t cb,void * ptr,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueReadBuffer(mQueue[device],buffer,blocking_read,offset,cb,ptr,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e,status);
}

clfft::Event clfft::Context::enqueueWrite(int device,cl_mem buffer,cl_bool blocking_write,size_t offset,size_t cb,const void * ptr,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueWriteBuffer(mQueue[device],buffer,blocking_write,offset,cb,ptr,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e);
}

clfft::Event clfft::Context::enqueueCopy(int device,cl_mem src,cl_mem dst,size_t src_offset,size_t dst_offset,size_t cb,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_event e;
  cl_int status;
  status = clEnqueueCopyBuffer(mQueue[device],src,dst,src_offset,dst_offset,cb,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e);
}

clfft::Event clfft::Context::enqueueUnpackReal1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_kernel kernel = mKernel[UNPACK_REAL1D_KERNEL];
  int index = 0;
  cl_int status;
  cl_int pp = (cl_int)p;
  cl_event e;
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&in); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&out); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_int),&pp); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&wg,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}

clfft::Event clfft::Context::enqueueUnpackComplex1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  cl_kernel kernel = mKernel[UNPACK_COMPLEX1D_KERNEL];
  int index = 0;
  cl_int status;
  cl_int pp = (cl_int)p;
  cl_event e;
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&in); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_mem),&out); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clSetKernelArg(kernel,index++,sizeof(cl_int),&pp); if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&wg,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}

clfft::Event clfft::Context::enqueueRadixRKernel(int device,size_t n,size_t batch,size_t p,size_t radix,Direction direction,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList)
{
  int kernelID = -1;
  switch (radix)
  {
  case 2: kernelID = FFT_RADIX2_KERNEL; break;
  case 4: kernelID = FFT_RADIX4_KERNEL; break;
  case 8: kernelID = FFT_RADIX8_KERNEL; break;
  case 16: kernelID = FFT_RADIX16_KERNEL; break;
  }
  if (kernelID < 0) return clfft::Event(CL_INVALID_ARG_VALUE);
  clearArgs(kernelID);
  pushArg(kernelID,in);
  pushArg(kernelID,out);
  pushArg<cl_int>(kernelID,(cl_int)p);
  return enqueueKernel(device,kernelID,n/radix,batch,wg,waitList);
}

clfft::Event clfft::Context::enqueueKernel(int device,int kernelID,size_t n,size_t batch,size_t wg,const EventVector & waitList)
{
  if (device < 0 || device >= (int)mDevice.size()) return clfft::Event(CL_INVALID_DEVICE);
  if (n <= 0 || batch <= 0) return clfft::Event(CL_INVALID_WORK_ITEM_SIZE);
  if (wg <= 0) return clfft::Event(CL_INVALID_WORK_GROUP_SIZE);
  cl_int status;

  // Limit WG if needed, by max device wg size, by max kernel wg size, by number of X threads
  wg = std::min(wg,n);
  wg = std::min(wg,mMaxWorkGroupSize[device]);
  size_t maxKWG;
  status = clGetKernelWorkGroupInfo(mKernel[kernelID],mDevice[device],CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&maxKWG,0);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  wg = std::min(wg,maxKWG);

  size_t wx = n; // Is a power of 2
  while ( (wx & wg) == 0 ) wx >>= 1; // remains a power of 2

  size_t nThreads[2];
  nThreads[0] = n;
  nThreads[1] = batch;
  size_t workGroup[2];
  workGroup[0] = wx;
  workGroup[1] = (size_t)1;
  cl_event e;
  status = clEnqueueNDRangeKernel(mQueue[device],mKernel[kernelID],(batch==(size_t)1)?1:2,0,nThreads,workGroup,waitList.size(),waitList.events(),&e);
  if (!CLFFT_CHECK_STATUS(status)) return clfft::Event(status);
  return clfft::Event(e); // OK
}


#if 0
cl_int CLFFTContext::enqueueFFT1D(int device,Direction direction,cl_mem in,cl_mem out,size_t n,cl_uint nEventsInWaitList,const cl_event * waitList,cl_event * e)
{
  if (e != 0) *e = 0; // Initialize return value
  int ln = log2(n);
  if (ln<0 || n<=1) return CL_INVALID_VALUE; // Invalid N

  // Buffers, 0 is IN and 1 is OUT
  cl_mem b[2] = { 0 };
  b[0] = in;
  b[1] = out;
  cl_event ev[2] = { 0 };
  cl_int status;

  // Loop on sequence size
  int current = 0; // Current buffer containing data
  size_t p = 1; // Length of combined sequences in buffer
  bool first = true;
  while (p < n)
  {
    // Select next radix of FFT to apply to the current sequences
    size_t logRadix = 1;
    if ( (p << 3) <= n ) logRadix = 3; else
    if ( (p << 2) <= n ) logRadix = 2; else
    logRadix = 1;

    // Get corresponding kernel
    int kernelID = -1;
    switch (logRadix)
    {
    case 1: kernelID = FFT_RADIX2_KERNEL; break;
    case 2: kernelID = FFT_RADIX4_KERNEL; break;
    case 3: kernelID = FFT_RADIX8_KERNEL; break;
    }
    if (kernelID < 0) { status = CL_INVALID_KERNEL; break; } // Bad log radix?

    // Enqueue kernel
    status = enqueueInOutP(device,kernelID,b[current],b[1-current],p,n >> logRadix,(first)?nEventsInWaitList:1,(first)?waitList:ev,ev+1);
    if (ev[0] != 0) clReleaseEvent(ev[0]);
    ev[0] = ev[1]; ev[1] = 0;
    if (!CLFFT_CHECK_STATUS(status)) break;

    // Prepare for next iteration
    current = 1 - current; // swap buffers
    p <<= logRadix; // increase sequence length
    first = false;
  }

  // Enqueue a final copy if the current buffer is not the output
  if (status == CL_SUCCESS && current != 1)
  {
    status = enqueueInOutP(device,UNPACK_COMPLEX1D_KERNEL,b[current],b[1-current],n,n,1,ev,ev+1);
    if (ev[0] != 0) clReleaseEvent(ev[0]);
    ev[0] = ev[1]; ev[1] = 0;
  }

  // Cleanup last event if needed
  if (status != CL_SUCCESS || e == 0)
  {
    if (ev[0] != 0) { clReleaseEvent(ev[0]); ev[0] = 0; }
  }
  // Set output event
  if (e != 0) *e = ev[0];

  return status;
}

cl_int CLFFTContext::enqueueInOutP(int device,int kernelID,cl_mem in,cl_mem out,size_t p,size_t n,cl_uint nEventsInWaitList,const cl_event * waitList,cl_event * e)
{
  if (e != 0) *e = 0; // Initialize return value
  if (device < 0 || device >= (int)mDevice.size()) return CL_INVALID_VALUE; // Invalid DEVICE
  int ln = log2(n);
  if (ln<0 || p<0 || p>n) return CL_INVALID_VALUE; // Invalid P or N

  cl_kernel kernel = mKernel[kernelID];
  cl_int status;

  // Args
  status = clSetKernelArg(kernel,0,sizeof(cl_mem),&in);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  status = clSetKernelArg(kernel,1,sizeof(cl_mem),&out);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  cl_int pp = (cl_int)p;
  status = clSetKernelArg(kernel,2,sizeof(cl_int),&pp);
  if (!CLFFT_CHECK_STATUS(status)) return status;

  // Group size
  size_t groupSize = n;
  size_t maxDeviceGroupSize = 0;
  size_t maxKernelGroupSize = 0;
  size_t maxGroupSize = (size_t)256; // force
  status = clGetDeviceInfo(mDevice[device],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&maxDeviceGroupSize,0);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  maxGroupSize = std::min(maxGroupSize,maxDeviceGroupSize);
  status = clGetKernelWorkGroupInfo(kernel,mDevice[device],CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&maxKernelGroupSize,0);
  if (!CLFFT_CHECK_STATUS(status)) return status;
  maxGroupSize = std::min(maxGroupSize,maxKernelGroupSize);
  while (groupSize > maxGroupSize) groupSize >>= (size_t)1; // keep it a divisor of N
  if (groupSize == 0) return CL_INVALID_WORK_GROUP_SIZE;

  // Enqueue
  status = clEnqueueNDRangeKernel(mQueue[device],kernel,1,0,&n,&groupSize,nEventsInWaitList,waitList,e);
  if (!CLFFT_CHECK_STATUS(status)) return status;

  return CL_SUCCESS; // OK
}
#endif
