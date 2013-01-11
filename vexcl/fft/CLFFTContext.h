// OpenCL FFT library
// Copyright 2011, Eric Bainville

#ifndef CLFFTContext_h
#define CLFFTContext_h

#include <string>
#include <vector>
#include <CL/cl.h>
#include "CLFFTEvents.h"

namespace clfft {

enum RealType {
  FLOAT_REAL_TYPE = 0,
  DOUBLE_REAL_TYPE,
  NB_CLFFT_REAL_TYPES
};

enum Direction {
  FORWARD_DIRECTION = -1,
  INVERSE_DIRECTION = 1
};

class Context
{
public:

  /** Create a new instance of the class, using the given OpenCL context.
      The object manages internally command queues and compile kernels for
      all devices attached to the context.

      @param context is the OpenCl context, we acquire a reference to it.
      @param realType is the real type.
      @param msgLen is the allocated size in MSG, may be 0.
      @param msg receives error messages on failure, may be 0.

      @return a non 0 instance on success, 0 otherwise. */
  static Context * create(cl_context context,RealType realType,std::string & errorMsg);

  /** Destructor. */
  virtual ~Context();

  /** Get number of devices and command queues attached to the creation context.

      @return number of devices/command queues. */
  int getNDevices() const;

  /** Get real type.

      @return one of XXX_REAL_TYPE. */
  int getRealType() const;

  /** Get real type size.
  
      @return size of real type (bytes). */
  size_t getRealTypeSize() const;

  /** Get OpenCL context.

      @return the OpenCL context passed at creation. */
  cl_context getOpenCLContext() const;

  /** Call clFinish on command queue.

      @param device is the device index (0 = first device of context, etc.).

      @return an OpenCL status. */
  cl_int finish(int device);

  /** Enqueue a barrier in command queue.
   *
      @param device is the device index (0 = first device of context, etc.).

      @return an OpenCL status. */
  cl_int enqueueBarrier(int device);

  /** Create a complex buffer.

      @param flags are the CL_MEM_XXX flags passed to clCreateBuffer.
      @param n is the size of the buffer in complex numbers.
      @param hostPtr is the host pointer passed to clCreateBuffer, may be 0.

      @return a valid cl_mem object on success, and 0 otherwise. */
  cl_mem createComplexBuffer(cl_mem_flags flags,size_t n,void * hostPtr);

  /** Enqueue a buffer read.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_read,offset,cb,ptr are arguments of clReadBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueRead(int device,cl_mem buffer,cl_bool blocking_read,size_t offset,size_t cb,void * ptr,const EventVector & waitList);

  /** Enqueue a buffer write.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_write,offset,cb,ptr are arguments of clWriteBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueWrite(int device,cl_mem buffer,cl_bool blocking_write,size_t offset,size_t cb,const void * ptr,const EventVector & waitList);

  /** Enqueue buffer copy.

      @param device is the device index (0 = first device of context, etc.).
      @param buffer,blocking_write,offset,cb,ptr are arguments of clWriteBuffer.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueCopy(int device,cl_mem src,cl_mem dst,size_t src_offset,size_t dst_offset,size_t cb,const EventVector & waitList);

  /** Enqueue kernel to unpack and pad P real values into a buffer of N complex values.

      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the number of real values in IN.
      @param in is the input OpenCL buffer, P*BATCH real numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueUnpackReal1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

  /** Enqueue kernel to pad P complex values into a buffer of N complex values.

      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the number of complex values in IN.
      @param in is the input OpenCL buffer, P*BATCH complex numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueUnpackComplex1D(int device,size_t n,size_t batch,size_t p,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

  /** Enqueue one radix-R kernel, one step of the computation of K parallel DFT-N.
  
      @param device is the device index (0 = first device of context, etc.).
      @param n is the size of the DFT, must be a power of 2.
      @param batch is the number of parallel DFT-N to process.
      @param p is the length of already transformed sequences in IN. RADIX*P is the length of transformed sequences in OUT.
      @param radix is the radix: 2, 4, or 8.
      @param direction is FORWARD_DIRECTION or INVERSE_DIRECTION.
      @param in is the input OpenCL buffer, N*BATCH complex numbers.
      @param out is the output OpenCL buffer, N*BATCH complex numbers.
      @param wg is the desired workgroup size.
      @param waitList is the event wait list.

      @return an OpenCL event. */
  Event enqueueRadixRKernel(int device,size_t n,size_t batch,size_t p,size_t radix,Direction direction,cl_mem in,cl_mem out,size_t wg,const EventVector & waitList);

private:

  // Constructor
  Context();

  enum Kernels {
    UNPACK_REAL1D_KERNEL = 0,
    UNPACK_COMPLEX1D_KERNEL,
    FFT_RADIX2_KERNEL,
    FFT_RADIX4_KERNEL,
    FFT_RADIX8_KERNEL,
    FFT_RADIX16_KERNEL,
    NB_KERNELS
  };

  // Functions used to setup kernel args. KERNELID is one of XXX_KERNEL.
  void clearArgs(int kernelID) { mKernelIndex[kernelID] = 0; }
  template <class T> cl_int pushArg(int kernelID,const T & x)
  {
    cl_int status = clSetKernelArg(mKernel[kernelID],mKernelIndex[kernelID]++,sizeof(T),&x);
    if (!CLFFT_CHECK_STATUS(status)) return status;
    return CL_SUCCESS;
  }
  // Run 1D or 2D kernel on device.
  Event enqueueKernel(int device,int kernelID,size_t nx,size_t ny,size_t wg,const EventVector & waitList);

  RealType mRealType; // Real type used
  size_t mRealTypeSize; // Size of the real type (bytes)
  cl_context mContext; // OpenCL context
  cl_program mProgram; // Program
  std::vector<cl_kernel> mKernel; // Kernels
  std::vector<cl_device_id> mDevice; // Devices in context
  std::vector<cl_command_queue> mQueue; // One command queue per context device
  std::vector<size_t> mMaxWorkGroupSize; // Max workgroup size for each device
  std::vector<int> mKernelIndex; // Current number of args set in each kernel
};

} // namespace

#endif // #ifndef CLFFTContext_h
