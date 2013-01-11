// Check and report OpenCL errors
// Copyright 2011, Eric Bainville

#ifndef CLFFTErrors_h
#define CLFFTErrors_h

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#ifdef WIN32
#define snprintf _snprintf
#endif

namespace clfft {

#define CLFFT_CHECK_STATUS(status) clfft::checkStatus(status,__FILE__,__LINE__)

/** Check OpenCL status value, and print error message if not success.
    Use CLFFT_CHECK_STATUS(status) to call this function.

    @param status is the OpenCL value to check.
    @param filename,line are used in message.

    @return TRUE if status is CL_SUCCESS, and FALSE otherwise. */
bool checkStatus(cl_int status,const char * filename,int line);

} // namespace

#endif // #ifndef CLFFTErrors_h
