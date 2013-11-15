#ifndef VEXCL_BACKEND_OPENCL_ERROR_HPP
#define VEXCL_BACKEND_OPENCL_ERROR_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl/backend/opencl/error.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Output OpenCL errors to a std::stream.
 */

#include <iostream>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.hpp>

namespace vex {
namespace backend {

typedef cl::Error error;

}
}

namespace std {

/// Sends description of an OpenCL error to the output stream.
inline std::ostream& operator<<(std::ostream &os, const cl::Error &e) {
    os << e.what() << "(";

#define CL_ERR2TXT(num, msg) case (num): os << (msg); break

    switch (e.err()) {
        CL_ERR2TXT(  0, "Success");
        CL_ERR2TXT( -1, "Device not found");
        CL_ERR2TXT( -2, "Device not available");
        CL_ERR2TXT( -3, "Compiler not available");
        CL_ERR2TXT( -4, "Mem object allocation failure");
        CL_ERR2TXT( -5, "Out of resources");
        CL_ERR2TXT( -6, "Out of host memory");
        CL_ERR2TXT( -7, "Profiling info not available");
        CL_ERR2TXT( -8, "Mem copy overlap");
        CL_ERR2TXT( -9, "Image format mismatch");
        CL_ERR2TXT(-10, "Image format not supported");
        CL_ERR2TXT(-11, "Build program failure");
        CL_ERR2TXT(-12, "Map failure");
        CL_ERR2TXT(-13, "Misaligned sub buffer offset");
        CL_ERR2TXT(-14, "Exec status error for events in wait list");
        CL_ERR2TXT(-30, "Invalid value");
        CL_ERR2TXT(-31, "Invalid device type");
        CL_ERR2TXT(-32, "Invalid platform");
        CL_ERR2TXT(-33, "Invalid device");
        CL_ERR2TXT(-34, "Invalid context");
        CL_ERR2TXT(-35, "Invalid queue properties");
        CL_ERR2TXT(-36, "Invalid command queue");
        CL_ERR2TXT(-37, "Invalid host ptr");
        CL_ERR2TXT(-38, "Invalid mem object");
        CL_ERR2TXT(-39, "Invalid image format descriptor");
        CL_ERR2TXT(-40, "Invalid image size");
        CL_ERR2TXT(-41, "Invalid sampler");
        CL_ERR2TXT(-42, "Invalid binary");
        CL_ERR2TXT(-43, "Invalid build options");
        CL_ERR2TXT(-44, "Invalid program");
        CL_ERR2TXT(-45, "Invalid program executable");
        CL_ERR2TXT(-46, "Invalid kernel name");
        CL_ERR2TXT(-47, "Invalid kernel definition");
        CL_ERR2TXT(-48, "Invalid kernel");
        CL_ERR2TXT(-49, "Invalid arg index");
        CL_ERR2TXT(-50, "Invalid arg value");
        CL_ERR2TXT(-51, "Invalid arg size");
        CL_ERR2TXT(-52, "Invalid kernel args");
        CL_ERR2TXT(-53, "Invalid work dimension");
        CL_ERR2TXT(-54, "Invalid work group size");
        CL_ERR2TXT(-55, "Invalid work item size");
        CL_ERR2TXT(-56, "Invalid global offset");
        CL_ERR2TXT(-57, "Invalid event wait list");
        CL_ERR2TXT(-58, "Invalid event");
        CL_ERR2TXT(-59, "Invalid operation");
        CL_ERR2TXT(-60, "Invalid gl object");
        CL_ERR2TXT(-61, "Invalid buffer size");
        CL_ERR2TXT(-62, "Invalid mip level");
        CL_ERR2TXT(-63, "Invalid global work size");
        CL_ERR2TXT(-64, "Invalid property");

        default:
            os << "Unknown error";
            break;
    }

#undef CL_ERR2TXT

    return os << ")";
}

}

#endif
