#ifndef VEXCL_BACKEND_HIP_ERROR_HPP
#define VEXCL_BACKEND_HIP_ERROR_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/backend/hip/error.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Output HIP errors to a std::stream.
 */

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <boost/config.hpp>

#ifdef BOOST_NO_NOEXCEPT
#  define noexcept throw()
#endif

#include <hip/hip_runtime.h>

#include <vexcl/detail/backtrace.hpp>

namespace std {

/// Send human-readable representation of hipError_t to the output stream.
inline std::ostream& operator<<(std::ostream &os, hipError_t rc) {
    os << "HIP Driver API Error (";
#define VEXCL_HIP_ERR2TXT(e) case e: os << static_cast<int>(e) << " - " << #e; break
    switch(rc) {
        VEXCL_HIP_ERR2TXT(hipSuccess);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidValue);
        VEXCL_HIP_ERR2TXT(hipErrorMemoryAllocation);
        VEXCL_HIP_ERR2TXT(hipErrorNotInitialized);
        VEXCL_HIP_ERR2TXT(hipErrorDeinitialized);
        VEXCL_HIP_ERR2TXT(hipErrorProfilerDisabled);
        VEXCL_HIP_ERR2TXT(hipErrorProfilerNotInitialized);
        VEXCL_HIP_ERR2TXT(hipErrorProfilerAlreadyStarted);
        VEXCL_HIP_ERR2TXT(hipErrorProfilerAlreadyStopped);
        VEXCL_HIP_ERR2TXT(hipErrorNoDevice);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidDevice);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidImage);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidContext);
        VEXCL_HIP_ERR2TXT(hipErrorContextAlreadyCurrent);
        VEXCL_HIP_ERR2TXT(hipErrorMapFailed);
        VEXCL_HIP_ERR2TXT(hipErrorUnmapFailed);
        VEXCL_HIP_ERR2TXT(hipErrorArrayIsMapped);
        VEXCL_HIP_ERR2TXT(hipErrorAlreadyMapped);
        VEXCL_HIP_ERR2TXT(hipErrorNoBinaryForGpu);
        VEXCL_HIP_ERR2TXT(hipErrorAlreadyAcquired);
        VEXCL_HIP_ERR2TXT(hipErrorNotMapped);
        VEXCL_HIP_ERR2TXT(hipErrorNotMappedAsArray);
        VEXCL_HIP_ERR2TXT(hipErrorNotMappedAsPointer);
        VEXCL_HIP_ERR2TXT(hipErrorECCNotCorrectable);
        VEXCL_HIP_ERR2TXT(hipErrorUnsupportedLimit);
        VEXCL_HIP_ERR2TXT(hipErrorContextAlreadyInUse);
        VEXCL_HIP_ERR2TXT(hipErrorPeerAccessUnsupported);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidSource);
        VEXCL_HIP_ERR2TXT(hipErrorFileNotFound);
        VEXCL_HIP_ERR2TXT(hipErrorSharedObjectSymbolNotFound);
        VEXCL_HIP_ERR2TXT(hipErrorSharedObjectInitFailed);
        VEXCL_HIP_ERR2TXT(hipErrorOperatingSystem);
        VEXCL_HIP_ERR2TXT(hipErrorInvalidResourceHandle);
        VEXCL_HIP_ERR2TXT(hipErrorNotFound);
        VEXCL_HIP_ERR2TXT(hipErrorNotReady);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_LAUNCH_FAILED);
        VEXCL_HIP_ERR2TXT(hipErrorLaunchOutOfResources);
        VEXCL_HIP_ERR2TXT(hipErrorLaunchTimeOut);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
        VEXCL_HIP_ERR2TXT(hipErrorPeerAccessAlreadyEnabled);
        VEXCL_HIP_ERR2TXT(hipErrorPeerAccessNotEnabled);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_PRIMARY_CONTEXT_ACTIVE);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_CONTEXT_IS_DESTROYED);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_ASSERT);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_TOO_MANY_PEERS);
        VEXCL_HIP_ERR2TXT(hipErrorHostMemoryAlreadyRegistered);
        VEXCL_HIP_ERR2TXT(hipErrorHostMemoryNotRegistered);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_NOT_PERMITTED);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_NOT_SUPPORTED);
	//        VEXCL_HIP_ERR2TXT(HIP_ERROR_UNKNOWN);
        default:
            os << "Unknown error " << static_cast<int>(rc);
    }
#undef VEXCL_HIP_ERR2TXT
    return os << ")";
}

} // namespace std

namespace vex {
namespace backend {
namespace hip {

/// HIP error class to be thrown as exception.
class error : public std::runtime_error {
    public:
        template <class ErrorCode>
        error(ErrorCode code, const char *file, int line)
            : std::runtime_error(get_msg(code, file, line))
        { }
    private:
        template <class ErrorCode>
        static std::string get_msg(ErrorCode code, const char *file, int line) {
            std::ostringstream s;
            s << file << ":" << line << "\n\t" << code;
            return s.str();
        }
};

inline void check(hipError_t rc, const char *file, int line) {
    if (rc != hipSuccess) {
        vex::detail::print_backtrace();
	throw error(rc, file, line);
    }
}

/// Throws if rc is not hipSuccess.
/**
 * Reports offending file and line number on standard error stream.
 */
#define hip_check(rc) vex::backend::check(rc, __FILE__, __LINE__)

} // namespace hip
} // namespace backend
} // namespace vex

namespace std {

/// Sends description of a HIP error to the output stream.
inline std::ostream& operator<<(std::ostream &os, const vex::backend::error &e) {
    return os << e.what();
}

} // namespace std

#endif
