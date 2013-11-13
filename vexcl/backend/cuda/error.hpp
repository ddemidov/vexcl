#ifndef VEXCL_BACKEND_CUDA_ERROR_HPP
#define VEXCL_BACKEND_CUDA_ERROR_HPP

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
 * \file   vexcl/backend/cuda/error.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Output CUDA errors to a std::stream.
 */

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <boost/config.hpp>

#ifdef BOOST_NO_NOEXCEPT
#  define noexcept throw()
#endif

#include <cuda.h>

namespace std {

inline std::ostream& operator<<(std::ostream &os, CUresult rc) {
#define CUDA_ERR2TXT(e) case e: return os << static_cast<int>(e) << " - " << #e
    switch(rc) {
        CUDA_ERR2TXT(CUDA_SUCCESS);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_VALUE);
        CUDA_ERR2TXT(CUDA_ERROR_OUT_OF_MEMORY);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_INITIALIZED);
        CUDA_ERR2TXT(CUDA_ERROR_DEINITIALIZED);
        CUDA_ERR2TXT(CUDA_ERROR_PROFILER_DISABLED);
        CUDA_ERR2TXT(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
        CUDA_ERR2TXT(CUDA_ERROR_PROFILER_ALREADY_STARTED);
        CUDA_ERR2TXT(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
        CUDA_ERR2TXT(CUDA_ERROR_NO_DEVICE);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_DEVICE);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_IMAGE);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_CONTEXT);
        CUDA_ERR2TXT(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
        CUDA_ERR2TXT(CUDA_ERROR_MAP_FAILED);
        CUDA_ERR2TXT(CUDA_ERROR_UNMAP_FAILED);
        CUDA_ERR2TXT(CUDA_ERROR_ARRAY_IS_MAPPED);
        CUDA_ERR2TXT(CUDA_ERROR_ALREADY_MAPPED);
        CUDA_ERR2TXT(CUDA_ERROR_NO_BINARY_FOR_GPU);
        CUDA_ERR2TXT(CUDA_ERROR_ALREADY_ACQUIRED);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_MAPPED);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
        CUDA_ERR2TXT(CUDA_ERROR_ECC_UNCORRECTABLE);
        CUDA_ERR2TXT(CUDA_ERROR_UNSUPPORTED_LIMIT);
        CUDA_ERR2TXT(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
        CUDA_ERR2TXT(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_SOURCE);
        CUDA_ERR2TXT(CUDA_ERROR_FILE_NOT_FOUND);
        CUDA_ERR2TXT(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
        CUDA_ERR2TXT(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
        CUDA_ERR2TXT(CUDA_ERROR_OPERATING_SYSTEM);
        CUDA_ERR2TXT(CUDA_ERROR_INVALID_HANDLE);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_FOUND);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_READY);
        CUDA_ERR2TXT(CUDA_ERROR_LAUNCH_FAILED);
        CUDA_ERR2TXT(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
        CUDA_ERR2TXT(CUDA_ERROR_LAUNCH_TIMEOUT);
        CUDA_ERR2TXT(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
        CUDA_ERR2TXT(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
        CUDA_ERR2TXT(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        CUDA_ERR2TXT(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
        CUDA_ERR2TXT(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        CUDA_ERR2TXT(CUDA_ERROR_ASSERT);
        CUDA_ERR2TXT(CUDA_ERROR_TOO_MANY_PEERS);
        CUDA_ERR2TXT(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
        CUDA_ERR2TXT(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_PERMITTED);
        CUDA_ERR2TXT(CUDA_ERROR_NOT_SUPPORTED);
        CUDA_ERR2TXT(CUDA_ERROR_UNKNOWN);
    }
#undef CUDA_ERR2TXT
    return os << "Unknown error";
}

} // namespace std

namespace vex {
namespace backend {

class error : public std::runtime_error {
    public:
        error(CUresult code) : std::runtime_error(get_msg(code)), code(code) { }

        CUresult err() const {
            return code;
        }
    private:
        static std::string get_msg(CUresult code) {
            std::ostringstream s;
            s << "CUDA error (" << code << ")";
            return s.str();
        }

        CUresult code;
};

inline void check(CUresult rc, const char *file, int line) {
    if (rc != CUDA_SUCCESS) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        throw error(rc);
    }
}

#define cuda_check(rc) vex::backend::check(rc, __FILE__, __LINE__)
} // namespace backend
} // namespace vex

namespace std {

inline std::ostream& operator<<(std::ostream &os, const vex::backend::error &e) {
    return os << e.what();
}

} // namespace std

#endif
