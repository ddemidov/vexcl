#ifndef VEXCL_BACKEND_CUDA_COMPILER_NVRTC_HPP
#define VEXCL_BACKEND_CUDA_COMPILER_NVRTC_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/backend/cuda/compiler_nvrtc.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  CUDA source code compilation wrapper.
 */

#include <cstdlib>
#include <cuda.h>
#include <nvrtc.h>

#include <vexcl/backend/common.hpp>
#include <vexcl/detail/backtrace.hpp>
#include <vexcl/backend/cuda/error.hpp>

namespace std {

/// Send human-readable representation of nvrtcResult to the output stream.
inline std::ostream& operator<<(std::ostream &os, nvrtcResult rc) {
    os << "NVRTC Error (";
#define VEXCL_NVRTC_ERR2TXT(e) case e: os << static_cast<int>(e) << " - " << #e; break
    switch(rc) {
        VEXCL_NVRTC_ERR2TXT(NVRTC_SUCCESS);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_OUT_OF_MEMORY);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_PROGRAM_CREATION_FAILURE);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_INVALID_INPUT);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_INVALID_PROGRAM);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_INVALID_OPTION);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_COMPILATION);
        VEXCL_NVRTC_ERR2TXT(NVRTC_ERROR_BUILTIN_OPERATION_FAILURE);
        default:
            os << "Unknown error";
    }
#undef VEXCL_NVRTC_ERR2TXT
    return os << ")";
}

} // namespace std

namespace vex {
namespace backend {

namespace cuda {

inline void check(nvrtcResult rc, const char *file, int line) {
    if (rc != NVRTC_SUCCESS) {
        vex::detail::print_backtrace();
	throw error(rc, file, line);
    }
}

/// Create and build a program from source string.
inline CUmodule build_sources(
        const command_queue &queue, const std::string &source,
        const std::string &options = ""
        )
{
#ifdef VEXCL_SHOW_KERNELS
    std::cout << source << std::endl;
#else
#  ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable: 4996)
#  endif
    if (getenv("VEXCL_SHOW_KERNELS"))
        std::cout << source << std::endl;
#  ifdef _MSC_VER
#    pragma warning(pop)
#  endif
#endif

    queue.context().set_current();

    nvrtcProgram prog;
    cuda_check( nvrtcCreateProgram(&prog, source.c_str(), NULL, 0, NULL, NULL) );

    try {
        cuda_check( nvrtcCompileProgram(prog, 0, NULL) );
    } catch (...) {
        size_t log_size;
        cuda_check( nvrtcGetProgramLogSize(prog, &log_size) );
        std::vector<char> log(log_size);
        cuda_check( nvrtcGetProgramLog(prog, log.data()) );

        std::cerr << log.data() << std::endl;
        throw;
    }

    size_t ptx_size;
    cuda_check( nvrtcGetPTXSize(prog, &ptx_size) );
    std::vector<char> ptx(ptx_size);
    cuda_check( nvrtcGetPTX(prog, ptx.data()) );
    cuda_check( nvrtcDestroyProgram(&prog) );

    // Load the compiled ptx.
    CUmodule module;
    cuda_check( cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0) );

    return module;
}

} // namespace cuda
} // namespace backend
} // namespace vex

#endif
