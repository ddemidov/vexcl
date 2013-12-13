#ifndef VEXCL_BACKEND_CUDA_COMPILER_HPP
#define VEXCL_BACKEND_CUDA_COMPILER_HPP

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
 * \file   vexcl/backend/cuda/compiler.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  CUDA source code compilation wrapper.
 */

#include <cstdlib>
#include <cuda.h>

#include <vexcl/backend/common.hpp>

namespace vex {
namespace backend {
namespace cuda {

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

    auto cc = queue.device().compute_capability();
    std::ostringstream fullsrc;
    fullsrc << "// Device:  " << queue.device().name() << "\n"
            << "// CC:      " << std::get<0>(cc) << "." << std::get<1>(cc) << "\n"
            << "// options: " << options << "\n"
            << source;

    // Write source to a .cu file
    std::string hash = sha1( fullsrc.str() );
    std::string basename = program_binaries_path(hash, true) + "kernel";
    std::string ptxfile  = basename + ".ptx";

    if ( !boost::filesystem::exists(ptxfile) ) {
        std::string cufile = basename + ".cu";


        {
            std::ofstream f(basename + ".cu");
            f << fullsrc.str();
        }

        // Compile the source to ptx.
        std::ostringstream cmdline;
        auto cc = queue.device().compute_capability();
        cmdline
            << "nvcc -ptx -O3"
            << " -arch=sm_" << std::get<0>(cc) << std::get<1>(cc)
            << " " << options
            << " -o " << ptxfile << " " << cufile;
        if (0 != system(cmdline.str().c_str()) ) {
#ifndef VEXCL_SHOW_KERNELS
            std::cerr << fullsrc.str() << std::endl;
#endif
            throw std::runtime_error("nvcc invocation failed");
        }
    }

    // Load the compiled ptx.
    CUmodule program;
    cuda_check( cuModuleLoad(&program, ptxfile.c_str()) );

    return program;
}

} // namespace cuda
} // namespace backend
} // namespace vex

#endif
