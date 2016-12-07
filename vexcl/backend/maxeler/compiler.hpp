#ifndef VEXCL_BACKEND_MAXELER_COMPILER_HPP
#define VEXCL_BACKEND_MAXELER_COMPILER_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/backend/maxeler/compiler.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Compilation of Maxeler kernels.
 */

#include <string>
#include <sstream>
#include <fstream>
#include <boost/dll/shared_library.hpp>

#include <vexcl/backend/common.hpp>
#include <vexcl/detail/backtrace.hpp>

namespace vex {
namespace backend {
namespace maxeler {

inline std::string dfe_compiler() {
    const char *cc = getenv("VEXCL_DFE_COMPILER");
    return cc ? cc : "vexcl_dfe_cc";
}

/// Compile and load a program from source string.
inline vex::backend::program build_sources(const command_queue &q,
        const std::tuple<std::string, std::string> &source,
        const std::string &options = ""
        )
{
    using std::get;

#ifndef VEXCL_SHOW_KERNELS
    if (getenv("VEXCL_SHOW_KERNELS"))
#endif
        std::cout
            << "// ----- MAXJ -----\n"
            << get<0>(source) << std::endl
            << "// ----- C++ -----\n"
            << get<1>(source) << std::endl
            ;

    static const std::string compiler = dfe_compiler();

    sha1_hasher sha1;
    sha1.process(get<0>(source))
        .process(get<1>(source))
        .process(compiler)
        ;

    std::string hash = static_cast<std::string>(sha1);

    // Write sources
    std::string root = program_binaries_path(hash, true);
    std::string lib_file = root + "vexcl_dfe_kernel.so";

    if (!boost::filesystem::exists(lib_file)) {
        boost::filesystem::create_directories(root + "src");

        {
            std::ofstream f(root + "src" + path_delim() + "vexcl_dfe_kernel.maxj");
            f << get<0>(source);
        }

        {
            std::ofstream f(root + "vexcl_dfe_kernel.cpp");
            f << get<1>(source);
        }

        // Compile the sources.
        std::ostringstream cmdline;

        cmdline << compiler << " -C " << root;

        if (0 != system(cmdline.str().c_str()) ) {
            vex::detail::print_backtrace();
            throw std::runtime_error("Kernel compilation failed");
        }
    }

    // Load the compiled shared library.
    return boost::dll::shared_library(lib_file);
}

} // namespace cuda
} // namespace backend
} // namespace vex

#endif
