#ifndef VEXCL_BACKEND_OPENCL_COMPILER_HPP
#define VEXCL_BACKEND_OPENCL_COMPILER_HPP

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
 * \file   vexcl/backend/opencl/compiler.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL source code compilation wrapper.
 */

#ifdef VEXCL_CACHE_KERNELS
#  include <fstream>
#  include <sstream>
#  include <iomanip>
#  include <cstdlib>
#  include <boost/uuid/sha1.hpp>
#  include <boost/optional.hpp>
#  include <boost/filesystem.hpp>
#endif

namespace vex {
namespace backend {

#ifdef VEXCL_CACHE_KERNELS
/// Path delimiter symbol.
inline const std::string& path_delim() {
    static const std::string delim = boost::filesystem::path("/").make_preferred().string();
    return delim;
}

/// Path to appdata folder.
inline const std::string& appdata_path() {
#ifdef WIN32
#  ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable: 4996)
#  endif
    static const std::string appdata = getenv("APPDATA") + path_delim() + "vexcl";
#  ifdef _MSC_VER
#    pragma warning(pop)
#  endif
#else
    static const std::string appdata = getenv("HOME") + path_delim() + ".vexcl";
#endif
    return appdata;
}

/// Path to cached binaries.
inline std::string program_binaries_path(const std::string &hash, bool create = false)
{
    std::string dir = appdata_path()    + path_delim()
                    + hash.substr(0, 2) + path_delim()
                    + hash.substr(2);
    if (create) boost::filesystem::create_directories(dir);
    return dir + path_delim();
}

/// Saves program binaries for future reuse.
inline void save_program_binaries(
        const std::string &hash, const cl::Program &program, const std::string &source
        )
{
    std::ofstream bfile(program_binaries_path(hash, true) + "kernel", std::ios::binary);
    if (!bfile) return;

    std::vector<size_t> sizes    = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    std::vector<char*>  binaries = program.getInfo<CL_PROGRAM_BINARIES>();

    assert(sizes.size() == 1);

    bfile.write((char*)&sizes[0], sizeof(size_t));
    bfile.write(binaries[0], sizes[0]);
    delete[] binaries[0];

    bfile << "\n" << source << "\n";
}

/// Tries to read program binaries from file cache.
inline boost::optional<cl::Program> load_program_binaries(
        const std::string &hash, const cl::Context &context,
        const std::vector<cl::Device> &device
        )
{
    std::ifstream bfile(program_binaries_path(hash) + "kernel", std::ios::binary);
    if (!bfile) return boost::optional<cl::Program>();

    size_t n;
    std::vector<char> buf;

    bfile.read((char*)&n, sizeof(size_t));
    buf.resize(n);
    bfile.read(buf.data(), n);

    cl::Program program(context, device, cl::Program::Binaries(
                1, std::make_pair(static_cast<const void*>(buf.data()), n)));

    try {
        program.build(device, "");
    } catch(const cl::Error&) {
        std::cerr << "Loading binaries failed:" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
            << std::endl;
        return boost::optional<cl::Program>();
    }

    return boost::optional<cl::Program>(program);
}

/// Returns SHA1 hash of the string parameter.
inline std::string sha1(const std::string &src) {
    boost::uuids::detail::sha1 sha1;
    sha1.process_bytes(src.c_str(), src.size());

    unsigned int hash[5];
    sha1.get_digest(hash);

    std::ostringstream buf;
    for(int i = 0; i < 5; ++i)
        buf << std::hex << std::setfill('0') << std::setw(8) << hash[i];

    return buf.str();
}
#endif

/// Create and build a program from source string.
/**
 * If VEXCL_CACHE_KERNELS macro is defined, then program binaries are cached
 * in filesystem and reused in the following runs.
 */
inline cl::Program build_sources(
        const cl::Context &context, const std::string &source,
        const std::string &options = ""
        )
{
#ifdef VEXCL_SHOW_KERNELS
    std::cout << source << std::endl;
#endif

    auto device = context.getInfo<CL_CONTEXT_DEVICES>();
    std::string compile_options = options + " " + backend::get_compile_options(device[0]);

#ifdef VEXCL_CACHE_KERNELS
    // Get unique (hopefully) hash string for the kernel.
    std::ostringstream hashsrc;

    hashsrc
        << "\n" << cl::Platform(device[0].getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>()
        << "\n" << device[0].getInfo<CL_DEVICE_NAME>()
        << "\noptions: " << compile_options
        << "\n" << source;

    std::string hash = sha1( hashsrc.str() );

    // Try to get cached program binaries:
    if (boost::optional<cl::Program> program = load_program_binaries(hash, context, device))
        return *program;
#endif

    // If cache is not available, just compile the sources.
    cl::Program program(context, cl::Program::Sources(
                1, std::make_pair(source.c_str(), source.size())
                ));

    try {
        program.build(device, (options + " " + get_compile_options(device[0])).c_str());
    } catch(const cl::Error&) {
        std::cerr << source
                  << std::endl
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
                  << std::endl;
        throw;
    }

#ifdef VEXCL_CACHE_KERNELS
    // Save program binaries for future reuse:
    save_program_binaries(hash, program, hashsrc.str());
#endif

    return program;
}

} // namespace backend
} // namespace vex

#endif
