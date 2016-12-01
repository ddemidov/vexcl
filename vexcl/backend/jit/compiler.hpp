#ifndef VEXCL_BACKEND_JIT_COMPILER_HPP
#define VEXCL_BACKEND_JIT_COMPILER_HPP

#include <string>
#include <sstream>
#include <fstream>
#include <boost/dll/shared_library.hpp>

#include <vexcl/backend/common.hpp>
#include <vexcl/detail/backtrace.hpp>

#ifndef VEXCL_JIT_COMPILER
#  define VEXCL_JIT_COMPILER "g++"
#endif

#ifndef VEXCL_JIT_COMPILER_OPTIONS
#  ifdef NDEBUG
#    define VEXCL_JIT_COMPILER_OPTIONS "-O3 -fPIC -shared -fopenmp"
#  else
#    define VEXCL_JIT_COMPILER_OPTIONS "-g -fPIC -shared -fopenmp"
#  endif
#endif

namespace vex {
namespace backend {
namespace jit {

inline vex::backend::program build_sources(const command_queue &q,
        const std::string &source, const std::string &options = ""
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

    std::string compile_options = options + " " + get_compile_options(q);

    sha1_hasher sha1;
    sha1.process(source).process(compile_options);

    std::string hash = static_cast<std::string>(sha1);

    // Write source to a .cpp file
    std::string basename = program_binaries_path(hash, true) + "kernel";
    std::string sofile   = basename + ".so";

    if ( !boost::filesystem::exists(sofile) ) {
        std::string cppfile = basename + ".cpp";

        {
            std::ofstream f(cppfile);
            f << source;
        }

        // Compile the source.
        std::ostringstream cmdline;
        cmdline << VEXCL_JIT_COMPILER << " -o " << sofile << " " << cppfile << " "
                << VEXCL_JIT_COMPILER_OPTIONS << " " << compile_options;

        if (0 != system(cmdline.str().c_str()) ) {
#ifndef VEXCL_SHOW_KERNELS
            std::cerr << source << std::endl;
#endif

            vex::detail::print_backtrace();
            throw std::runtime_error("Kernel compilation failed");
        }
    }

    // Load the compiled shared library.
    return boost::dll::shared_library(sofile);
}

} // namespace cuda
} // namespace backend
} // namespace vex

#endif
