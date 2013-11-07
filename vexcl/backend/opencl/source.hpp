#ifndef VEXCL_BACKEND_OPENCL_SOURCE_HPP
#define VEXCL_BACKEND_OPENCL_SOURCE_HPP

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
 * \file   vexcl/backend/opencl/source.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Helper class for OpenCL source code generation.
 */

#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

namespace vex {
namespace backend {

enum device_options_kind {
    compile_options,
    program_header
};

/// Global program options holder
template <device_options_kind kind>
struct device_options {
    static const std::string& get(const cl::Device &dev) {
        if (options[dev()].empty()) options[dev()].push_back("");

        return options[dev()].back();
    }

    static void push(const cl::Device &dev, const std::string &str) {
        options[dev()].push_back(str);
    }

    static void pop(const cl::Device &dev) {
        if (!options[dev()].empty()) options[dev()].pop_back();
    }

    private:
        static std::map<cl_device_id, std::vector<std::string> > options;
};

template <device_options_kind kind>
std::map<cl_device_id, std::vector<std::string> > device_options<kind>::options;

inline std::string get_compile_options(const cl::Device &dev) {
    return device_options<compile_options>::get(dev);
}

inline std::string get_program_header(const cl::Device &dev) {
    return device_options<program_header>::get(dev);
}

/// Set global OpenCL compilation options for a given device.
/**
 * This replaces any previously set options. To roll back, call
 * pop_compile_options().
 */
inline void push_compile_options(const cl::Device &dev, const std::string &str) {
    device_options<compile_options>::push(dev, str);
}

/// Rolls back changes to compile options.
inline void pop_compile_options(const cl::Device &dev) {
    device_options<compile_options>::pop(dev);
}

/// Set global OpenCL program header for a given device.
/**
 * This replaces any previously set header. To roll back, call
 * pop_program_header().
 */
inline void push_program_header(const cl::Device &dev, const std::string &str) {
    device_options<program_header>::push(dev, str);
}

/// Rolls back changes to compile options.
inline void pop_program_header(const cl::Device &dev) {
    device_options<program_header>::pop(dev);
}

/// Set global OpenCL compilation options for each device in queue list.
inline void push_compile_options(const std::vector<cl::CommandQueue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::push(qdev(*q), str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_compile_options(const std::vector<cl::CommandQueue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::pop(qdev(*q));
}

/// Set global OpenCL program header for each device in queue list.
inline void push_program_header(const std::vector<cl::CommandQueue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::push(qdev(*q), str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_program_header(const std::vector<cl::CommandQueue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::pop(qdev(*q));
}

/// Returns standard OpenCL program header.
/**
 * Defines pragmas necessary to work with double precision and anything
 * provided by the user with help of push_program_header().
 */
inline std::string standard_kernel_header(const cl::Device &dev) {
    return std::string(
        "#if defined(cl_khr_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
        "#elif defined(cl_amd_fp64)\n"
        "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
        "#endif\n"
        ) + get_program_header(dev);
}

class source_generator {
    private:
        unsigned           indent;
        bool               first_prm, cpu;
        std::ostringstream src;

    public:
        source_generator(const cl::CommandQueue &queue)
            : indent(0), first_prm(true), cpu( is_cpu(qdev(queue)) )
        {
            src << standard_kernel_header(qdev(queue));
        }

        source_generator& new_line() {
            src << "\n" << std::string(2 * indent, ' ');
            return *this;
        }

        source_generator& open(const char *bracket) {
            new_line() << bracket;
            ++indent;
            return *this;
        }

        source_generator& close(const char *bracket) {
            assert(indent > 0);
            --indent;
            new_line() << bracket;
            return *this;
        }

        template <class Return>
        source_generator& function(const std::string &name) {
            first_prm = true;
            new_line() << type_name<Return>() << " " << name;
            return *this;
        }

        source_generator& kernel(const std::string &name) {
            first_prm = true;
            new_line() << "kernel void " << name;
            return *this;
        }

        template <class Prm>
        source_generator& parameter(const std::string &name) {
            if (first_prm)
                first_prm = false;
            else
                src << ",";

            new_line() << type_name<Prm>() << " " << name;
            return *this;
        }

        source_generator& grid_stride_loop(
                const std::string &idx = "idx", const std::string &bnd = "n"
                )
        {
            if ( cpu ) {
                new_line() << "size_t chunk_size  = (" << bnd
                           << " + get_global_size(0) - 1) / get_global_size(0);";
                new_line() << "size_t chunk_start = get_global_id(0) * chunk_size;";
                new_line() << "size_t chunk_end   = min(" << bnd
                           << ", chunk_start + chunk_size);";
                new_line() << "for(size_t " << idx << " = chunk_start; "
                           << idx << " < chunk_end; ++" << idx << ")";
            } else {
                new_line() <<
                    "for(size_t " << idx << " = get_global_id(0);"
                    " " << idx << " < " << bnd << ";"
                    " " << idx << " += get_global_size(0))";
            }
            return *this;
        }

        std::string str() const {
            return src.str();
        }
    private:
        template <class T>
        friend inline
        source_generator& operator<<(source_generator &src, T &&t) {
            src.src << t;
            return src;
        }
};

} // namespace backend
} // namespace vex

#endif
