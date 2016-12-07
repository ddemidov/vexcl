#ifndef VEXCL_BACKEND_MAXELER_SOURCE_HPP
#define VEXCL_BACKEND_MAXELER_SOURCE_HPP

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
 * \file   vexcl/backend/maxeler/source.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Source code generation for the Maxeler backend.
 */

#include <map>
#include <string>
#include <list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cassert>

#include <vexcl/backend/common.hpp>
#include <vexcl/types.hpp>

namespace vex {
namespace backend {
namespace maxeler {

inline std::string standard_kernel_header(const command_queue &q) {
    return get_program_header(q);
}

namespace detail {

template <class T, class Enable = void>
struct maxeler_type_impl;

template <>
struct maxeler_type_impl<float> {
    static std::string get() {
        return "dfeFloat(8, 24)";
    }
};

template <>
struct maxeler_type_impl<double> {
    static std::string get() {
        return "dfeFloat(11, 53)";
    }
};

template <class T>
inline std::string maxeler_type() {
    return maxeler_type_impl<typename std::decay<T>::type>::get();
}

} // namespace detail

class source_generator {
    private:
        unsigned indent;
        bool first_prm;
        bool input_prm;

        enum {
            undefined,
            inside_function,
            inside_kernel
        } prm_state;

        std::ostringstream src;
        std::string kernel_name;
        std::ostringstream output_section;

    public:
        source_generator() : indent(0), first_prm(true), input_prm(true), prm_state(undefined)
        { }

        source_generator(const command_queue &q, bool include_standard_header = true)
            : indent(0), first_prm(true), input_prm(true), prm_state(undefined)
        {
            if (include_standard_header) src << standard_kernel_header(q);

            src <<
                "package vexcl;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;\n"
                "import com.maxeler.maxcompiler.v2.managers.standard.Manager;\n"
                "import com.maxeler.maxcompiler.v2.managers.standard.Manager.IOType;\n"
                "import com.maxeler.maxcompiler.v2.build.EngineParameters;\n"
                ;
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

        source_generator& begin_function(const std::string &return_type, const std::string &name) {
            new_line() << "class " << name;
            open("{");
            new_line() << "public static " << return_type << " apply";
            return *this;
        }

        template <class Return>
        source_generator& begin_function(const std::string &name) {
            return begin_function("DFEVar", name);
        }

        source_generator& begin_function_parameters() {
            prm_state = inside_function;
            first_prm = true;
            return open("(");
        }

        source_generator& end_function_parameters() {
            prm_state = undefined;
            return close(")").open("{");
        }

        source_generator& end_function() {
            return close("}").close("}");
        }

        source_generator& begin_kernel(const std::string &name) {
            kernel_name = name;

            new_line() << "class " << name << "_kernel extends Kernel";
            open("{");
            new_line() << "name" << "_kernel(KernelParameters parameters)";
            open("{");
            new_line() << "super(parameters);";

            return *this;
        }

        source_generator& begin_kernel_parameters() {
            prm_state = inside_kernel;
            return *this;
        }

        source_generator& input_parameters() {
            input_prm = true;
            return *this;
        }

        source_generator& output_parameters() {
            input_prm = false;
            return *this;
        }

        source_generator& end_kernel_parameters() {
            prm_state = undefined;
            return *this;
        }

        source_generator& end_kernel() {
            src << output_section.str();

            return close("}").close("}");
        }

        template <class Prm>
        source_generator& parameter(const std::string &name) {
            switch(prm_state) {
                case inside_kernel:
                    return kernel_parameter<Prm>(name);
                case inside_function:
                    return function_parameter(name);
                default:
                    throw std::logic_error("parameter definition outside of parameter block");
            }
        }

        template <class Prm>
        source_generator& smem_parameter(const std::string& = "smem") {
            return *this;
        }

        template <class Prm>
        source_generator& smem_declaration(const std::string &name = "smem") {
            return *this;
        }

        source_generator& grid_stride_loop(
                const std::string &idx = "idx", const std::string &bnd = "n"
                )
        {
            return *this;
        }

        std::string global_id(int d) const {
            return "";
        }

        std::string global_size(int d) const {
            return "";
        }

        std::string local_id(int d) const {
            return "";
        }

        std::string local_size(int d) const {
            return "";
        }

        std::string group_id(int d) const {
            return "";
        }

        source_generator& barrier(bool /*global*/ = false) {
            return *this;
        }

        source_generator& smem_static_var(const std::string &type, const std::string &name) {
            return *this;
        }

        std::string str() {
            new_line() << "class vexcl_manager";
            open("{");
            new_line() << "public static main(String[] args);";
            open("{");
            new_line() << "EngineParameters params = new EngineParameters(args);";
            new_line() << "Manager manager = new Manager(params);";

            new_line() << "Kernel " << kernel_name << " = new "
                << kernel_name << "_kernel(manager.makeKernelParameters(\""
                << kernel_name << "\"));";

            new_line() << "manager.setKernel(" << kernel_name << ");";
            new_line() << "manager.setIO(IOType.ALL_CPU);";
            new_line() << "manager.createSLiCinterface();";
            new_line() << "manager.build();";
            close("}");
            close("}");

            return src.str();
        }

    private:
        template <class T>
        friend inline
        source_generator& operator<<(source_generator &src, const T &t) {
            src.src << t;
            return src;
        }

        source_generator& prm_separator() {
            if (first_prm)
                first_prm = false;
            else
                src << ",";

            return *this;
        }

        source_generator& function_parameter(const std::string &name) {
            prm_separator().new_line() << "DFEVar " << name;
            return *this;
        }

        template <class Prm>
        source_generator& kernel_parameter(const std::string &name) {
            if (input_prm) {
                new_line() << "DFEVar " << name << " = io.input(\""
                    << name << "\", "
                    << detail::maxeler_type<typename std::remove_pointer<Prm>::type>()
                    << ");";
            } else {
                new_line() << "DFEVar " << name << ";";
                output_section << "\n" << std::string(2 * indent, ' ')
                    << "io.output(\"" << name << "\", " << name << ", "
                    << detail::maxeler_type<typename std::remove_pointer<Prm>::type>()
                    <<");";
            }
            return *this;
        }
};


} // namespace maxeler
} // namespace backend
} // namespace vex

#endif
