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
#include <type_traits>
#include <cassert>

#include <vexcl/util.hpp>
#include <vexcl/backend/common.hpp>
#include <vexcl/types.hpp>

namespace vex {

template <class T> struct global_ptr {};
template <class T> struct shared_ptr {};
template <class T> struct regstr_ptr {};
template <class T> struct constant_ptr {};

template <class T>
struct type_name_impl <global_ptr<T> > {
    static std::string get() {
        std::ostringstream s;
        s << type_name<T>() << " *";
        return s.str();
    }
};

template <class T>
struct type_name_impl < global_ptr<const T> > {
    static std::string get() {
        std::ostringstream s;
        s << "const " << type_name<T>() << " *";
        return s.str();
    }
};

template <class T>
struct type_name_impl <shared_ptr<T> >
  : type_name_impl<global_ptr<T>>
{};

template <class T>
struct type_name_impl <regstr_ptr<T> >
  : type_name_impl<global_ptr<T>>
{};

template <class T>
struct type_name_impl <constant_ptr<T> >
  : type_name_impl<global_ptr<const typename std::decay<T>::type> >
{};

template<typename T>
struct type_name_impl<T*>
  : type_name_impl<global_ptr<T>>
{};

namespace backend {
namespace maxeler {

inline std::string standard_kernel_header(const command_queue &q) {
    return get_program_header(q);
}

namespace detail {

template <class T, class Enable = void>
struct maxeler_type_impl;

#define VEXCL_DEFINE_MAXTYPE(ctype, maxtype)                                   \
  template <> struct maxeler_type_impl<ctype> {                                \
    static std::string get() { return #maxtype; }                              \
  }

VEXCL_DEFINE_MAXTYPE(float,        dfeFloat(8, 24));
VEXCL_DEFINE_MAXTYPE(double,       dfeFloat(11, 53));
VEXCL_DEFINE_MAXTYPE(int,          dfeInt(32));
VEXCL_DEFINE_MAXTYPE(unsigned int, dfeUInt(32));
VEXCL_DEFINE_MAXTYPE(size_t,       dfeUInt(64));

#undef VEXCL_DEFINE_MAXTYPE

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

        std::ostringstream src, c_src;
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
                "package vexcl_dfe_kernel;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;\n"
                "import com.maxeler.maxcompiler.v2.managers.standard.Manager;\n"
                "import com.maxeler.maxcompiler.v2.managers.standard.Manager.IOType;\n"
                "import com.maxeler.maxcompiler.v2.build.EngineParameters;\n"
                ;

            c_src <<
                "#include <boost/config.hpp>\n"
                "#include <MaxSLiCInterface.h>\n"
                "#include \"vexcl_dfe_kernel.h\"\n\n"
                "struct kernel_api {\n"
                "    virtual void execute(char*) const = 0;\n"
                "};\n\n"
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
            return close("}").close("}\n");
        }

        source_generator& begin_kernel(const std::string &name) {
            kernel_name = name;

            new_line() << "class " << name << " extends Kernel";
            open("{");
            new_line() << name << "(KernelParameters parameters)";
            open("{");
            new_line() << "super(parameters);";

            c_src <<
                "struct " << name << "_t : public kernel_api {\n"
                "  void execute(char*) const;\n"
                "};\n\n"
                "extern \"C\" BOOST_SYMBOL_EXPORT " << name << "_t " << name << ";\n"
                << name << "_t " << name << ";\n"
                "void " << name << "_t::execute(char *_p) const {";

            return *this;
        }

        source_generator& begin_kernel_parameters() {
            prm_state = inside_kernel;
            c_src << "\n  vexcl_dfe_kernel_actions_t kprm;";
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
            c_src << "\n  static max_file_t *mf = vexcl_dfe_kernel_init();";
            c_src << "\n  max_engine_t *me = max_load(mf, \"*\");";
            c_src << "\n  vexcl_dfe_kernel_run(me, &kprm);";
            c_src << "\n  max_unload(me);";

            prm_state = undefined;
            return *this;
        }

        source_generator& end_kernel() {
            src << output_section.str();
            close("}").close("}\n");

            c_src << "\n}\n";

            return *this;
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
            return "control.count.simpleCounter(64)";
        }

        std::string global_size(int d) const {
            return "n";
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

        std::tuple<std::string, std::string> str() {
            new_line() << "class vexcl_manager";
            open("{");
            new_line() << "public static void main(String[] args)";
            open("{");
            new_line() << "EngineParameters params = new EngineParameters(args);";
            new_line() << "Manager manager = new Manager(params);";

            new_line() << "Kernel k = new "
                << kernel_name << "(manager.makeKernelParameters(\""
                << kernel_name << "\"));";

            new_line() << "manager.setKernel(k);";
            new_line() << "manager.setIO(IOType.ALL_CPU);";
            new_line() << "manager.createSLiCinterface();";
            new_line() << "manager.build();";
            close("}");
            close("}");

            return std::make_tuple(src.str(), c_src.str());
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

        template <class Prm, class Enable = void>
        friend struct kernel_parameter_impl;

        template <class Prm, class Enable = void>
        struct kernel_parameter_impl {};

        template <class T>
        struct kernel_parameter_impl< global_ptr<T> > {
            static void apply(source_generator &s, const std::string &name) {
                if (s.input_prm) {
                    s.new_line() << "DFEVar " << name << " = io.input(\""
                        << name << "\", " << detail::maxeler_type<T>() << ");";
                } else {
                    s.new_line() << "DFEVar " << name << ";";
                    s.output_section << "\n" << std::string(2 * s.indent, ' ')
                        << "io.output(\"" << name << "\", " << name << ", "
                        << detail::maxeler_type<T>()
                        <<");";
                }
            }
        };

        template <class T>
        struct kernel_parameter_impl< T* >
            : kernel_parameter_impl< global_ptr<T> >
        {};

        template <class T>
        struct kernel_parameter_impl<T,
            typename std::enable_if< std::is_arithmetic<T>::value >::type
            >
        {
            static void apply(source_generator &s, const std::string &name) {
                // "n" is special; kernels get their size automatically.
                if (name == "n") return;

                precondition(s.input_prm, "Scalar output is not supported");
                s.new_line() << "DFEVar " << name << " = io.scalarInput(\""
                    << name << "\", " << detail::maxeler_type<T>() << ");";
            }
        };

        template <class Prm>
        source_generator& kernel_parameter(const std::string &name) {
            kernel_parameter_impl<Prm>::apply(*this, name);

            if (name == "n") {
                c_src << "\n  kprm.param_N";
            } else if (input_prm) {
                if (std::is_arithmetic<Prm>::value) {
                    c_src << "\n  kprm.inscalar_" << kernel_name << "_" << name;
                } else {
                    c_src << "\n  kprm.instream_" << name;
                }
            } else {
                c_src << "\n  kprm.outstream_" << name;
            }

            c_src << " = *reinterpret_cast<" << type_name<Prm>() << "*>(_p);"
                  << " _p+= sizeof(" << type_name<Prm>() << ");";

            return *this;
        }
};


} // namespace maxeler
} // namespace backend
} // namespace vex

#endif
