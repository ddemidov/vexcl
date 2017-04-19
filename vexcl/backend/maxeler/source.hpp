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
struct maxeler_dfe_type_impl;

template <class T, class Enable = void>
struct maxeler_cpu_type_impl;

#define VEXCL_DEFINE_MAXTYPE(ctype, dfetype, cputype)                          \
  template <> struct maxeler_dfe_type_impl<ctype> {                            \
    static std::string get() { return #dfetype; }                              \
  };                                                                           \
  template <> struct maxeler_cpu_type_impl<ctype> {                            \
    static std::string get() { return #cputype; }                              \
  }

VEXCL_DEFINE_MAXTYPE(float,        dfeFloat(8, 24),  CPUTypes.FLOAT);
VEXCL_DEFINE_MAXTYPE(double,       dfeFloat(11, 53), CPUTypes.DOUBLE);
VEXCL_DEFINE_MAXTYPE(int,          dfeInt(32),       CPUTypes.INT32);
VEXCL_DEFINE_MAXTYPE(unsigned int, dfeUInt(32),      CPUTypes.UINT32);
VEXCL_DEFINE_MAXTYPE(size_t,       dfeUInt(64),      CPUTypes.UINT64);

#undef VEXCL_DEFINE_MAXTYPE

template <class T>
inline std::string maxeler_dfe_type() {
    return maxeler_dfe_type_impl<typename std::decay<T>::type>::get();
}

template <class T>
inline std::string maxeler_cpu_type() {
    return maxeler_cpu_type_impl<typename std::decay<T>::type>::get();
}

} // namespace detail

class source_generator {
    private:
        unsigned indent_size;
        bool first_prm;

        enum {
            in_prm,
            out_prm,
            inout_prm
        } prm_kind;

        enum {
            undefined,
            inside_function,
            inside_kernel
        } prm_state;


        static const int kernel_freq = 180; 		// reasonable kernel freq on MAX4
        static const int mem_freq = 533; 			// Max LMEM freq on MAX4 without quarter rate mode

        static const int first_cost_table = 1;   	// which cost tables to build from 1..32 available
        static const int last_cost_table = 8;
        static const int cost_table_threads = 4; 	// how many cost tables to build in parallel
        static const int near_miss_threshold = 500; // retry the cost table if score is below this threshold

        std::ostringstream src, c_src;
        std::string kernel_name;
        std::ostringstream kernel_output, stream_design;
        std::ostringstream write_lmem_j, read_lmem_j, execute_j;
        std::ostringstream write_lmem_c, read_lmem_c, execute_c;
        int input_streams, output_streams;

    public:
        source_generator()
            : indent_size(0), first_prm(true), prm_kind(in_prm), prm_state(undefined),
              input_streams(0), output_streams(0)
        { }

        source_generator(const command_queue &q, bool include_standard_header = true)
            : indent_size(0), first_prm(true), prm_kind(in_prm), prm_state(undefined),
              input_streams(0), output_streams(0)
        {
            if (include_standard_header) src << standard_kernel_header(q);

            src <<
                "package vexcl_dfe_kernel;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;\n"
                "import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;\n"
                "import com.maxeler.maxcompiler.v2.build.EngineParameters;\n"
                "import com.maxeler.maxcompiler.v2.managers.custom.CustomManager;\n"
                "import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;\n"
                "import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemCommandGroup;\n"
                "import com.maxeler.maxcompiler.v2.managers.custom.stdlib.LMemInterface;\n"
                "import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;\n"
                "import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;\n"
                "import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface.Direction;\n"
                "import com.maxeler.maxcompiler.v2.managers.engine_interfaces.InterfaceParam;\n"
                "import com.maxeler.maxcompiler.v2.managers.BuildConfig;\n"
                ;

            c_src <<
                "#include <boost/config.hpp>\n"
                "#include <MaxSLiCInterface.h>\n"
                "#include \"vexcl_dfe_kernel.h\"\n\n"
                "struct kernel_api {\n"
                "    virtual void write_lmem(char*) const = 0;\n"
                "    virtual void read_lmem(char*) const = 0;\n"
                "    virtual void execute(char*) const = 0;\n"
                "};\n\n"
                "max_file_t* vexcl_dfe_max_file() {\n"
                "  static max_file_t *mf = vexcl_dfe_kernel_init();\n"
                "  return mf;\n"
                "}\n\n"
                "template <bool dummy = true>\n"
                "struct max_loader {\n"
                "  static max_engine_t* load(max_file_t* f) {\n"
                "    static max_file_t   *last_f = 0;\n"
                "    static max_engine_t *last_e = 0;\n"
                "    if (f != last_f) {\n"
                "      if (last_e) max_unload(last_e);\n"
                "      last_e = max_load(f, \"*\");\n"
                "      last_f = f;\n"
                "    }\n"
                "    return last_e;\n"
                "  }\n"
                "};\n\n"
                ;
        }

        std::string indent() const {
            return std::string(2 * indent_size, ' ');
        }

        std::string indent(int size) const {
            return std::string(2 * size, ' ');
        }

        source_generator& new_line() {
            src << "\n" << indent();
            return *this;
        }

        source_generator& open(const char *bracket) {
            new_line() << bracket;
            ++indent_size;
            return *this;
        }

        source_generator& close(const char *bracket) {
            assert(indent_size > 0);
            --indent_size;
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

            return *this;
        }

        source_generator& begin_kernel_parameters() {
            prm_state = inside_kernel;
            return *this;
        }

        source_generator& in_params() {
            prm_kind = in_prm;
            return *this;
        }

        source_generator& out_params() {
            prm_kind = out_prm;
            return *this;
        }

        source_generator& inout_params() {
            prm_kind = inout_prm;
            return *this;
        }

        source_generator& end_kernel_parameters() {
            prm_state = undefined;
            return *this;
        }

        source_generator& end_kernel() {
            src << kernel_output.str();
            close("}").close("}\n");

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

        std::tuple<std::string, std::string> sources() {
            new_line() << "class vexcl_manager extends CustomManager";
            open("{");
            new_line() << "vexcl_manager(EngineParameters ep)";
            open("{");
            new_line() << "super(ep);";

            new_line() << "config.setDefaultStreamClockFrequency(" << kernel_freq << ");";
            new_line() << "config.setOnCardMemoryFrequency(" << mem_freq <<");";

            new_line() << "KernelBlock k = addKernel(new " << kernel_name << "(makeKernelParameters(\"" << kernel_name << "\")));";
            new_line() << "LMemInterface lmem = addLMemInterface();"
                       << stream_design.str();
            close("}");

            new_line() << "private static EngineInterface write_lmem()";
            open("{");
            new_line() << "EngineInterface ei = new EngineInterface(\"write_lmem\");";
            new_line() << "InterfaceParam size = ei.addParam(\"N\", " << detail::maxeler_cpu_type<int>() << ");";
            new_line() << "InterfaceParam head = ei.addConstant(0l);"
                       << write_lmem_j.str();
            new_line() << "ei.ignoreAll(Direction.IN_OUT);";
            new_line() << "return ei;";
            close("}");
            new_line() << "private static EngineInterface read_lmem()";
            open("{");
            new_line() << "EngineInterface ei = new EngineInterface(\"read_lmem\");";
            new_line() << "InterfaceParam size = ei.addParam(\"N\", " << detail::maxeler_cpu_type<int>() << ");";
            new_line() << "InterfaceParam head = ei.addConstant(0l);"
                       << read_lmem_j.str();
            new_line() << "ei.ignoreAll(Direction.IN_OUT);";
            new_line() << "return ei;";
            close("}");
            new_line() << "private static EngineInterface execute()";
            open("{");
            new_line() << "EngineInterface ei = new EngineInterface(\"execute\");";
            new_line() << "InterfaceParam size = ei.addParam(\"N\", " << detail::maxeler_cpu_type<int>() << ");";
            new_line() << "InterfaceParam head = ei.addConstant(0l);";
            new_line() << "ei.setTicks(\"" << kernel_name << "\", size);"
                       << execute_j.str();
            new_line() << "ei.ignoreAll(Direction.IN_OUT);";
            new_line() << "return ei;";
            close("}");
            new_line() << "public static void main(String[] args)";
            open("{");
            new_line() << "vexcl_manager m = new vexcl_manager(new EngineParameters(args));";
            new_line() << "m.createSLiCinterface(write_lmem());";
            new_line() << "m.createSLiCinterface(read_lmem());";
            new_line() << "m.createSLiCinterface(execute());";
            new_line() << "BuildConfig buildConfig = m.getBuildConfig();";
            new_line() << "buildConfig.setMPPRCostTableSearchRange(" << first_cost_table << "," << last_cost_table << ");";
            new_line() << "buildConfig.setMPPRParallelism(" << cost_table_threads << ");";
            new_line() << "buildConfig.setMPPRRetryNearMissesThreshold(" << near_miss_threshold << ");";
            new_line() << "m.build();";
            close("}");
            close("}");

            // TODO: finish c_src
            c_src <<
                "struct " << kernel_name << "_t : public kernel_api {\n"
                "  void write_lmem(char*) const;\n"
                "  void read_lmem(char*) const;\n"
                "  void execute(char*) const;\n"
                "};\n\n"
                "extern \"C\" BOOST_SYMBOL_EXPORT " << kernel_name << "_t " << kernel_name << ";\n"
                << kernel_name << "_t " << kernel_name << ";\n\n";

            c_src << "void " << kernel_name << "_t::write_lmem(char *_p) const {\n";
            if (input_streams) {
                c_src <<
                    "  vexcl_dfe_kernel_write_lmem_actions_t kprm;"
                    << write_lmem_c.str() << "\n"
                    "  max_file_t *mf = vexcl_dfe_max_file();\n"
                    "  max_engine_t *me = max_loader<>::load(mf);\n"
                    "  vexcl_dfe_kernel_write_lmem_run(me, &kprm);\n";
            }
            c_src <<
                "}\n\n"
                "void " << kernel_name << "_t::read_lmem(char *_p) const {\n";
            if (output_streams) {
                c_src <<
                    "  vexcl_dfe_kernel_read_lmem_actions_t kprm;"
                    << read_lmem_c.str() << "\n"
                    "  max_file_t *mf = vexcl_dfe_max_file();\n"
                    "  max_engine_t *me = max_loader<>::load(mf);\n"
                    "  vexcl_dfe_kernel_read_lmem_run(me, &kprm);\n";
            }
            c_src <<
                "}\n\n"
                "void " << kernel_name << "_t::execute(char *_p) const {\n"
                "  vexcl_dfe_kernel_execute_actions_t kprm;"
                << execute_c.str() << "\n"
                "  max_file_t *mf = vexcl_dfe_max_file();\n"
                "  max_engine_t *me = max_loader<>::load(mf);\n"
                "  vexcl_dfe_kernel_execute_run(me, &kprm);\n"
                "}\n";

            return std::make_tuple(src.str(), c_src.str());
        }

        std::string str() const {
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

        template <class Prm, class Enable = void>
        friend struct kernel_parameter_impl;

        template <class Prm, class Enable = void>
        struct kernel_parameter_impl {};

        template <class T>
        struct kernel_parameter_impl< global_ptr<T> > {
            static void apply(source_generator &s, const std::string &name) {
                switch(s.prm_kind) {
                    case source_generator::in_prm:
                        s.input_streams++;

                        s.new_line() << "DFEVar " << name << " = io.input(\""
                            << name << "\", " << detail::maxeler_dfe_type<T>() << ");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "k.getInput(\"" << name << "\")"
                            " <== "
                            "lmem.addStreamFromLMem(\"" << name << "\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);";

                        s.stream_design << "\n" << s.indent(2) <<
                            "lmem.addStreamToLMem(\"" << name << "_to_lmem\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)"
                            " <== "
                            "addStreamFromCPU(\"" << name << "_from_cpu\");";

                        s.write_lmem_j << "\n" << s.indent(2) <<
                            "ei.setStream(\"" << name << "_from_cpu\", " << detail::maxeler_cpu_type<T>() << ", size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_to_lmem\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.read_lmem_j << "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.execute_j << "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.write_lmem_c << "\n" << s.indent(1) <<
                            "kprm.instream_" << name << "_from_cpu = " << "*reinterpret_cast<" << type_name<typename std::decay<T>::type*>() << "*>(_p);"
                            "\n" << s.indent(1) << "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.read_lmem_c << "\n" << s.indent(1) <<
                            "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.execute_c << "\n" << s.indent(1) <<
                            "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";

                        break;
                    case source_generator::out_prm:
                        s.output_streams++;

                        s.new_line() << "DFEVar " << name << ";";

                        s.kernel_output << "\n" << s.indent()
                            << "io.output(\"" << name << "\", " << name << ", "
                            << detail::maxeler_dfe_type<T>()
                            <<");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "lmem.addStreamToLMem(\"" << name << "\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)"
                            " <== "
                            "k.getOutput(\"" << name << "\");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "addStreamToCPU(\"" << name << "_to_cpu\")"
                            " <== "
                            "lmem.addStreamFromLMem(\"" << name << "_from_lmem\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);";

                        s.write_lmem_j << "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.read_lmem_j << "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_from_lmem\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "ei.setStream(\"" << name << "_to_cpu\", " << detail::maxeler_cpu_type<T>() << ", size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.execute_j << "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.write_lmem_c << "\n" << s.indent(1) <<
                            "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.read_lmem_c << "\n" << s.indent(1) <<
                            "kprm.outstream_" << name << "_to_cpu = " << "*reinterpret_cast<" << type_name<typename std::decay<T>::type*>() << "*>(_p);"
                            "\n" << s.indent(1) << "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.execute_c << "\n" << s.indent(1) <<
                            "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";

                        break;
                    case source_generator::inout_prm:
                        s.input_streams++;
                        s.output_streams++;

                        s.new_line() << "DFEVar " << name << " = io.input(\""
                            << name << "_in\", " << detail::maxeler_dfe_type<T>() << ");";

                        s.kernel_output << "\n" << s.indent()
                            << "io.output(\"" << name << "_out\", " << name << ", "
                            << detail::maxeler_dfe_type<T>()
                            <<");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "k.getInput(\"" << name << "_in\")"
                            " <== "
                            "lmem.addStreamFromLMem(\"" << name << "_in\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);";

                        s.stream_design << "\n" << s.indent(2) <<
                            "lmem.addStreamToLMem(\"" << name << "_out\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)"
                            " <== "
                            "k.getOutput(\"" << name << "_out\");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "lmem.addStreamToLMem(\"" << name << "_to_lmem\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D)"
                            " <== "
                            "addStreamFromCPU(\"" << name << "_from_cpu\");";

                        s.stream_design << "\n" << s.indent(2) <<
                            "addStreamToCPU(\"" << name << "_to_cpu\")"
                            " <== "
                            "lmem.addStreamFromLMem(\"" << name << "_from_lmem\", LMemCommandGroup.MemoryAccessPattern.LINEAR_1D);";

                        s.write_lmem_j << "\n" << s.indent(2) <<
                            "ei.setStream(\"" << name << "_from_cpu\", " << detail::maxeler_cpu_type<T>() << ", size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_to_lmem\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.read_lmem_j << "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_from_lmem\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "ei.setStream(\"" << name << "_to_cpu\", " << detail::maxeler_cpu_type<T>() << ", size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.execute_j << "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_in\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "ei.setLMemLinear(\"" << name << "_out\", head, size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes());"
                            "\n" << s.indent(2) <<
                            "head = head + size * " << detail::maxeler_cpu_type<T>() << ".sizeInBytes();";

                        s.write_lmem_c << "\n" << s.indent(1) <<
                            "kprm.instream_" << name << "_from_cpu = " << "*reinterpret_cast<" << type_name<typename std::decay<T>::type*>() << "*>(_p);"
                            "\n" << s.indent(1) << "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.read_lmem_c << "\n" << s.indent(1) <<
                            "kprm.outstream_" << name << "_to_cpu = " << "*reinterpret_cast<" << type_name<typename std::decay<T>::type*>() << "*>(_p);"
                            "\n" << s.indent(1) << "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";
                        s.execute_c << "\n" << s.indent(1) <<
                            "_p+= sizeof(" << type_name<typename std::decay<T>::type>() << "*);";

                        break;
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
                // "n" is special:
                if (name == "n") {
                    s.write_lmem_c <<
                        "\n" << s.indent(1) << "kprm.param_N = " << "*reinterpret_cast<" << type_name<T>() << "*>(_p);"
                        "\n" << s.indent(1) << "_p+= sizeof(" << type_name<T>() << ");";
                    s.read_lmem_c <<
                        "\n" << s.indent(1) << "kprm.param_N = " << "*reinterpret_cast<" << type_name<T>() << "*>(_p);"
                        "\n" << s.indent(1) << "_p+= sizeof(" << type_name<T>() << ");";
                    s.execute_c <<
                        "\n" << s.indent(1) << "kprm.param_N = " << "*reinterpret_cast<" << type_name<T>() << "*>(_p);"
                        "\n" << s.indent(1) << "_p+= sizeof(" << type_name<T>() << ");";
                    return;
                }

                precondition(s.prm_kind == source_generator::in_prm, "Scalar output is not supported");

                s.new_line() << "DFEVar " << name << " = io.scalarInput(\""
                    << name << "\", " << detail::maxeler_dfe_type<T>() << ");";

                s.execute_j << "\n" << s.indent(2) <<
                    "InterfaceParam " << name << " = ei.addParam(\"" << name << "\", " << detail::maxeler_cpu_type<T>() << ");"
                    "\n" << s.indent(2) <<
                    "ei.setScalar(\"" << s.kernel_name << "\", \"" << name << "\", " << name << ");";

                s.write_lmem_c << "\n" << s.indent(1) <<
                    "_p+= sizeof(" << type_name<T>() << ");";
                s.read_lmem_c << "\n" << s.indent(1) <<
                    "_p+= sizeof(" << type_name<T>() << ");";
                s.execute_c << "\n" << s.indent(1) <<
                    "kprm.param_" << name << " = " << "*reinterpret_cast<" << type_name<T>() << "*>(_p);"
                    "\n" << s.indent(1) << "_p+= sizeof(" << type_name<T>() << ");";
            }
        };

        template <class Prm>
        source_generator& kernel_parameter(const std::string &name) {
            kernel_parameter_impl<Prm>::apply(*this, name);
            return *this;
        }
};


} // namespace maxeler
} // namespace backend
} // namespace vex

#endif
