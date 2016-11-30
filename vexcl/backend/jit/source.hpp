#ifndef VEXCL_BACKEND_JIT_SOURCE_HPP
#define VEXCL_BACKEND_JIT_SOURCE_HPP

#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cassert>

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
namespace jit {

inline std::string standard_kernel_header(const command_queue &q) {
    return std::string(R"(
#include <boost/config.hpp>
#include <cmath>
struct ndrange {
    size_t x,y,z;
};
struct grid_info {
    ndrange grid_dim;
    ndrange block_dim;
};
struct thread_info {
    ndrange block_id;
    ndrange thread_id;
};
struct kernel_api {
    virtual void execute(const grid_info *_g, const thread_info *_t, char *_p) const = 0;
};
#define KERNEL_PARAMETER(type, name) \
    type name = *reinterpret_cast<type*>(_p); _p+= sizeof(type)
)") + get_program_header(q);
}

class source_generator {
    private:
        unsigned indent;
        bool first_prm;

        enum {
            undefined,
            inside_function,
            inside_kernel
        } prm_state;

        std::ostringstream src;

    public:
        source_generator()
            : indent(0), first_prm(true), prm_state(undefined) {}

        source_generator(const command_queue &q, bool include_standard_header = true)
            : indent(0), first_prm(true), prm_state(undefined)
        {
            if (include_standard_header) src << standard_kernel_header(q);
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
            first_prm = true;
            new_line() << return_type << " " << name;
            return *this;
        }

        template <class Return>
        source_generator& begin_function(const std::string &name) {
            return function(type_name<Return>(), name);
        }

        source_generator& begin_function_parameters() {
            first_prm = true;
            return open("(");
        }

        source_generator& end_function_parameters() {
            prm_state = inside_function;
            return close(")").open("{");
        }

        source_generator& end_function() {
            prm_state = undefined;
            return close("}");
        }

        source_generator& begin_kernel(const std::string &name) {
            new_line() << "struct " << name << "_t : public kernel_api {";
            new_line() << "  void execute(const grid_info*, const thread_info*, char*) const;";
            new_line() << "};";
            new_line() << "extern \"C\" BOOST_SYMBOL_EXPORT " << name << "_t " << name << ";";
            new_line() << name << "_t " << name << ";";
            new_line() << "void " << name << "_t::execute(const grid_info *_g, const thread_info *_t, char *_p) const";
            open("{");
            return *this;
        }

        source_generator& begin_kernel_parameters() {
            prm_state = inside_kernel;
            return *this;
        }

        source_generator& end_kernel_parameters() {
            prm_state = undefined;
            return *this;
        }

        source_generator& end_kernel() {
            return close("}");
        }

        source_generator& parameter(const std::string &prm_type, const std::string &name) {
            switch(prm_state) {
                case inside_kernel:
                    return kernel_parameter(prm_type, name);
                case inside_function:
                    return function_parameter(prm_type, name);
                default:
                    throw std::logic_error("parameter definition outside of parameter block");
            }
        }

        template <class Prm>
        source_generator& parameter(const std::string &name) {
            return parameter(type_name<typename std::decay<Prm>::type>(), name);
        }

        source_generator& grid_stride_loop(
                const std::string &idx = "idx", const std::string &bnd = "n"
                )
        {
            new_line() << "size_t global_size = _g->grid_dim.x * _g->block_dim.x;";
            new_line() << "size_t global_id   = _t->block_id.x * _g->block_dim.x + _t->thread_id.x;";
            new_line() << "size_t chunk_size = (n + global_size - 1) / global_size;";
            new_line() << "size_t chunk_start = chunk_size * global_id;";
            new_line() << "size_t chunk_end = chunk_start + chunk_size;";
            new_line() << "if (n < chunk_end) chunk_end = n;";
            new_line() << "for(size_t idx = chunk_start; idx < chunk_end; ++idx)";

            return *this;
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

        source_generator& function_parameter(const std::string &prm_type, const std::string &name) {
            prm_separator().new_line() << prm_type << " " << name;
            return *this;
        }

        source_generator& kernel_parameter(const std::string &prm_type, const std::string &name) {
            new_line() << "KERNEL_PARAMETER(" << prm_type << ", " << name << ");";
            return *this;
        }
};

} // namespace jit
} // namespace backend
} // namespace vex

#endif
