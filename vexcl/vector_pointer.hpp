#ifndef VEXCL_VECTOR_POINTER_HPP
#define VEXCL_VECTOR_POINTER_HPP

#include <vexcl/vector.hpp>
#include <vexcl/operations.hpp>

namespace vex {

template <typename T>
struct vector_pointer {
    const vector<T> &v;

    vector_pointer(const vector<T> &v) : v(v) {}
};

template <typename T>
#ifdef DOXYGEN
vector_pointer<T>
#else
inline typename boost::proto::result_of::as_expr<vector_pointer<T>, vector_domain>::type
#endif
raw_pointer(const vector<T> &v) {
    precondition(
            v.nparts() == 1,
            "raw_pointer is not supported for multi-device contexts"
            );

    return boost::proto::as_expr<vector_domain>(vector_pointer<T>(v));
}

namespace traits {

template <typename T>
struct is_vector_expr_terminal< vector_pointer<T> > : std::true_type {};

template <typename T>
struct kernel_param_declaration< vector_pointer<T> >
{
    static std::string get(const vector_pointer<T>&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        std::ostringstream s;
        s << ",\n\tglobal " << type_name<T>() << " * " << prm_name;
        return s.str();
    }
};

template <typename T>
struct partial_vector_expr< vector_pointer<T> >
{
    static std::string get(const vector_pointer<T>&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        return prm_name ;
    }
};

template <typename T>
struct kernel_arg_setter< vector_pointer<T> >
{
    static void set(const vector_pointer<T> &term,
            cl::Kernel &kernel, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &position, detail::kernel_generator_state_ptr)
    {
        kernel.setArg(position++, term.v(0));
    }
};

template <typename T>
struct expression_properties< vector_pointer<T> >
{
    static void get(const vector_pointer<T> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.v.queue_list();
        partition  = term.v.partition();
        size       = term.v.size();
    }
};

} // namespace traits
}

#endif
