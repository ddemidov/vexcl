#ifndef VEXCL_SPMAT_INLINE_SPMV_HPP
#define VEXCL_SPMAT_INLINE_SPMV_HPP

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
 * \file   vexcl/spmat/inline_spmv.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Inline SpMV operation.
 */

#include <vexcl/spmat.hpp>

namespace vex {

/// \cond INTERNAL
struct inline_spmv_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< inline_spmv_terminal >::type
    > inline_spmv_terminal_expression;

template <typename val_t, typename col_t, typename idx_t>
struct inline_spmv : inline_spmv_terminal_expression {
    typedef spmv<val_t, col_t, idx_t> Base;
    const typename Base::mat &A;
    const typename Base::vec &x;

    inline_spmv(const Base &base) : A(base.A), x(base.x) {}
};
/// \endcond

template <typename val_t, typename col_t, typename idx_t>
inline_spmv<val_t, col_t, idx_t> make_inline(const spmv<val_t, col_t, idx_t> &base) {
    precondition(base.x.nparts() == 1, "Can not inline multi-device SpMV operation.");

    return inline_spmv<val_t, col_t, idx_t>(base);
}

/// \cond INTERNAL
// Allow inline products to participate in vector expressions:
template <>
struct is_vector_expr_terminal< inline_spmv_terminal >
    : std::true_type
{ };

template <typename val_t, typename col_t, typename idx_t>
struct kernel_name< inline_spmv<val_t, col_t, idx_t> > {
    static std::string get() {
        return "spmv_";
    }
};

template <typename val_t, typename col_t, typename idx_t>
struct partial_vector_expr< inline_spmv<val_t, col_t, idx_t> > {
    static std::string get(const cl::Device &device, int component, int position, kernel_generator_state &state) {
        return SpMat<val_t, col_t, idx_t>::inline_expression(device, component, position, state);
    }
};

template <typename val_t, typename col_t, typename idx_t>
struct terminal_preamble< inline_spmv<val_t, col_t, idx_t> > {
    static std::string get(const cl::Device &device, int component, int position, kernel_generator_state &state) {
        return SpMat<val_t, col_t, idx_t>::inline_preamble(device, component, position, state);
    }
};

template <typename val_t, typename col_t, typename idx_t>
struct kernel_param_declaration< inline_spmv<val_t, col_t, idx_t> > {
    static std::string get(const cl::Device &device, int component, int position, kernel_generator_state &state) {
        return SpMat<val_t, col_t, idx_t>::inline_parameters(device, component, position, state);
    }
};

template <typename val_t, typename col_t, typename idx_t>
struct kernel_arg_setter< inline_spmv<val_t, col_t, idx_t> > {
    static void set(cl::Kernel &kernel, uint device, size_t index_offset,
            uint &position, const inline_spmv<val_t, col_t, idx_t> &term, kernel_generator_state &state)
    {
        SpMat<val_t, col_t, idx_t>::inline_arguments(
                kernel, device, index_offset, position,
                term.A, term.x, state
                );
    }
};

template <typename val_t, typename col_t, typename idx_t>
struct expression_properties< inline_spmv<val_t, col_t, idx_t> > {
    static void get(const inline_spmv<val_t, col_t, idx_t> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        expression_properties< vector<val_t> >::get(term.x, queue_list, partition, size);
    }
};

/// \endcond

} // namespace vex

#endif
