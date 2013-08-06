#ifndef VEXCL_REDUCTOR_HPP
#define VEXCL_REDUCTOR_HPP

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
 * \file   vexcl/reductor.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Vector expression reduction.
 */

#ifdef _MSC_VER
#  define NOMINMAX
#endif

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <numeric>
#include <limits>

#include <vexcl/operations.hpp>

namespace vex {

/// Summation. Should be used as a template parameter for Reductor class.
struct SUM {
    template <typename T>
    static T initial() {
        return T();
    };

    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return prm1 + prm2;"; }
    };

    template <class Iterator>
    static typename std::iterator_traits<Iterator>::value_type
    reduce(Iterator begin, Iterator end) {
        return std::accumulate(begin, end,
                initial<typename std::iterator_traits<Iterator>::value_type>()
                );
    }
};

/// Maximum element. Should be used as a template parameter for Reductor class.
struct MAX {
    template <typename T>
    static T initial() {
        // Strictly speaking, this should fail for unsigned types.
        // But negating maximum possible unsigned value gives 0 on
        // 2s complement systems, so...
        return -std::numeric_limits<T>::max();
    };

    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return max(prm1, prm2);"; }
    };

    template <class Iterator>
    static typename std::iterator_traits<Iterator>::value_type
    reduce(Iterator begin, Iterator end) {
        return *std::max_element(begin, end);
    }
};

/// Minimum element. Should be used as a template parameter for Reductor class.
struct MIN {
    template <typename T>
    static T initial() {
        return std::numeric_limits<T>::max();
    };

    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return min(prm1, prm2);"; }
    };

    template <class Iterator>
    static typename std::iterator_traits<Iterator>::value_type
    reduce(Iterator begin, Iterator end) {
        return *std::min_element(begin, end);
    }
};

/// Parallel reduction of arbitrary expression.
/**
 * Reduction uses small temporary buffer on each device present in the queue
 * parameter. One Reductor class for each reduction kind is enough per thread
 * of execution.
 */
template <typename real, class RDC>
class Reductor {
    public:
        /// Constructor.
        Reductor(const std::vector<cl::CommandQueue> &queue
#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
                = current_context().queue()
#endif
                );

        /// Compute reduction of a vector expression.
        template <class Expr>
#ifdef DOXYGEN
        real
#else
        typename std::enable_if<
            boost::proto::matches<Expr, vector_expr_grammar>::value,
            real
        >::type
#endif
        operator()(const Expr &expr) const;

        /// Compute reduction of a multivector expression.
        template <class Expr>
#ifdef DOXYGEN
        std::array<real, N>
#else
        typename std::enable_if<
            boost::proto::matches<Expr, multivector_expr_grammar>::value,
            std::array<real, std::result_of<traits::multiex_dimension(Expr)>::type::value>
        >::type
#endif
        operator()(const Expr &expr) const;
    private:
        const std::vector<cl::CommandQueue> &queue;
        std::vector<size_t> idx;
        std::vector<cl::Buffer> dbuf;

        mutable std::vector<real> hbuf;
        mutable std::vector<cl::Event> event;

        template <size_t I, size_t N, class Expr>
        typename std::enable_if<I == N, void>::type
        assign_subexpressions(std::array<real, N> &, const Expr &) const
        { }

        template <size_t I, size_t N, class Expr>
        typename std::enable_if<I < N, void>::type
        assign_subexpressions(std::array<real, N> &result, const Expr &expr) const
        {
            result[I] = (*this)(detail::extract_subexpression<I>()(expr));

            assign_subexpressions<I + 1, N, Expr>(result, expr);
        }
};

#ifndef DOXYGEN
template <typename real, class RDC>
Reductor<real,RDC>::Reductor(const std::vector<cl::CommandQueue> &queue)
    : queue(queue), event(queue.size())
{
    idx.reserve(queue.size() + 1);
    idx.push_back(0);

    for(auto q = queue.begin(); q != queue.end(); q++) {
        cl::Context context = qctx(*q);
        cl::Device  device  = qdev(*q);

        size_t bufsize = num_workgroups(device);
        idx.push_back(idx.back() + bufsize);

        dbuf.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, bufsize * sizeof(real)));
    }

    hbuf.resize(idx.back());
}

template <typename real, class RDC> template <class Expr>
typename std::enable_if<
    boost::proto::matches<Expr, vector_expr_grammar>::value,
    real
>::type
Reductor<real,RDC>::operator()(const Expr &expr) const {
    using namespace detail;

    static kernel_cache cache;

    get_expression_properties prop;
    extract_terminals()(expr, prop);

    for(unsigned d = 0; d < queue.size(); ++d) {
        cl::Context context = qctx(queue[d]);
        cl::Device  device  = qdev(queue[d]);

        auto kernel = cache.find( context() );

        if (kernel == cache.end()) {
            std::ostringstream increment_line;

            output_local_preamble loc_init(increment_line, device);
            boost::proto::eval(expr, loc_init);

            vector_expr_context expr_ctx(increment_line, device);

            increment_line << "\t\tmySum = reduce_operation(mySum, ";
            boost::proto::eval(expr, expr_ctx);
            increment_line << ");\n";

            std::ostringstream source;
            source << standard_kernel_header(device);

            typedef typename RDC::template function<real> fun;
            fun::define(source, "reduce_operation");

            output_terminal_preamble termpream(source, device);
            boost::proto::eval(boost::proto::as_child(expr),  termpream);

            source << "kernel void vexcl_reductor_kernel(\n\t"
                << type_name<size_t>() << " n";

            extract_terminals()( expr, declare_expression_parameter(source, device) );

            source << ",\n\tglobal " << type_name<real>() << " *g_odata,\n"
                "\tlocal  " << type_name<real>() << " *sdata\n"
                "\t)\n"
                "{\n";
            if ( is_cpu(device) ) {
                source <<
                    "    size_t grid_size  = get_global_size(0);\n"
                    "    size_t chunk_size = (n + grid_size - 1) / grid_size;\n"
                    "    size_t chunk_id   = get_global_id(0);\n"
                    "    size_t start      = min(n, chunk_size * chunk_id);\n"
                    "    size_t stop       = min(n, chunk_size * (chunk_id + 1));\n"
                    "    " << type_name<real>() << " mySum = " << RDC::template initial<real>() << ";\n"
                    "    for (size_t idx = start; idx < stop; idx++) {\n"
                    << increment_line.str() <<
                    "    }\n"
                    "\n"
                    "    g_odata[get_group_id(0)] = mySum;\n"
                    "}\n";
            } else {
                source <<
                    "    size_t tid        = get_local_id(0);\n"
                    "    size_t block_size = get_local_size(0);\n"
                    "    size_t p          = get_group_id(0) * block_size * 2 + tid;\n"
                    "    size_t gridSize   = get_global_size(0) * 2;\n"
                    "    size_t idx;\n"
                    "    " << type_name<real>() << " mySum = " << RDC::template initial<real>() << ";\n"
                    "    while (p < n) {\n"
                    "        idx = p;\n"
                    "        " << increment_line.str() <<
                    "        idx = p + block_size;\n"
                    "        if (idx < n)\n"
                    "            " << increment_line.str() <<
                    "        p += gridSize;\n"
                    "    }\n"
                    "    sdata[tid] = mySum;\n"
                    "\n"
                    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                    "    if (block_size >= 1024) { if (tid < 512) { sdata[tid] = mySum = reduce_operation(mySum, sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
                    "    if (block_size >=  512) { if (tid < 256) { sdata[tid] = mySum = reduce_operation(mySum, sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
                    "    if (block_size >=  256) { if (tid < 128) { sdata[tid] = mySum = reduce_operation(mySum, sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
                    "    if (block_size >=  128) { if (tid <  64) { sdata[tid] = mySum = reduce_operation(mySum, sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
                    "\n"
                    "    if (tid < 32) {\n"
                    "        local volatile " << type_name<real>() << "* smem = sdata;\n"
                    "        if (block_size >=  64) { smem[tid] = mySum = reduce_operation(mySum, smem[tid + 32]); }\n"
                    "        if (block_size >=  32) { smem[tid] = mySum = reduce_operation(mySum, smem[tid + 16]); }\n"
                    "        if (block_size >=  16) { smem[tid] = mySum = reduce_operation(mySum, smem[tid +  8]); }\n"
                    "        if (block_size >=   8) { smem[tid] = mySum = reduce_operation(mySum, smem[tid +  4]); }\n"
                    "        if (block_size >=   4) { smem[tid] = mySum = reduce_operation(mySum, smem[tid +  2]); }\n"
                    "        if (block_size >=   2) { smem[tid] = mySum = reduce_operation(mySum, smem[tid +  1]); }\n"
                    "    }\n"
                    "    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];\n"
                    "}\n";
            }

            auto program = build_sources(context, source.str());

            cl::Kernel krn(program, "vexcl_reductor_kernel");
            size_t wgs;
            if (is_cpu(device)) {
                wgs = 1;
            } else {
                wgs = kernel_workgroup_size(krn, device);

                size_t smem = static_cast<size_t>(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>())
                            - static_cast<size_t>(krn.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device));
                while(wgs * sizeof(real) > smem)
                    wgs /= 2;
            }

            kernel = cache.insert(std::make_pair(
                        context(), kernel_cache_entry(krn, wgs)
                        )).first;
        }

        if (size_t psize = prop.part_size(d)) {
            size_t w_size = kernel->second.wgsize;
            size_t g_size = (idx[d + 1] - idx[d]) * w_size;
            auto   lmem   = vex::Local(w_size * sizeof(real));

            unsigned pos = 0;
            kernel->second.kernel.setArg(pos++, psize);

            extract_terminals()(
                    expr,
                    set_expression_argument(kernel->second.kernel, d, pos, prop.part_start(d))
                    );

            kernel->second.kernel.setArg(pos++, dbuf[d]);
            kernel->second.kernel.setArg(pos++, lmem);

            queue[d].enqueueNDRangeKernel(kernel->second.kernel,
                    cl::NullRange, g_size, w_size);
        }
    }

    std::fill(hbuf.begin(), hbuf.end(), RDC::template initial<real>());

    for(unsigned d = 0; d < queue.size(); d++) {
        if (prop.part_size(d))
            queue[d].enqueueReadBuffer(dbuf[d], CL_FALSE,
                    0, sizeof(real) * (idx[d + 1] - idx[d]), &hbuf[idx[d]], 0, &event[d]);
    }

    for(unsigned d = 0; d < queue.size(); d++)
        if (prop.part_size(d)) event[d].wait();

    return RDC::reduce(hbuf.begin(), hbuf.end());
}

template <typename real, class RDC> template <class Expr>
typename std::enable_if<
    boost::proto::matches<Expr, multivector_expr_grammar>::value,
    std::array<real, std::result_of<traits::multiex_dimension(Expr)>::type::value>
>::type
Reductor<real,RDC>::operator()(const Expr &expr) const {
    const size_t dim = std::result_of<traits::multiex_dimension(Expr)>::type::value;
    std::array<real, dim> result;

    assign_subexpressions<0, dim, Expr>(result, expr);

    return result;
}
#endif

/// Returns a reference to a static instance of vex::Reductor<T,R>
template <typename T, class R>
const vex::Reductor<T, R>& get_reductor(const std::vector<cl::CommandQueue> &queue)
{
    // We will hold one static reductor per set of queues (or, rather, contexts):
    static std::map< std::vector<cl_context>, vex::Reductor<T, R> > cache;

    // Extract OpenCL context handles from command queues:
    std::vector<cl_context> ctx;
    ctx.reserve(queue.size());
    for(auto q = queue.begin(); q != queue.end(); ++q)
        ctx.push_back( vex::qctx(*q)() );

    // See if there is suitable instance of reductor already:
    auto r = cache.find(ctx);

    // If not, create new instance and move it to the cache.
    if (r == cache.end())
        r = cache.insert( std::make_pair(
                    std::move(ctx), vex::Reductor<T, R>(queue)
                    ) ).first;

    return r->second;
}

} // namespace vex

#endif
