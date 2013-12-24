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
    /* In order to define a reduction kind for vex::Reductor, one should:
     *
     * 1. Define initial value (e.g. 0 for sums, 1 for products, plus-minus
     *    infinity for extrema):
     */
    template <typename T>
    static T initial() {
        return T();
    }

    /*
     * 2. Provide an OpenCL function that will be used on compute device to do
     *    incremental reductions. That is nested struct "function":
     */
    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return prm1 + prm2;"; }
    };

    /*
     * 3. Provide a host-side function that will be used for final reduction of
     *    small result vector on host:
     */
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
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4146)
#endif
        if (std::is_unsigned<T>::value)
            return static_cast<T>(0);
        else
            return -std::numeric_limits<T>::max();
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
    }

    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return prm1 > prm2 ? prm1 : prm2;"; }
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
    }

    template <typename T>
    struct function : UserFunction<function<T>, T(T, T)> {
        static std::string body() { return "return prm1 < prm2 ? prm1 : prm2;"; }
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
        Reductor(const std::vector<backend::command_queue> &queue
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
            boost::proto::matches<Expr, multivector_expr_grammar>::value &&
            !boost::proto::matches<Expr, vector_expr_grammar>::value,
            std::array<real, std::result_of<traits::multiex_dimension(Expr)>::type::value>
        >::type
#endif
        operator()(const Expr &expr) const;
    private:
        const std::vector<backend::command_queue> &queue;
        std::vector<size_t> idx;
        std::vector< backend::device_vector<real> > dbuf;

        mutable std::vector<real> hbuf;

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
Reductor<real,RDC>::Reductor(const std::vector<backend::command_queue> &queue)
    : queue(queue)
{
    idx.reserve(queue.size() + 1);
    idx.push_back(0);

    for(auto q = queue.begin(); q != queue.end(); q++) {
        size_t bufsize = backend::kernel::num_workgroups(*q);
        idx.push_back(idx.back() + bufsize);

        dbuf.push_back(backend::device_vector<real>(*q, bufsize));
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

    // If expression is of zero size, then there is nothing to do. Hurray!
    if (prop.size == 0) return RDC::template initial<real>();

    // Sometimes the expression only knows its size:
    if (prop.size && prop.part.empty())
        prop.part = vex::partition(prop.size, queue);

    for(unsigned d = 0; d < queue.size(); ++d) {
        auto key    = backend::cache_key(queue[d]);
        auto kernel = cache.find(key);

        backend::select_context(queue[d]);

        if (kernel == cache.end()) {
            backend::source_generator source(queue[d]);

            typedef typename RDC::template function<real> fun;
            fun::define(source, "reduce_operation");

            output_terminal_preamble termpream(source, queue[d], "prm", empty_state());
            boost::proto::eval(boost::proto::as_child(expr),  termpream);

            source.kernel("vexcl_reductor_kernel")
                .open("(").parameter<size_t>("n");

            extract_terminals()( expr, declare_expression_parameter(source, queue[d], "prm", empty_state()) );

            source
                .template parameter< global_ptr<real> >("g_odata")
                .template smem_parameter<real>()
                .close(")");

#define VEXCL_INCREMENT_MY_SUM                                                 \
  {                                                                            \
    output_local_preamble loc_init(source, queue[d], "prm", empty_state());    \
    boost::proto::eval(expr, loc_init);                                        \
    vector_expr_context expr_ctx(source, queue[d], "prm", empty_state());      \
    source.new_line() << "mySum = reduce_operation(mySum, ";                   \
    boost::proto::eval(expr, expr_ctx);                                        \
    source << ");";                                                            \
  }

            source.open("{");
            source.smem_declaration<real>();
            source.new_line() << type_name< shared_ptr<real> >() << " sdata = smem;";

            if ( backend::is_cpu(queue[d]) ) {
                source.new_line() << "size_t grid_size  = " << source.global_size(0) << ";";
                source.new_line() << "size_t chunk_size = (n + grid_size - 1) / grid_size;";
                source.new_line() << "size_t chunk_id   = " << source.global_id(0) << ";";
                source.new_line() << "size_t start      = min(n, chunk_size * chunk_id);";
                source.new_line() << "size_t stop       = min(n, chunk_size * (chunk_id + 1));";
                source.new_line() << type_name<real>() << " mySum = (" << type_name<real>() << ")" << RDC::template initial<real>() << ";";
                source.new_line() << "for (size_t idx = start; idx < stop; idx++)";
                source.open("{");
                VEXCL_INCREMENT_MY_SUM
                source.close("}");
                source.new_line() << "g_odata[" << source.group_id(0) << "] = mySum;";
                source.close("}");

                backend::kernel krn(queue[d], source.str(), "vexcl_reductor_kernel");
                kernel = cache.insert(std::make_pair(key, krn)).first;
            } else {
                source.new_line() << "size_t tid = " << source.local_id(0) << ";";
                source.new_line() << "size_t block_size = " << source.local_size(0) << ";";
                source.new_line() << type_name<real>() << " mySum = " << RDC::template initial<real>() << ";";

                source.grid_stride_loop().open("{");
                VEXCL_INCREMENT_MY_SUM
                source.close("}");
                source.new_line() << "sdata[tid] = mySum;";
                source.new_line().barrier();
                for(unsigned bs = 512; bs > 32; bs /= 2) {
                    source.new_line() << "if (block_size >= " << bs * 2 << ")";
                    source.open("{").new_line() << "if (tid < " << bs << ") "
                        "{ sdata[tid] = mySum = reduce_operation(mySum, sdata[tid + " << bs << "]); }";
                    source.new_line().barrier().close("}");
                }
                source.new_line() << "if (tid < 32)";
                source.open("{");
                source.new_line() << "volatile " << type_name< shared_ptr<real> >() << " smem = sdata;";
                for(unsigned bs = 32; bs > 0; bs /= 2) {
                    source.new_line() << "if (block_size >= " << 2 * bs << ") "
                        "{ smem[tid] = mySum = reduce_operation(mySum, smem[tid + " << bs << "]); }";
                }
                source.close("}");
                source.new_line() << "if (tid == 0) g_odata[" << source.group_id(0) << "] = sdata[0];";
                source.close("}");

                backend::kernel krn(queue[d], source.str(), "vexcl_reductor_kernel", sizeof(real));
                kernel = cache.insert(std::make_pair(key, krn)).first;
            }
        }

#undef VEXCL_INCREMENT_MY_SUM

        if (size_t psize = prop.part_size(d)) {
            kernel->second.push_arg(psize);

            extract_terminals()(
                    expr,
                    set_expression_argument(kernel->second, d, prop.part_start(d), empty_state())
                    );

            kernel->second.push_arg(dbuf[d]);
            kernel->second.set_smem([](size_t wgs){ return wgs * sizeof(real); });

            kernel->second(queue[d]);
        }
    }

    std::fill(hbuf.begin(), hbuf.end(), RDC::template initial<real>());

    for(unsigned d = 0; d < queue.size(); d++) {
        if (prop.part_size(d))
            dbuf[d].read(queue[d], 0, idx[d + 1] - idx[d], &hbuf[idx[d]]);
    }

    for(unsigned d = 0; d < queue.size(); d++)
        if (prop.part_size(d)) queue[d].finish();

    return RDC::reduce(hbuf.begin(), hbuf.end());
}

template <typename real, class RDC> template <class Expr>
typename std::enable_if<
    boost::proto::matches<Expr, multivector_expr_grammar>::value &&
    !boost::proto::matches<Expr, vector_expr_grammar>::value,
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
const vex::Reductor<T, R>& get_reductor(const std::vector<backend::command_queue> &queue)
{
    // We will hold one static reductor per set of queues (or, rather, contexts):
    static std::map< std::vector<backend::kernel_cache_key>, vex::Reductor<T, R> > cache;

    // Extract OpenCL context handles from command queues:
    std::vector<backend::kernel_cache_key> ctx;
    ctx.reserve(queue.size());
    for(auto q = queue.begin(); q != queue.end(); ++q)
        ctx.push_back( backend::cache_key(*q) );

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
