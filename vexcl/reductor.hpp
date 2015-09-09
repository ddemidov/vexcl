#ifndef VEXCL_REDUCTOR_HPP
#define VEXCL_REDUCTOR_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Vector expression reduction.
 */

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <numeric>
#include <limits>

#include <vexcl/vector.hpp>
#include <vexcl/operations.hpp>

namespace vex {

/// Summation. Should be used as a template parameter for Reductor class.
struct SUM {
    // In order to define a reduction kind for vex::Reductor, one should define
    // a struct like the following:
    template <class T>
    struct impl {
        // Initial value for the operation.
        static T initial() {
            return T();
        }

        // Device-side reduction function.
        struct device : UserFunction<device, T(T, T)> {
            static std::string name() { return "SUM_" + type_name<T>(); }
            static std::string body() { return "return prm1 + prm2;"; }
        };

        // Host-side reduction function.
        T operator()(T a, T b) const {
            return a + b;
        }
    };
};

/// Compensated summation.
/**
 * Reduces the numerical error in the result with <a
 * href="http://en.wikipedia.org/wiki/Kahan_summation_algorithm">
 * Kahan summation algorithm</a>.
 */
struct SUM_Kahan : SUM {};

/// Maximum element. Should be used as a template parameter for Reductor class.
struct MAX {
    template <class T>
    struct impl {
        // Initial value for the operation.
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

        struct device : UserFunction<device, T(T, T)> {
            static std::string name() { return "MAX_" + type_name<T>(); }
            static std::string body() { return "return prm1 > prm2 ? prm1 : prm2;"; }
        };

        T operator()(T a, T b) const {
            return a > b ? a : b;
        }
    };
};

/// Minimum element. Should be used as a template parameter for Reductor class.
struct MIN {
    template <class T>
    struct impl {
        // Initial value for the operation.
        static T initial() {
            return std::numeric_limits<T>::max();
        }

        struct device : UserFunction<device, T(T, T)> {
            static std::string name() { return "MIN_" + type_name<T>(); }
            static std::string body() { return "return prm1 < prm2 ? prm1 : prm2;"; }
        };

        T operator()(T a, T b) const {
            return a < b ? a : b;
        }
    };
};

/// Parallel reduction of arbitrary expression.
/**
 * Reduction uses small temporary buffer on each device present in the queue
 * parameter. One Reductor class for each reduction kind is enough per thread
 * of execution.
 */
template <typename real, class RDC = SUM>
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
        mutable std::vector<backend::command_queue> queue;

        struct reductor_data {
            std::vector<real>            hbuf;
            backend::device_vector<real> dbuf;

            reductor_data(const backend::command_queue &q)
                : hbuf(backend::kernel::num_workgroups(q)),
                  dbuf(q, backend::kernel::num_workgroups(q))
            { }
        };

        typedef
            detail::object_cache<detail::index_by_queue, reductor_data>
            reductor_data_cache;

        static reductor_data_cache& get_data_cache() {
            static reductor_data_cache cache;
            return cache;
        }

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

        template <class Expr, class OP>
        struct local_sum {
            static void get(const backend::command_queue &q, const Expr &expr,
                    backend::source_generator &source)
            {
                using namespace detail;

                typedef typename OP::template impl<real>::device fun;
                real initial = OP::template impl<real>::initial();

                source.new_line()
                    << type_name<real>() << " mySum = ("
                    << type_name<real>() << ")" << initial << ";";
                source.grid_stride_loop().open("{");

                output_local_preamble loc_init(source, q, "prm", empty_state());
                boost::proto::eval(expr, loc_init);
                source.new_line() << "mySum = " << fun::name() << "(mySum, ";
                vector_expr_context expr_ctx(source, q, "prm", empty_state());
                boost::proto::eval(expr, expr_ctx);
                source << ");";

                source.close("}");
            }
        };

        // http://en.wikipedia.org/wiki/Kahan_summation_algorithm
        template <class Expr>
        struct local_sum<Expr, SUM_Kahan> {
            static void get(const backend::command_queue &q, const Expr &expr,
                    backend::source_generator &source)
            {
                using namespace detail;

                source.new_line()
                    << type_name<real>() << " mySum = ("
                    << type_name<real>() << ")0, c = ("
                    << type_name<real>() << ")0;";
                source.grid_stride_loop().open("{");

                output_local_preamble loc_init(source, q, "prm", empty_state());
                boost::proto::eval(expr, loc_init);

                source.new_line() << type_name<real>() << " y = (";
                vector_expr_context expr_ctx(source, q, "prm", empty_state());
                boost::proto::eval(expr, expr_ctx);
                source << ") - c;";

                source.new_line() << type_name<real>() << " t = mySum + y;";
                source.new_line() << "c = (t - mySum) - y;";
                source.new_line() << "mySum = t;";

                source.close("}");
            }
        };
};

#ifndef DOXYGEN
template <typename real, class RDC>
Reductor<real,RDC>::Reductor(const std::vector<backend::command_queue> &queue)
    : queue(queue)
{ }

template <typename real, class RDC> template <class Expr>
typename std::enable_if<
    boost::proto::matches<Expr, vector_expr_grammar>::value,
    real
>::type
Reductor<real,RDC>::operator()(const Expr &expr) const {
    using namespace detail;

    static kernel_cache cache;

    auto &data_cache = get_data_cache();

    get_expression_properties prop;
    extract_terminals()(expr, prop);

    real initial = RDC::template impl<real>::initial();

    // If expression is of zero size, then there is nothing to do. Hurray!
    if (prop.size == 0) return initial;

    // Sometimes the expression only knows its size:
    if (prop.size && prop.part.empty())
        prop.part = vex::partition(prop.size, queue);

    for(unsigned d = 0; d < queue.size(); ++d) {
        auto kernel = cache.find(queue[d]);

        backend::select_context(queue[d]);

        if (kernel == cache.end()) {
            backend::source_generator source(queue[d]);

            output_terminal_preamble termpream(source, queue[d], "prm", empty_state());
            boost::proto::eval(boost::proto::as_child(expr),  termpream);

            typedef typename RDC::template impl<real>::device fun;
            boost::proto::eval(boost::proto::as_child( fun()( real(), real()) ), termpream);

            source.kernel("vexcl_reductor_kernel")
                .open("(").template parameter<size_t>("n");

            extract_terminals()( expr, declare_expression_parameter(source, queue[d], "prm", empty_state()) );

            source.template parameter< global_ptr<real> >("g_odata");

            if (!backend::is_cpu(queue[d]))
                source.template smem_parameter<real>();

            source.close(")");


            source.open("{");

            local_sum<Expr, RDC>::get(queue[d], expr, source);

            if ( backend::is_cpu(queue[d]) ) {
                source.new_line() << "g_odata[" << source.group_id(0) << "] = mySum;";
                source.close("}");

                kernel = cache.insert(queue[d], backend::kernel(
                            queue[d], source.str(), "vexcl_reductor_kernel"));
            } else {
                source.smem_declaration<real>();
                source.new_line() << type_name< shared_ptr<real> >() << " sdata = smem;";

                source.new_line() << "size_t tid = " << source.local_id(0) << ";";
                source.new_line() << "size_t block_size = " << source.local_size(0) << ";";

                source.new_line() << "sdata[tid] = mySum;";
                source.new_line().barrier();
                for(unsigned bs = 512; bs > 0; bs /= 2) {
                    source.new_line() << "if (block_size >= " << bs * 2 << ")";
                    source.open("{").new_line() << "if (tid < " << bs << ") "
                        "{ sdata[tid] = mySum = " << fun::name() << "(mySum, sdata[tid + " << bs << "]); }";
                    source.new_line().barrier().close("}");
                }
                source.new_line() << "if (tid == 0) g_odata[" << source.group_id(0) << "] = sdata[0];";
                source.close("}");

                kernel = cache.insert(queue[d], backend::kernel(
                            queue[d], source.str(), "vexcl_reductor_kernel",
                            sizeof(real)));
            }
        }

        if (size_t psize = prop.part_size(d)) {
            auto data = data_cache.find(queue[d]);
            if (data == data_cache.end())
                data = data_cache.insert(queue[d], reductor_data(queue[d]));

            kernel->second.push_arg(psize);

            extract_terminals()(
                    expr,
                    set_expression_argument(kernel->second, d, prop.part_start(d), empty_state())
                    );

            kernel->second.push_arg(data->second.dbuf);

            if (!backend::is_cpu(queue[d]))
                kernel->second.set_smem(
                        [](size_t wgs){
                            return wgs * sizeof(real);
                        });

            kernel->second(queue[d]);
        }
    }

    for(unsigned d = 0; d < queue.size(); d++) {
        if (prop.part_size(d)) {
            auto data = data_cache.find(queue[d]);

            data->second.dbuf.read(queue[d], 0, data->second.hbuf.size(), data->second.hbuf.data());
        }
    }

    real result = initial;
    typename RDC::template impl<real> rdc;
    for(unsigned d = 0; d < queue.size(); d++) {
        if (prop.part_size(d)) {
            auto data = data_cache.find(queue[d]);

            queue[d].finish();

            result = rdc(result, std::accumulate(
                        data->second.hbuf.begin(), data->second.hbuf.end(),
                        initial, rdc));
        }
    }

    return result;
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

/// Returns an instance of vex::Reductor<T,R>
/**
 * \deprecated
 */
template <typename T, class R>
vex::Reductor<T, R> get_reductor(const std::vector<backend::command_queue> &queue)
{
    return vex::Reductor<T, R>(queue);
}

} // namespace vex

#endif
