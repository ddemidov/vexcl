#ifndef VEXCL_MBA_HPP
#define VEXCL_MBA_HPP

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
 * \file   vexcl/mba.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Scattered data interpolation with multilevel B-Splines.
 */

#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cassert>

#include <boost/tuple/tuple.hpp>
#include <boost/fusion/adapted/boost_tuple.hpp>

#include <vexcl/operations.hpp>

// Include boost.preprocessor header if variadic templates are not available.
// Also include it if we use gcc v4.6.
// This is required due to bug http://gcc.gnu.org/bugzilla/show_bug.cgi?id=35722
#if defined(BOOST_NO_VARIADIC_TEMPLATES) || (defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 4 && __GNUC_MINOR__ == 6)
#  include <boost/preprocessor/repetition.hpp>
#  ifndef VEXCL_MAX_ARITY
#    define VEXCL_MAX_ARITY BOOST_PROTO_MAX_ARITY
#  endif
#endif
namespace vex {

/// \cond INTERNAL
struct mba_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< mba_terminal >::type
    > mba_terminal_expression;

template <class MBA, class ExprTuple>
struct mba_interp : public mba_terminal_expression {
    typedef typename MBA::value_type value_type;

    const MBA      &cloud;
    const ExprTuple coord;

    mba_interp(const MBA &cloud, const ExprTuple coord)
        : cloud(cloud), coord(coord) {}
};

namespace detail {
    // Compile time value of N^M.
    template <size_t N, size_t M>
    struct power : std::integral_constant<size_t, N * power<N, M-1>::value> {};

    template <size_t N>
    struct power<N, 0> : std::integral_constant<size_t, 1> {};

    // Nested loop counter of compile-time size (M loops of size N).
    template <size_t N, size_t M>
    class scounter {
        public:
            scounter() : idx(0) {
                std::fill(i.begin(), i.end(), static_cast<size_t>(0));
            }

            size_t operator[](size_t d) const {
                return i[d];
            }

            scounter& operator++() {
                for(size_t d = M; d--; ) {
                    if (++i[d] < N) break;
                    i[d] = 0;
                }

                ++idx;

                return *this;
            }

            operator size_t() const {
                return idx;
            }

            bool valid() const {
                return idx < power<N, M>::value;
            }
        private:
            size_t idx;
            std::array<size_t, M> i;
    };

    // Nested loop counter of run-time size (M loops of given sizes).
    template <size_t M>
    class dcounter {
        public:
            dcounter(const std::array<size_t, M> &N)
                : idx(0),
                  size(std::accumulate(N.begin(), N.end(),
                            static_cast<size_t>(1), std::multiplies<size_t>())),
                  N(N)
            {
                std::fill(i.begin(), i.end(), static_cast<size_t>(0));
            }

            size_t operator[](size_t d) const {
                return i[d];
            }

            dcounter& operator++() {
                for(size_t d = M; d--; ) {
                    if (++i[d] < N[d]) break;
                    i[d] = 0;
                }

                ++idx;

                return *this;
            }

            operator size_t() const {
                return idx;
            }

            bool valid() const {
                return idx < size;
            }
        private:
            size_t idx, size;
            std::array<size_t, M> N, i;
    };
} // namespace detail

/// \endcond

/// Scattered data interpolation with multilevel B-Splines.
/**
 * This is an implementation of the MBA algorithm from [1]. This is a fast
 * algorithm for scattered N-dimensional data interpolation and approximation.
 * Multilevel B-splines are used to compute a C2-continuous surface
 * through a set of irregularly spaced points. The algorithm makes use of a
 * coarse-to-fine hierarchy of control lattices to generate a sequence of
 * bicubic B-spline functions whose sum approaches the desired interpolation
 * function. Large performance gains are realized by using B-spline refinement
 * to reduce the sum of these functions into one equivalent B-spline function.
 * High-fidelity reconstruction is possible from a selected set of sparse and
 * irregular samples.
 *
 * [1] S. Lee, G. Wolberg, and S. Y. Shin. Scattered data interpolation with
 *     multilevel B-Splines. IEEE Transactions on Visualization and
 *     Computer Graphics, 3:228–244, 1997.
 */
template <size_t NDIM, typename real = double>
class mba {
    public:
        typedef real value_type;
        typedef std::array<real,   NDIM> point;
        typedef std::array<size_t, NDIM> index;

        static const size_t ndim = NDIM;

        std::vector<cl::CommandQueue> queue;
        std::vector<cl::Buffer> phi;
        point xmin, hinv;
        index n, stride;

        /**
         * \param cmin   corner of bounding box with smallest coordinates.
         * \param cmax   corner of bounding box with largest coordinates.
         * \param coo    coordinates of data points.
         * \param val    values of data points.
         * \param grid   initial control lattice size (excluding boundary points).
         * \param levels number of levels in hierarchy.
         * \param tol    stop if residual is less than this.
         */
        mba(
                const std::vector<cl::CommandQueue> &queue,
                const point &cmin, const point &cmax,
                const std::vector<point> &coo, std::vector<real> val,
                std::array<size_t, NDIM> grid, size_t levels = 8, real tol = 1e-8
           ) : queue(queue)
        {
#ifndef NDEBUG
            assert(coo.size() == val.size());

            for(size_t k = 0; k < NDIM; ++k)
                assert(grid[k] > 1);
#endif

            double res0 = std::accumulate(val.begin(), val.end(), static_cast<real>(0),
                    [](real sum, real v) { return sum + v * v; });

            std::unique_ptr<lattice> psi( new lattice(cmin, cmax, grid, coo, val) );
            double res = psi->update_data(coo, val);
#ifdef VEXCL_MBA_VERBOSE
            std::cout << "level  0: res = " << std::scientific << res << std::endl;
#endif

            for (size_t k = 1; (res > res0 * tol) && (k < levels); ++k) {
                for(size_t d = 0; d < NDIM; ++d) grid[d] = 2 * grid[d] - 1;

                std::unique_ptr<lattice> f( new lattice(cmin, cmax, grid, coo, val) );
                res = f->update_data(coo, val);
#ifdef VEXCL_MBA_VERBOSE
                std::cout << "level " << k << std::scientific << ": res = " << res << std::endl;
#endif

                f->append_refined(*psi);
                psi = std::move(f);
            }

            xmin   = psi->xmin;
            hinv   = psi->hinv;
            n      = psi->n;
            stride = psi->stride;

            phi.reserve(queue.size());

            for(auto q = queue.begin(); q != queue.end(); ++q)
                phi.push_back( cl::Buffer(
                            qctx(*q), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(real) * psi->phi.size(), psi->phi.data()
                            ) );
        }

#if !defined(BOOST_NO_VARIADIC_TEMPLATES) && ((!defined(__GNUC__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 6)) || defined(__clang__))
        /// Provide interpolated values at given coordinates.
        template <class... Expr>
        mba_interp< mba, boost::tuple<const Expr&...> >
        operator()(const Expr&... expr) const {
            static_assert(sizeof...(Expr) == NDIM, "Wrong number of parameters");
            return mba_interp< mba, boost::tuple<const Expr&...> >(*this, boost::tie(expr...));
        }
#else

#define PRINT_PARAM(z, n, data) const Expr ## n &expr ## n
#define PRINT_TEMPL(z, n, data) const Expr ## n &
#define FUNCALL_OPERATOR(z, n, data) \
        template < BOOST_PP_ENUM_PARAMS(n, class Expr) > \
        mba_interp< mba, boost::tuple<BOOST_PP_ENUM(n, PRINT_TEMPL, ~)> > \
        operator()( BOOST_PP_ENUM(n, PRINT_PARAM, ~) ) { \
            return mba_interp< mba, boost::tuple<BOOST_PP_ENUM(n, PRINT_TEMPL, ~)> >( \
                    *this, boost::tie( BOOST_PP_ENUM_PARAMS(n, expr) )); \
        }

BOOST_PP_REPEAT_FROM_TO(1, VEXCL_MAX_ARITY, FUNCALL_OPERATOR, ~)

#undef PRINT_TEMPL
#undef PRINT_PARAM
#undef FUNCALL_OPERATOR
#endif
    private:
        // Control lattice.
        struct lattice {
            point xmin, hinv;
            index n, stride;
            std::vector<real> phi;

            lattice(
                    const point &cmin, const point &cmax, std::array<size_t, NDIM> grid,
                    const std::vector<point> &coo, const std::vector<real> &val
                   ) : xmin(cmin), n(grid)
            {
                for(size_t d = 0; d < NDIM; ++d) {
                    hinv[d] = (grid[d] - 1) / (cmax[d] - cmin[d]);
                    xmin[d] -= 1 / hinv[d];
                    n[d]    += 2;
                }

                stride[NDIM - 1] = 1;
                for(size_t d = NDIM - 1; d--; )
                    stride[d] = stride[d + 1] * n[d + 1];

                std::vector<real> delta(n[0] * stride[0], 0.0);
                std::vector<real> omega(n[0] * stride[0], 0.0);

                auto p = coo.begin();
                auto v = val.begin();
                for(; p != coo.end(); ++p, ++v) {
                    if (!contained(cmin, cmax, *p)) continue;

                    index i;
                    point s;

                    for(size_t d = 0; d < NDIM; ++d) {
                        real u = ((*p)[d] - xmin[d]) * hinv[d];
                        i[d] = std::floor(u) - 1;
                        s[d] = u - std::floor(u);
                    }

                    std::array<real, detail::power<4, NDIM>::value> w;
                    real sw2 = 0;

                    for(detail::scounter<4, NDIM> d; d.valid(); ++d) {
                        real buf = 1;
                        for(size_t k = 0; k < NDIM; ++k)
                            buf *= B(d[k], s[k]);

                        w[d] = buf;
                        sw2 += buf * buf;
                    }

                    for(detail::scounter<4, NDIM> d; d.valid(); ++d) {
                        real phi = (*v) * w[d] / sw2;

                        size_t idx = 0;
                        for(size_t k = 0; k < NDIM; ++k) {
                            assert(i[k] + d[k] < n[k]);

                            idx += (i[k] + d[k]) * stride[k];
                        }

                        real w2 = w[d] * w[d];

                        assert(idx < delta.size());

                        delta[idx] += w2 * phi;
                        omega[idx] += w2;
                    }
                }

                phi.resize(omega.size());

                for(auto w = omega.begin(), d = delta.begin(), f = phi.begin();
                        w != omega.end();
                        ++w, ++d, ++f
                   )
                {
                    if (std::fabs(*w) < 1e-32)
                        *f = 0;
                    else
                        *f = (*d) / (*w);
                }
            }

            // Get interpolated value at given position.
            real operator()(const point &p) const {
                index i;
                point s;

                for(size_t d = 0; d < NDIM; ++d) {
                    real u = (p[d] - xmin[d]) * hinv[d];
                    i[d] = std::floor(u) - 1;
                    s[d] = u - std::floor(u);
                }

                real f = 0;

                for(detail::scounter<4, NDIM> d; d.valid(); ++d) {
                    real w = 1;
                    for(size_t k = 0; k < NDIM; ++k)
                        w *= B(d[k], s[k]);

                    f += w * get(i, d);
                }

                return f;
            }

            // Subtract interpolated values from data points.
            real update_data(const std::vector<point> &coo, std::vector<real> &val) const {
                auto c = coo.begin();
                auto v = val.begin();

                real res = 0;

                for(; c != coo.end(); ++c, ++v) {
                    *v -= (*this)(*c);

                    res += (*v) * (*v);
                }

                return res;
            }

            // Refine r and append it to the current control lattice.
            void append_refined(const lattice &r) {
                static const std::array<real, 5> s = {{
                    0.125, 0.500, 0.750, 0.500, 0.125
                }};

                for(detail::dcounter<NDIM> i(r.n); i.valid(); ++i) {
                    real f = r.phi[i];
                    for(detail::scounter<5, NDIM> d; d.valid(); ++d) {
                        index j;
                        bool skip = false;
                        size_t idx = 0;
                        for(size_t k = 0; k < NDIM; ++k) {
                            j[k] = 2 * i[k] + d[k] - 3;
                            if (j[k] >= n[k]) { skip = true; break; }

                            idx += j[k] * stride[k];
                        }

                        if (skip) continue;

                        real c = 1;
                        for(size_t k = 0; k < NDIM; ++k) c *= s[d[k]];

                        phi[idx] += f * c;
                    }
                }
            }

            private:
                // Value of k-th B-Spline at t.
                static inline real B(size_t k, real t) {
                    assert(0 <= t && t < 1);
                    assert(k < 4);

                    switch (k) {
                        case 0:
                            return (t * (t * (-t + 3) - 3) + 1) / 6;
                        case 1:
                            return (t * t * (3 * t - 6) + 4) / 6;
                        case 2:
                            return (t * (t * (-3 * t + 3) + 3) + 1) / 6;
                        case 3:
                            return t * t * t / 6;
                        default:
                            return 0;
                    }
                }

                // x is within [xmin, xmax].
                static bool contained(
                        const point &xmin, const point &xmax, const point &x)
                {
                    for(size_t d = 0; d < NDIM; ++d) {
                        static const real eps = 1e-12;

                        if (x[d] - eps <  xmin[d]) return false;
                        if (x[d] + eps >= xmax[d]) return false;
                    }

                    return true;
                }

                // Get value of phi at index (i + d).
                template <class Shift>
                inline real get(const index &i, const Shift &d) const {
                    size_t idx = 0;

                    for(size_t k = 0; k < NDIM; ++k) {
                        size_t j = i[k] + d[k];

                        if (j >= n[k]) return 0;
                        idx += j * stride[k];
                    }

                    return phi[idx];
                }
        };
};

/// \cond INTERNAL

namespace traits {

template <>
struct is_vector_expr_terminal< mba_terminal > : std::true_type {};

template <>
struct proto_terminal_is_value< mba_terminal > : std::true_type {};

template <class MBA, class ExprTuple>
struct terminal_preamble< mba_interp<MBA, ExprTuple> > {
    static std::string get(const mba_interp<MBA, ExprTuple>&,
            const cl::Device&, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;

        std::string B    = prm_name + "_B";
        std::string real = type_name<typename MBA::value_type>();

        s << real << " " << B << "0(" << real << " t) { return (t * (t * (-t + 3) - 3) + 1) / 6; }\n"
          << real << " " << B << "1(" << real << " t) { return (t * t * (3 * t - 6) + 4) / 6; }\n"
          << real << " " << B << "2(" << real << " t) { return (t * (t * (-3 * t + 3) + 3) + 1) / 6; }\n"
          << real << " " << B << "3(" << real << " t) { return t * t * t / 6; }\n\n"
          << real << " " << prm_name << "_mba(\n";

        for(size_t k = 0; k < MBA::ndim; ++k)
            s << "    " << real << " x" << k << ",\n";

        for(size_t k = 0; k < MBA::ndim; ++k)
            s <<
            "    " << real << " c" << k << ",\n"
            "    " << real << " h" << k << ",\n"
            "    " << type_name<size_t>() << " n" << k << ",\n"
            "    " << type_name<size_t>() << " m" << k << ",\n";

        s <<
            "    global const " << real << " *phi\n"
            ")\n"
            "{\n"
            "    " << real << " u;\n"
            "\n";
        for(size_t k = 0; k < MBA::ndim; ++k)
            s <<
            "    u = (x" << k << " - c" << k << ") * h" << k << ";\n"
            "    " << type_name<size_t>() << " i" << k << " = floor(u) - 1;\n"
            "    " << real << " s" << k << " = u - floor(u);\n"
            "\n";
        s <<
            "    " << real << " f = 0;\n"
            "    " << type_name<size_t>() << " j, idx;\n"
            "\n";

        for(detail::scounter<4,MBA::ndim> d; d.valid(); ++d) {
            s << "    idx = 0;\n";
            for(size_t k = 0; k < MBA::ndim; ++k) {
                s <<
                    "    j = i" << k << " + " << d[k] << ";\n"
                    "    if (j < n" << k << ") {\n"
                    "    idx += j * m" << k << ";\n";
            }

            s << "    f += ";
            for(size_t k = 0; k < MBA::ndim; ++k) {
                if (k) s << " * ";
                s << B << d[k] << "(s" << k << ")";
            }

            s << " * phi[idx];\n";

            for(size_t k = 0; k < MBA::ndim; ++k)
                s << "    }\n";
        }
        s <<
            "\n"
            "    return f;\n"
            "}\n";

        return s.str();
    }
};

template <class MBA, class ExprTuple>
struct kernel_param_declaration< mba_interp<MBA, ExprTuple> > {
    static std::string get(const mba_interp<MBA, ExprTuple> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;

        boost::fusion::for_each(term.coord, prmdecl(s, dev, prm_name));

        for(size_t k = 0; k < MBA::ndim; ++k) {
            s << ",\n\t" << type_name<typename MBA::value_type>() << " " << prm_name << "_c" << k
              << ",\n\t" << type_name<typename MBA::value_type>() << " " << prm_name << "_h" << k
              << ",\n\t" << type_name<size_t>() << " " << prm_name << "_n" << k
              << ",\n\t" << type_name<size_t>() << " " << prm_name << "_m" << k;
        }

        s << ",\n\tglobal const " << type_name<typename MBA::value_type>() << " * " << prm_name << "_phi";

        return s.str();
    }

    struct prmdecl {
        std::ostream &s;
        const cl::Device &dev;
        const std::string &prm_name;
        mutable int pos;

        prmdecl(std::ostream &s,
                const cl::Device &dev, const std::string &prm_name
            ) : s(s), dev(dev), prm_name(prm_name), pos(0)
        {}

        template <class Expr>
        void operator()(const Expr &expr) const {
            detail::declare_expression_parameter ctx(s, dev, 1, prm_name + "_x" + std::to_string(pos) + "_");
            detail::extract_terminals()(boost::proto::as_child(expr), ctx);

            pos++;
        }
    };
};

template <class MBA, class ExprTuple>
struct partial_vector_expr< mba_interp<MBA, ExprTuple> > {
    static std::string get(const mba_interp<MBA, ExprTuple> &term,
            const cl::Device &dev, const std::string &prm_name,
            detail::kernel_generator_state&)
    {
        std::ostringstream s;

        s << prm_name << "_mba(";

        boost::fusion::for_each(term.coord, buildexpr(s, dev, prm_name));

        for(size_t k = 0; k < MBA::ndim; ++k) {
            s << ", " << prm_name << "_c" << k
              << ", " << prm_name << "_h" << k
              << ", " << prm_name << "_n" << k
              << ", " << prm_name << "_m" << k;
        }

        s << ", " << prm_name << "_phi)";

        return s.str();
    }

    struct buildexpr {
        std::ostream &s;
        const cl::Device &dev;
        const std::string &prm_name;
        mutable int pos;

        buildexpr(std::ostream &s,
                const cl::Device &dev, const std::string &prm_name
            ) : s(s), dev(dev), prm_name(prm_name), pos(0)
        {}

        template <class Expr>
        void operator()(const Expr &expr) const {
            if(pos) s << ", ";

            detail::vector_expr_context ctx(s, dev, 1, prm_name + "_x" + std::to_string(pos) + "_");
            boost::proto::eval(boost::proto::as_child(expr), ctx);

            pos++;
        }
    };
};

template <class MBA, class ExprTuple>
struct kernel_arg_setter< mba_interp<MBA, ExprTuple> > {
    static void set(const mba_interp<MBA, ExprTuple> &term,
            cl::Kernel &kernel, unsigned device, size_t index_offset,
            unsigned &position, detail::kernel_generator_state&)
    {

        boost::fusion::for_each(term.coord, setargs(kernel, device, index_offset, position));

        for(size_t k = 0; k < MBA::ndim; ++k) {
            kernel.setArg(position++, term.cloud.xmin[k]);
            kernel.setArg(position++, term.cloud.hinv[k]);
            kernel.setArg(position++, term.cloud.n[k]);
            kernel.setArg(position++, term.cloud.stride[k]);
        }
        kernel.setArg(position++, term.cloud.phi[device]);
    }

    struct setargs {
        cl::Kernel &kernel;
        unsigned device;
        size_t index_offset;
        unsigned &position;

        setargs(
                cl::Kernel &kernel, unsigned device, size_t index_offset, unsigned &position
               ) : kernel(kernel), device(device), index_offset(index_offset), position(position)
        {}

        template <class Expr>
        void operator()(const Expr &expr) const {
            detail::set_expression_argument ctx(kernel, device, position, index_offset);
            detail::extract_terminals()( boost::proto::as_child(expr), ctx);
        }
    };
};

template <class MBA, class ExprTuple>
struct expression_properties< mba_interp<MBA, ExprTuple> > {
    static void get(const mba_interp<MBA, ExprTuple> &term,
            std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
    }

    struct extrprop {
        std::vector<cl::CommandQueue> &queue_list;
        std::vector<size_t> &partition;
        size_t &size;

        extrprop(std::vector<cl::CommandQueue> &queue_list,
            std::vector<size_t> &partition, size_t &size
            ) : queue_list(queue_list), partition(partition), size(size)
        {}

        template <class Expr>
        void operator()(const Expr &expr) const {
            if (queue_list.empty()) {
                detail::get_expression_properties prop;
                detail::extract_terminals()(boost::proto::as_child(expr), prop);

                queue_list = prop.queue;
                partition  = prop.part;
                size       = prop.size;
            }
        }
    };
};

} //namespace traits

/// \endcond

} // namespace vex


#endif
