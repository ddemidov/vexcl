#ifndef VEXCL_FFT_HPP
#define VEXCL_FFT_HPP

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
 * \file   fft.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Fast Fourier Transformation.
 */

#include <memory>
#include <vexcl/vector.hpp>
#include <vexcl/fft/plan.hpp>

namespace vex {

template <class F, class Expr>
struct fft_expr
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    typedef typename F::input_t T0;
    typedef typename F::value_type value_type;
    value_type scale;

    F &f;
    const Expr &input;

    fft_expr(F &f, const Expr &x) : scale(1), f(f), input(x) {}

    template <bool negate, bool append>
    void apply(vector<typename F::output_t> &output) const {
        f.template execute<negate, append>(input, output, scale);
    }
};

template <class F, class E>
struct is_scalable<fft_expr<F, E>> : std::true_type {};


namespace fft {

enum direction {
    forward, inverse
};

}

/// Fast Fourier Transform.
/**
 * Usage:
 * \code
 * FFT<cl_double2> fft(ctx, length);
 * output = fft(input); // out-of-place transform
 * data = fft(data);    // in-place transform
 * \endcode
 */
template <typename T0, typename T1 = T0, class Planner = fft::planner>
struct FFT {
    typedef T0 input_t;
    typedef T1 output_t;
    typedef typename cl_scalar_of<T1>::type value_type;

    fft::plan<T0, T1, Planner> plan;

    /// 1D constructor
    FFT(const std::vector<cl::CommandQueue> &queues,
        size_t length, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, std::vector<size_t>(1, length), dir == fft::inverse, planner) {}

    FFT(size_t length, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), std::vector<size_t>(1, length), dir == fft::inverse, planner) {}

    /// N-D constructors
    FFT(const std::vector<cl::CommandQueue> &queues,
        const std::vector<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, lengths, dir == fft::inverse, planner) {}

    FFT(const std::vector<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, dir == fft::inverse, planner) {}

#ifndef BOOST_NO_INITIALIZER_LISTS
    FFT(const std::vector<cl::CommandQueue> &queues,
        const std::initializer_list<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, lengths, dir == fft::inverse, planner) {}

    FFT(const std::initializer_list<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, dir == fft::inverse, planner) {}
#endif

    template <bool negate, bool append, class Expr>
    void execute(const Expr &input, vector<T1> &output, value_type scale) {
        plan(input, output, append, negate ? -scale : scale);
    }


    // User call
    template <class Expr>
    fft_expr<FFT<T0, T1, Planner>, Expr> operator()(const Expr &x) {
        return fft_expr< FFT<T0, T1, Planner>, Expr>(*this, x);
    }
};

}

#endif
