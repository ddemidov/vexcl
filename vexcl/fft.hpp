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

#ifdef USE_AMD_FFT
#   include <clAmdFft.h>
#else
#   include <vexcl/fft/plan.hpp>
#endif

namespace vex {

#ifdef USE_AMD_FFT
template <class F>
struct fft_expr
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    typedef typename F::input_t T0;

    F &f;
    vector<T0> &input;

    fft_expr(F &f, vector<T0> &x) : f(f), input(x) {}

    template <bool negate, bool append>
    void apply(vector<typename F::output_t> &output) const
    {
        static_assert(!append, "Appending not implemented yet.");
        static_assert(!negate, "Negation not implemented yet.");
        f.execute(input, output);
    }
};


enum fft_direction {
   forward = CLFFT_FORWARD,
   inverse = CLFFT_BACKWARD
};

void fft_check_error(clAmdFftStatus status) {
    if(status != CL_SUCCESS)
        throw cl::Error(status, "AMD FFT");
}

// AMD FFT needs Setup/Teardown calls for the whole library.
// Sequential Setup/Teardowns are OK, but overlapping is not.
template <bool dummy = true>
struct amd_context {
    static_assert(dummy, "dummy parameter should be true");

    static void init() {
        if (!ctx) ctx.rest(new ctx_t());
    }

    // Not really needed, but who knows.
    static void destroy() {
        ctx.reset(0);
    }

    private:
        struct ctx_t {
            ctx_t() { fft_check_error(clAmdFftSetup(NULL)); }
            ~ctx_t() { fft_check_error(clAmdFftTeardown()); }
        };

        static std::unique_ptr<ctx_t> ctx;
};

template <bool dummy>
std::unique_ptr<typename amd_context<dummy>::ctx_t> amd_context<dummy>::ctx;

/**
 * An FFT functor. Assumes the vector is in row major format and densely packed.
 * Only supports a single device, only 2^a 3^b 5^c sizes, only single precision.
 * 1-3 dimensions.
 * Usage:
 * \code
 * FFT<cl_float2> fft(ctx.queue(), {width, height});
 * output = fft(input); // out-of-place transform
 * data = fft(data); // in-place transform
 * \endcode
 */
template <typename T0, typename T1 = T0>
struct FFT {
    static_assert(std::is_same<T0, T1>::value &&
        std::is_same<T0, cl_float2>::value,
        "Only single precision Complex-to-Complex transformations implemented.");

    typedef FFT<T0, T1> this_t;
    typedef T0 input_t;
    typedef T1 output_t;

    const std::vector<cl::CommandQueue> &queues;
    clAmdFftPlanHandle plan;
    fft_direction dir;

    template <class Array>
    FFT(const std::vector<cl::CommandQueue> &queues,
        const Array &lengths, fft_direction dir = forward)
        : queues(queues), plan(0), dir(dir) {
        init(lengths);
    }

    FFT(const std::vector<cl::CommandQueue> &queues,
        size_t length, fft_direction dir = forward)
        : queues(queues), plan(0), dir(dir) {
        std::array<size_t, 1> lengths = {{length}};
        init(lengths);
    }

#ifndef BOOST_NO_INITIALIZER_LISTS
    FFT(const std::vector<cl::CommandQueue> &queues,
        std::initializer_list<size_t> lengths, fft_direction dir = forward)
        : queues(queues), plan(0), dir(dir) {
        init(lengths);
    }
#endif

    template <class Array>
    void init(const Array &lengths) {
        assert(lengths.size() >= 1 && lengths.size() <= 3);
        assert(queues.size() == 1);
        amd_context<>::init();
        size_t _lengths[3];
        std::copy(std::begin(lengths), std::end(lengths), _lengths);
        cl::Context context = qctx(queues[0]);
        check_error(clAmdFftCreateDefaultPlan(&plan, context(),
            static_cast<clAmdFftDim>(lengths.size()), _lengths));
        check_error(clAmdFftSetPlanPrecision(plan, CLFFT_SINGLE));
        check_error(clAmdFftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
    }

    ~FFT() {
        if(plan) check_error(clAmdFftDestroyPlan(&plan));
    }


    void execute(const vector<T0> &input, vector<T1> &output) const {
        cl_mem input_buf = input(0)();
        cl_mem output_buf = output(0)();
        fft_check_error(clAmdFftSetResultLocation(plan,
            input_buf == output_buf ? CLFFT_INPLACE : CLFFT_OUTOFPLACE));
        cl_command_queue queue = queues[0]();
        fft_check_error(clAmdFftEnqueueTransform(plan,
            static_cast<clAmdFftDirection>(dir),
            1, &queue, /* wait events */0, NULL, /* out events */NULL,
            &input_buf, &output_buf, NULL));
    }


    // User call
    fft_expr<this_t> operator()(vector<T0> &x) {
        return {*this, x};
    }
};
#else // USE_AMD_FFT


template <class F>
struct fft_expr
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    typedef typename F::input_t T0;
    typedef typename F::value_type value_type;
    value_type scale;

    F &f;
    const vector<T0> &input;

    fft_expr(F &f, const vector<T0> &x) : scale(1), f(f), input(x) {}

    template <bool negate, bool append>
    void apply(vector<typename F::output_t> &output) const {
        f.template execute<negate, append>(input, output, scale);
    }
};

template <class F>
struct is_scalable<fft_expr<F>> : std::true_type {};


enum direction {
    forward, inverse
};

template <typename T0, typename T1 = T0>
struct FFT {
    typedef T0 input_t;
    typedef T1 output_t;
    typedef typename cl_scalar_of<T1>::type value_type;

    fft::plan<T0, T1> plan;

    /// 1D constructor
    FFT(const std::vector<cl::CommandQueue> &queues,
        size_t length, direction dir = forward)
        : plan(queues, {length}, dir == inverse) {}

    /// N-D constructors
    FFT(const std::vector<cl::CommandQueue> &queues,
        const std::vector<size_t> &lengths, direction dir = forward)
        : plan(queues, lengths, dir == inverse) {}

#ifndef BOOST_NO_INITIALIZER_LISTS
    FFT(const std::vector<cl::CommandQueue> &queues,
        const std::initializer_list<size_t> &lengths, direction dir = forward)
        : plan(queues, lengths, dir == inverse) {}
#endif

    template <bool negate, bool append>
    void execute(const vector<T0> &input, vector<T1> &output, value_type scale) {
        plan(input, output, append, negate ? -scale : scale);
    }


    // User call
    fft_expr<FFT<T0, T1>> operator()(const vector<T0> &x) {
        return {*this, x};
    }
};

#endif


}

#endif
