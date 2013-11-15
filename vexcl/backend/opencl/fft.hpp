#ifndef VEXCL_BACKEND_OPENCL_FFT_HPP
#define VEXCL_BACKEND_OPENCL_FFT_HPP

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
 * \file   vexcl/backend/opencl/fft.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Fast Fourier Transformation.
 */

#include <memory>
#include <vexcl/vector.hpp>
#include <vexcl/backend/opencl/fft/plan.hpp>

namespace vex {

/// Fast Fourier Transform.
/**
 * Usage:
 * \code
 * FFT<cl_double2> fft(ctx, length);
 * output = fft(input); // out-of-place transform
 * data = fft(data);    // in-place transform
 * FFT<cl_double2> ifft({width, height}, fft::inverse); // implicit context
 * input = ifft(output); // backward transform
 * \endcode
 * To batch multiple transformations, use `fft::none` as the first kind:
 * \code
 * FFT<cl_double2> fft({batch, n}, {fft::none, fft::forward});
 * output = fft(input);
 * \endcode
 */
template <typename Tin, typename Tout = Tin, class Planner = fft::planner>
struct FFT {
    typedef typename cl_scalar_of<Tin>::type value_type;

    fft::plan<Tin, Planner> plan;

    /// 1D constructor
    FFT(const std::vector<backend::command_queue> &queues,
        size_t length, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, std::vector<size_t>(1, length), std::vector<fft::direction>(1, dir), planner) {}

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
    FFT(size_t length, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), std::vector<size_t>(1, length), std::vector<fft::direction>(1, dir), planner) {}
#endif

    /// \defgroup  N-dimensional constructors.
    /** @{ */
    FFT(const std::vector<backend::command_queue> &queues,
        const std::vector<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, lengths, std::vector<fft::direction>(lengths.size(), dir), planner) {}

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
    FFT(const std::vector<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, std::vector<fft::direction>(lengths.size(), dir), planner) {}
#endif

    FFT(const std::vector<backend::command_queue> &queues,
        const std::vector<size_t> &lengths,
        const std::vector<fft::direction> &dirs,
        const Planner &planner = Planner())
        : plan(queues, lengths, dirs, planner) {}

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
    FFT(const std::vector<size_t> &lengths,
        const std::vector<fft::direction> &dirs,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, dirs, planner) {}
#endif


#ifndef BOOST_NO_INITIALIZER_LISTS
    FFT(const std::vector<backend::command_queue> &queues,
        const std::initializer_list<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(queues, lengths, std::vector<fft::direction>(lengths.size(), dir), planner) {}

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
    FFT(const std::initializer_list<size_t> &lengths, fft::direction dir = fft::forward,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, std::vector<fft::direction>(lengths.size(), dir), planner) {}
#endif

    FFT(const std::vector<backend::command_queue> &queues,
        const std::initializer_list<size_t> &lengths,
        const std::initializer_list<fft::direction> &dirs,
        const Planner &planner = Planner())
        : plan(queues, lengths, dirs, planner) {}

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
    FFT(const std::initializer_list<size_t> &lengths,
        const std::initializer_list<fft::direction> &dirs,
        const Planner &planner = Planner())
        : plan(current_context().queue(), lengths, dirs, planner) {}
#endif
#endif
    /** @} */

    // User call
    template <class Expr>
    auto operator()(const Expr &x) -> decltype(plan.template apply<Tout>(x))
    {
        return plan.template apply<Tout>(x);
    }
};

}

#endif
