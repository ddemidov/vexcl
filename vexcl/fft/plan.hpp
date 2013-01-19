#ifndef VEXCL_FFT_PLAN_HPP
#define VEXCL_FFT_PLAN_HPP

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
 * \file   fft/plan.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  FFT plan, stores kernels and buffers for one configuration.
 */


#include <vexcl/vector.hpp>
#include <vexcl/fft/kernels.hpp>
#include <boost/lexical_cast.hpp>

namespace vex {
namespace fft {


template <class T2>
struct plan {
    static_assert(std::is_same<T2, cl_float2>::value || std::is_same<T2, cl_double2>::value,
        "Only float2 and double2 are supported.");
    typedef typename cl_scalar_of<T2>::type T;

    const std::vector<cl::CommandQueue> &queues;
    const bool inverse;
    size_t total_n;

    std::vector<kernel_call> kernels;

    vector<T2> temp[2];
    size_t input, output;

    // \param sizes
    //  1D case: {n}.
    //  2D case: {h, w} in row-major format: x + y * w. (like FFTw)
    //  etc.
    plan(const std::vector<cl::CommandQueue> &queues, const std::vector<size_t> sizes, bool inverse)
        : queues(queues), inverse(inverse) {
        assert(sizes.size() >= 1);
        assert(queues.size() == 1);
        auto queue = queues[0];
        auto context = qctx(queue);
        auto device = qdev(queue);

        total_n = 1;
        for(auto x : sizes) total_n *= x;

        temp[0] = vector<T2>(queues, total_n);
        temp[1] = vector<T2>(queues, total_n);
        size_t current = 0, other = 1;

        // Build the list of kernels.
        input = current;
        for(auto d = sizes.rbegin() ; d != sizes.rend() ; d++) {
            const size_t w = *d, h = total_n / w;

            // 1D, each row.
            if(w > 1) {
                size_t p = 1;
                while(p < w) {
                    size_t radix = get_radix(p, w);
                    kernels.push_back(radix_kernel<T>(queue, w, h,
                        inverse, radix, p, temp[current](0), temp[other](0)));
                    std::swap(current, other);
                    p *= radix;
                }
            }

            // transpose.
            if(w > 1 && h > 1) {
                kernels.push_back(transpose_kernel<T>(queue, w, h,
                    temp[current](0), temp[other](0)));
                std::swap(current, other);
            }
        }
        output = current;
    }

    // returns the next radix to use for stage p, size n.
    static size_t get_radix(size_t p, size_t n) {
        const size_t rs[] = {16, 8, 4, 2};
        for(auto r : rs) if(p * r <= n) return r;
        throw std::runtime_error("Unsupported FFT size.");
    }

    /// Use to set the input
    vector<T2> &in() { return temp[input]; }

    /// Execute the complete transformation.
    void operator()() {
        for(auto run : kernels)
            queues[0].enqueueNDRangeKernel(run.kernel, cl::NullRange,
                run.global, run.local);
        if(inverse)
            temp[output] *= (T)1 / total_n;
    }

    /// Use to get the output
    const vector<T2> &out() const { return temp[output]; }
};


}
}
#endif
