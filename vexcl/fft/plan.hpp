#ifndef VEXCL_FFT_PLAN_HPP
#define VEXCL_FFT_PLAN_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   plan.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  FFT plan, stores kernels and buffers for one configuration.
 */


#include <vexcl/vector.hpp>
#include <vexcl/fft/kernels.hpp>
#include <boost/lexical_cast.hpp>

namespace vex {
namespace fft {


/// Return the greatest k = 2^m <= n
static int pow2_floor(int n) {
    if(n == 0) return 0;
    for(size_t m = 0 ; ; m++)
        if((1 << (m + 1)) > n)
            return 1 << m;
}

extern const char r2c_f_oper[] = "return (float2)(prm1, 0);";
extern const char r2c_d_oper[] = "return (double2)(prm1, 0);";
extern const char c2r_oper[] = "return prm1.x;";

template <class T>
struct helpers {};

template <>
struct helpers<cl_float> {
    typedef UserFunction<r2c_f_oper, cl_float2(cl_float)> r2c;
    typedef UserFunction<c2r_oper, cl_float(cl_float2)> c2r;
};

template <>
struct helpers<cl_double> {
    typedef UserFunction<r2c_d_oper, cl_double2(cl_double)> r2c;
    typedef UserFunction<c2r_oper, cl_double(cl_double2)> c2r;
};




template <class T>
struct plan {
    typedef typename cl::vector_of<T, 2>::type T2;

    typename helpers<T>::r2c r2c;
    typename helpers<T>::c2r c2r;

    const std::vector<cl::CommandQueue> &queues;
    T scale;

    struct kernel_call {
        cl::Kernel kernel;
        cl::NDRange global, local;
        kernel_call(cl::Kernel k, cl::NDRange g, cl::NDRange l) : kernel(k), global(g), local(l) {}
    };
    std::vector<kernel_call> kernels;

    vector<T2> temp[2];
    size_t input, output;

    // \param sizes
    //  1D case: {n}.
    //  2D case: {h, w} in row-major format: x + y * w. (like FFTw)
    //  etc.
    plan(const std::vector<cl::CommandQueue> &queues, const std::vector<size_t> sizes, bool inverse)
        : queues(queues) {
        assert(sizes.size() >= 1);
        assert(queues.size() == 1);
        auto queue = queues[0];
        auto context = qctx(queue);
        auto device = qdev(queue);

        size_t total_n = 1;
        for(auto x : sizes) total_n *= x;
        scale = inverse ? ((T)1 / total_n) : 1;

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
                    size_t m = w / radix;
                    auto kernel = radix_kernel<T>(context, inverse, radix, p, temp[current](0), temp[other](0));
                    size_t wg = pow2_floor(std::min(m, (size_t)kernel_workgroup_size(kernel, device)));
                    kernels.emplace_back(kernel, cl::NDRange(m, h), cl::NDRange(wg, 1));
                    std::swap(current, other);
                    p *= radix;
                }
            }

            // transpose.
            if(w > 1 && h > 1) {
                const size_t block_dim = 16;
                auto kernel = transpose_kernel<T>(context, w, h, block_dim, temp[current](0), temp[other](0));
                kernels.emplace_back(kernel, cl::NDRange(w, h),
                    cl::NDRange(std::min(w, block_dim), std::min(h, block_dim)));
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
    
    /// Execute the complete transformation.
    /// Converts real-valued input and output, supports multiply-adding to output.
    template <class In, class Out>
    void operator()(const vector<In> &in, vector<Out> &out, bool append, T ex_scale) {
        static_assert(std::is_same<In, T>::value || std::is_same<In, T2>::value, "Invalid input type.");
        static_assert(std::is_same<Out, T>::value || std::is_same<Out, T2>::value, "Invalid output type.");
        if(std::is_same<In, T>::value) temp[input] = r2c(in);
        else temp[input] = in;
        for(auto run : kernels)
            queues[0].enqueueNDRangeKernel(run.kernel, cl::NullRange,
                run.global, run.local);
        if(std::is_same<Out, T>::value) {
            if(append) out += c2r(temp[output]) * (ex_scale * scale);
            else out = c2r(temp[output]) * (ex_scale * scale);
        } else {
            if(append) out += temp[output] * (ex_scale * scale);
            else out = temp[output] * (ex_scale * scale);
        }
    }
};


}
}
#endif
