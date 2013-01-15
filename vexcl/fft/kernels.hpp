#ifndef VEXCL_FFT_KERNELS_HPP
#define VEXCL_FFT_KERNELS_HPP

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
 * \file   kernels.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Kernel generator for FFT.
 */


// OpenCL FFT kernels, original by Eric Bainville

namespace vex {
namespace fft {

// return the `s` most significant bits from an integer in reverse.
// <http://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious>
template <class T>
T bit_reverse(T v, size_t s = sizeof(T) * CHAR_BIT) {
    T r = v & 1;
    s--;
    for(v >>= 1 ; v ; v >>= 1) {
        r <<= 1;
        r |= v & 1;
        s--;
    }
    return r << s;
}

// calculate log2(2^k) = k.
template <class T>
T log2(T n) {
    int k = 0;
    if(n <= 0) throw std::logic_error("not a positive power of 2.");
    while(n != 1) {
        if(n & 1) throw std::logic_error("not a power of 2.");
        n >>= 1;
        k++;
    }
    return k;
}


// generates "(vfrom,vfrom+1,...,vto)"
void param_list(std::ostringstream &o, size_t from, size_t to) {
    o << '(';
    for(size_t i = from ; i != to ; i++) {
        if(i != from) o << ',';
        o << 'v' << i;
    } o << ')';
}

template <class T>
void kernel_radix(std::ostringstream &o, size_t radix, bool invert) {
    typedef typename cl::vector_of<T,2>::type T2;
    const size_t log2_radix = log2(radix);

    // inline DFT macro.
    if(radix == 2) {
        o << R"(#define DFT2(v0,v1) { \
                real2_t tmp = v0 - v1; \
                v0 += v1; \
                v1 = tmp; \
            }
        )";
    } else {
        const size_t half_radix = radix / 2;
        // parameters
        o << "#define DFT" << radix;
        param_list(o, 0, radix);
        o << '{';
        // leaves
        for(size_t i = 0 ; i < half_radix ; i++) {
            o << "DFT2(v" << i << ",v" << (i + half_radix) << ");";
            if(i != 0) {
                const size_t j = i + half_radix;
                if(2 * i == half_radix) {
                    o << 'v' << j << "=twiddle_1_2(v" << j << ");";
                } else {
                    T factor = (invert ? 1 : -1) * (T)M_PI * i / half_radix;
                    T2 twiddle = {{std::cos(factor), std::sin(factor)}};            
                    o << 'v' << j << "=mul(v" << j << ',' << std::setprecision(25) << twiddle << ");";
                }
            }
        }
        // next stage
        o << "DFT" << half_radix; param_list(o, 0, half_radix); o << ';';
        o << "DFT" << half_radix ; param_list(o, half_radix, radix); o << ';';
        o << "}\n";
    }

    // kernel.
    o << "__kernel void radix" << radix << "(__global const real2_t *x, __global real2_t *y, uint p) {";
    o << "const size_t threads = get_global_size(0);"
        << "const size_t i = get_global_id(0);"
        << "const size_t k = i & (p - 1);" // index in input sequence, in 0..P-1
        << "const size_t j = ((i - k) * " << radix << ") + k;" // output index
        << "const size_t batch_offset = get_global_id(1) * threads * " << radix << ';'
        << "x += i + batch_offset; y += j + batch_offset;"
        << "real_t alpha = -FFT_PI * (real_t)k / (real_t)(p * " << radix << "/ 2);";

    // read
    o << "real2_t v0 = x[0];";
    for(size_t i = 1 ; i < radix ; i++)
        o << "real2_t v" << i << "=twiddle(x[" << i << "* threads]," << i << ",alpha);";
    // inplace DFT
    o << "DFT" << radix; param_list(o, 0, radix); o << ';';
    // write back
    for(size_t i = 0 ; i < radix ; i++) {
        size_t j = bit_reverse(i, log2_radix);
        o << "y[" << i << "*p]=v" << j << ';';
    }
    o << "}\n";
}



template <class T>
std::string kernel_code(bool invert, std::vector<size_t> radixes) {
    std::ostringstream o;
    if(std::is_same<T, cl_double>::value) {
        o << standard_kernel_header;
        o << "typedef double real_t; typedef double2 real2_t;\n";
        o << "#define FFT_PI M_PI\n";
    } else {
        o << "typedef float real_t; typedef float2 real2_t;\n";
        o << "#define FFT_PI M_PI_F\n";
    }

    // from NVIDIA SDK.
    o << R"(
        __kernel void transpose(
            __global const real2_t *input, __global real2_t *output, __local real2_t *block,
            const uint width, const uint height, const uint block_size) {
            const size_t
                global_x = get_global_id(0), global_y = get_global_id(1),
                local_x = get_local_id(0), local_y = get_local_id(1),
                group_x = get_group_id(0), group_y = get_group_id(1),
                target_x = local_y + group_y * block_size,
                target_y = local_x + group_x * block_size;
            // copy from input to local memory
            block[local_x + local_y * block_size] = input[global_x + global_y * width];
            // wait until the whole block is filled
            barrier(CLK_LOCAL_MEM_FENCE);
            // transpose local block to target
            output[target_x + target_y * height] = block[local_x + local_y * block_size];
        }
    )";

    // Return A*B (complex multiplication)
    o << R"(
        real2_t mul(real2_t a, real2_t b) {
            return (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
    )";

    // A * exp(k * alpha * I) == cos(k * alpha) + I * sin(k * alpha)
    o << "real2_t twiddle(real2_t a, int k, real_t alpha) {"
        << "real_t cs, sn;"
        << "sn = sincos(" << (invert ? '-' : ' ') << "k * alpha, &cs);"
        << "return mul(a, (real2_t)(cs, sn));"
        << "}";

    // real(twiddle) = 0
    o << "real2_t twiddle_1_2(real2_t a){"
        << "return (real2_t)(" << (invert ? "-a.y, a.x" : "a.y, -a.x") << ");}\n";

    // kernels.
    for(auto r : radixes)
        kernel_radix<T>(o, r, invert);

    return o.str();
}



} // namespace fft
} // namespace vex


#endif
