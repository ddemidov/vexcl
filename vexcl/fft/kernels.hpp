#ifndef VEXCL_FFT_KERNELS_HPP
#define VEXCL_FFT_KERNELS_HPP

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
 * \file   fft/kernels.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Kernel generator for FFT.
 */


// OpenCL FFT kernels, original by Eric Bainville

namespace vex {
namespace fft {

struct kernel_call {
    cl::Program program;
    cl::Kernel kernel;
    cl::NDRange global, local;
    kernel_call(cl::Program p, cl::Kernel k, cl::NDRange g, cl::NDRange l) : program(p), kernel(k), global(g), local(l) {}
};

// Store v=b^e as components.
struct pow {
    size_t base, exponent, value;
    pow(size_t b, size_t e) : base(b), exponent(e), value(std::pow(b, e)) {}
};


/// Return the greatest k = 2^m <= n
static int pow2_floor(int n) {
    if(n == 0) return 0;
    for(size_t m = 0 ; ; m++)
        if((1 << (m + 1)) > n)
            return 1 << m;
}

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
void in_place_dft(std::ostringstream &o, bool invert, pow radix) {
    typedef typename cl_vector_of<T,2>::type T2;
    // inline DFT macro.
    if(radix.value == 2) {
        o << R"(#define DFT2(v0,v1) { \
                real2_t tmp = v0 - v1; \
                v0 += v1; \
                v1 = tmp; \
            }
        )";
    } else if(radix.base == 2) {
        const size_t half_radix = radix.value / 2;
        // parameters
        o << "#define DFT" << radix.value;
        param_list(o, 0, radix.value);
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
        o << "DFT" << half_radix ; param_list(o, half_radix, radix.value); o << ';';
        o << "}\n";
    } else if(radix.value == 3) {
        const T sq = std::sqrt(3) / 2;
        o <<
            "#define DFT3(v0,v1,v2) { "
            "   real2_t w0 = " << std::setprecision(25) << (T2){{-0.5, invert ? sq : -sq}} << "; "
            "   real2_t w1 = " << std::setprecision(25) << (T2){{-0.5, invert ? -sq : sq}} << "; "
            "   real2_t t0 = v0 + v1 + v2; "
            "   real2_t t1 = v0 + mul(v1, w0) + mul(v2, w1);"
            "   v2 = v0 + mul(v1, w1) + mul(v2, w0);"
            "   v0 = t0;"
            "   v1 = t1;"
            "} \n";
    }
}

template <class T>
void kernel_radix(std::ostringstream &o, bool invert, pow radix, size_t p, size_t threads) {
    for(size_t e = 0 ; e <= radix.exponent ; e++)
        in_place_dft<T>(o, invert, pow(radix.base, e));

    // kernel.
    o << "__kernel void radix(__global const real2_t *x, __global real2_t *y) {";
    o << "const size_t i = get_global_id(0);";
        // index in input sequence, in 0..P-1
    if(radix.base == 2)
        o << "const size_t k = i & " << (p - 1) << ";";
    else
        o << "const size_t k = i % " << p << ";";

    o << "const size_t j = ((i - k) * " << radix.value << ") + k;" // output index
        << "const size_t batch_offset = get_global_id(1) * " << (threads * radix.value) << ';'
        << "x += i + batch_offset; y += j + batch_offset;";

    // read
    for(size_t i = 0 ; i < radix.value ; i++)
        o << "real2_t v" << i << " = x[" << (i * threads) << "];";
    // twiddle
    for(size_t i = 1 ; i < radix.value ; i++) {
        const T alpha = -M_PI * i / (p * radix.value / radix.base);
        o << "v" << i << "=twiddle(v" << i << ",(real_t)" << std::setprecision(25) << alpha << " * k);";
    }
    // inplace DFT
    o << "DFT" << radix.value; param_list(o, 0, radix.value); o << ';';
    // write back
    if(radix.base == 2) {
        for(size_t i = 0 ; i < radix.value ; i++) {
            size_t j = bit_reverse(i, radix.exponent);
            o << "y[" << (i * p) << "]=v" << j << ';';
        }
    } else if(radix.value == 3) {
        for(size_t i = 0 ; i < radix.value ; i++)
            o << "y[" << (i * p) << "]=v" << i << ";";
    }
    o << "}\n";
}


template <class T>
void kernel_common(std::ostringstream &o) {
    if(std::is_same<T, cl_double>::value) {
        o << standard_kernel_header;
        o << "typedef double real_t; typedef double2 real2_t;\n";
        o << "#define FFT_PI M_PI\n";
    } else {
        o << "typedef float real_t; typedef float2 real2_t;\n";
        o << "#define FFT_PI M_PI_F\n";
    }
}


template <class T>
kernel_call radix_kernel(cl::CommandQueue &queue, size_t n, size_t batch, bool invert, pow radix, size_t p, cl::Buffer in, cl::Buffer out) {
    std::ostringstream o;
    kernel_common<T>(o);

    // Return A*B (complex multiplication)
    o << R"(
        real2_t mul(real2_t a, real2_t b) {
            return (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
    )";

    // A * exp(alpha * I) == A  * (cos(alpha) + I * sin(alpha))
    o << "real2_t twiddle(real2_t a, real_t alpha) {"
        << "real_t cs, sn;"
        << "sn = sincos(" << (invert ? '-' : ' ') << "alpha, &cs);"
        << "return mul(a, (real2_t)(cs, sn));"
        << "}";

    // real(twiddle) = 0
    o << "real2_t twiddle_1_2(real2_t a){"
        << "return (real2_t)(" << (invert ? "-a.y, a.x" : "a.y, -a.x") << ");}\n";

    const size_t m = n / radix.value;
    kernel_radix<T>(o, invert, radix, p, m);

    auto program = build_sources(qctx(queue), o.str(), "-cl-mad-enable -cl-fast-relaxed-math");
    cl::Kernel kernel(program, "radix");
    kernel.setArg(0, in);
    kernel.setArg(1, out);

    if(m % 2 != 0) wg = 1; // ?!

    return kernel_call(program, kernel, cl::NDRange(m, batch), cl::NDRange(wg, 1));
}


template <class T>
kernel_call transpose_kernel(cl::CommandQueue &queue, size_t width, size_t height, cl::Buffer in, cl::Buffer out) {
    std::ostringstream o;
    kernel_common<T>(o);

    // determine max block size to fit into __local memory.
    size_t block_size = 128;
    const auto local_size = qdev(queue).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    while(block_size * block_size * sizeof(T) * 2 > local_size) block_size /= 2;
    block_size /= 2;

    // from NVIDIA SDK.
    o << "__kernel void transpose("
        "__global const real2_t *input, __global real2_t *output) {"
        "const size_t "
        "global_x = get_global_id(0), global_y = get_global_id(1),"
        "local_x = get_local_id(0), local_y = get_local_id(1),"
        "group_x = get_group_id(0), group_y = get_group_id(1),"
        "width =" << width << ", height = " << height << ", block_size = " << block_size << ","
        "target_x = local_y + group_y * block_size,"
        "target_y = local_x + group_x * block_size;"
        // local memory
        "__local real2_t block[" << (block_size * block_size) << "];"
        // copy from input to local memory
        "block[local_x + local_y * block_size] = input[global_x + global_y * width];"
        // wait until the whole block is filled
        "barrier(CLK_LOCAL_MEM_FENCE);"
        // transpose local block to target
        "output[target_x + target_y * height] = block[local_x + local_y * block_size];"
        "}";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "transpose");
    kernel.setArg(0, in);
    kernel.setArg(1, out);

    return kernel_call(program, kernel, cl::NDRange(width, height),
        cl::NDRange(std::min(width, block_size), std::min(height, block_size)));
}



} // namespace fft
} // namespace vex


#endif
