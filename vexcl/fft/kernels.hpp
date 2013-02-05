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
void in_place_dft(std::ostringstream &o, bool invert, size_t radix) {
    // inline DFT macro.
    if(radix == 2) {
        o << "#define DFT2(v0,v1) { \\\n"
             "\t\treal2_t tmp = v0 - v1; \\\n"
             "\t\tv0 += v1; \\\n"
             "\t\tv1 = tmp; \\\n"
             "\t}\n\n";
    } else {
        const size_t half_radix = radix / 2;
        // parameters
        o << "#define DFT" << radix;
        param_list(o, 0, radix);
        o << "{ \\\n";
        // leaves
        for(size_t i = 0 ; i < half_radix ; i++) {
            o << "\t\tDFT2(v" << i << ",v" << (i + half_radix) << "); \\\n";
            if(i != 0) {
                const size_t j = i + half_radix;
                if(2 * i == half_radix) {
                    o << "\t\tv" << j << " = twiddle_1_2(v" << j << "); \\\n";
                } else {
                    T factor = (invert ? 1 : -1) * (T)M_PI * i / half_radix;
                    typename cl_vector_of<T,2>::type twiddle =
                        {{std::cos(factor), std::sin(factor)}};
                    o << "\t\tv" << j << " = mul(v" << j << ',' << std::setprecision(25) << twiddle << "); \\\n";
                }
            }
        }
        // next stage
        o << "\t\tDFT" << half_radix; param_list(o, 0, half_radix); o << "; \\\n";
        o << "\t\tDFT" << half_radix ; param_list(o, half_radix, radix); o << "; \\\n";
        o << "\t}\n\n";
    }
}

template <class T>
void kernel_radix(std::ostringstream &o, bool invert, size_t radix, size_t p, size_t threads) {
    for(size_t r = radix ; r >= 2 ; r /= 2)
        in_place_dft<T>(o, invert, r);

    // kernel.
    o << "kernel void radix(global const real2_t *x, global real2_t *y) {\n";
    o << "\tconst size_t i = get_global_id(0);\n"
      << "\tconst size_t k = i & " << (p - 1) << ";\n" // index in input sequence, in 0..P-1
      << "\tconst size_t j = ((i - k) * " << radix << ") + k;\n" // output index
      << "\tconst size_t batch_offset = get_global_id(1) * " << (threads * radix) << ";\n"
      << "\tx += i + batch_offset; y += j + batch_offset;\n";

    // read
    for(size_t i = 0 ; i < radix ; i++)
        o << "\treal2_t v" << i << " = x[" << (i * threads) << "];\n";
    // twiddle
    for(size_t i = 1 ; i < radix ; i++) {
        const T alpha = -static_cast<T>(M_PI) * i / (p * radix / 2);
        o << "\tv" << i << "=twiddle(v" << i << ",(real_t)" << std::setprecision(25) << alpha << " * k);\n";
    }
    // inplace DFT
    o << "\tDFT" << radix; param_list(o, 0, radix); o << ";\n";
    // write back
    for(size_t i = 0 ; i < radix ; i++) {
        size_t j = bit_reverse(i, log2(radix));
        o << "\ty[" << (i * p) << "]=v" << j << ";\n";
    }
    o << "}\n\n";
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
kernel_call radix_kernel(cl::CommandQueue &queue, size_t n, size_t batch, bool invert, size_t radix, size_t p, cl::Buffer in, cl::Buffer out) {
    std::ostringstream o;
    kernel_common<T>(o);

    // Return A*B (complex multiplication)
    o << "real2_t mul(real2_t a, real2_t b) {\n"
         "\treturn (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n"
         "}\n\n";

    // A * exp(alpha * I) == A  * (cos(alpha) + I * sin(alpha))
    o << "real2_t twiddle(real2_t a, real_t alpha) {\n"
      << "\treal_t cs, sn;\n"
      << "\tsn = sincos(" << (invert ? '-' : ' ') << "alpha, &cs);\n"
      << "\treturn mul(a, (real2_t)(cs, sn));\n"
      << "}\n\n";

    // real(twiddle) = 0
    o << "real2_t twiddle_1_2(real2_t a) {\n"
      << "\treturn (real2_t)(" << (invert ? "-a.y, a.x" : "a.y, -a.x") << ");\n}\n\n";

    const size_t m = n / radix;
    kernel_radix<T>(o, invert, radix, p, m);

#ifdef VEXCL_SHOW_KERNELS
    std::cout << o.str() << std::endl;
#endif

    auto program = build_sources(qctx(queue), o.str(), "-cl-mad-enable -cl-fast-relaxed-math");
    cl::Kernel kernel(program, "radix");
    kernel.setArg(0, in);
    kernel.setArg(1, out);

    size_t wg = pow2_floor(std::min(m,
        (size_t)kernel_workgroup_size(kernel, qdev(queue))));

    return kernel_call(program, kernel, cl::NDRange(m, batch), cl::NDRange(wg, 1));
}


template <class T>
kernel_call transpose_kernel(cl::CommandQueue &queue, size_t width, size_t height, cl::Buffer in, cl::Buffer out) {
    std::ostringstream o;
    kernel_common<T>(o);

    // determine max block size to fit into local memory.
    size_t block_size = 128;
    const auto local_size = qdev(queue).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    while(block_size * block_size * sizeof(T) * 2 > local_size) block_size /= 2;
    block_size /= 2;

    // from NVIDIA SDK.
    o << "kernel void transpose(\n"
        "\tglobal const real2_t *input, global real2_t *output) {\n"
        "\tconst size_t "
        "\tglobal_x = get_global_id(0), global_y = get_global_id(1),\n"
        "\t\tlocal_x = get_local_id(0), local_y = get_local_id(1),\n"
        "\t\tgroup_x = get_group_id(0), group_y = get_group_id(1),\n"
        "\t\twidth =" << width << ", height = " << height << ", block_size = " << block_size << ",\n"
        "\t\ttarget_x = local_y + group_y * block_size,\n"
        "\t\ttarget_y = local_x + group_x * block_size;\n"
        // local memory
        "\tlocal real2_t block[" << (block_size * block_size) << "];\n"
        // copy from input to local memory
        "\tblock[local_x + local_y * block_size] = input[global_x + global_y * width];\n"
        // wait until the whole block is filled
        "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"
        // transpose local block to target
        "\toutput[target_x + target_y * height] = block[local_x + local_y * block_size];\n"
        "}\n\n";

#ifdef VEXCL_SHOW_KERNELS
    std::cout << o.str() << std::endl;
#endif

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
