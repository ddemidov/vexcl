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

// Store v=b^e as components.
struct pow {
    size_t base, exponent, value;
    pow(size_t b, size_t e) : base(b), exponent(e), value(std::pow(b, e)) {}
};

inline std::ostream &operator<<(std::ostream &o, const pow &p) {
    return o << p.base << '^' << p.exponent << '=' << p.value;
}

/// Reverse the digits of a number in the base given.
/** 0 <= i < m,
 * i is converted into a base m.base-number,
 * its digits reversed, and converted back into an integer. */
size_t digit_reverse(size_t i, pow m) {
    assert(i < m.value);
    // convert to other base, digits[0]=least significant
    // convert back with digits[0]=most significant.
    size_t j = 0;
    for(size_t k = 0 ; k < m.exponent ; k++) {
        j = j * m.base + (i % m.base);
        i /= m.base;
    }
    return j;
}

/// a is a power of 2, b is an arbitrary number.
/** returns a power of 2 <= a that is <= b. */
size_t min_pow2(size_t a, size_t b) {
    while(a > b)
        a >>= 1;
    return a;
}

/// ceil(x/m) * m
size_t int_ceil(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}


struct kernel_call {
    std::string desc;
    cl::Program program;
    cl::Kernel kernel;
    cl::NDRange global, local;
    kernel_call(std::string d, cl::Program p, cl::Kernel k, cl::NDRange g, cl::NDRange l) : desc(d), program(p), kernel(k), global(g), local(l) {}
};




// generates "(prefix vfrom,vfrom+1,...,vto)"
inline void param_list(std::ostringstream &o, std::string prefix, size_t from, size_t to, size_t step = 1) {
    o << '(';
    for(size_t i = from ; i != to ; i += step) {
        if(i != from) o << ", ";
        o << prefix << 'v' << i;
    } o << ')';
}

// omega(n,k) = exp(-i 2 pi k / n)
template<class T>
inline typename cl_vector_of<T,2>::type omega(int k, int n) {
    const T alpha = -2 * M_PI * (k % n) / n;
    return {{std::cos(alpha), std::sin(alpha)}};
}

template <class T>
void in_place_dft(std::ostringstream &o, pow radix) {
    typedef typename cl_vector_of<T,2>::type T2;
    o << "void dft" << radix.value;
    param_list(o, "real2_t *", 0, radix.value);
    o << "{\n";

    if(radix.exponent == 1) { // prime bases
        if(radix.value == 2) {
            o << "  real2_t tmp = *v0 - *v1;\n"
              << "  *v0 += *v1;\n"
              << "  *v1 = tmp;\n";
        } else {
            // naive DFT
            for(size_t i = 1 ; i < radix.base ; i++)
                o << "  real2_t w" << i << " = " << omega<T>(i, radix.base) << ";\n";
            for(size_t i = 0 ; i < radix.base ; i++) {
                o << "  real2_t t" << i << " = *v0";
                for(size_t j = 1 ; j < radix.base ; j++) {
                    o << " + ";
                    if(i == 0)
                        o << "*v" << j;
                    else
                        o << "mul(*v" << j << ", w" << ((i * j) % radix.base) << ")";
                }
                o << ";\n";
            }
            // write back
            for(size_t i = 0 ; i < radix.base ; i++)
                o << "  *v" << i << " = t" << i << ";\n";
        }
    } else { // recursive prime power DFTs
        const size_t prev_radix = radix.value / radix.base;
        // leaves
        for(size_t i = 0 ; i < prev_radix ; i++) {
            o << "  dft" << radix.base;
            param_list(o, "", i, i + radix.value, prev_radix);
            o << ";\n";
        }
        // twiddle
        for(size_t i = 1 ; i <= (prev_radix-1) * (radix.base-1) ; i++)
            o << "  real2_t w" << i << " = " << omega<T>(i, radix.value) << ";\n";
        for(size_t i = 1 ; i < prev_radix ; i++) {
            for(size_t j = 1 ; j < radix.base ; j++) {
                const size_t k = i + j * prev_radix;
                o << "  *v" << k << " = mul(*v" << k << ", w" << ((i * j) % radix.value) << ");\n";
            }
        }
        // next stage
        for(size_t i = 0 ; i < radix.base ; i++) {
            o << "  dft" << prev_radix;
            param_list(o, "", i * prev_radix, (i + 1) * prev_radix);
            o << ";\n";
        }
    }
    o << "}\n";
}


template <class T>
void kernel_radix(std::ostringstream &o, pow radix) {
    for(size_t e = 1 ; e <= radix.exponent ; e++)
        in_place_dft<T>(o, pow(radix.base, e));
    // kernel.
    o << "__kernel void radix(__global const real2_t *x, __global real2_t *y, uint p) {\n"
      << "  const size_t i = get_global_id(0);\n"
      << "  const size_t threads = get_global_size(0);\n"
        // index in input sequence, in 0..P-1
      << "  const size_t k = i % p;\n"
      << "  const size_t batch_offset = get_global_id(1) * threads * " << radix.value << ";\n";

    // read
    o << "  x += i + batch_offset;\n";
    for(size_t i = 0 ; i < radix.value ; i++)
        o << "  real2_t v" << i << " = x[" << i << " * threads];\n";

    // twiddle
    o << "  if(p != 1) {\n";
    for(size_t i = 1 ; i < radix.value ; i++) {
        const T alpha = -2 * M_PI * i / radix.value;
        o << "    v" << i << " = twiddle(v" << i << ", "
          << "(real_t)" << alpha << " * k / p);\n";
    }
    o << "  }\n";

    // inplace DFT
    o << "  dft" << radix.value;
    param_list(o, "&", 0, radix.value);
    o << ";\n";

    // write back
    o << "  const size_t j = k + (i - k) * " << radix.value << ";\n";
    o << "  y += j + batch_offset;\n";
    for(size_t i = 0 ; i < radix.value ; i++)
        o << "  y[" << i << " * p] = v" << digit_reverse(i, radix) << ";\n";
    o << "}\n";
}


template <class T>
void kernel_common(std::ostringstream &o) {
    if(std::is_same<T, cl_double>::value) {
        o << standard_kernel_header
          << "typedef double real_t;\n"
          << "typedef double2 real2_t;\n";
    } else {
        o << "typedef float real_t;\n"
          << "typedef float2 real2_t;\n";
    }
}


template <class T>
kernel_call radix_kernel(cl::CommandQueue &queue, size_t n, size_t batch, bool invert, pow radix, size_t p, cl::Buffer in, cl::Buffer out) {
    std::ostringstream o;
    o << std::setprecision(25);
    kernel_common<T>(o);

    // Return A*B (complex multiplication)
    o << "real2_t mul(real2_t a, real2_t b) {\n";
    if(invert) // conjugate b
        o << "  return (real2_t)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);\n";
    else
        o << "  return (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n";
    o << "}\n";

    // A * exp(alpha * I) == A  * (cos(alpha) + I * sin(alpha))
    // native_cos(), native_sin() is a *lot* faster than sincos, on nVidia.
    o << "real2_t twiddle(real2_t a, real_t alpha) {\n";
    o << "  real_t cs, sn;\n";
    if(std::is_same<T, cl_double>::value)
        // use sincos with double since we probably want higher precision
        o << "  sn = sincos(alpha, &cs);\n";
    else
        // use native with float since we probably want higher performance
        o << "  cs = native_cos(alpha); sn = native_sin(alpha);\n";
    o << "  return mul(a, (real2_t)(cs, sn));\n"
      << "}\n";

    const size_t m = n / radix.value;
    kernel_radix<T>(o, radix);

#ifdef VEXCL_SHOW_KERNELS
    std::cout << o.str() << std::endl;
#endif

    auto program = build_sources(qctx(queue), o.str(), "-cl-mad-enable -cl-fast-relaxed-math");
    cl::Kernel kernel(program, "radix");
    kernel.setArg(0, in);
    kernel.setArg(1, out);
    kernel.setArg<cl_uint>(2, p);

    std::ostringstream desc;
    desc << "dft{r=" << radix << ", p=" << p << ", n=" << n << ", batch=" << batch << ", threads=" << m << "}";

    return kernel_call(desc.str(), program, kernel, cl::NDRange(m, batch), cl::NullRange);
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
      << "__global const real2_t *input, __global real2_t *output, uint width, uint height) {\n"
      << "  const size_t "
      << "    global_x = get_global_id(0), global_y = get_global_id(1),\n"
      << "    local_x = get_local_id(0), local_y = get_local_id(1),\n"
      << "    group_x = get_group_id(0), group_y = get_group_id(1),\n"
      << "    block_size = " << block_size << ",\n"
      << "    target_x = local_y + group_y * block_size,\n"
      << "    target_y = local_x + group_x * block_size;\n"
      << "  const bool range = global_x < width && global_y < height;\n"
        // local memory
      << "  __local real2_t block[" << (block_size * block_size) << "];\n"
        // copy from input to local memory
      << "  if(range)\n"
      << "    block[local_x + local_y * block_size] = input[global_x + global_y * width];\n"
        // wait until the whole block is filled
      << "  barrier(CLK_LOCAL_MEM_FENCE);\n"
        // transpose local block to target
      << "  if(range)\n"
      << "    output[target_x + target_y * height] = block[local_x + local_y * block_size];\n"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "transpose");
    kernel.setArg(0, in);
    kernel.setArg(1, out);
    kernel.setArg<cl_uint>(2, width);
    kernel.setArg<cl_uint>(3, height);

    // range multiple of wg size, last block maybe not completely filled.
    size_t r_w = int_ceil(width, block_size);
    size_t r_h = int_ceil(height, block_size);

    std::ostringstream desc;
    desc << "transpose{"
         << "w=" << width << "(" << r_w << "), "
         << "h=" << height << "(" << r_h << "), "
         << "bs=" << block_size << "}";

    return kernel_call(desc.str(), program, kernel, cl::NDRange(r_w, r_h),
        cl::NDRange(block_size, block_size));
}



} // namespace fft
} // namespace vex


#endif
