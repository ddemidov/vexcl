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

#include <cmath>

namespace vex {
namespace fft {

// Store v=b^e as components.
struct pow {
    size_t base, exponent, value;
    pow(size_t b, size_t e) : base(b), exponent(e),
	value(static_cast<size_t>(std::pow(static_cast<double>(b), static_cast<double>(e)))) {}
};

inline std::ostream &operator<<(std::ostream &o, const pow &p) {
    o << p.base;
    if(p.exponent != 1) o << '^' << p.exponent;
    return o;
}

/// ceil(x/m) * m
inline size_t int_ceil(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}


struct kernel_call {
    bool once;
    size_t count;
    std::string desc;
    cl::Program program;
    cl::Kernel kernel;
    cl::NDRange global, local;
    kernel_call(bool o, std::string d, cl::Program p, cl::Kernel k, cl::NDRange g, cl::NDRange l) : once(o), count(0), desc(d), program(p), kernel(k), global(g), local(l) {}
};




// generates "(prefix vfrom,vfrom+1,...,vto)"
inline void param_list(std::ostringstream &o, std::string prefix, size_t from, size_t to, size_t step = 1) {
    o << '(';
    for(size_t i = from ; i != to ; i += step) {
        if(i != from) o << ", ";
        o << prefix << 'v' << i;
    } o << ')';
}

template <class T>
inline void kernel_radix(std::ostringstream &o, pow radix, bool invert) {
    o << in_place_dft(radix.value, invert);

    // kernel.
    o << "__kernel void radix(__global const real2_t *x, __global real2_t *y, uint p, uint threads) {\n"
      << "  const size_t i = get_global_id(0);\n"
      << "  if(i >= threads) return;\n"
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
        const T alpha = -2 * static_cast<T>(M_PI) * i / radix.value;
        o << "    v" << i << " = mul(v" << i << ", twiddle("
          << "(real_t)" << alpha << " * k / p));\n";
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
        o << "  y[" << i << " * p] = v" << i << ";\n";
    o << "}\n";
}


template <class T>
inline void kernel_common(std::ostringstream &o) {
    if(std::is_same<T, cl_double>::value) {
        o << standard_kernel_header
          << "typedef double real_t;\n"
          << "typedef double2 real2_t;\n";
    } else {
        o << "typedef float real_t;\n"
          << "typedef float2 real2_t;\n";
    }
}

inline void mul_code(std::ostringstream &o, bool invert) {
    // Return A*B (complex multiplication)
    o << "real2_t mul(real2_t a, real2_t b) {\n";
    if(invert) // conjugate b
        o << "  return (real2_t)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);\n";
    else
        o << "  return (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n";
    o << "}\n";
}

template <class T>
inline void twiddle_code(std::ostringstream &o) {
    // A * exp(alpha * I) == A  * (cos(alpha) + I * sin(alpha))
    // native_cos(), native_sin() is a *lot* faster than sincos, on nVidia.
    o << "real2_t twiddle(real_t alpha) {\n";
    if(std::is_same<T, cl_double>::value)
        // use sincos with double since we probably want higher precision
        o << "  real_t cs, sn = sincos(alpha, &cs);\n"
          << "  return (real2_t)(cs, sn);\n";
    else
        // use native with float since we probably want higher performance
        o << "  return (real2_t)(native_cos(alpha), native_sin(alpha));\n";
    o << "}\n";
}


template <class T>
inline kernel_call radix_kernel(bool once, const cl::CommandQueue &queue, size_t n, size_t batch, bool invert, pow radix, size_t p, const cl::Buffer &in, const cl::Buffer &out) {
    std::ostringstream o;
    o << std::setprecision(25);
    kernel_common<T>(o);
    mul_code(o, invert);
    twiddle_code<T>(o);

    const size_t m = n / radix.value;
    kernel_radix<T>(o, radix, invert);

    auto program = build_sources(qctx(queue), o.str(), "-cl-mad-enable -cl-fast-relaxed-math");
    cl::Kernel kernel(program, "radix");
    kernel.setArg(0, in);
    kernel.setArg(1, out);
    kernel.setArg<cl_uint>(2, p);
    kernel.setArg<cl_uint>(3, m);

    const auto device = qdev(queue);
    const size_t wg_mul = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
    //const size_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    //const size_t max_wg = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    size_t wg = wg_mul;
    //while(wg * max_cu < max_wg) wg += wg_mul;
    //wg -= wg_mul;
    const size_t threads = int_ceil(m, wg);

    std::ostringstream desc;
    desc << "dft{r=" << radix << ", p=" << p << ", n=" << n << ", batch=" << batch << ", threads=" << m << "(" << threads << "), wg=" << wg << ", in=" << in() << ", out=" << out() << "}";

    return kernel_call(once, desc.str(), program, kernel, cl::NDRange(threads, batch), cl::NDRange(wg, 1));
}


template <class T>
inline kernel_call transpose_kernel(const cl::CommandQueue &queue, size_t width, size_t height, const cl::Buffer &in, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);

    // determine max block size to fit into local memory/workgroup
    size_t block_size = 128;
    {
        const auto dev = qdev(queue);
        const auto local_size = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        const auto workgroup = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        while(block_size * block_size * sizeof(T) * 2 > local_size) block_size /= 2;
        while(block_size * block_size > workgroup) block_size /= 2;
    }

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
         << "bs=" << block_size << ", in=" << in() << ", out=" << out() << "}";

    return kernel_call(false, desc.str(), program, kernel, cl::NDRange(r_w, r_h),
        cl::NDRange(block_size, block_size));
}



template <class T>
inline kernel_call bluestein_twiddle(const cl::CommandQueue &queue, cl_uint n, bool inverse, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);
    twiddle_code<T>(o);

    o << standard_kernel_header;

    o << "__kernel void bluestein_twiddle(__global real2_t *output) {\n"
      << "  const size_t x = get_global_id(0), n = get_global_size(0);\n"
      << "  const int sign = " << (inverse ? "+1" : "-1") << ";\n"
      << "  const size_t xx = ((ulong)x * x) % (2 * n);\n"
      << "  output[x] = twiddle(sign * M_PI * xx / n);\n"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "bluestein_twiddle");
    kernel.setArg(0, out);

    std::ostringstream desc;
    desc << "bluestein_twiddle{n=" << n << ", inverse=" << inverse << ", out=" << out() << "}";
    return kernel_call(true, desc.str(), program, kernel, cl::NDRange(n), cl::NullRange);
}

template <class T>
inline kernel_call bluestein_pad_kernel(const cl::CommandQueue &queue, size_t n, size_t m, const cl::Buffer &in, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);

    o << "real2_t conj(real2_t v) {\n"
      << "  return (real2_t)(v.x, -v.y);\n"
      << "}\n";
    o << "__kernel void bluestein_pad_kernel("
      << "__global real2_t *input, __global real2_t *output, uint n, uint m) {\n"
      << "  const size_t x = get_global_id(0);\n"
      << "  if(x < n || m - x < n)\n"
      << "    output[x] = conj(input[min(x, m - x)]);\n"
      << "  else\n"
      << "    output[x] = (real2_t)(0,0);\n"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "bluestein_pad_kernel");
    kernel.setArg(0, in);
    kernel.setArg(1, out);
    kernel.setArg<cl_uint>(2, n);
    kernel.setArg<cl_uint>(3, m);

    std::ostringstream desc;
    desc << "bluestein_pad_kernel{n=" << n << ", m=" << m << ", in=" << in() << ", out=" << out() << "}";
    return kernel_call(true, desc.str(), program, kernel, cl::NDRange(m), cl::NullRange);
}

template <class T>
inline kernel_call bluestein_mul_in(const cl::CommandQueue &queue, bool inverse, cl_uint batch, cl_uint radix, cl_uint p, cl_uint threads, cl_uint stride, const cl::Buffer &data, const cl::Buffer &exp, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);
    mul_code(o, false);
    twiddle_code<T>(o);

    o << "__kernel void bluestein_mul_in("
      << "__global const real2_t *data, __global const real2_t *exp, __global real2_t *output, "
      << "uint radix, uint p) {\n"
      << "  const size_t\n"
      << "    thread = get_global_id(0), threads = get_global_size(0),\n"
      << "    batch = get_global_id(1),\n"
      << "    element = get_global_id(2), out_stride = get_global_size(2),\n"
      << "    in_off = thread + batch * radix * threads + element * threads,\n"
      << "    out_off = thread * out_stride + batch * out_stride * threads + element;\n"
      << "  if(element < radix) {\n"
      << "    real2_t w = exp[element];"
      << "    if(p != 1) {\n"
      << "      const int sign = " << (inverse ? "+1" : "-1") << ";\n"
      << "      ulong a = (ulong)element * (thread % p);\n"
      << "      ulong b = (ulong)radix * p;\n"
      << "      real2_t t = twiddle(2 * sign * M_PI * (a % (2 * b)) / b);\n"
      << "      w = mul(w, t);\n"
      << "    }\n"
      << "    output[out_off] = mul(data[in_off], w);\n"
      << "  } else\n"
      << "    output[out_off] = (real2_t)(0,0);"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "bluestein_mul_in");
    kernel.setArg(0, data);
    kernel.setArg(1, exp);
    kernel.setArg(2, out);
    kernel.setArg(3, radix);
    kernel.setArg(4, p);

    std::ostringstream desc;
    desc << "bluestein_mul_in{batch=" << batch << ", radix=" << radix << ", p=" << p << ", threads=" << threads << ", stride=" << stride << ", data=" << data() << ", exp=" << exp() << ", out=" << out() << "}";
    return kernel_call(false, desc.str(), program, kernel, cl::NDRange(threads, batch, stride), cl::NullRange);
}

template <class T>
inline kernel_call bluestein_mul_out(const cl::CommandQueue &queue, cl_uint batch, cl_uint p, cl_uint radix, cl_uint threads, cl_uint stride, const cl::Buffer &data, const cl::Buffer &exp, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);
    mul_code(o, false);

    o << "__kernel void bluestein_mul_out("
      << "__global const real2_t *data, __global const real2_t *exp, __global real2_t *output, "
      << "real_t div, uint p, uint in_stride) {\n"
      << "  const size_t\n"
      << "    i = get_global_id(0), threads = get_global_size(0),\n"
      << "    b = get_global_id(1),\n"
      << "    l = get_global_id(2), radix = get_global_size(2),\n"
      << "    k = i % p,\n"
      << "    j = k + (i - k) * radix,\n"
      << "    in_off = i * in_stride + b * in_stride * threads + l,\n"
      << "    out_off = j + b * threads * radix + l * p;\n"
      << "  output[out_off] = mul(data[in_off] * div, exp[l]);\n"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "bluestein_mul_out");
    kernel.setArg(0, data);
    kernel.setArg(1, exp);
    kernel.setArg(2, out);
    kernel.setArg<T>(3, 1.0 / stride);
    kernel.setArg(4, p);
    kernel.setArg(5, stride);

    std::ostringstream desc;
    desc << "bluestein_mul_out{r=" << radix << ", batch=" << batch << ", p=" << p << ", rx=" << radix << ", thr=" << threads << ", stride=" << stride << ", data=" << data() << ", exp=" << exp() << ", out=" << out() << "}";
    return kernel_call(false, desc.str(), program, kernel, cl::NDRange(threads, batch, radix), cl::NullRange);
}

template <class T>
inline kernel_call bluestein_mul(const cl::CommandQueue &queue, cl_uint n, cl_uint batch, const cl::Buffer &data, const cl::Buffer &exp, const cl::Buffer &out) {
    std::ostringstream o;
    kernel_common<T>(o);
    mul_code(o, false);

    o << "__kernel void bluestein_mul("
      << "__global const real2_t *data, __global const real2_t *exp, __global real2_t *output) {\n"
      << "  const size_t x = get_global_id(0), y = get_global_id(1);\n"
      << "  const size_t stride = get_global_size(0);\n"
      << "  const size_t off = x + stride * y;"
      << "  output[off] = mul(data[off], exp[x]);\n"
      << "}\n";

    auto program = build_sources(qctx(queue), o.str());
    cl::Kernel kernel(program, "bluestein_mul");
    kernel.setArg(0, data);
    kernel.setArg(1, exp);
    kernel.setArg(2, out);

    std::ostringstream desc;
    desc << "bluestein_mul{n=" << n << ", batch=" << batch << ", data=" << data() << ", exp=" << exp() << ", out=" << out() << "}";
    return kernel_call(false, desc.str(), program, kernel, cl::NDRange(n, batch), cl::NullRange);
}


} // namespace fft
} // namespace vex


#endif
