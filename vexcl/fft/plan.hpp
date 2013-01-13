#ifndef VEXCL_FFT_PLAN_HPP
#define VEXCL_FFT_PLAN_HPP

#include <vexcl/vector.hpp>
#include <vexcl/fft/kernels.hpp>
#include <boost/lexical_cast.hpp>

namespace vex {
namespace fft {


/// Return the greatest k = 2^m <= n
size_t pow2_floor(size_t n) {
    if(n == 0) return 0;
    for(size_t m = 0 ; ; m++)
        if(std::pow(2, m + 1) > n)
            return std::pow(2, m);
    throw std::runtime_error("overflow");
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

    typedef typename helpers<T>::r2c r2c;
    typedef typename helpers<T>::c2r c2r;

    const std::vector<cl::CommandQueue> &queue;
    cl::Program program;
    const std::vector<size_t> sizes;
    const bool inverse;

    std::map<size_t, cl::Kernel> radix_k;
    cl::Kernel transpose_k;

    vector<T2> temp[2];
    size_t current, other;
    size_t total_n;

    std::vector<size_t> radixes;

    // \param sizes
    //  1D case: {n}.
    //  2D case: {h, w} in row-major format: x + y * w. (like FFTw)
    //  etc.
    plan(const std::vector<cl::CommandQueue> &queues, const std::vector<size_t> sizes, bool inverse)
        : queue(queues), sizes(sizes), inverse(inverse), current(1), other(0) {
        assert(sizes.size() >= 1);
        assert(queues.size() == 1);
        auto queue = queues[0];
        radixes = {{2,4,8,16}};
        auto code = kernel_code<T>(inverse, radixes);
        program = build_sources(qctx(queue), code,
            "-cl-fast-relaxed-math -Werror");
        for(auto radix : radixes) {
            std::ostringstream n; n << "radix" << radix;
            radix_k[radix] = cl::Kernel(program, n.str().c_str());
        }
        transpose_k = cl::Kernel(program, "transpose");
        total_n = 1;
        for(size_t i = 0 ; i < sizes.size() ; i++)
            total_n *= sizes[i];
        temp[0] = vector<T2>(queues, total_n);
        temp[1] = vector<T2>(queues, total_n);
    }

    void enqueue_radix(size_t m, size_t batch, size_t p, size_t radix) {
        auto kernel = radix_k[radix];
        kernel.setArg(0, temp[current](0));
        kernel.setArg(1, temp[other](0));
        kernel.setArg<cl_uint>(2, p);

        size_t wg = pow2_floor(std::min(m,
            (size_t)kernel_workgroup_size(kernel, qdev(queue[0]))));
        wg = std::min(wg, (size_t)128);
        if(batch == 1)
            queue[0].enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange(m), cl::NDRange(wg));
        else
            queue[0].enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange(m, batch), cl::NDRange(wg, 1));
    }

    // returns the next radix to use for stage p, size n.
    size_t get_radix(size_t p, size_t n) {
        for(auto r = radixes.rbegin() ; r != radixes.rend() ; r++)
            if(p * (*r) <= n) return *r;
        throw std::runtime_error("Unsupported FFT size.");
    }
    
    // Execute 1D transforms. Input and output will be in temp[current].
    void execute_1d(size_t n, size_t batch) {
        if(n == 1) return;
        size_t p = 1;
        while(p < n) {
            size_t radix = get_radix(p, n);
            enqueue_radix(n / radix, batch, p, radix);
            p *= radix;
            std::swap(current, other);
        }
    }

    // Transpose the array (width <-> height)
    void transpose(size_t w, size_t h) {
        if(w == 1 || h == 1) return;
        const size_t block_dim = 16;
        transpose_k.setArg(0, temp[current](0));
        transpose_k.setArg(1, temp[other](0));
        transpose_k.setArg(2, sizeof(T2) * block_dim * (block_dim + 1), NULL);
        transpose_k.setArg<cl_uint>(3, w);
        transpose_k.setArg<cl_uint>(4, h);
        transpose_k.setArg<cl_uint>(5, block_dim);

        queue[0].enqueueNDRangeKernel(transpose_k, cl::NullRange,
            cl::NDRange(w, h),
            cl::NDRange(std::min(w,block_dim), std::min(h,block_dim)));
        std::swap(current, other);
    }

    // Execute all transforms
    void execute() {
        // FFT each row, transpose so that columns are rows, repeat
        for(auto d = sizes.rbegin() ; d != sizes.rend() ; d++) {
            const size_t w = *d, h = total_n / w;
            execute_1d(w, h);
            transpose(w, h);
        }
        if(inverse)
            temp[current] *= (T)1 / total_n;
    }

    // Execute the complete transformation. C2C
    void operator()(vector<T2> &input, vector<T2> &output) {
        temp[current] = input;
        execute();
        output = temp[current];
    }

    // R2C
    void operator()(vector<T> &input, vector<T2> &output) {
        temp[current] = r2c(input);
        execute();
        output = temp[current];
    }

    // C2R
    void operator()(vector<T2> &input, vector<T> &output) {
        temp[current] = input;
        execute();
        output = c2r(temp[current]);
    }

    // R2R
    void operator()(vector<T> &input, vector<T> &output) {
        temp[current] = r2c(input);
        execute();
        output = c2r(temp[current]);
    }
};


}
}
#endif
