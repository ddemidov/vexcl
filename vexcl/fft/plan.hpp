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

#include <cmath>

#include <vexcl/vector.hpp>
#include <vexcl/fft/kernels.hpp>
#include <boost/lexical_cast.hpp>

namespace vex {
namespace fft {



// for arbitrary x, return smallest y=a^b c^d e^f...>=x where a,c,e are iterated over (assumed to be prime)
template<class Iterator>
inline size_t next_prime_power(Iterator begin, Iterator end, size_t target, size_t n = 1, size_t best = -1) {
    const size_t prime = *begin++;
    for(; n < target ; n *= prime) {
        if(begin != end)
            best = next_prime_power(begin, end, target, n, best);
    }
    return std::min(n, best);
}



struct simple_planner {
    const size_t max_size;
    simple_planner(size_t s = 25) : max_size(s) {}

    // prime factors to use
    virtual std::vector<size_t> primes() const {
	static const size_t data[] = {2, 3, 5, 7, 11};
	return std::vector<size_t>(data, data + sizeof(data)/sizeof(data[0]));
    }

    // returns the size the data must be padded to.
    virtual size_t best_size(size_t n) const {
        auto ps = primes();
        return next_prime_power(ps.begin(), ps.end(), n);
    }

    // splits n into a list of powers 2^a 2^b 2^c 3^d 5^e...
    virtual std::vector<pow> factor(size_t n) const {
        std::vector<pow> fs;
        auto ps = primes();
        for(auto p = ps.begin() ; p != ps.end() ; p++)
            if(n % *p == 0) {
                size_t e = 1;
                while(n % size_t(std::pow(static_cast<double>(*p), static_cast<double>(e + 1))) == 0) e += 1;
                n /= static_cast<size_t>(std::pow(static_cast<double>(*p), static_cast<double>(e)));
                // split exponent into reasonable parts.
                auto qs = stages(pow(*p, e));
                std::copy(qs.begin(), qs.end(), std::back_inserter(fs));
            }
        if(n != 1) throw std::runtime_error("Unsupported FFT size");
        return fs;
    }

    // use largest radixes, i.e. 2^4 2^4 2^1
    virtual std::vector<pow> stages(pow p) const {
        size_t t = static_cast<size_t>(std::log(max_size + 1.0) / std::log(static_cast<double>(p.base)));
        std::vector<pow> fs;
        for(size_t e = p.exponent ; ; ) {
            if(e > t) {
                fs.push_back(pow(p.base, t));
                e -= t;
            } else if(e <= t) {
                fs.push_back(pow(p.base, e));
                break;
            }
        }
        return fs;
    }
};


struct even_planner : simple_planner {
    even_planner(size_t s = 25) : simple_planner(s) {}

    // avoid very small radixes, i.e. 2^3 2^3 2^3
    virtual std::vector<pow> stages(pow p) const {
        size_t t = static_cast<size_t>(std::log(max_size + 1.0) / std::log(static_cast<double>(p.base)));
        // number of parts
        size_t m = (p.exponent + t - 1) / t;
        // two levels.
        size_t r = t * m - p.exponent;
        size_t u = m * (r / m);
        size_t v = t - (r / m);
        std::vector<pow> fs;
        for(size_t i = 0 ; i < m - r + u ; i++)
            fs.push_back(pow(p.base, v));
        for(size_t i = 0 ; i < r - u ; i++)
            fs.push_back(pow(p.base, v - 1));
        return fs;
    }
};

typedef even_planner default_planner;

template <class T0, class T1, class Planner = default_planner>
struct plan {
    typedef typename cl_scalar_of<T0>::type T0s;
    typedef typename cl_scalar_of<T1>::type T1s;
    static_assert(boost::is_same<T0s, T1s>::value, "Input and output must have same precision.");
    typedef T0s T;
    static_assert(boost::is_same<T, cl_float>::value || boost::is_same<T, cl_double>::value,
        "Only float and double data supported.");

    typedef typename cl_vector_of<T, 2>::type T2;

    VEX_FUNCTION(r2c, T2(T), "return (" + type_name<T2>() + ")(prm1, 0);");
    VEX_FUNCTION(c2r, T(T2), "return prm1.x;");

    const std::vector<cl::CommandQueue> &queues;
    Planner planner;
    T scale;

    std::vector<kernel_call> kernels;

    size_t input, output;
    std::vector<cl::Buffer> bufs;

    // \param sizes
    //  1D case: {n}.
    //  2D case: {h, w} in row-major format: x + y * w. (like FFTw)
    //  etc.
    plan(const std::vector<cl::CommandQueue> &_queues, const std::vector<size_t> sizes, bool inverse, const Planner &planner = Planner())
        : queues(_queues), planner(planner) {
        assert(sizes.size() >= 1);
        assert(queues.size() == 1);
        auto queue = queues[0];
        auto context = qctx(queue);
        auto device = qdev(queue);

        size_t total_n = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
        scale = inverse ? ((T)1 / total_n) : 1;

        size_t current = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * total_n));
        size_t other = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * total_n));

        // Build the list of kernels.
        input = current;
        for(auto d = sizes.rbegin() ; d != sizes.rend() ; d++) {
            const size_t w = *d, h = total_n / w;

            // 1D, each row.
            if(w > 1) {
                if(w == planner.best_size(w)) { // use Cooley-Tukey.
                    plan_cooley_tukey(inverse, w, h, current, other, false);
                } else { // use Bluestein Z-Chirp.
                    plan_bluestein(inverse, w, h, current, other);
                }
            }

            // transpose.
            if(w > 1 && h > 1) {
                kernels.push_back(transpose_kernel<T>(queue, w, h, bufs[current], bufs[other]));
                std::swap(current, other);
            }
        }
        output = current;
    }

    void plan_cooley_tukey(bool inverse, size_t n, size_t batch, size_t &current, size_t &other, bool once) {
        size_t p = 1;
        auto rs = planner.factor(n);
        for(auto r = rs.begin() ; r != rs.end() ; r++) {
            kernels.push_back(radix_kernel<T>(once, queues[0], n, batch,
                inverse, *r, p, bufs[current], bufs[other]));
            std::swap(current, other);
            p *= r->value;
        }
    }

    // this as a numeric error in O(exp(n)), as opposed to O(sqrt(n)),
    // which means it's a very bad idea to use this with float for sizes > 100,
    // double's worst case is still better than float's best case though.
    void plan_bluestein(bool inverse, size_t n, size_t batch, size_t &current, size_t &other) {
        size_t conv_n = planner.best_size(2 * n - 1);
        auto context = qctx(queues[0]);

        size_t b_original = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * n));
        size_t b_other = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * conv_n));
        size_t b_current = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * conv_n));
        size_t a_current = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * conv_n * batch));
        size_t a_other = bufs.size(); bufs.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T2) * conv_n * batch));

        kernels.push_back(bluestein_twiddle<T>(queues[0], n, inverse,
            bufs[b_original])); // once

        kernels.push_back(bluestein_pad_kernel<T>(queues[0], n, conv_n,
            bufs[b_original], bufs[b_current])); // once

        plan_cooley_tukey(false, conv_n, 1, b_current, b_other, /*once*/true);

        kernels.push_back(bluestein_mul<T>(queues[0], conv_n, batch, /*pad*/n, /*in s*/n, /*out s*/conv_n,
            bufs[current], bufs[b_original], bufs[a_current]));

        plan_cooley_tukey(false, conv_n, batch, a_current, a_other, false);

        kernels.push_back(bluestein_mul<T>(queues[0], conv_n, batch, /*pad*/conv_n, /*in s*/conv_n, /*outs*/conv_n,
            bufs[a_current], bufs[b_current], bufs[a_other]));
        std::swap(a_current, a_other);

        plan_cooley_tukey(true, conv_n, batch, a_current, a_other, false);

        kernels.push_back(bluestein_mul<T>(queues[0], n, batch, /*pad*/n, /*in s*/conv_n, /*out s*/n,
            bufs[a_current], bufs[b_original], bufs[other], /*div*/conv_n));
        std::swap(current, other);
    }

    /// Execute the complete transformation.
    /// Converts real-valued input and output, supports multiply-adding to output.
    template<class Expr>
    void operator()(const Expr &in, vector<T1> &out, bool append, T ex_scale) {
        vector<T2> in_c(queues[0], bufs[input]);
        if(std::is_same<T0, T>::value) in_c = r2c(in);
        else in_c = in;

        for(auto run = kernels.begin(); run != kernels.end(); ++run) {
            if(!run->once || run->count == 0) {
                queues[0].enqueueNDRangeKernel(run->kernel, cl::NullRange,
                    run->global, run->local);
                run->count++;
            }
        }

        vector<T2> out_c(queues[0], bufs[output]);
        if(std::is_same<T1, T>::value) {
            if(append) out += c2r(out_c) * (ex_scale * scale);
            else out = c2r(out_c) * (ex_scale * scale);
        } else {
            if(append) out += out_c * (ex_scale * scale);
            else out = out_c * (ex_scale * scale);
        }
    }
};


template <class T0, class T1, class P>
inline std::ostream &operator<<(std::ostream &o, const plan<T0,T1,P> &p) {
    o << "FFT[\n";
    for(auto k = p.kernels.begin() ; k != p.kernels.end() ; k++) {
        o << "  ";
        if(k->once) o << "once: ";
        o << k->desc << "\n";
    }
    return o << "]";
}

} // namespace fft
} // namespace vex
#endif
