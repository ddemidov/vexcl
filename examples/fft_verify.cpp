#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
#include <random>
#include <fftw3.h>

using namespace vex;

typedef double T;
typedef cl_vector_of<T, 2>::type T2;

std::vector<cl_double2> random_vec(size_t n) {
    std::vector<cl_double2> data(n);
    for(size_t i = 0 ; i < n ; i++) {
        data[i].s[0] = 1.0 * rand() / RAND_MAX - 0.5;
        data[i].s[1] = 1.0 * rand() / RAND_MAX - 0.5;
    }
    return data;
}


bool test(Context &ctx, std::vector<size_t> ns) {
    std::ostringstream name;
    for(size_t i = 0 ; i < ns.size() ; i++) {
        if(i > 0) name << 'x';
        name << ns[i];
    }
    char fc = std::cout.fill('.');
    std::cout << name.str() << ": " << std::setw(62 - name.str().size()) << "." << std::flush;
    std::cout.fill(fc);

    size_t n = 1;
    for(size_t i = 0 ; i < ns.size() ; i++) n *= ns[i];

    // random data.
    std::vector<cl_double2> input_h = random_vec(n);

    // reference.
    std::vector<cl_double2> ref_h(n);
    std::vector<int> ns_(ns.begin(), ns.end());
    fftw_plan p1 = fftw_plan_dft(ns_.size(), ns_.data(),
        reinterpret_cast<fftw_complex *>(&input_h[0]),
        reinterpret_cast<fftw_complex *>(&ref_h[0]),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_destroy_plan(p1);

    // convert to clFFT precision
    std::vector<T2> input_h_t(n), ref_h_t(n);
    for(size_t i = 0; i != n ; i++) {
        input_h_t[i] = cl_convert<T2>(input_h[i]);
        ref_h_t[i] = cl_convert<T2>(ref_h[i]);
    }

    // test
    vector<T2> input(ctx.queue(), input_h_t);
    vector<T2> output(ctx.queue(), n);
    vector<T2> ref(ctx.queue(), ref_h_t);
    vector<T2> back(ctx.queue(), n);

    Reductor<T, SUM> sum(ctx.queue());
    #define rms(a,b) (std::sqrt(sum(dot(a - b, a - b))) / std::sqrt(sum(dot(b, b))))

    try {
        FFT<T2>  fft(ctx.queue(), ns);
        output = fft(input);

        bool rc = true;
        const T rms_fft = rms(output, ref);
        rc &= rms_fft < 0.1;

        FFT<T2> ifft(ctx.queue(), ns, inverse);
        back = ifft(output);
        const T rms_inv = rms(back, input);
        rc &= rms_inv < 1e-3;

        std::cout << (rc ? " success." : " failed.") << '\n';

        if(!rc) {
            std::cout << "  fftw-clfft      " << rms_fft << "%" << '\n';
            std::cout << "  x-ifft(fft(x))  " << rms_inv << "%" << '\n';
            std::cout << fft.plan << '\n';
        }

        return rc;

    } catch(cl::Error e) {
        std::cerr << "FFT error " << ": " << e << std::endl;
        throw;
    }
}


double skew_rand(double p) {
    return std::pow(1.0 * rand() / RAND_MAX, p);
}

int main() {
    Context ctx(Filter::Env && Filter::Count(1), CL_QUEUE_PROFILING_ENABLE);
    std::cout << ctx << std::endl;
    srand(23);

    bool rc = true;

    const size_t max = 1 << 20;

    fft::default_planner p;
    for(size_t i = 0 ; i < 100 ; i++) {
        // random number of dimensions, mostly 1.
        size_t dims = 1 + size_t(skew_rand(3) * 5);
        // random size.
        std::vector<size_t> n;
        size_t d_max = std::pow(max, 1.0 / dims);
        size_t total = 1;
        for(size_t d = 0 ; d < dims ; d++) {
            size_t sz = 1 + size_t(skew_rand(dims == 1 ? 3 : 1) * d_max);
            if(rand() % 7 != 0)
                sz = p.best_size(sz);
            n.push_back(sz);
            total *= sz;
        }
        // run
        if(total <= max)
            rc &= test(ctx, n);
    }


    return rc ? EXIT_SUCCESS : EXIT_FAILURE;
}

