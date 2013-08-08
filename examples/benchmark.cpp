#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <random>
#include <vexcl/vexcl.hpp>

using namespace vex;

#define BENCHMARK_VECTOR
#define BENCHMARK_REDUCTOR
#define BENCHMARK_STENCIL
#define BENCHMARK_SPMAT
#define BENCHMARK_CPU

#ifdef WIN32
#  pragma warning(disable : 4267)
#endif

template <typename real>
struct name_of {
};

template <>
struct name_of<double> {
  static constexpr const char* name="double";
};

template <>
struct name_of<float> {
  static constexpr const char* name="float";
};

template <typename real>
std::vector<real> random_vector(size_t n) {
    static std::default_random_engine rng( std::rand() );
    static std::uniform_real_distribution<real> rnd(0.0, 1.0);

    std::vector<real> x(n);
    std::generate(x.begin(), x.end(), []() { return rnd(rng); });

    return x;
}

//---------------------------------------------------------------------------
template <typename real>
std::pair<double,double> benchmark_vector(
        const std::vector<cl::CommandQueue> &queue, profiler<> &prof,
        bool benchmark_cpu
        )
{
    prof.tic_cpu(name_of<real>::name);

    const size_t N = 1024 * 1024;
    const size_t M = 1024;
    double time_elapsed;

    std::vector<real> A(N, 0);
    std::vector<real> B = random_vector<real>(N);
    std::vector<real> C = random_vector<real>(N);
    std::vector<real> D = random_vector<real>(N);

    vex::vector<real> a(queue, A);
    vex::vector<real> b(queue, B);
    vex::vector<real> c(queue, C);
    vex::vector<real> d(queue, D);

    a += b + c * d;
    a = 0;

    prof.tic_cl("OpenCL");
    for(size_t i = 0; i < M; i++)
        a += b + c * d;
    time_elapsed = prof.toc("OpenCL");

    double gflops = (3.0 * N * M) / time_elapsed / 1e9;
    double bwidth = (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9;

    std::cout
        << "Vector arithmetic (" << name_of<real>::name << ")\n"
        << "  OpenCL"
        << "\n    GFLOPS:    " << gflops
        << "\n    Bandwidth: " << bwidth
        << std::endl;

#ifdef BENCHMARK_CPU
    if (benchmark_cpu) {

    prof.tic_cpu("C++");
    for(size_t i = 0; i < M; i++)
        for(size_t j = 0; j < N; j++)
            A[j] += B[j] + C[j] * D[j];
    time_elapsed = prof.toc("C++");

    {
        double gflops = (3.0 * N * M) / time_elapsed / 1e9;
        double bwidth = (5.0 * N * M * sizeof(real)) / time_elapsed / 1e9;

        std::cout
            << "  C++"
            << "\n    GFLOPS:    " << gflops
            << "\n    Bandwidth: " << bwidth
            << std::endl;
    }

    vex::copy(A, b);
    Reductor<real,SUM> sum(queue);

    a -= b;
    std::cout << "  res = " << sum(a * a)
              << std::endl << std::endl;

    } // benchmark_cpu
#endif

    prof.toc(name_of<real>::name);

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
template <typename real>
std::pair<double, double> benchmark_reductor(
        const std::vector<cl::CommandQueue> &queue, profiler<> &prof,
        bool benchmark_cpu
        )
{
    prof.tic_cpu(name_of<real>::name);

    const size_t N = 16 * 1024 * 1024;
    const size_t M = 1024 / 16;
    double time_elapsed;

    std::vector<real> A = random_vector<real>(N);
    std::vector<real> B = random_vector<real>(N);

    vex::vector<real> a(queue, A);
    vex::vector<real> b(queue, B);

    Reductor<real,SUM> sum(queue);

    real sum_cl = sum(a * b);
    sum_cl = 0;

    prof.tic_cl("OpenCL");
    for(size_t i = 0; i < M; i++)
        sum_cl += sum(a * b);
    time_elapsed = prof.toc("OpenCL");

    double gflops = 2.0 * N * M / time_elapsed / 1e9;
    double bwidth = 2.0 * N * M * sizeof(real) / time_elapsed / 1e9;

    std::cout
        << "Reduction (" << name_of<real>::name << ")\n"
        << "  OpenCL"
        << "\n    GFLOPS:    " << gflops
        << "\n    Bandwidth: " << bwidth
        << std::endl;

#ifdef BENCHMARK_CPU
    if (benchmark_cpu) {

    real sum_cpp = 0;
    prof.tic_cpu("C++");
    for(size_t i = 0; i < M; i++)
        sum_cpp += std::inner_product(A.begin(), A.end(), B.begin(), 0.0);
    time_elapsed = prof.toc("C++");

    {
        double gflops = 2.0 * N * M / time_elapsed / 1e9;
        double bwidth = 2.0 * N * M * sizeof(real) / time_elapsed / 1e9;

        std::cout
            << "  C++"
            << "\n    GFLOPS:    " << gflops
            << "\n    Bandwidth: " << bwidth
            << std::endl;
    }

    std::cout << "  res = " << std::fabs(sum_cl - sum_cpp)
              << std::endl << std::endl;

    } // benchmark_cpu
#endif

    prof.toc(name_of<real>::name);

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
template <typename real>
std::pair<double, double> benchmark_stencil(
        const std::vector<cl::CommandQueue> &queue, profiler<> &prof,
        bool benchmark_cpu
        )
{
    prof.tic_cpu(name_of<real>::name);

    const long N = 1024 * 1024;
    const long M = 1024;
    double time_elapsed;

    std::vector<real> A = random_vector<real>(N);
    std::vector<real> B(N);

    std::vector<real> S(21, 1.0 / 21);
    long center = S.size() / 2;
    vex::stencil<real> s(queue, S, center);

    vex::vector<real> a(queue, A);
    vex::vector<real> b(queue, N);

    b = a * s;

    prof.tic_cl("OpenCL");
    for(long i = 0; i < M; i++)
        b = a * s;
    time_elapsed = prof.toc("OpenCL");

    double gflops = 2.0 * S.size() * N * M / time_elapsed / 1e9;
    double bwidth = 2.0 * S.size() * N * M * sizeof(real) / time_elapsed / 1e9;

    std::cout
        << "Stencil convolution (" << name_of<real>::name << ")\n"
        << "  OpenCL"
        << "\n    GFLOPS:    " << gflops
        << "\n    Bandwidth: " << bwidth
        << std::endl;

#ifdef BENCHMARK_CPU
    if (benchmark_cpu) {

    prof.tic_cpu("C++");
    for(long j = 0; j < M; j++) {
        for(long i = 0; i < N; i++) {
            real sum = 0;
            for(long k = 0; k < (long)S.size(); k++)
                sum += S[k] * A[std::min<long>(N-1, std::max<long>(0, i + k - center))];
            B[i] = sum;
        }
    }
    time_elapsed = prof.toc("C++");

    {
        double gflops = 2.0 * S.size() * N * M / time_elapsed / 1e9;
        double bwidth = 2.0 * S.size() * N * M * sizeof(real) / time_elapsed / 1e9;

        std::cout
            << "  C++"
            << "\n    GFLOPS:    " << gflops
            << "\n    Bandwidth: " << bwidth
            << std::endl;
    }

    Reductor<real,MAX> max(queue);
    copy(B, a);

    std::cout << "  res = " << max(fabs(a - b))
              << std::endl << std::endl;

    } // benchmark_cpu
#endif

    prof.toc(name_of<real>::name);

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
template <typename real>
std::pair<double,double> benchmark_spmv(
        const std::vector<cl::CommandQueue> &queue, profiler<> &prof,
        bool benchmark_cpu
        )
{
    prof.tic_cpu(name_of<real>::name);

    // Construct matrix for 3D Poisson problem in cubic domain.
    const size_t n = 128;
    const size_t N = n * n * n;
    const size_t M = 1024;

    double time_elapsed;

    const real h2i = (n - 1) * (n - 1);

    std::vector<size_t> row;
    std::vector<uint>   col;
    std::vector<real>   val;
    std::vector<real>   X(n * n * n, 1e-2);
    std::vector<real>   Y(n * n * n, 0);

    row.reserve(n * n * n + 1);
    col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
    val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

    row.push_back(0);
    for(size_t k = 0, idx = 0; k < n; k++) {
        for(size_t j = 0; j < n; j++) {
            for(size_t i = 0; i < n; i++, idx++) {
                if (
                        i == 0 || i == (n - 1) ||
                        j == 0 || j == (n - 1) ||
                        k == 0 || k == (n - 1)
                   )
                {
                    col.push_back(idx);
                    val.push_back(1);
                    row.push_back(row.back() + 1);
                } else {
                    col.push_back(idx - n * n);
                    val.push_back(-h2i);

                    col.push_back(idx - n);
                    val.push_back(-h2i);

                    col.push_back(idx - 1);
                    val.push_back(-h2i);

                    col.push_back(idx);
                    val.push_back(6 * h2i);

                    col.push_back(idx + 1);
                    val.push_back(-h2i);

                    col.push_back(idx + n);
                    val.push_back(-h2i);

                    col.push_back(idx + n * n);
                    val.push_back(-h2i);

                    row.push_back(row.back() + 7);
                }
            }
        }
    }

    size_t nnz = row.back();

    // Transfer data to compute devices.
    vex::SpMat<real,uint> A(queue, n * n * n, n * n * n, row.data(), col.data(), val.data());

    vex::vector<real> x(queue, X);
    vex::vector<real> y(queue, Y);

    // Get timings.
    y += A * x;
    y = 0;

    prof.tic_cl("OpenCL");
    for(size_t i = 0; i < M; i++)
        y += A * x;
    time_elapsed = prof.toc("OpenCL");

    double gflops = (2.0 * nnz + N) * M / time_elapsed / 1e9;
    double bwidth = M * (nnz * (2 * sizeof(real) + sizeof(size_t)) + 4 * N * sizeof(real)) / time_elapsed / 1e9;

    std::cout
        << "SpMV (" << name_of<real>::name << ")\n"
        << "  OpenCL"
        << "\n    GFLOPS:    " << gflops
        << "\n    Bandwidth: " << bwidth
        << std::endl;

#ifdef BENCHMARK_CPU
    if (benchmark_cpu) {

    prof.tic_cpu("C++");
    for(size_t k = 0; k < M; k++)
        for(size_t i = 0; i < N; i++) {
            real s = 0;
            for(size_t j = row[i]; j < row[i + 1]; j++)
                s += val[j] * X[col[j]];
            Y[i] += s;
        }
    time_elapsed = prof.toc("C++");

    {
        double gflops = (2.0 * nnz + N) * M / time_elapsed / 1e9;
        double bwidth = M * (nnz * (2 * sizeof(real) + sizeof(size_t)) + 4 * N * sizeof(real)) / time_elapsed / 1e9;

        std::cout
            << "  C++"
            << "\n    GFLOPS:    " << gflops
            << "\n    Bandwidth: " << bwidth
            << std::endl;
    }

    copy(Y, x);

    y -= x;

    Reductor<real,SUM> sum(queue);

    std::cout << "  res = " << sum(y * y) << std::endl << std::endl;

    } // benchmark_cpu
#endif

    prof.toc(name_of<real>::name);

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
template <typename real>
std::pair<double,double> benchmark_spmv_ccsr(
        const std::vector<cl::CommandQueue> &queue, profiler<> &prof,
        bool benchmark_cpu
        )
{
    prof.tic_cpu(name_of<real>::name);

    // Construct matrix for 3D Poisson problem in cubic domain.
    const uint n = 128;
    const uint N = n * n * n;
    const uint M = 1024;

    double time_elapsed;

    const real h2i = (n - 1) * (n - 1);

    std::vector<size_t> idx;
    std::vector<size_t> row(3);
    std::vector<int>    col(8);
    std::vector<real>   val(8);

    std::vector<real>   X(n * n * n, 1e-2);
    std::vector<real>   Y(n * n * n, 0);

    idx.reserve(n * n * n);

    row[0] = 0;
    row[1] = 1;
    row[2] = 8;

    col[0] = 0;
    val[0] = 1;

    col[1] = -static_cast<int>(n * n);
    col[2] = -static_cast<int>(n);
    col[3] =    -1;
    col[4] =     0;
    col[5] =     1;
    col[6] =     n;
    col[7] =  (n * n);

    val[1] = -h2i;
    val[2] = -h2i;
    val[3] = -h2i;
    val[4] =  h2i * 6;
    val[5] = -h2i;
    val[6] = -h2i;
    val[7] = -h2i;

    for(size_t k = 0; k < n; k++) {
        for(size_t j = 0; j < n; j++) {
            for(size_t i = 0; i < n; i++) {
                if (
                        i == 0 || i == (n - 1) ||
                        j == 0 || j == (n - 1) ||
                        k == 0 || k == (n - 1)
                   )
                {
                    idx.push_back(0);
                } else {
                    idx.push_back(1);
                }
            }
        }
    }

    size_t nnz = 6 * (n - 2) * (n - 2) * (n - 2) + n * n * n;

    // Transfer data to compute devices.
    vex::SpMatCCSR<real,int> A(queue[0], n * n * n, 2,
            idx.data(), row.data(), col.data(), val.data());

    std::vector<cl::CommandQueue> q1(1, queue[0]);
    vex::vector<real> x(q1, X);
    vex::vector<real> y(q1, Y);

    // Get timings.
    y += A * x;
    y = 0;

    prof.tic_cl("OpenCL");
    for(size_t i = 0; i < M; i++)
        y += A * x;
    time_elapsed = prof.toc("OpenCL");

    double gflops = (2.0 * nnz + N) * M / time_elapsed / 1e9;
    double bwidth = M * (nnz * (2 * sizeof(real) + sizeof(int)) + 4 * N * sizeof(real)) / time_elapsed / 1e9;

    std::cout
        << "SpMV (CCSR) (" << name_of<real>::name << ")\n"
        << "  OpenCL"
        << "\n    GFLOPS:    " << gflops
        << "\n    Bandwidth: " << bwidth
        << std::endl;

#ifdef BENCHMARK_CPU
    if (benchmark_cpu) {

    prof.tic_cpu("C++");
    for(size_t k = 0; k < M; k++)
        for(size_t i = 0; i < N; i++) {
            real s = 0;
            for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
                s += val[j] * X[i + col[j]];
            Y[i] += s;
        }
    time_elapsed = prof.toc("C++");

    {
        double gflops = (2.0 * nnz + N) * M / time_elapsed / 1e9;
        double bwidth = M * (nnz * (2 * sizeof(real) + sizeof(int)) + 4 * N * sizeof(real)) / time_elapsed / 1e9;

        std::cout
            << "  C++"
            << "\n    GFLOPS:    " << gflops
            << "\n    Bandwidth: " << bwidth
            << std::endl;
    }

    copy(Y, x);

    y -= x;

    Reductor<real,SUM> sum(q1);

    std::cout << "  res = " << sum(y * y) << std::endl << std::endl;

    } // benchmark_cpu
#endif

    prof.toc(name_of<real>::name);

    return std::make_pair(gflops, bwidth);
}

//---------------------------------------------------------------------------
int main() {
    try {
        vex::Context ctx_d(Filter::DoublePrecision && Filter::Env);
        vex::Context ctx_f(Filter::Env);

        if (!ctx_d.size() && !ctx_f.size()) {
            std::cerr << "No compute devices found" << std::endl;
            return 1;
        }

        std::cout << "Double precision device(s):" << std::endl;
        std::cout << ctx_d << std::endl;

        std::cout << "Single precision device(s):" << std::endl;
        std::cout << ctx_f << std::endl;

        std::ofstream log("profiling.dat", std::ios::app);

        log << ctx_d.size() << " ";

        double gflops, bwidth;

        profiler<> prof(ctx_f);

#ifdef BENCHMARK_VECTOR
        prof.tic_cpu("Vector arithmetic");
        std::tie(gflops, bwidth) = benchmark_vector<double>(ctx_d, prof, true);
        benchmark_vector<float>(ctx_d, prof, true);
        prof.toc("Vector arithmetic");

        log << gflops << " " << bwidth << " ";
#endif

#ifdef BENCHMARK_REDUCTOR
        prof.tic_cpu("Reduction");
        std::tie(gflops, bwidth) = benchmark_reductor<double>(ctx_d, prof, true);
        benchmark_reductor<float>(ctx_d, prof, true);
        prof.toc("Reduction");

        log << gflops << " " << bwidth << " ";
#endif

#ifdef BENCHMARK_STENCIL
        prof.tic_cpu("Stencil");
        std::tie(gflops, bwidth) = benchmark_stencil<double>(ctx_d, prof, true);
        benchmark_stencil<float>(ctx_d, prof, true);
        prof.toc("Stencil");

        log << gflops << " " << bwidth << " ";
#endif

#ifdef BENCHMARK_SPMAT
        prof.tic_cpu("SpMV");
        std::tie(gflops, bwidth) = benchmark_spmv<double>(ctx_d, prof, true);
        benchmark_spmv<float>(ctx_d, prof, true);
        prof.toc("SpMV");

        log << gflops << " " << bwidth << std::endl;

        prof.tic_cpu("SpMV (CCSR)");
        std::tie(gflops, bwidth) = benchmark_spmv_ccsr<double>(ctx_d, prof, true);
        benchmark_spmv_ccsr<float>(ctx_d, prof, true);
        prof.toc("SpMV (CCSR)");
#endif

        std::cout << prof << std::endl;
    } catch (const cl::Error &e) {
        std::cerr << e << std::endl;
        return 1;
    }
}

// vim: et
