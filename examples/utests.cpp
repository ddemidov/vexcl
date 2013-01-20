#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include <numeric>
#include <cmath>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>
#include <vexcl/random.hpp>
#include <vexcl/fft.hpp>

using namespace vex;

static bool all_passed = true;

bool run_test(const std::string &name, std::function<bool()> test) {
    char fc = std::cout.fill('.');
    std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc = test();
    all_passed = all_passed && rc;
    std::cout << (rc ? " success." : " failed.") << std::endl;
    return rc;
}

VEX_FUNCTION(greater, size_t(double, double), "return prm1 > prm2 ? 1 : 0;");

template <class state_type>
void sys_func(const state_type &x, state_type &dx, double dt) {
    dx = dt * sin(x);
}

template <class state_type, class SysFunction>
void runge_kutta_4(SysFunction sys, state_type &x, double dt) {
    state_type xtmp, k1, k2, k3, k4;

    sys(x, k1, dt);

    xtmp = x + 0.5 * k1;
    sys(xtmp, k2, dt);

    xtmp = x + 0.5 * k2;
    sys(xtmp, k3, dt);

    xtmp = x + k3;
    sys(xtmp, k4, dt);

    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

int main(int argc, char *argv[]) {
    try {
        vex::Context ctx(Filter::DoublePrecision && Filter::Env);
        std::cout << ctx << std::endl;

        if (ctx.queue().empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::vector<cl::CommandQueue> single_queue(
                ctx.queue().begin(), ctx.queue().begin() + 1);

        uint seed = argc > 1 ? atoi(argv[1]) : static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl << std::endl;
        srand(seed);

#if 1
        run_test("Empty vector construction", [&]() -> bool {
                bool rc = true;
                vex::vector<double> x;
                rc = rc && (x.size() == 0);
                rc = rc && (x.end() - x.begin() == 0);
                return rc;
                });
#endif

#if 1
        run_test("Vector construction from size", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                rc = rc && (x.size() == N);
                rc = rc && (x.end() == x.begin() + N);
                return rc;
                });
#endif

#if 1
        run_test("Vector construction from std::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N, 42);
                std::vector<double> y(N);
                vex::vector<double> X(ctx.queue(), x);
                rc = rc && (X.size() == x.size());
                rc = rc && (X.end() == X.begin() + x.size());
                copy(X, y);
                std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                    [](double a, double b) { return a - b; });
                rc = rc && std::all_of(y.begin(), y.end(),
                    [](double a) {return a == 0; });
                return rc;
                });
#endif

#if 1
        run_test("Vector construction from size and host pointer", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N, 42);
                std::vector<double> y(N);
                vex::vector<double> X(ctx.queue(), N, x.data());
                rc = rc && (X.size() == x.size());
                rc = rc && (X.end() == X.begin() + x.size());
                copy(X, y);
                std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                    [](double a, double b) { return a - b; });
                rc = rc && std::all_of(y.begin(), y.end(),
                    [](double a) {return a == 0; });
                return rc;
                });
#endif

#if 1
        run_test("Vector copy construction", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x1;
                vex::vector<double> x2(x1);

                vex::vector<double> y1(ctx.queue(), N);
                vex::vector<double> y2(y1);
                rc = rc && (x1.size() == x2.size() && x1.size() == 0);
                rc = rc && (y1.size() == y2.size() && y1.size() == N);
                return rc;
                });
#endif

#if 1
        run_test("Vector move construction from vex::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                x = 42;
                vex::vector<double> y = std::move(x);
                rc = rc && (y.size() == N);
                rc = rc && (y.end() == y.begin() + N);
                Reductor<double,MIN> min(ctx.queue());
                Reductor<double,MAX> max(ctx.queue());
                rc = rc && min(y) == 42;
                rc = rc && max(y) == 42;
                return rc;
                });
#endif

#if 1
        run_test("Vector move assignment", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N, 42);
                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y = std::move(X);
                rc = rc && (Y.size() == x.size());
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(Y != x[0]) == 0;
                return rc;
                });
#endif

#if 1
        run_test("Vector swap", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                vex::vector<double> y(ctx.queue(), N/2);
                x = 42;
                y = 67;
                swap(x, y);
                rc = rc && (y.size() == N);
                rc = rc && (x.size() == N/2);
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(y != 42) == 0;
                rc = rc && sum(x != 67) == 0;
                return rc;
                });
#endif

#if 1
        run_test("Vector resize from std::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N, 42);
                vex::vector<double> X;
                X.resize(ctx.queue(), x);
                rc = rc && (X.size() == x.size());
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(X != 42) == 0;
                return rc;
                });
#endif

#if 1
        run_test("Vector resize vex::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                x = 42;
                vex::vector<double> y;
                y.resize(x);
                rc = rc && (y.size() == x.size());
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(x != y) == 0;
                return rc;
                });
#endif

#if 1
        run_test("Iterate over vex::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                x = 42;
                rc = rc && std::all_of(x.begin(), x.end(),
                    [](double a) { return a == 42; });
                return rc;
                });
#endif

#if 1
        run_test("Access vex::vector elements", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                for(uint i = 0; i < N; i++)
                    x[i] = 42;
                for(uint i = 0; i < N; i++)
                    rc = rc && (x[i] == 42);
                return rc;
                });
#endif

#if 1
        run_test("Copy vex::vector to std::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N);
                vex::vector<double> X(ctx.queue(), N);
                X = 42;
                copy(X, x);
                rc = rc && std::all_of(x.begin(), x.end(),
                    [](double a) { return a == 42; });
                X = 67;
                vex::copy(X.begin(), X.end(), x.begin());
                rc = rc && std::all_of(x.begin(), x.end(),
                    [](double a) { return a == 67; });
                return rc;
                });
#endif

#if 1
        run_test("Copy std::vector to vex::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N, 42);
                vex::vector<double> X(ctx.queue(), N);
                copy(x, X);
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(X != 42) == 0;
                std::fill(x.begin(), x.end(), 67);
                vex::copy(x.begin(), x.end(), X.begin());
                rc = rc && sum(X != 67) == 0;
                return rc;
                });
#endif

#if 1
        run_test("Assign expression to vex::vector", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                vex::vector<double> y(ctx.queue(), N);
                vex::vector<double> z(ctx.queue(), N);
                y = 42;
                z = 67;
                x = 5 * sin(y) + z;
                Reductor<double,MAX> max(ctx.queue());
                rc = rc && max(fabs(x - (5 * sin(42.0) + 67))) < 1e-12;
                return rc;
                });
#endif


#if 1
        run_test("Reduction", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });
                vex::vector<double> X(ctx.queue(), x);
                Reductor<double,SUM> sum(ctx.queue());
                Reductor<double,MIN> min(ctx.queue());
                Reductor<double,MAX> max(ctx.queue());
                rc = rc && fabs(sum(X) - std::accumulate(x.begin(), x.end(), 0.0)) < 1e-6;
                rc = rc && fabs(min(X) - *std::min_element(x.begin(), x.end())) < 1e-6;
                rc = rc && fabs(max(X) - *std::max_element(x.begin(), x.end())) < 1e-6;
                return rc;
                });
#endif


#if 1
        run_test("Sparse matrix-vector product", [&]() -> bool {
                bool rc = true;
                const size_t n   = 32;
                const double h2i = (n - 1) * (n - 1);

                std::vector<size_t> row;
                std::vector<size_t> col;
                std::vector<double> val;

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

                std::vector<double> x(n * n * n);
                std::vector<double> y(n * n * n);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

                vex::SpMat <double> A(ctx.queue(), x.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), x.size());

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(size_t i = 0; i < x.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(sum - y[i]));
                }

                rc = rc && res < 1e-8;

                Y = 42 * (A * X);
                copy(Y, y);

                res = 0;
                for(size_t i = 0; i < x.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(42 * sum - y[i]));
                }

                rc = rc && res < 1e-8;

                Y = X + A * X;
                copy(Y, y);

                res = 0;
                for(size_t i = 0; i < x.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(sum + x[i] - y[i]));
                }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif


#if 1
        run_test("Sparse matrix-vector product for non-square matrix", [&]() -> bool {
                bool rc = true;
                const size_t n = 1 << 10;
                const size_t m = 2 * n;
                const size_t nnz_per_row = 16;

                std::vector<size_t> row;
                std::vector<size_t> col;
                std::vector<double> val;

                row.reserve(n + 1);
                col.reserve(n * nnz_per_row);
                val.reserve(n * nnz_per_row);

                row.push_back(0);
                for(size_t k = 0; k < n; k++) {
                    std::set<size_t> cs;
                    while(cs.size() < nnz_per_row)
                        cs.insert(rand() % m);

                    for(auto c = cs.begin(); c != cs.end(); c++) {
                        col.push_back(*c);
                        val.push_back(static_cast<double>(rand()) / RAND_MAX);
                    }

                    row.push_back(col.size());
                }

                std::vector<double> x(m);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

                vex::SpMat <double> A(ctx.queue(), y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), y.size());

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(size_t i = 0; i < y.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(sum - y[i]));
                }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Sparse matrix-vector product with nondefault types", [&]() -> bool {
                bool rc = true;
                const size_t n = 1 << 10;
                const size_t nnz_per_row = 16;

                std::vector<unsigned int> row;
                std::vector<unsigned int> col;
                std::vector<double> val;

                row.reserve(n + 1);
                col.reserve(n * nnz_per_row);
                val.reserve(n * nnz_per_row);

                row.push_back(0);
                for(size_t k = 0; k < n; k++) {
                    std::set<unsigned int> cs;
                    while(cs.size() < nnz_per_row)
                        cs.insert(rand() % n);

                    for(auto c = cs.begin(); c != cs.end(); c++) {
                        col.push_back(*c);
                        val.push_back(static_cast<double>(rand()) / RAND_MAX);
                    }

                    row.push_back(col.size());
                }

                std::vector<double> x(n);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

                vex::SpMat <double,unsigned int, unsigned int> A(ctx.queue(), y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), y.size());

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(size_t i = 0; i < y.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(sum - y[i]));
                }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Sparse matrix-vector product for empty-row matrix", [&]() -> bool {
                bool rc = true;
                const size_t n = 1 << 20;
                const size_t m = 1 << 10;
                const size_t nnz_per_row = 2;
                const size_t start_empty_row = 1 << 8;

                std::vector<size_t> row;
                std::vector<size_t> col;
                std::vector<double> val;

                row.reserve(n + 1);
                col.reserve(start_empty_row * nnz_per_row);
                val.reserve(start_empty_row * nnz_per_row);

                row.push_back(0);
                for(size_t k = 0; k < n; k++) {
                    if (k > start_empty_row) {
                        row.push_back(col.size());
                        continue;
                    }
                    std::set<size_t> cs;
                    while(cs.size() < nnz_per_row)
                        cs.insert(rand() % m);

                    for(auto c = cs.begin(); c != cs.end(); c++) {
                        col.push_back(*c);
                        val.push_back(static_cast<double>(rand()) / RAND_MAX);
                    }

                    row.push_back(col.size());
                }

                std::vector<double> x(m);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

                vex::SpMat <double> A(ctx.queue(), y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), y.size());

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(size_t i = 0; i < y.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[i]; j < row[i + 1]; j++)
                        sum += val[j] * x[col[j]];
                    res = std::max(res, fabs(sum - y[i]));
                }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Sparse matrix-vector product (CCSR format)", [&]() -> bool {
                bool rc = true;
                const uint n   = 32;
                const double h2i = (n - 1) * (n - 1);

                std::vector<size_t> idx;
                std::vector<size_t> row(3);
                std::vector<int>    col(8);
                std::vector<double> val(8);

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

                std::vector<double> x(n * n * n);
                std::vector<double> y(n * n * n);
                std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

                std::vector<cl::CommandQueue> q1(1, ctx.queue()[0]);

                vex::SpMatCCSR<double,int> A(q1[0], x.size(), row.size() - 1,
                        idx.data(), row.data(), col.data(), val.data());

                vex::vector<double> X(q1, x);
                vex::vector<double> Y(q1, x.size());

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(size_t i = 0; i < x.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
                        sum += val[j] * x[i + col[j]];
                    res = std::max(res, fabs(sum - y[i]));
                }

                rc = rc && res < 1e-8;

                Y = X + A * X;
                copy(Y, y);

                res = 0;
                for(size_t i = 0; i < x.size(); i++) {
                    double sum = 0;
                    for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
                        sum += val[j] * x[i + col[j]];
                    res = std::max(res, fabs(sum + x[i] - y[i]));
                }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Builtin function with one argument", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
                vex::vector<double> X(ctx.queue(), x);
                Reductor<double,SUM> sum(ctx.queue());
                rc = rc && 1e-8 > fabs(sum(sin(X)) -
                    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
                        return s + sin(v);
                        }));
                rc = rc && 1e-8 > fabs(sum(cos(X)) -
                    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
                        return s + cos(v);
                        }));
                return rc;
                });
#endif

#if 1
        run_test("Empty multivector construction", [&]() -> bool {
                bool rc = true;
                vex::multivector<double,3> m;
                rc = rc && m.size() == 0;
                rc = rc && m.end() == m.begin();
                return rc;
        });
#endif

#if 1
        run_test("Multivector construction from a copy", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                vex::multivector<double,3> m(ctx.queue(), n);

                m(0) = 1;
                m(1) = 2;
                m(2) = 3;

                vex::multivector<double,3> c(m);

                c -= m;

                Reductor<double,SUM> sum(ctx.queue());

                rc = rc && m.size() == c.size();
                rc = rc && fabs(sum(c(1))) < 1e-8;
                return rc;
        });
#endif

#if 1
        run_test("Access multivector's elements, copy data", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<float> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (float)rand() / RAND_MAX; });
                multivector<float, m> x(ctx.queue(), n);
                copy(host, x);
                for(size_t i = 0; i < n; i++) {
                    std::array<float,m> val = x[i];
                    for(uint j = 0; j < m; j++) {
                        rc = rc && val[j] == host[j * n + i];
                        val[j] = 0;
                    }
                    x[i] = val;
                }
                copy(x, host);
                rc = rc && 0 == *std::min_element(host.begin(), host.end());
                rc = rc && 0 == *std::max_element(host.begin(), host.end());
                return rc;
        });
#endif

#if 1
        run_test("Simple arithmetic with multivectors", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<float> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (float)rand() / RAND_MAX; });
                multivector<float, m> x(ctx.queue(), n);
                multivector<float, m> y(ctx.queue(), host);
                multivector<float, m> z(ctx.queue(), host);
                Reductor<float,MIN> min(ctx.queue());
                Reductor<float,MAX> max(ctx.queue());

                std::array<int, m> v;
                for(uint i = 0; i < m; i++) v[i] = i;
                x = v;
                std::array<float, m> xmin = min(x);
                std::array<float, m> xmax = max(x);
                for(uint i = 0; i < m; i++) {
                    rc = rc && xmin[i] == v[i];
                    rc = rc && xmax[i] == v[i];
                }

                x = 2 * y + z;
                std::transform(host.begin(), host.end(), host.begin(), [](float x) {
                    return 2 * x + x;
                    });
                for(uint i = 0; i < m; i++) {
                    rc = rc && min(x(i)) == *min_element(
                        host.begin() + i * n, host.begin() + (i + 1) * n);
                    rc = rc && max(x(i)) == *max_element(
                        host.begin() + i * n, host.begin() + (i + 1) * n);
                }
                return rc;
        });
#endif

#if 1
        run_test("Multiexpressions with multivectors", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<float> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (float)rand() / RAND_MAX; });
                multivector<float, m> x(ctx.queue(), n);
                multivector<float, m> y(ctx.queue(), host);
                multivector<float, m> z(ctx.queue(), host);
                Reductor<float,MIN> min(ctx.queue());
                Reductor<float,MAX> max(ctx.queue());

                std::array<int, m> v;
                for(uint i = 0; i < m; i++) v[i] = i;
                x = v;
                std::array<float, m> xmin = min(x);
                std::array<float, m> xmax = max(x);
                for(uint i = 0; i < m; i++) {
                    rc = rc && xmin[i] == v[i];
                    rc = rc && xmax[i] == v[i];
                }

                x = std::tie(
                        2 * y(0) + z(0),
                        2 * y(1) + z(1),
                        2 * y(2) + z(2),
                        2 * y(3) + z(3)
                        );

                std::transform(host.begin(), host.end(), host.begin(), [](float x) {
                    return 2 * x + x;
                    });
                for(uint i = 0; i < m; i++) {
                    rc = rc && min(x(i)) == *min_element(
                        host.begin() + i * n, host.begin() + (i + 1) * n);
                    rc = rc && max(x(i)) == *max_element(
                        host.begin() + i * n, host.begin() + (i + 1) * n);
                }
                return rc;
        });
#endif

#if 1
        run_test("Tie vectors into a multivector", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                std::vector<double> host(n);
                std::generate(host.begin(), host.end(),
                    [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> x(ctx.queue(), host);
                vex::vector<double> y(ctx.queue(), host);

                vex::vector<double> a(ctx.queue(), n);
                vex::vector<double> b(ctx.queue(), n);

                vex::tie(a, b) = std::tie(x + y, x - y);

                std::vector<double> A(n);
                std::vector<double> B(n);

                vex::copy(a, A);
                vex::copy(b, B);

                double res = 0;

                for(uint i = 0; i < n; i++) {
                    res = std::max(res, fabs(A[i] - (host[i] + host[i])));
                    res = std::max(res, fabs(B[i] - (host[i] - host[i])));
                }

                rc = rc && (res < 1e-8);

                return rc;
        });
#endif

#if 1
        run_test("One-argument builtin function call for multivector", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<double> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (double)rand() / RAND_MAX; });
                multivector<double, m> x(ctx.queue(), n);
                multivector<double, m> y(ctx.queue(), host);
                x = cos(y);
                for(size_t k = 0; k < 10; k++) {
                    size_t i = rand() % n;
                    std::array<double,m> val = x[i];
                    for(uint j = 0; j < m; j++)
                        rc = rc && fabs(val[j] - cos(host[j * n + i])) < 1e-8;
                }
                return rc;
        });
#endif

#if 1
        run_test("Reduction of multivector", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<double> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (double)rand() / RAND_MAX; });
                multivector<double, m> x(ctx.queue(), host);
                Reductor<double,SUM> sum(ctx.queue());
                std::array<double,m> s = sum(x);
                for(uint i = 0; i < m; i++) {
                    rc = rc && fabs(
                        s[i] - std::accumulate(
                            host.begin() + i * n, host.begin() + (i + 1) * n, 0.0)
                        ) < 1e-6;
                }
                return rc;
                });
#endif

#if 1
        run_test("Sparse matrix-multivector product", [&]() -> bool {
                bool rc = true;
                const size_t n   = 32;
                const size_t N   = n * n * n;
                const size_t m   = 2;
                const double h2i = (n - 1) * (n - 1);

                std::vector<size_t> row;
                std::vector<size_t> col;
                std::vector<double> val;

                row.reserve(N + 1);
                col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + N);
                val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + N);

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

                std::vector<double> x(N * m);
                std::vector<double> y(N * m);
                std::generate(x.begin(), x.end(), []() {
                        return (double)rand() / RAND_MAX;
                        });

                vex::SpMat <double> A(ctx.queue(), N, N,
                        row.data(), col.data(), val.data());

                vex::multivector<double,m> X(ctx.queue(), x);
                vex::multivector<double,m> Y(ctx.queue(), N);

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(uint k = 0; k < m; k++)
                    for(size_t i = 0; i < N; i++) {
                        double sum = 0;
                        for(size_t j = row[i]; j < row[i + 1]; j++)
                            sum += val[j] * x[col[j] + k * N];
                        res = std::max(res, fabs(sum - y[i + k * N]));
                    }

                rc = rc && res < 1e-8;

                Y = X + A * X;
                copy(Y, y);

                res = 0;
                for(uint k = 0; k < m; k++)
                    for(size_t i = 0; i < N; i++) {
                        double sum = 0;
                        for(size_t j = row[i]; j < row[i + 1]; j++)
                            sum += val[j] * x[col[j] + k * N];
                        res = std::max(res, fabs(sum + x[i + k * N] - y[i + k * N]));
                    }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Sparse matrix-multivector product (CCSR format)", [&]() -> bool {
                bool rc = true;
                const uint n     = 32;
                const uint N     = n * n * n;
                const uint m     = 2;
                const double h2i = (n - 1) * (n - 1);

                std::vector<size_t> idx;
                std::vector<size_t> row(3);
                std::vector<int>    col(8);
                std::vector<double> val(8);

                idx.reserve(N);

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

                std::vector<double> x(N * m);
                std::vector<double> y(N * m);
                std::generate(x.begin(), x.end(), []() {
                        return (double)rand() / RAND_MAX;
                        });

                std::vector<cl::CommandQueue> q1(1, ctx.queue()[0]);

                vex::SpMatCCSR<double,int> A(q1[0], N, row.size() - 1,
                        idx.data(), row.data(), col.data(), val.data());

                vex::multivector<double,m> X(q1, x);
                vex::multivector<double,m> Y(q1, N);

                Y = A * X;
                copy(Y, y);

                double res = 0;
                for(uint k = 0; k < m; k++)
                    for(size_t i = 0; i < N; i++) {
                        double sum = 0;
                        for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
                            sum += val[j] * x[i + col[j] + k * N];
                        res = std::max(res, fabs(sum - y[i + k * N]));
                    }

                rc = rc && res < 1e-8;

                Y = X + A * X;
                copy(Y, y);

                res = 0;
                for(uint k = 0; k < m; k++)
                    for(size_t i = 0; i < N; i++) {
                        double sum = 0;
                        for(size_t j = row[idx[i]]; j < row[idx[i] + 1]; j++)
                            sum += val[j] * x[i + col[j] + k * N];
                        res = std::max(res, fabs(sum + x[i + k * N] - y[i + k * N]));
                    }

                rc = rc && res < 1e-8;

                return rc;
        });
#endif

#if 1
        run_test("Builtin function with two arguments", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                std::vector<double> x(N);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
                vex::vector<double> X(ctx.queue(), x);
                Reductor<double,SUM> sum(ctx.queue());
                rc = rc && 1e-8 > fabs(sum(pow(X, 2.0)) -
                    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
                        return s + pow(v, 2.0);
                        }));
                return rc;
                });
#endif

#if 1
        run_test("Custom function", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                vex::vector<double> y(ctx.queue(), N);
                x = 1;
                y = 2;
                Reductor<size_t,SUM> sum(ctx.queue());
                rc = rc && sum(greater(x, y)) == 0;
                rc = rc && sum(greater(y, x)) == N;
                rc = rc && sum(x > y) == 0;
                rc = rc && sum(x < y) == N;
                return rc;
                });
#endif

#if 1
        run_test("Custom functions with same signature", [&]() -> bool {
                const size_t N = 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                Reductor<size_t,SUM> sum(ctx.queue());
                x = 1;
                VEX_FUNCTION(times2, double(double), "return prm1 * 2;");
                VEX_FUNCTION(times4, double(double), "return prm1 * 4;");
                rc = rc && sum(times2(x)) == 2 * N;
                rc = rc && sum(times4(x)) == 4 * N;
                return rc;
            });
#endif

#if 1
        run_test("Two-arguments builtin function call for multivector", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                std::vector<double> host(n * m);
                std::generate(host.begin(), host.end(),
                    [](){ return (double)rand() / RAND_MAX; });
                multivector<double, m> x(ctx.queue(), n);
                multivector<double, m> y(ctx.queue(), host);
                x = pow(y, 2.0);
                for(size_t k = 0; k < 10; k++) {
                    size_t i = rand() % n;
                    std::array<double,m> val = x[i];
                    for(uint j = 0; j < m; j++)
                        rc = rc && fabs(val[j] - pow(host[j * n + i], 2.0)) < 1e-8;
                }
                return rc;
        });
#endif

#if 1
        run_test("Custom function for multivector", [&]() -> bool {
                bool rc = true;
                const size_t n = 1024;
                const size_t m = 4;
                multivector<double, m> x(ctx.queue(), n);
                multivector<double, m> y(ctx.queue(), n);
                x = 1;
                y = 2;
                x = greater(x, y);
                for(size_t k = 0; k < 10; k++) {
                    size_t i = rand() % n;
                    std::array<double,m> val = x[i];
                    for(uint j = 0; j < m; j++)
                        rc = rc && val[j] == 0;
                }
                return rc;
                });
#endif

#if 1
        run_test("Stencil convolution", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 20;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx.queue(), s, center);

                std::vector<double> x(n);
                std::vector<double> y(n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), y);

                Y += X * S;

                copy(Y, y);

                double res = 0;
                for(int i = 0; i < n; i++) {
                    double sum = 1;
                    for(int k = -center; k < (int)s.size() - center; k++)
                        sum += s[k + center] * x[std::min(n-1,std::max(0, i + k))];
                    res = std::max(res, fabs(sum - y[i]));
                }
                rc = rc && res < 1e-8;

                Y = 42 * (X * S);

                copy(Y, y);

                res = 0;
                for(int i = 0; i < n; i++) {
                    double sum = 0;
                    for(int k = -center; k < (int)s.size() - center; k++)
                        sum += s[k + center] * x[std::min(n-1,std::max(0, i + k))];
                    res = std::max(res, fabs(42 * sum - y[i]));
                }
                rc = rc && res < 1e-8;

                return rc;
                });
#endif

#if 1
        run_test("Two Stencil convolutions in one expression", [&]() -> bool {
                const int n = 32;
                std::vector<double> s(5);
                stencil<double> S(ctx.queue(), s, 3);
                vex::vector<double> X(ctx.queue(), n);
                vex::vector<double> Y(ctx.queue(), n);
                Y = X * S + X * S;
                return true;
            });
#endif

#if 1
        run_test("Stencil convolution with small vector", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 7;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx.queue(), s, center);

                std::vector<double> x(n);
                std::vector<double> y(n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), y);

                Y += X * S;

                copy(Y, y);

                double res = 0;
                for(int i = 0; i < n; i++) {
                    double sum = 1;
                    for(int k = -center; k < (int)s.size() - center; k++)
                        sum += s[k + center] * x[std::min(n-1,std::max(0, i + k))];
                    res = std::max(res, fabs(sum - y[i]));
                }
                rc = rc && res < 1e-8;
                return rc;
                });
#endif

#if 1
        run_test("Stencil convolution with multivector", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 16;
                const int m = 20;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx.queue(), s.begin(), s.end(), center);

                std::vector<double> x(m * n);
                std::vector<double> y(m * n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::multivector<double,m> X(ctx.queue(), x);
                vex::multivector<double,m> Y(ctx.queue(), y);

                Y += X * S;

                copy(Y, y);

                for(int c = 0; c < m; c++) {
                    double res = 0;
                    for(int i = 0; i < n; i++) {
                        double sum = 1;
                        for(int k = -center; k < (int)s.size() - center; k++)
                            sum += s[k + center] * x[c * n + std::min(n-1,std::max(0, i + k))];
                        res = std::max(res, fabs(sum - y[i + c * n]));
                    }
                    rc = rc && res < 1e-8;
                }

                Y = 42 * (X * S);

                copy(Y, y);

                for(int c = 0; c < m; c++) {
                    double res = 0;
                    for(int i = 0; i < n; i++) {
                        double sum = 0;
                        for(int k = -center; k < (int)s.size() - center; k++)
                            sum += s[k + center] * x[c * n + std::min(n-1,std::max(0, i + k))];
                        res = std::max(res, fabs(42 * sum - y[i + c * n]));
                    }
                    rc = rc && res < 1e-8;
                }

                return rc;
                });
#endif

#if 1
        run_test("Big stencil convolution", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 16;

                std::vector<double> s(2048);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx.queue(), s, center);

                std::vector<double> x(n);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), n);

                Y = X * S;

                copy(Y, y);

                double res = 0;
                for(int i = 0; i < n; i++) {
                    double sum = 0;
                    for(int k = -center; k < (int)s.size() - center; k++)
                        sum += s[k + center] * x[std::min(n-1,std::max(0, i + k))];
                    res = std::max(res, fabs(sum - y[i]));
                }
                rc = rc && res < 1e-8;
                return rc;
                });
#endif

#if 1
        run_test("User-defined stencil operator", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 20;

                VEX_STENCIL_OPERATOR(pow3_op,
                    double, 3, 1,  "return X[0] + pow(X[-1] + X[1], 3.0);",
                    ctx.queue());

                std::vector<double> x(n);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);
                vex::vector<double> Y(ctx.queue(), n);

                Y = pow3_op(X);

                copy(Y, y);

                double res = 0;
                for(int i = 0; i < n; i++) {
                    int left  = std::max(0, i - 1);
                    int right = std::min(n - 1, i + 1);

                    double sum = x[i] + pow(x[left] + x[right], 3.0);
                    res = std::max(res, fabs(sum - y[i]));
                }

                Y = 41 * pow3_op(X) + pow3_op(X);

                copy(Y, y);

                res = 0;
                for(int i = 0; i < n; i++) {
                    int left  = std::max(0, i - 1);
                    int right = std::min(n - 1, i + 1);

                    double sum = x[i] + pow(x[left] + x[right], 3.0);
                    res = std::max(res, fabs(42 * sum - y[i]));
                }

                rc = rc && res < 1e-8;
                return rc;
                });
#endif

#if 1
        run_test("Kernel auto-generation", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 20;

                std::ostringstream body;
                generator::set_recorder(body);

                typedef generator::symbolic<double> sym_state;

                double dt = 0.01;
                sym_state sym_x(sym_state::VectorParameter);

                // Record expression sequience.
                runge_kutta_4(sys_func<sym_state>, sym_x, dt);

                // Build kernel.
                auto kernel = generator::build_kernel(ctx.queue(),
                    "rk4_stepper", body.str(), sym_x);

                std::vector<double> x(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);

                // Make 100 iterations on CPU with x[0].
                for(int i = 0; i < 100; i++)
                    runge_kutta_4(sys_func<double>, x[0], dt);

                // Make 100 iterations on GPU with full X vector.
                for(int i = 0; i < 100; i++)
                    kernel(X);

                // Compare results.
                rc = rc && fabs(x[0] - X[0]) < 1e-8;
                return rc;
                });
#endif

#if 1
        run_test("Gather scattered points from vector", [&]() -> bool {
                const size_t N = 1 << 20;
                const size_t M = 100;
                bool rc = true;

                std::vector<double> x(N);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx.queue(), x);

                std::vector<size_t> i(M);
                std::generate(i.begin(), i.end(), [N](){ return rand() % N; });
                std::sort(i.begin(), i.end());
                i.resize( std::unique(i.begin(), i.end()) - i.begin() );

                std::vector<double> data(i.size());

                vex::gather<double> get(ctx.queue(), x.size(), i);

                get(X, data);

                for(size_t p = 0; p < i.size(); ++p)
                    rc = rc && data[p] == x[i[p]];

                return rc;
                });
#endif

#if 1
        run_test("Use element index in a vector expression", [&]() -> bool {
                const size_t N = 16 * 1024;
                bool rc = true;
                vex::vector<double> x(ctx.queue(), N);
                x = sin(0.5 * element_index());
                for(int i = 0; i < 100; ++i) {
                    size_t idx = rand() % N;
                    rc = rc && fabs(x[idx] - sin(0.5 * idx)) < 1e-8;
                }
                return rc;
                });
#endif

#if 1
        run_test("Use element index in a multivector expression", [&]() -> bool {
                const size_t N = 16 * 1024;
                bool rc = true;
                vex::multivector<double, 2> x(ctx.queue(), N);
                vex::multivector<double, 2> y(ctx.queue(), N);
                x = std::tie(
                    sin(0.5 * element_index()),
                    cos(0.5 * element_index())
                    );
                y = 0.5 * element_index();

                for(int i = 0; i < 100; ++i) {
                    size_t idx = rand() % N;

                    std::array<double,2> vx = x[idx];
                    std::array<double,2> vy = y[idx];

                    rc = rc && fabs(vx[0] - sin(0.5 * idx)) < 1e-8;
                    rc = rc && fabs(vx[1] - cos(0.5 * idx)) < 1e-8;

                    rc = rc && fabs(vy[0] - 0.5 * idx) < 1e-8;
                    rc = rc && fabs(vy[1] - 0.5 * idx) < 1e-8;
                }
                return rc;
                });
#endif

#if 1
        run_test("Arithmetic with OpenCL vector values", [&]() -> bool {
                const size_t N = 16 * 1024;
                bool rc = true;

                vex::vector<cl_int4> X(ctx.queue(), N);

                cl_int4 c = {{1, 2, 3, 4}};
                VEX_FUNCTION(make_int4, cl_int4(int), "return (int4)(prm1, prm1, prm1, prm1);");
                X = c * (make_int4(5 + element_index()));

                for(int i = 0; i < 100; ++i) {
                    size_t idx = rand() % N;

                    cl_int4 v = X[idx];

                    for(int j = 0; j < 4; ++j)
                    rc = rc && (v.s[j] - c.s[j] * (5 + idx)) == 0;
                }

                return rc;
                });
#endif

#if 1
        run_test("Random generator", [&]() -> bool {
                const size_t N = 1024 * 1024;
                bool rc = true;
                Reductor<size_t,SUM> sumi(ctx.queue());
                Reductor<double,SUM> sumd(ctx.queue());

                vex::vector<cl_uint> x0(ctx.queue(), N);
                Random<cl_int> rand0;
                x0 = rand0(element_index(), rand());

                vex::vector<cl_float8> x1(ctx.queue(), N);
                Random<cl_float8> rand1;
                x1 = rand1(element_index(), rand());

                vex::vector<cl_double4> x2(ctx.queue(), N);
                Random<cl_double4> rand2;
                x2 = rand2(element_index(), rand());

                vex::vector<cl_double> x3(ctx.queue(), N);
                Random<cl_double> rand3;
                x3 = rand3(element_index(), rand());
                // X in [0,1]
                rc = rc && sumi(x3 > 1) == 0;
                rc = rc && sumi(x3 < 0) == 0;
                // mean = 0.5
                rc = rc && std::abs((sumd(x3) / N) - 0.5) < 1e-2;

                vex::vector<cl_double> x4(ctx.queue(), N);
                RandomNormal<cl_double> rand4;
                x4 = rand4(element_index(), rand());
                // E(X ~ N(0,s)) = 0
                rc = rc && std::abs(sumd(x4)/N) < 1e-2;
                // E(abs(X) ~ N(0,s)) = sqrt(2/M_PI) * s
                rc = rc && std::abs(sumd(fabs(x4))/N - std::sqrt(2 / M_PI)) < 1e-2;

                vex::vector<cl_double> x5(ctx.queue(), N);
                Random<cl_double, random::threefry> rand5;
                x5 = rand5(element_index(), rand());
                rc = rc && std::abs(sumd(x5)/N - 0.5) < 1e-2;

                vex::vector<cl_double4> x6(ctx.queue(), N);
                Random<cl_double, random::threefry> rand6;
                x6 = rand6(element_index(), rand());
                return rc;
                });
#endif

#if 1
        run_test("Nested function invocation", [&]() -> bool {
                bool rc = true;
                const size_t N = 1024;
                VEX_FUNCTION(f, int(int), "return 2 * prm1;");
                VEX_FUNCTION(g, int(int), "return 3 * prm1;");

                vex::vector<int> data(ctx.queue(), N);

                data = 1;
                data = f(f(data));
                rc = rc && data[0] == 4;

                data = 1;
                data = g(f(data));
                rc = rc && data[0] == 6;

                return rc;
                }); 
#endif

#if 1
        run_test("FFT", [&]() -> bool {
               bool rc = true;
               const size_t N = 1024;

               {
                   vex::vector<cl_float> data(single_queue, N);
                   FFT<cl_float> fft(single_queue, N);
                   // should compile
                   data += fft(data * data) * 5;
               }

               {
                   vex::vector<cl_float> in(single_queue, N);
                   vex::vector<cl_float2> out(single_queue, N);
                   vex::vector<cl_float> back(single_queue, N);
                   Random<cl_float> randf;
                   in = randf(element_index(), rand());
                   FFT<cl_float, cl_float2> fft(single_queue, N);
                   FFT<cl_float2, cl_float> ifft(single_queue, N, inverse);
                   out = fft(in);
                   back = ifft(out);
                   Reductor<cl_float, SUM> sum(single_queue);
                   float rms = std::sqrt(sum(pow(in - back, 2)) / N);
                   rc = rc && rms < 1e-3;
               }

               return rc;
            });
#endif
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error: " << err << std::endl;
        return 1;
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }

    return !all_passed;
}

// vim: et
