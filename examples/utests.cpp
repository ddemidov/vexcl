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

#define TESTS_ON

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

        if (ctx.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::vector<cl::CommandQueue> single_queue(1, ctx.queue(0));

        uint seed = argc > 1 ? atoi(argv[1]) : static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl << std::endl;
        srand(seed);

#ifdef TESTS_ON
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

                vex::SpMat <double> A(ctx, x.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, x.size());

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


#ifdef TESTS_ON
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

                vex::SpMat <double> A(ctx, y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, y.size());

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

#ifdef TESTS_ON
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

                vex::SpMat <double,unsigned int, unsigned int> A(ctx, y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, y.size());

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

#ifdef TESTS_ON
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

                vex::SpMat <double> A(ctx, y.size(), x.size(), row.data(), col.data(), val.data());
                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, y.size());

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

#ifdef TESTS_ON
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

                vex::SpMatCCSR<double,int> A(ctx.queue(0), x.size(), row.size() - 1,
                        idx.data(), row.data(), col.data(), val.data());

                vex::vector<double> X(single_queue, x);
                vex::vector<double> Y(single_queue, x.size());

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

#ifdef TESTS_ON
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

                vex::SpMat <double> A(ctx, N, N,
                        row.data(), col.data(), val.data());

                vex::multivector<double,m> X(ctx, x);
                vex::multivector<double,m> Y(ctx, N);

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

#ifdef TESTS_ON
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

                vex::SpMatCCSR<double,int> A(ctx.queue(0), N, row.size() - 1,
                        idx.data(), row.data(), col.data(), val.data());

                vex::multivector<double,m> X(single_queue, x);
                vex::multivector<double,m> Y(single_queue, N);

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

#ifdef TESTS_ON
        run_test("Stencil convolution", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 20;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx, s, center);

                std::vector<double> x(n);
                std::vector<double> y(n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, y);

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

#ifdef TESTS_ON
        run_test("Two Stencil convolutions in one expression", [&]() -> bool {
                const int n = 32;
                std::vector<double> s(5);
                stencil<double> S(ctx, s, 3);
                vex::vector<double> X(ctx, n);
                vex::vector<double> Y(ctx, n);
                Y = X * S + X * S;
                return true;
            });
#endif

#ifdef TESTS_ON
        run_test("Stencil convolution with small vector", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 7;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx, s, center);

                std::vector<double> x(n);
                std::vector<double> y(n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, y);

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

#ifdef TESTS_ON
        run_test("Stencil convolution with multivector", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 16;
                const int m = 20;

                std::vector<double> s(rand() % 64 + 1);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx, s.begin(), s.end(), center);

                std::vector<double> x(m * n);
                std::vector<double> y(m * n, 1);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::multivector<double,m> X(ctx, x);
                vex::multivector<double,m> Y(ctx, y);

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

#ifdef TESTS_ON
        run_test("Big stencil convolution", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 16;

                std::vector<double> s(2048);
                std::generate(s.begin(), s.end(), [](){ return (double)rand() / RAND_MAX; });

                int center = rand() % s.size();

                stencil<double> S(ctx, s, center);

                std::vector<double> x(n);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, n);

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

#ifdef TESTS_ON
        run_test("User-defined stencil operator", [&]() -> bool {
                bool rc = true;
                const int n = 1 << 20;

                VEX_STENCIL_OPERATOR(pow3_op,
                    double, 3, 1,  "return X[0] + pow(X[-1] + X[1], 3.0);",
                    ctx);

                std::vector<double> x(n);
                std::vector<double> y(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);
                vex::vector<double> Y(ctx, n);

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

#ifdef TESTS_ON
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
                auto kernel = generator::build_kernel(ctx,
                    "rk4_stepper", body.str(), sym_x);

                std::vector<double> x(n);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);

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

#ifdef TESTS_ON
        run_test("Gather scattered points from vector", [&]() -> bool {
                const size_t N = 1 << 20;
                const size_t M = 100;
                bool rc = true;

                std::vector<double> x(N);
                std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

                vex::vector<double> X(ctx, x);

                std::vector<size_t> i(M);
                std::generate(i.begin(), i.end(), [N](){ return rand() % N; });
                std::sort(i.begin(), i.end());
                i.resize( std::unique(i.begin(), i.end()) - i.begin() );

                std::vector<double> data(i.size());

                vex::gather<double> get(ctx, x.size(), i);

                get(X, data);

                for(size_t p = 0; p < i.size(); ++p)
                    rc = rc && data[p] == x[i[p]];

                return rc;
                });
#endif

#ifdef TESTS_ON
        run_test("Random generator", [&]() -> bool {
                const size_t N = 1024 * 1024;
                bool rc = true;
                Reductor<size_t,SUM> sumi(ctx);
                Reductor<double,SUM> sumd(ctx);

                vex::vector<cl_uint> x0(ctx, N);
                Random<cl_int> rand0;
                x0 = rand0(element_index(), rand());

                vex::vector<cl_float8> x1(ctx, N);
                Random<cl_float8> rand1;
                x1 = rand1(element_index(), rand());

                vex::vector<cl_double4> x2(ctx, N);
                Random<cl_double4> rand2;
                x2 = rand2(element_index(), rand());

                vex::vector<cl_double> x3(ctx, N);
                Random<cl_double> rand3;
                x3 = rand3(element_index(), rand());
                // X in [0,1]
                rc = rc && sumi(x3 > 1) == 0;
                rc = rc && sumi(x3 < 0) == 0;
                // mean = 0.5
                rc = rc && std::abs((sumd(x3) / N) - 0.5) < 1e-2;

                vex::vector<cl_double> x4(ctx, N);
                RandomNormal<cl_double> rand4;
                x4 = rand4(element_index(), rand());
                // E(X ~ N(0,s)) = 0
                rc = rc && std::abs(sumd(x4)/N) < 1e-2;
                // E(abs(X) ~ N(0,s)) = sqrt(2/M_PI) * s
                rc = rc && std::abs(sumd(fabs(x4))/N - std::sqrt(2 / M_PI)) < 1e-2;

                vex::vector<cl_double> x5(ctx, N);
                Random<cl_double, random::threefry> rand5;
                x5 = rand5(element_index(), rand());
                rc = rc && std::abs(sumd(x5)/N - 0.5) < 1e-2;

                vex::vector<cl_double4> x6(ctx, N);
                Random<cl_double, random::threefry> rand6;
                x6 = rand6(element_index(), rand());
                return rc;
                });
#endif

#ifdef TESTS_ON
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
                   float rms = std::sqrt(sum(pow(in - back, 2.0f)) / N);
                   rc = rc && rms < 1e-3;
               }

               return rc;
            });
#endif

#ifdef TESTS_ON
        run_test("Test global program header", [&]() -> bool {
               bool rc = true;
               const size_t N = 1 << 20;

               vex::vector<int> x(ctx, N);

               vex::set_program_header(ctx, "#define THE_ANSWER 42\n");

               VEX_FUNCTION(answer, int(int), "return prm1 * THE_ANSWER;");

               x = answer(1);

               for(int i = 0; i < 100; ++i) {
                   size_t idx = rand() % N;
                   rc = rc && x[idx] == 42;
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
