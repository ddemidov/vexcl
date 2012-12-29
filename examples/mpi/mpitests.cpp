#define _GLIBCXX_USE_NANOSLEEP
#include <iostream>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/mpi/mpi.hpp>

static bool all_passed = true;
vex::mpi::comm_data mpi;

bool run_test(const std::string &name, std::function<bool()> test) {
    char fc = std::cout.fill('.');
    if (mpi.rank == 0)
        std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc = test();
    bool glob_rc;

    MPI_Allreduce(&rc, &glob_rc, 1, MPI_C_BOOL, MPI_LAND, mpi.comm);

    all_passed = all_passed && glob_rc;
    if (mpi.rank == 0)
        std::cout << (rc ? " success." : " failed.") << std::endl;
    return rc;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    mpi = vex::mpi::comm_data(MPI_COMM_WORLD);

    if (mpi.rank == 0) std::cout << "World size: " << mpi.size << std::endl;

    try {
        vex::Context ctx( vex::Filter::Exclusive(
                    vex::Filter::Env && vex::Filter::Count(1) ) );

        mpi.precondition(!ctx.empty(), "No OpenCL devices found");

        for(int i = 0; i < mpi.size; ++i) {
            if (i == mpi.rank)
                std::cout << mpi.rank << ": " << ctx.device(0) << std::endl;

            MPI_Barrier(mpi.comm);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (mpi.rank == 0) std::cout << std::endl;

        run_test("Allocate mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);

                rc = rc && x.local_size() == n;
                rc = rc && x.global_size() == n * mpi.size;

                return rc;
                });

        run_test("Assign constant to mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);

                x = 42;

                rc = rc && x[n/2] == 42;

                return rc;
                });

        run_test("Copy constructor for mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);
                x = 42;

                vex::mpi::vector<double> y = x;

                rc = rc && y[n/2] == 42;

                return rc;
                });

        run_test("Assign arithmetic expression to mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> y(mpi.comm, ctx.queue(), n);

                x = 42;
                y = vex::cos(x / 7);

                rc = rc && fabs(y[n/2] - cos(6.0)) < 1e-8;

                return rc;
                });

        run_test("Reduce mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);
                vex::mpi::Reductor<double, vex::SUM> sum(mpi.comm, ctx.queue());

                x = 1;

                rc = rc && fabs(sum(x) - x.global_size()) < 1e-8;

                return rc;
                });

        run_test("Allocate mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(mpi.comm, ctx.queue(), n);

                rc = rc && x.local_size() == n;
                rc = rc && x.global_size() == n * mpi.size;

                return rc;
                });

        run_test("Assign constant to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(mpi.comm, ctx.queue(), n);

                x = 42;

                rc = rc && x(0)[n/2] == 42;
                rc = rc && x(1)[n/2] == 42;
                rc = rc && x(2)[n/2] == 42;

                x = std::make_tuple(6, 7, 42);

                rc = rc && x(0)[n/2] == 6;
                rc = rc && x(1)[n/2] == 7;
                rc = rc && x(2)[n/2] == 42;

                return rc;
                });

        run_test("Assign arithmetic expression to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(mpi.comm, ctx.queue(), n);
                vex::mpi::multivector<double,m> y(mpi.comm, ctx.queue(), n);

                x = std::make_tuple(6, 7, 42);
                y = vex::cos(x / 7);

                rc = rc && fabs(y(0)[n/2] - cos(6.0  / 7.0)) < 1e-8;
                rc = rc && fabs(y(1)[n/2] - cos(7.0  / 7.0)) < 1e-8;
                rc = rc && fabs(y(2)[n/2] - cos(42.0 / 7.0)) < 1e-8;

                return rc;
                });

        run_test("Reduce mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double, m> x(mpi.comm, ctx.queue(), n);
                vex::mpi::Reductor<double, vex::SUM> sum(mpi.comm, ctx.queue());

                x = std::make_tuple(1, 2, 3);

                auto s = sum(x);

                rc = rc && fabs(s[0] - 1 * x.global_size()) < 1e-8;
                rc = rc && fabs(s[1] - 2 * x.global_size()) < 1e-8;
                rc = rc && fabs(s[2] - 3 * x.global_size()) < 1e-8;

                return rc;
                });

        run_test("Assign multiexpression to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double, m> x(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> y0(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> y1(mpi.comm, ctx.queue(), n);

                y0 = 1;
                y1 = 2;

                int the_answer = 42;

                x = std::tie(y1 - y0, y1 + y0, the_answer);

                rc = rc && x(0)[42] == 1;
                rc = rc && x(1)[42] == 3;
                rc = rc && x(2)[42] == 42;

                return rc;
                });

        run_test("Assign multiexpression to tied mpi::vectors", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x0(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> x1(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> x2(mpi.comm, ctx.queue(), n);

                vex::mpi::vector<double> y0(mpi.comm, ctx.queue(), n);
                vex::mpi::vector<double> y1(mpi.comm, ctx.queue(), n);

                y0 = 1;
                y1 = 2;

                int the_answer = 42;

                vex::tie(x0, x1, x2) = std::tie(y1 - y0, y1 + y0, the_answer);

                rc = rc && x0[42] == 1;
                rc = rc && x1[42] == 3;
                rc = rc && x2[42] == 42;

                return rc;
                });
        run_test("Matrix-vector product", [&]() -> bool {
                const size_t n  = 1024;
                const size_t m  = 3;
                const size_t n2 = n * n;
                bool rc = true;

                size_t chunk_size  = (n2 + mpi.size - 1) / mpi.size;
                size_t chunk_start = chunk_size * mpi.rank;
                size_t chunk_end   = std::min(n2, chunk_start + chunk_size);

                chunk_size = chunk_end - chunk_start;

                auto part = mpi.restore_partitioning(chunk_size);

                std::vector<int>    row;
                std::vector<int>    col;
                std::vector<double> val;

                row.reserve(chunk_size + 1);
                col.reserve(5 * chunk_size);
                val.reserve(5 * chunk_size);

                row.push_back(0);

                for(size_t idx = part[mpi.rank]; idx < part[mpi.rank + 1]; ++idx) {
                    size_t i = idx % n;
                    size_t j = idx / n;

                    if (i == 0 || i + 1 == n || j == 0 || j + 1 == n) {
                        col.push_back(idx);
                        val.push_back(0);
                    } else {
                        col.push_back(idx - n);
                        val.push_back(-1);

                        col.push_back(idx - 1);
                        val.push_back(-1);

                        col.push_back(idx);
                        val.push_back(4.5);

                        col.push_back(idx + 1);
                        val.push_back(-1);

                        col.push_back(idx + n);
                        val.push_back(-1);
                    }

                    row.push_back(col.size());
                }

                vex::mpi::SpMat<double, int, int> A(mpi.comm, ctx.queue(),
                        chunk_size, chunk_size,
                        row.data(), col.data(), val.data()
                        );

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), chunk_size);
                vex::mpi::vector<double> y(mpi.comm, ctx.queue(), chunk_size);

                vex::mpi::multivector<double, m> mx(mpi.comm, ctx.queue(), chunk_size);
                vex::mpi::multivector<double, m> my(mpi.comm, ctx.queue(), chunk_size);

                x = 1;
                y = A * x;

                mx = std::make_tuple(1, 2, 3);
                my = A * mx;

                vex::mpi::Reductor<double, vex::MIN> min(mpi.comm, ctx.queue());
                vex::mpi::Reductor<double, vex::MAX> max(mpi.comm, ctx.queue());
                vex::mpi::Reductor<double, vex::SUM> sum(mpi.comm, ctx.queue());

                rc = rc && min(y) == 0;
                rc = rc && max(y) == 0.5;
                rc = rc && sum(y) == 0.5 * (n - 2) * (n - 2);

                auto vmin = min(my);
                auto vmax = max(my);
                auto vsum = sum(my);

                for(int i = 0; i < m; ++i) {
                    rc = rc && vmin[i] == 0;
                    rc = rc && vmax[i] == 0.5 * (i + 1);
                    rc = rc && vsum[i] == 0.5 * (i + 1) * (n - 2) * (n - 2);
                }

                return rc;
                });

        run_test("Use element_index in mpi::vector expression", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(mpi.comm, ctx.queue(), n);

                auto part = mpi.restore_partitioning(n);

                x = vex::element_index(part[mpi.rank]);

                for (int i = 0; i < 100; ++i) {
                    size_t idx = rand() % n;
                    rc = rc && x[idx] == part[mpi.rank] + idx;
                }

                return rc;
                });

    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error (" << mpi.rank << "): " << err << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error (" << mpi.rank << "): " << err.what() << std::endl;
    }

    MPI_Finalize();
    return !all_passed;
}
