#define _GLIBCXX_USE_NANOSLEEP
#include <iostream>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/mpi/mpi.hpp>

using namespace vex;

int mpi_rank;
int mpi_size;

static bool all_passed = true;

bool run_test(const std::string &name, std::function<bool()> test) {
    char fc = std::cout.fill('.');
    if (mpi_rank == 0)
        std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc = test();
    bool glob_rc;

    MPI_Allreduce(&rc, &glob_rc, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    all_passed = all_passed && glob_rc;
    if (mpi_rank == 0)
        std::cout << (rc ? " success." : " failed.") << std::endl;
    return rc;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0) 
        std::cout << "World size: " << mpi_size << std::endl;

    try {
        vex::Context ctx( Filter::Exclusive(
                    Filter::Env && Filter::Count(1)
                    ) );

        vex::mpi::precondition(MPI_COMM_WORLD, ctx.size() > 0, "No OpenCL device found");

        for(int i = 0; i < mpi_size; ++i) {
            if (i == mpi_rank)
                std::cout << mpi_rank << ": "
                          << ctx.device(0).getInfo<CL_DEVICE_NAME>()
                          << std::endl;

            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (mpi_rank == 0) std::cout << std::endl;

        run_test("Allocate mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);

                rc = rc && x.local_size() == n;
                rc = rc && x.size() == n * mpi_size;

                return rc;
                });

        run_test("Assign constant to mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);

                x = 42;

                rc = rc && x.data()[n/2] == 42;

                return rc;
                });

        run_test("Copy constructor for mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);
                x = 42;

                vex::mpi::vector<double> y = x;

                rc = rc && y.data()[n/2] == 42;

                return rc;
                });

        run_test("Assign arithmetic expression to mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> y(MPI_COMM_WORLD, ctx.queue(), n);

                x = 42;
                y = cos(x / 7);

                rc = rc && fabs(y.data()[n/2] - cos(6.0)) < 1e-8;

                return rc;
                });

        run_test("Reduce mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::Reductor<double, vex::SUM> sum(MPI_COMM_WORLD, ctx.queue());

                x = 1;

                rc = rc && fabs(sum(x) - x.size()) < 1e-8;

                return rc;
                });

        run_test("Allocate mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(MPI_COMM_WORLD, ctx.queue(), n);

                rc = rc && x.local_size() == n;
                rc = rc && x.size() == n * mpi_size;

                return rc;
                });

        run_test("Assign constant to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(MPI_COMM_WORLD, ctx.queue(), n);

                x = 42;

                rc = rc && x.data()(0)[n/2] == 42;
                rc = rc && x.data()(1)[n/2] == 42;
                rc = rc && x.data()(2)[n/2] == 42;

                x = std::make_tuple(6, 7, 42);

                rc = rc && x.data()(0)[n/2] == 6;
                rc = rc && x.data()(1)[n/2] == 7;
                rc = rc && x.data()(2)[n/2] == 42;

                return rc;
                });

        run_test("Assign arithmetic expression to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double,m> x(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::multivector<double,m> y(MPI_COMM_WORLD, ctx.queue(), n);

                x = std::make_tuple(6, 7, 42);
                y = cos(x / 7);

                rc = rc && fabs(y.data()(0)[n/2] - cos(6.0  / 7.0)) < 1e-8;
                rc = rc && fabs(y.data()(1)[n/2] - cos(7.0  / 7.0)) < 1e-8;
                rc = rc && fabs(y.data()(2)[n/2] - cos(42.0 / 7.0)) < 1e-8;

                return rc;
                });

        run_test("Reduce mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double, m> x(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::Reductor<double, vex::SUM> sum(MPI_COMM_WORLD, ctx.queue());

                x = std::make_tuple(1, 2, 3);

                auto s = sum(x);

                rc = rc && fabs(s[0] - 1 * x.size()) < 1e-8;
                rc = rc && fabs(s[1] - 2 * x.size()) < 1e-8;
                rc = rc && fabs(s[2] - 3 * x.size()) < 1e-8;

                return rc;
                });

        run_test("Assign multiexpression to mpi::multivector", [&]() -> bool {
                const size_t n = 1024;
                const size_t m = 3;
                bool rc = true;

                vex::mpi::multivector<double, m> x(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> y0(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> y1(MPI_COMM_WORLD, ctx.queue(), n);

                y0 = 1;
                y1 = 2;

                int the_answer = 42;

                x = std::tie(y1 - y0, y1 + y0, the_answer);

                rc = rc && x(0).data()[42] == 1;
                rc = rc && x(1).data()[42] == 3;
                rc = rc && x(2).data()[42] == 42;

                return rc;
                });

        run_test("Assign multiexpression to tied mpi::vectors", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x0(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> x1(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> x2(MPI_COMM_WORLD, ctx.queue(), n);

                vex::mpi::vector<double> y0(MPI_COMM_WORLD, ctx.queue(), n);
                vex::mpi::vector<double> y1(MPI_COMM_WORLD, ctx.queue(), n);

                y0 = 1;
                y1 = 2;

                int the_answer = 42;

                vex::tie(x0, x1, x2) = std::tie(y1 - y0, y1 + y0, the_answer);

                rc = rc && x0.data()[42] == 1;
                rc = rc && x1.data()[42] == 3;
                rc = rc && x2.data()[42] == 42;

                return rc;
                });
        run_test("Matrix-vector product", [&]() -> bool {
                const size_t n  = 1024;
                const size_t n2 = n * n;
                bool rc = true;

                size_t chunk_size  = (n2 + mpi_size - 1) / mpi_size;
                size_t chunk_start = chunk_size * mpi_rank;
                size_t chunk_end   = std::min(n2, chunk_start + chunk_size);

                chunk_size = chunk_end - chunk_start;

                auto part = vex::mpi::restore_partitioning(MPI_COMM_WORLD, chunk_size);

                std::vector<int>    row;
                std::vector<int>    col;
                std::vector<double> val;


                row.reserve(chunk_size + 1);
                col.reserve(5 * chunk_size);
                val.reserve(5 * chunk_size);

                row.push_back(0);

                for(size_t idx = part[mpi_rank]; idx < part[mpi_rank + 1]; ++idx) {
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

                vex::mpi::SpMat<double, int, int> A(MPI_COMM_WORLD, ctx.queue(),
                        chunk_size, chunk_size,
                        row.data(), col.data(), val.data()
                        );

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), chunk_size);
                vex::mpi::vector<double> y(MPI_COMM_WORLD, ctx.queue(), chunk_size);

                x = 1;
                y = 1;

                A.mul(x, y);

                vex::mpi::Reductor<double, vex::MIN> min(MPI_COMM_WORLD, ctx.queue());
                vex::mpi::Reductor<double, vex::MAX> max(MPI_COMM_WORLD, ctx.queue());
                vex::mpi::Reductor<double, vex::SUM> sum(MPI_COMM_WORLD, ctx.queue());

                rc = rc && min(y) == 0;
                rc = rc && max(y) == 0.5;
                rc = rc && sum(y) == 0.5 * (n - 2) * (n - 2);

                return rc;
                });

    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error (" << mpi_rank << "): " << err << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error (" << mpi_rank << "): " << err.what() << std::endl;
    }

    MPI_Finalize();
    return !all_passed;
}
