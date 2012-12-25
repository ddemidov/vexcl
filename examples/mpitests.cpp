#define _GLIBCXX_USE_NANOSLEEP
#include <iostream>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/mpi/vector.hpp>
#include <vexcl/mpi/reduce.hpp>

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

        if (!ctx.size())
            throw std::runtime_error("No OpenCL device found");

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
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error (" << mpi_rank << "): " << err << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error (" << mpi_rank << "): " << err.what() << std::endl;
    }

    MPI_Finalize();
    return !all_passed;
}
