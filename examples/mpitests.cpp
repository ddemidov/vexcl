#include <iostream>
#include <mpi.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/mpi/vector.hpp>

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
        vex::Context ctx( vex::Filter::Exclusive(
                    vex::Filter::Env && vex::Filter::Count(1)
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

        if (mpi_rank == 0) std::cout << std::endl;

        run_test("Allocate mpi::vector", [&]() -> bool {
                const size_t n = 1024;
                bool rc = true;

                vex::mpi::vector<double> x(MPI_COMM_WORLD, ctx.queue(), n);

                rc = rc && x.local_size() == n;
                rc = rc && x.size() == n * mpi_size;

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
