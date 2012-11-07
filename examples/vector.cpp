#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>

#ifdef WIN32
#  pragma warning (disable : 4244)
#endif

using namespace vex;

int main() {
    srand(time(0));

    try {
        // Select every device supporting double precision.
        vex::Context ctx(Filter::All);

        if (ctx.queue().empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::cout << ctx << std::endl;

        // Allocate host vector, fill it with random values.
        const size_t N = 1024 * 1024;
        std::vector<double> host_vec(N);
        std::generate(host_vec.begin(), host_vec.end(), []() {
                return double(rand()) / RAND_MAX;
                });

        // Allocate device vector initialized with host vector data.
        // Device vector will be partitioned between selected devices.
        vex::vector<double> x(ctx.queue(), host_vec);

        // Allocate uninitialized device vectors.
        vex::vector<double> y(ctx.queue(), N);
        vex::vector<double> z(ctx.queue(), N);

        // Appropriate kernels are compiled (once) and called automagically:
        // Fill device vector with constant value...
        y = 42;
        // ... or with some expression:
        z = sqrt(2 * x) + cos(y);

        Reductor<double,SUM> sum(ctx.queue());

        std::cout << "y * y = " << sum(y * y) << std::endl;

        // Check results at random location.
        size_t pos = rand() % N;

        // You can read or write device vector elements with [] notation (very
        // ineffective but sometimes convenient).
        std::cout << y[pos] << std::endl;
        std::cout << "res = " << static_cast<double>(z[pos]) - sqrt(2 * static_cast<double>(x[pos])) - cos(static_cast<double>(y[pos])) << std::endl;

        // Or you can read the entire vector to host:
        copy(z, host_vec);

        vex::copy(host_vec.begin(), host_vec.end(), z.begin());
        vex::copy(z.begin(), z.end(), host_vec.data());
    } catch (const cl::Error &e) {
        std::cout << "OpenCL error: " << e << std::endl;
    }
}

// vim: set et
