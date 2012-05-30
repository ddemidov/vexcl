vexcl
=======

VexCL is vector expression template library for OpenCL. It has been created for
ease of C++ based OpenCL development.  Multi-device (and multi-platform)
computations are supported.  See Doxygen-generated documentation at
http://ddemidov.github.com/vexcl.

Selection of compute devices
----------------------------

You can select any number of available compute devices, which satisfy provided
filters. Filter is a functor returning bool and acting on a `cl::Device`
parameter. Several standard filters are provided, such as device type or name
filter, double precision support etc. Filters can be combined with logical
operators. In the example below all devices with names matching "Radeon" and
supporting double precision are selected:
```C++
#include <iostream>
#include <vexcl/vexcl.hpp>
using namespace vex;
int main() {
    std::vector<cl::Device> device = device_list(
        Filter::Name("Radeon") && Filter::DoublePrecision()
        );
    std::cout << device << std::endl;
}
```

Often you want not just device list, but initialized OpenCL context with
command queue on each available device. This may be achieved with `queue_list`
function:
```C++
std::vector<cl::Context>      context;
std::vector<cl::CommandQueue> queue;
// Select no more than 2 NVIDIA GPUs:
std::tie(context, queue) = queue_list(
    [](const cl::Device &d) {
        return d.getInfo<CL_DEVICE_VENDOR>() == "NVIDIA Corporation";
    } && Filter::Count(2)
    );
std::cout << queue << std::endl;
```

Memory allocation and vector arithmetic
---------------------------------------

Once you got queue list, you can allocate OpenCL buffers on the associated
devices. `vex::vector` constructor accepts `std::vector` of `cl::CommandQueue`.
The contents of the created vector will be partitioned between each queue
(presumably, each of the provided queues is linked with separate device).
Size of each partition will be proportional to relative device bandwidth unless
macro `VEXCL_DUMB_PARTITIONING` is defined, in which case equal partitioning
scheme will be applied. Device bandwidth is measured first time it is requested
by launch of small test kernel.

Multi-platform computation is supported (that is, you
can spread your vectors across devices by different vendors), but should be
used with caution: all computations will be performed with the speed of the
slowest device selected.

In the example below host vector is allocated and initialized, then copied to
all devices obtained with the queue_list() call. A couple of empty device
vectors are allocated as well:
```C++
const uint n = 1 << 20;
std::vector<double> x(n);
std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

std::vector<cl::Context>      context;
std::vector<cl::CommandQueue> queue;
std::tie(context, queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU));

vex::vector<double> X(queue, CL_MEM_READ_ONLY,  x);
vex::vector<double> Y(queue, CL_MEM_READ_WRITE, n);
vex::vector<double> Z(queue, CL_MEM_READ_WRITE, n);
```

You can now use simple vector arithmetic with device vectors. For every
expression you use, appropriate kernel is compiled (first time it is
encountered in your program) and called automagically.

Vectors are processed in parallel across all devices they were allocated on:
```C++
Y = Const(42);
Z = Sqrt(Const(2) * X) + Cos(Y);
```

You can copy the result back to host or you can use `vector::operator[]` to
read (or write) vector elements directly. Though latter technique is very
ineffective and should be used for debugging purposes only.
```C++
copy(Z, x);
assert(x[42] == Z[42]);
```

Another frequently performed operation is reduction of a vector expression to
single value, such as summation. This can be done with `Reductor` class:
```C++
Reductor<double,SUM> sum(queue);
Reductor<double,MAX> max(queue);

std::cout << max(Abs(X) - Const(0.5)) << std::endl;
std::cout << sum(Sqrt(Const(2) * X) + Cos(Y)) << std::endl;
```

Sparse matrix-vector multiplication
-----------------------------------

One of the most common operations in linear algebra is matrix-vector
multiplication. Class `SpMat` holds representation of a sparse matrix,
spanning several GPUs. In the example below it is used for solution of a system
of linear equations with conjugate gradients method:
```C++
typedef double real;
// Solve system of linear equations A u = f with conjugate gradients method.
// Input matrix is represented in CSR format (parameters row, col, and val).
void cg_gpu(
        const std::vector<uint> &row,   // Indices to col and val vectors.
        const std::vector<uint> &col,   // Column numbers of non-zero elements.
        const std::vector<real> &val,   // Values of non-zero elements.
        const std::vector<real> &rhs,   // Right-hand side.
        std::vector<real> &x            // In: initial approximation; out: result.
        )
{
    // Init OpenCL
    std::vector<cl::Context>      context;
    std::vector<cl::CommandQueue> queue;
    std::tie(context, queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU));

    // Move data to GPU(s)
    uint n = x.size();
    vex::SpMat<real>  A(queue, n, row.data(), col.data(), val.data());
    vex::vector<real> f(queue, CL_MEM_READ_ONLY,  rhs);
    vex::vector<real> u(queue, CL_MEM_READ_WRITE, x);
    vex::vector<real> r(queue, CL_MEM_READ_WRITE, n);
    vex::vector<real> p(queue, CL_MEM_READ_WRITE, n);
    vex::vector<real> q(queue, CL_MEM_READ_WRITE, n);

    Reductor<real,MAX> max(queue);

    // Solve equation Au = f with conjugate gradients method.
    real rho1, rho2;
    r = f - A * u;

    for(uint iter = 0; max(Abs(r)) > 1e-8 && iter < n; iter++) {
        rho1 = inner_product(r, r);

        if (iter == 0) {
            p = r;
        } else {
            real beta = rho1 / rho2;
            p = r + Const(beta) * p;
        }

        q = A * p;

        real alpha = rho1 / inner_product(p, q);

        u += Const(alpha) * p;
        r -= Const(alpha) * q;

        rho2 = rho1;
    }

    // Get result to host.
    copy(u, x);
}
```

Using custom kernels
--------------------

Custom kernels are of course possible as well. `vector::operator(uint)` returns
`cl::Buffer` object for a specified device:
```C++
std::vector<cl::Context>      context;
std::vector<cl::CommandQueue> queue;
std::tie(context, queue) = queue_list(Filter::Vendor("NVIDIA"));

const uint n = 1 << 20;
vex::vector<float> x(queue, CL_MEM_WRITE_ONLY, n);

auto program = build_sources(context[0], std::string(
    "kernel void dummy(uint size, global float *x)\n"
    "{\n"
    "    uint i = get_global_id(0);\n"
    "    if (i < size) x[i] = 4.2;\n"
    "}\n"
    ));

for(uint d = 0; d < queue.size(); d++) {
    auto dummy = cl::Kernel(program, "dummy").bind(queue[d], alignup(n, 256), 256);
    dummy((uint)x.part_size(d), x(d));
}

std::cout << sum(x) << std::endl;
```
