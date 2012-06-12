#ifndef VEXCL_VEXCL_HPP
#define VEXCL_VEXCL_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Vector expression template library for OpenCL.
 */

/**
\mainpage VexCL

VexCL is vector expression template library for OpenCL. It has been created for
ease of C++ based OpenCL development.  Multi-device (and multi-platform)
computations are supported. The source code is available at
https://github.com/ddemidov/vexcl.

\section devlist Selection of compute devices

You can select any number of available compute devices, which satisfy provided
filters. Filter is a functor returning bool and acting on a cl::Device
parameter. Several standard filters are provided, such as device type or name
filter, double precision support etc. Filters can be combined with logical
operators. In the example below all devices with names matching "Radeon" and
supporting double precision are selected:
\code
#include <vexcl/vexcl.hpp>
using namespace vex;
int main() {
    vex::Context ctx(Filter::Name("Radeon") && Filter::DoublePrecision);
    std::cout << ctx << std::endl;
}
\endcode
vex::Context object holds list of initialized OpenCL contexts and command
queues for each filtered device. If you just need list of available devices
without creating contexts and queues on them, then look for device_list()
function in documenation.

\section vector Memory allocation and vector arithmetic

Once you got queue list, you can allocate OpenCL buffers on the associated
devices. vex::vector constructor accepts std::vector of cl::CommandQueue.  The
contents of the created vector will be partitioned between each queue
(presumably, each of the provided queues is linked with separate device).  Size
of each partition will be proportional to relative device bandwidth unless
macro VEXCL_DUMB_PARTITIONING is defined, in which case equal partitioning
scheme will be applied. Device bandwidth is measured first time it is requested
by launch of small test kernel.

Multi-platform computation is supported (that is, you can spread your vectors
across devices by different vendors), but should be used with caution: all
computations will be performed with the speed of the slowest device selected.

In the example below host vector is allocated and initialized, then copied to
all GPU devices found in the system. A couple of empty device vectors are
allocated as well:
\code
const size_t n = 1 << 20;
std::vector<double> x(n);
std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });

vex::Context ctx(Filter::Type(CL_DEVICE_TYPE_GPU));

vex::vector<double> X(ctx.queue(), x);
vex::vector<double> Y(ctx.queue(), n);
vex::vector<double> Z(ctx.queue(), n);
\endcode

You can now use simple vector arithmetic with device vector. For every
expression you use, appropriate kernel is compiled (first time it is
encountered in your program) and called automagically. If you want to see
sources of the generated kernels on the standard output, define
VEXCL_SHOW_KERNELS macro before including VexCL headers.

Vectors are processed in parallel across all devices they were allocated on:
\code
Y = 42;
Z = sqrt(2 * X) + cos(Y);
\endcode

You can copy the result back to host or you can use vector::operator[] to
read (or write) vector elements directly. Though latter technique is very
ineffective and should be used for debugging purposes only.
\code
copy(Z, x);
assert(x[42] == Z[42]);
\endcode

Another frequently performed operation is reduction of a vector expression to
single value, such as summation. This can be done with vex::Reductor class:
\code
Reductor<double> sum(ctx.queue());

std::cout << sum(Z) << std::endl;
std::cout << sum(sqrt(2 * X) + cos(Y)) << std::endl;
\endcode

\section spmv Sparse matrix-vector multiplication

One of the most common operations in linear algebra is matrix-vector
multiplication. Class vex::SpMat holds representation of a sparse matrix,
spanning several devices. In the example below it is used for solution of a
system of linear equations with conjugate gradients method:
\code
typedef double real;
// Solve system of linear equations A u = f with conjugate gradients method.
// Input matrix is represented in CSR format (parameters row, col, and val).
void cg_gpu(
	const std::vector<size_t> &row, // Indices to col and val vectors.
	const std::vector<size_t> &col, // Column numbers of non-zero elements.
	const std::vector<real>   &val, // Values of non-zero elements.
	const std::vector<real>   &rhs, // Right-hand side.
	std::vector<real> &x            // In: initial approximation; out: result.
	)
{
    // Init OpenCL.
    vex::Context ctx(Filter::Type(CL_DEVICE_TYPE_GPU));

    // Move data to compute devices.
    size_t n = x.size();
    vex::SpMat<real>  A(ctx.queue(), n, row.data(), col.data(), val.data());
    vex::vector<real> f(ctx.queue(), rhs);
    vex::vector<real> u(ctx.queue(), x);
    vex::vector<real> r(ctx.queue(), n);
    vex::vector<real> p(ctx.queue(), n);
    vex::vector<real> q(ctx.queue(), n);

    Reductor<real,MAX> max(ctx.queue());
    Reductor<real,SUM> sum(ctx.queue());

    // Solve equation Au = f with conjugate gradients method.
    real rho1, rho2;
    r = f - A * u;

    for(uint iter = 0; max(fabs(r)) > 1e-8 && iter < n; iter++) {
	rho1 = sum(r * r);

	if (iter == 0) {
	    p = r;
	} else {
	    real beta = rho1 / rho2;
	    p = r + beta * p;
	}

	q = A * p;

	real alpha = rho1 / sum(p * q);

	u += alpha * p;
	r -= alpha * q;

	rho2 = rho1;
    }

    // Get result to host.
    copy(u, x);
}
\endcode

\section userfun User-defined functions

Simple arithmetic expressions are sometimes not enough. Imagine that you need
to count how many elements in vector x are greater that their counterparts in
vector y. This may be achieved by introduction of custom function. In order
to build such a function, you need to supply its body, its return type and
types of its arguments. After that, you can apply the function to any valid
vector expressions:
\code
// Function body has to be defined at global scope, and it has to be of `extern
// const char[]` type. This allows us to use it as a template parameter.
extern const char one_greater_than_other[] = "return prm1 > prm2 ? 1 : 0;";
UserFunction<one_greater_than_other, size_t(float, float)> greater;

size_t count_if_greater(
    const Reductor<size_t, SUM> &sum,
    const vex:vector<float> &x,
    const vex::vector<float> &y
    )
{
    return sum(greater(x, y));
}
\endcode

You could also write sum(greater(x + y, 5 * y)), or use any other expressions
as parameters to the greater() call. Note that in the function body
parameters are always named as prm1, prm2, etc.


\section multivector Multi-component vectors

Class template vex::multivector<T,N> allows to store several equally sized
device vectors and perform computations on all components synchronously.
Operations are delegated to the underlying vectors. Expressions may include
std::array<T,N> values, where N is equal to the number of multivector
components. Each component gets corresponding element of std::array<> when
expression is applied.
\code
const size_t n = 1 << 20;
std::vector<double> host(n * 3);
std::generate(host.begin(), host.end(), rand);

vex::multivector<double,3> x(ctx.queue(), host);
vex::multivector<double,3> y(ctx.queue(), n);

std::array<int, 3> c = {4, 5, 6};

y = 2 * cos(x) - c;

std::array<double,3> v = y[42];
assert(fabs(v[1] - (2 * cos(host[n + 42]) - c[1])) < 1e-8);
\endcode

Components of a multivector may be accessed with operator():
\code
vex::vector<double> z = y(1);
\endcode

\section custkern Using custom kernels

Custom kernels are of course possible as well. vector::operator(uint)
returns cl::Buffer object for a specified device:
\code
vex::Context ctx(Filter::Type(CL_DEVICE_TYPE_GPU));

const size_t n = 1 << 20;
vex::vector<float> x(ctx.queue(), n);

auto program = build_sources(context, std::string(
    "kernel void dummy(ulong size, global float *x)\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < size) x[i] = 4.2;\n"
    "}\n"
    ));

for(uint d = 0; d < ctx.size(); d++) {
    auto dummy = cl::Kernel(program, "dummy").bind(ctx.queue()[d], alignup(n, 256), 256);
    dummy((cl_ulong)x.part_size(d), x(d));
}

Reductor<float,SUM> sum(ctx.queue());
std::cout << sum(x) << std::endl;
\endcode

\section scalability Scalability

In the images below, scalability of the library with respect to number of
compute devices is shown. Effective performance (GFLOPS) and bandwidth (GB/sec)
were measured by launching big number of test kernels on one, two, or three
Nvidia Tesla C2070 cards. The results shown are averaged over 20 runs.

The details of the experiments may be found in <a
href="https://github.com/ddemidov/vexcl/blob/master/examples/benchmark.cpp">
examples/benchmark.cpp</a> file.  Basically, performance of the following code
was measured:

\code
// Vector arithmetic
a += b + c * d;

// Reduction
double s = sum(a * b);

// SpMV
y += A * x;
\endcode

\image html perf.png ""

\section compilers Supported compilers

VexCL makes heavy use of C++11 features, so your compiler has to be modern
enough. GCC version 4.6 and above is fully supported. Microsoft Visual C++ 2010
manages to compile the project with some features disabled: since it does not
support variadic templates, only one-argument builtin functions are enabled;
user functions are not available at all.
*/

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4290)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>

#include <vexcl/util.hpp>
#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/spmat.hpp>
#include <vexcl/reduce.hpp>
#include <vexcl/profiler.hpp>

#ifdef WIN32
#  pragma warning(pop)
#endif
#endif
