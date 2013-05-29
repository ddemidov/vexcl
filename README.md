# VexCL

VexCL is a vector expression template library for OpenCL. It has been created
for ease of OpenCL development with C++. VexCL strives to reduce amount of
boilerplate code needed to develop OpenCL applications. The library provides
convenient and intuitive notation for vector arithmetic, reduction, sparse
matrix-vector products, etc. Multi-device and even multi-platform computations
are supported. The source code of the library is distributed under very
permissive MIT license.

The code is available at https://github.com/ddemidov/vexcl.

Doxygen-generated documentation: http://ddemidov.github.io/vexcl.

Slides from VexCL-related talks:

* [Meeting C++ 2012, Dusseldorf](https://github.com/ddemidov/vexcl/blob/master/doc/mcpp_vexcl_2012_slides.pdf?raw=true)
* [SIAM CSE 2013, Boston](https://github.com/ddemidov/vexcl/blob/master/doc/vexcl_cse13_slides.pdf?raw=true)
* [FOSDEM 2013, Brussels](https://fosdem.org/2013/schedule/event/odes_cuda_opencl)

The paper [Programming CUDA and OpenCL: A Case Study Using Modern C++
Libraries](http://arxiv.org/abs/1212.6326) compares both convenience and
performance of several GPGPU libraries, including VexCL.

### Table of contents

* [Context initialization](#context-initialization)
* [Memory allocation](#memory-allocation)
* [Copies between host and devices](#copies-between-host-and-devices)
* [Vector expressions](#vector-expressions)
    * [Builtin operations](#builtin-operations)
    * [Element indices](#element-indices)
    * [User-defined functions](#user-defined-functions)
    * [Random number generation](#random-number-generation)
    * [Permutations](#permutations)
    * [Slicing](#slicing)
* [Reductions](#reductions)
* [Sparse matrix-vector products](#sparse-matrix-vector-products)
* [Stencil convolutions](#stencil-convolutions)
* [Fast Fourier Transform](#fast-fourier-transform)
* [Multivectors](#multivectors)
* [Converting generic C++ algorithms to OpenCL](#converting-generic-c-algorithms-to-opencl)
* [Custom kernels](#custom-kernels)
* [Interoperability with other libraries](#interoperability-with-other-libraries)
* [Supported compilers](#supported-compilers)

## Context initialization

VexCL can transparently work with multiple compute devices that are present in
the system. VexCL context is initialized with a device filter, which is just a
functor that takes a reference to `cl::Device` and returns a `bool`. Several
[standard filters][filters] are provided, but one can easily add a custom
functor. Filters may be combined with logical operators. All compute devices
that satisfy the provided filter are added to the created context. In the
example below all GPU devices that support double precision arithmetics are
selected:
~~~{.cpp}
#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>

int main() {
    vex::Context ctx( vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::DoublePrecision );

    if (!ctx) throw std::runtime_error("No devices available.");

    // Print out list of selected devices:
    std::cout << ctx << std::endl;
}
~~~

[filters]:    http://ddemidov.github.io/vexcl/namespacevex_1_1Filter.html
[env-filter]: http://ddemidov.github.io/vexcl/structvex_1_1Filter_1_1EnvFilter.html

One of the most convenient filters is [vex::Filter::Env][env-filter] which
selects compute devices based on environment variables. It allows to switch
compute device without need to recompile the program.

## Memory allocation

The `vex::vector<T>` class constructor accepts a const reference to
`std::vector<cl::CommandQueue>`. A `vex::Context` instance may be conveniently
converted to the type, but it is also possible to initialize the command queues
elsewhere, thus completely eliminating the need to create a `vex::Context`.
Each command queue in the list should uniquely identify a single compute
device.

The contents of the created vector will be partitioned across all devices that
were present in the queue list.  Size of each partition will be proportional to
the device bandwidth, which is measured the first time the device is used. All
vectors of the same size are guaranteed to be partitioned consistently, which
allows to minimize inter-device communication.

In the example below, three device vectors of the same size are allocated.
Vector `A` is copied from host vector `a`, and the other vectors are created
uninitialized:
~~~{.cpp}
const size_t n = 1024 * 1024;
vex::Context ctx( vex::Filter::All );

std::vector<double> a(n, 1.0);

vex::vector<double> A(ctx, a);
vex::vector<double> B(ctx, n);
vex::vector<double> C(ctx, n);
~~~
Assuming that the current system has an NVIDIA and an AMD GPUs along with an
Intel CPU installed, possible partitioning may look as in the following figure:

![Partitioning](https://raw.github.com/ddemidov/vexcl/master/doc/figures/partitioning.png)

## Copies between host and devices

Function `vex::copy()` allows to copy data between host and device memories.
There are two forms of the function -- simple one and an STL-like:
~~~{.cpp}
std::vector<double> h(n);       // Host vector.
vex::vector<double> d(ctx, n);  // Device vector.

// Simple form:
vex::copy(h, d);    // Copy data from host to device.
vex::copy(d, h);    // Copy data from device to host.

// STL-like form:
vex::copy(h.begin(), h.end(), d.begin()); // Copy data from host to device.
vex::copy(d.begin(), d.end(), h.begin()); // Copy data from device to host.
~~~

The STL-like variant allows to copy sub-ranges of the vectors, or copy data
from/to raw host pointers.

Vectors also overload array subscript operator, so that users may have direct
read or write access to individual vector elements. But this operation is
highly ineffective and should be used with caution. Iterators allow for element
access as well, so that STL algorithms may in principle be used with device
vectors. This would be very slow but may be used as a temporary building
blocks.

## Vector expressions

VexCL allows to use convenient and intuitive notation for vector operations. In
order to be used in the same expression, all vectors have to be _compatible_:
* Have same size;
* Span same set of compute devices.

If the conditions are satisfied, then vectors may be combined with rich set of
available expressions. Vector expressions are processed in parallel across all
devices they were allocated on. One should keep in mind that in case several
OpenCL command queues are used, then the queues of the vector that is being
assigned to will be employed. Each vector expression results in launch of a
single OpenCL kernel. The kernel is automatically generated and launched the
first time the expression is encountered in the program. If
`VEXCL_SHOW_KERNELS` macro is defined, then the sources of all generated
kernels will be dumped to the standard output. For example, the expression:
~~~{.cpp}
X = 2 * Y - sin(Z);
~~~
will lead to the launch of the following OpenCL kernel:
~~~{.c}
kernel void minus_multiplies_term_term_sin_term_(
    ulong n,
    global double *res,
    int prm_1,
    global double *prm_2,
    global double *prm_3
)
{
    for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {
        res[idx] = ( ( prm_1 * prm_2[idx] ) - sin( prm_3[idx] ) );
    }
}
~~~
Here and in the rest of examples `X`, `Y`, and `Z` are compatible instances
of `vex::vector<double>`.

### Builtin operations

VexCL expressions may combine device vectors and scalars with arithmetic,
logic, or bitwise operators as well as with builtin OpenCL functions. If some
builtin operator or function is unavailable, it should be considered a bug.
Please do not hesitate to open an issue in this case.

~~~{.cpp}
Z = sqrt(2 * X) + pow(cos(Y), 2.0);
~~~

### Element indices

Function `vex::element_index(size_t offset = 0)` allows to use an index of each
vector element inside vector expressions. The numbering is continuous across
the compute devices and starts with an optional `offset`.

~~~{.cpp}
// Linear function:
double x0 = 0.0, dx = 1.0 / (X.size() - 1);
X = x0 + dx * vex::element_index();

// Single period of sine function:
Y = sin(2 * M_PI * vex::element_index() / Y.size());
~~~

### User-defined functions

Users may define custom functions to use in vector expressions. One has to
define function signature and function body. The body may contain any number of
lines of valid OpenCL code. Function parameters are named `prm1`, `prm2`, etc.
The most convenient way to define a function is `VEX_FUNCTION` macro:

~~~{.cpp}
VEX_FUNCTION(squared_radius, double(double, double), "return prm1 * prm1 + prm2 * prm2;");
Z = sqrt(squared_radius(X, Y));
~~~
The resulting `squared_radius` function object is stateless; only its type is
used for kernel generation. Hence, it is safe to put commonly used functions in
global scope.

Custom functions may be used not only for convenience, but also for performance
reasons. The above example could in principle be rewritten as:
~~~{.cpp}
Z = sqrt(X * X + Y * Y);
~~~
The drawback of the latter variant is that `X` and `Y` will be read _twice_.

Note that any valid vector expression may be passed as a function parameter:
~~~{.cpp}
Z = squared_radius(sin(X + Y), cos(X - Y));
~~~

### Random number generation

VexCL provides counter-based random number generators from [Random123][] suite,
in which  Nth random number is obtained by applying a stateless mixing function
to N instead of the conventional approach of using N iterations of a stateful
transformation. This technique is easily parallelizable and is well suited for
use in GPGPU applications.

[Random123]: http://www.deshawresearch.com/resources_random123.html

For integral types, generated values span the complete range;
for floating point types, generated values are in [0,1] interval.

In order to use a random number sequence in a vector expression, user has to
declare an instance of either `vex::Random` or `vex::RandomNormal` class
template as in the following example:
~~~{.cpp}
vex::Random<double, vex::random::threefry> rnd;

// X will contain random numbers from [-1, 1]:
X = 2 * rnd(vex::element_index(), std::rand()) - 1;
~~~
Note that `vex::element_index()` here provides the random number generator with
a sequence position N.

### Permutations

`vex::permutation` allows to use permuted vector in a vector expression. The
class constructor accepts `vex::vector<size_t>` of indices. The following
example assigns reversed vector `X` to `Y`:

~~~{.cpp}
vex::vector<size_t> I(ctx, N);
I = N - 1 - vex::element_index();

vex::permutation reverse(I);

Y = reverse(X);
~~~

_Permutation operation is only supported in single-device contexts._

### Slicing

An instance of `vex::slicer<NDIM>` class allows to conveniently access
sub-blocks of multi-dimensional arrays that are stored in `vex::vector` in
row-major order. The constructor of the class accepts dimensions of an array to
be sliced. The following example extracts every other element from interval
`[100, 200)` of one-dimensional vector X:

~~~{.cpp}
vex::vector<double> X(ctx, n);
vex::vector<double> Y(ctx, 50);

vex::slicer<1> slice({n});

Y = slice[vex::range(100, 2, 200)](X);
~~~

And the example below shows how to work with two-dimensional matrix:

~~~{.cpp}
using vex::range;

vex::vector<double> X(ctx, n * n); // n-by-n matrix stored in row-major order.
vex::vector<double> Y(ctx, n);
vex::vector<double> Z(ctx, 100);

vex::slicer<2> slice({n, n});

Y = slice[42](X);          // Put 42-th row of X into Y.
Y = slice[range()][42](X); // Put 42-th column of X into Y;

// Assign sub-block [10,20)x[30,40) of X to Z:
Z = slice[range(10, 20)][range(30, 40)](X);
~~~

_Slicing is only supported in single-device contexts._

## Reductions

An instance of `vex::Reductor<T, OP>` allows to reduce an arbitrary vector
expression to a single value of type T. Supported reduction operations are
`SUM`, `MIN`, and `MAX`. Reductor objects receive a list of command queues at
construction and should only be applied to vectors residing on the same
compute devices.

In the following example an inner product of two vectors is computed:
~~~{.cpp}
vex::Reductor<double, vex::SUM> sum(ctx);

double s = sum(x * y);
~~~
And here is an easy way to compute an approximate value of π with Monte-Carlo
method:
~~~{.cpp}
VEX_FUNCTION(squared_radius, double(double, double), "return prm1 * prm1 + prm2 * prm2;");

vex::Reductor<size_t, vex::SUM> sum(ctx);
vex::Random<double, vex::random::threefry> rnd;

X = 2 * rnd(vex::element_index(), std::rand()) - 1;
Y = 2 * rnd(vex::element_index(), std::rand()) - 1;

double pi = 4.0 * sum(squared_radius(X, Y) < 1) / X.size();
~~~

## Sparse matrix-vector products

One of the most common operations in linear algebra is matrix-vector
multiplication. An instance of `vex::SpMat` class holds representation of a
sparse matrix. Its constructor accepts sparse matrix in common [CRS][] format.
In the example below a `vex::SpMat` is constructed from an [Eigen][] [sparse
matrix][eigen-spmat]:

[CRS]: http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
[Eigen]: http://eigen.tuxfamily.org/
[eigen-spmat]: http://eigen.tuxfamily.org/dox/TutorialSparse.html

~~~{.cpp}
Eigen::SparseMatrix<double, Eigen::RowMajor, int> E;

vex::SpMat<double, int> A(ctx, E.rows(), E.cols(),
    E.outerIndexPtr(), E.innerIndexPtr(), E.valuesPtr());
~~~

The matrix-vector products may be used in vector expressions. The only
restriction is that the expressions have to be additive. This is due to the
fact that matrix representation may span several compute devices. Hence,
a matrix-vector product operation may require several kernel launches and
inter-device communication.

~~~{.cpp}
// Compute residual value for a system of linear equations:
Z = Y - A * X;
~~~

## Stencil convolutions

Stencil convolution is another common operation that may be used, for example,
to represent a signal filter, or a (one-dimensional) differential operator.
VexCL implements two stencil kinds. The first one is a simple linear stencil
that holds linear combination coefficients. The example below computes moving
average of a vector with a 5-point window:
~~~{.cpp}
vex::stencil<double> S(ctx, /*coefficients:*/{0.2, 0.2, 0.2, 0.2, 0.2}, /*center:*/2);

Y = X * S;
~~~

Users may also define custom stencil operators. This may be of use if, for
example, the operator is nonlinear. The definition of a stencil operator looks
very similar to a definition of a custom function. The only difference is that
stencil operator constructor accepts vector of OpenCL command queues. The
following example implements non-linear operator `y(i) = sin(x(i) - x(i - 1)) +
sin(x(i+1) - sin(x(i))`:
~~~{.cpp}
VEX_STENCIL_OPERATOR(S, /*return type:*/double, /*window width:*/3, /*center:*/1,
    "return sin(X[0] - X[-1]) + sin(X[1] - X[0]);"
    );

Z = S(Y);
~~~

The current window is available inside the body of the operator through the `X`
array that is indexed relatively to the stencil center.

Stencil convolution operations, similar to the matrix-vector products, are only
allowed in additive expressions.

## Fast Fourier Transform

VexCL provides implementation of Fast Fourier Transform (FFT) that accepts
arbitrary vector expressions as input, allows to perform multidimensional
transforms (of any number of dimensions), and supports arbitrary sized vectors:

~~~{.cpp}
vex::FFT<double, cl_double2> fft(ctx, n);
vex::FFT<cl_double2, double> ifft(ctx, n, vex::fft::inverse);

vex::vector<double> in(ctx, n), back(ctx, n);
vex::vector<cl_double2> out(ctx, n);

// ...

out  = fft (in);
back = ifft(out);

Z = fft(sin(X) + cos(Y));
~~~

FFT is another example of operation that is only available in additive
expressions. Another restriction is that FFT currently only supports contexts
with a single compute device.

## Multivectors

Class template `vex::multivector<T,N>` allows to store several equally sized
device vectors and perform computations on all components synchronously.  Each
operation is delegated to the underlying vectors, but usually results in the
launch of a single fused kernel. Expressions may include values of
`std::array<T,N>` type, where N is equal to the number of multivector
components. Each component gets corresponding element of `std::array<>` when
expression is applied. Similarly, array subscript operator or reduction of a
multivector returns an `std::array<T,N>`.  In order to access k-th component of
a multivector, one can use overloaded `operator()`:

~~~{.cpp}
VEX_FUNCTION(between, bool(double, double, double), "return prm1 <= prm2 && prm2 <= prm3;");

vex::Reductor<double, vex::SUM> sum(ctx);
vex::SpMat<double> A(ctx, ... );
std::array<double, 2> v = {6.0, 7.0};

vex::multivector<double, 2> X(ctx, N), Y(ctx, N);

// ...

X = sin(v * Y + 1);             // X(k) = sin(v[k] * Y(k) + 1);
v = sum( between(0, X, Y) );    // v[k] = sum( between( 0, X(k), Y(k) ) );
X = A * Y;                      // X(k) = A * Y(k);
~~~

Some operations can not be expressed with simple multivector arithmetics. For
example, an operation of two dimensional rotation mixes components in the right
hand side expressions:
~~~
y0 = x0 * cos(alpha) - x1 * sin(alpha);
y1 = x0 * sin(alpha) + x1 * cos(alpha);
~~~

This may in principle be implemented as:
~~~{.cpp}
double alpha;
vex::multivector<double, 2> X(ctx, N), Y(ctx, N);

Y(0) = X(0) * cos(alpha) - X(1) * sin(alpha);
Y(1) = X(0) * sin(alpha) + X(1) * cos(alpha);
~~~
But this would result in two kernel launches. VexCL allows to assign a tuple of
expressions to a multivector, which will lead to the launch of a single fused
kernel:
~~~
Y = std::tie( X(0) * cos(alpha) - X(1) * sin(alpha),
              X(0) * sin(alpha) + X(1) * cos(alpha) );
~~~

## Converting generic C++ algorithms to OpenCL

CUDA and OpenCL differ in their handling of compute kernels compilation. In
NVIDIA's framework the compute kernels are compiled to PTX code together with
the host program. In OpenCL the compute kernels are compiled at runtime from
high-level C-like sources, adding an overhead which is particularly noticeable
for smaller sized problems. This distinction leads to higher initialization
cost of OpenCL programs, but at the same time it allows to generate better
optimized kernels for the problem at hand. VexCL allows to exploit the
possibility with help of its kernel generator mechanism.

An instance of `vex::generator::symbolic<T>` dumps to output stream any
arithmetic operations it is being subjected to. For example, this code snippet:
~~~{.cpp}
vex::generator::symbolic<double> x = 6, y = 7;
x = sin(x * y);
~~~
results in the following output:
~~~
double var1 = 6;
double var2 = 7;
var1 = sin( ( var1 * var2 ) );
~~~

The symbolic type allows to record a sequence of arithmetic operations made by
a generic C++ algorithm. To illustrate the idea, consider the generic
implementation of a 4th order Runge-Kutta ODE stepper:
~~~{.cpp}
template <class state_type, class SysFunction>
void runge_kutta_4(SysFunction sys, state_type &x, double dt) {
    state_type k1 = dt * sys(x);
    state_type k2 = dt * sys(x + 0.5 * k1);
    state_type k3 = dt * sys(x + 0.5 * k2);
    state_type k4 = dt * sys(x + k3);

    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}
~~~
This function takes a system function `sys`, state variable `x`, and advances
`x` by time step `dt`. For example, to model the equation `dx/dt = sin(x)`, one
has to provide the following system function:
~~~{.cpp}
template <class state_type>
state_type sys_func(const state_type &x) {
    return sin(x);
}
~~~

The following code snippet makes a hundred of RK4 iterations for a single
`double` value on a CPU:
~~~{.cpp}
double x = 1, dt = 0.01;

for(int step = 0; step < 100; ++step)
    runge_kutta_4(sys_func<double>, x, dt);
~~~

Let us now generate the kernel for a single RK4 step and apply the kernel to a
`vex::vector<double>` (by doing this we essentially simultaneously solve big
number of same ODEs with different initial conditions).
~~~{.cpp}
// Set recorder for expression sequence.
std::ostringstream body;
vex::generator::set_recorder(body);

// Create symbolic variable.
typedef vex::generator::symbolic<double> sym_state;
sym_state sym_x(sym_state::VectorParameter);

// Record expression sequience for a single RK4 step.
double dt = 0.01;
runge_kutta_4(sys_func<sym_state>, sym_x, dt);

// Build kernel from the recorded sequence.
auto kernel = vex::generator::build_kernel(ctx, "rk4_stepper", body.str(), sym_x);

// Create initial state.
const size_t n = 1024 * 1024;
vex::vector<double> x(ctx, n);
x = 10.0 * vex::element_index() / n;

// Make 100 RK4 steps.
for(int i = 0; i < 100; i++) kernel(x);
~~~

This approach has some obvious restrictions. Namely, the C++ code has to be
embarrassingly parallel and is not allowed to contain any branching or
data-dependent loops. Nevertheless, the kernel generation facility may save
substantial amount of both human and machine time when applicable.

## Custom kernels

As [Kozma Prutkov](http://en.wikipedia.org/wiki/Kozma_Prutkov) repeatedly said,
"One cannot embrace the unembraceable". So in order to be usable, VexCL has to
support custom kernels. `vex::vector::operator()(uint k)` returns `cl::Buffer`
that holds vector data on k-th compute device.  If the result depends on the
neighbor points, one has to keep in mind that these points are possibly located
on a different compute device.  In this case the exchange of these halo points
has to be arranged manually.

The following example builds and launches a custom kernel for each device in
the context:
~~~{.cpp}
std::vector<cl::Kernel> kernel(ctx.size());

// Compile and store the kernels for later use.
for(uint d = 0; d < ctx.size(); d++) {
    cl::Program program = vex::build_sources(ctx.context(d),
        "kernel void dummy(ulong size, global float *x)\n"
        "{\n"
        "    size_t i = get_global_id(0);\n"
        "    if (i < size) x[i] = 4.2;\n"
        "}\n"
        );
    kernel[d] = cl::Kernel(program, "dummy");
}

// Apply the kernels to the vector partitions on each device.
for(uint d = 0; d < ctx.size(); d++) {
    cl_ulong n = x.part_size();
    kernel[d].setArg(0, n);
    kernel[d].setArg(1, x(d));
    ctx.queue(d).enqueueNDRangeKernel(kernel[d], cl::NullRange, n, cl::NullRange);
}
~~~

## Interoperability with other libraries

Since VexCL is built upon standard Khronos OpenCL C++ bindings, it is
easily interoperable with other OpenCL libraries. In particular, VexCL provides
some glue code for [ViennaCL][] and for [Boost.compute][] libraries.

[ViennaCL]: http://viennacl.sourceforge.net/
[Boost.compute]: https://github.com/kylelutz/compute

[ViennaCL][] (The Vienna Computing Library) is a scientific computing library
written in C++.  It provides OpenCL, CUDA, and OpenMP compute backends.  The
programming interface is compatible with Boost.uBLAS and allows for simple,
high-level access to the vast computing resources available on parallel
architectures such as GPUs.  The library's primary focus is on common linear
algebra operations (BLAS levels 1, 2 and 3) and the solution of large sparse
systems of equations by means of iterative methods with optional
preconditioners.

It is possible to use generic ViennaCL's solvers with VexCL types. See
[examples/viennacl/solvers.cpp](https://github.com/ddemidov/vexcl/blob/master/examples/viennacl/solvers.cpp)
for an example.

[Boost.compute][] is a GPU/parallel-computing library for C++ based on OpenCL.
The core library is a thin C++ wrapper over the OpenCL C API and provides
access to compute devices, contexts, command queues and memory buffers.  On top
of the core library is a generic, STL-like interface providing common
algorithms (e.g. `transform()`, `accumulate()`, `sort()`) along with common
containers (e.g. `vector<T>`, `flat_set<T>`). It also features a number of
extensions including parallel-computing algorithms (e.g. `exclusive_scan()`,
`scatter()`, `reduce()`) and a number of fancy iterators (e.g.
`transform_iterator<>`, `permutation_iterator<>`, `zip_iterator<>`).

[vexcl/external/boost_compute.hpp](https://github.com/ddemidov/vexcl/blob/master/vexcl/external/boost_compute.hpp)
provides an example of using Boost.compute algorithms with VexCL vectors.
Namely, it implements parallel sort and inclusive scan primitives on top of the
corresponding Boost.compute algorithms.

## Supported compilers

VexCL makes heavy use of C++11 features, so your compiler has to be modern
enough. The compilers that have been tested and supported:

* GCC v4.6 and higher.
* Clang v3.1 and higher.
* Microsoft Visual C++ 2010 and higher.

VexCL uses standard OpenCL bindings for C++ from Khronos group. The cl.hpp file
should be included with the OpenCL implementation on your system, but it is
also provided with the library.

----------------------------
_This work is a joint effort of [Supercomputer Center of Russian Academy of
Sciences][jscc] (Kazan branch) and [Kazan Federal University][kpfu]. It is
partially supported by RFBR grants No 12-07-0007 and 12-01-00033._

[jscc]: http://www.jscc.ru/eng/index.shtml
[kpfu]: http://www.kpfu.ru
