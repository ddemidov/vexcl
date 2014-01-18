# VexCL

[<img src="https://travis-ci.org/ddemidov/vexcl.png?branch=master">](https://travis-ci.org/ddemidov/vexcl)
[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/ddemidov/vexcl/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

VexCL is a vector expression template library for OpenCL/CUDA. It has been
created for ease of GPGPU development with C++. VexCL strives to reduce amount
of boilerplate code needed to develop GPGPU applications. The library provides
convenient and intuitive notation for vector arithmetic, reduction, sparse
matrix-vector products, etc. Multi-device and even multi-platform computations
are supported. The source code of the library is distributed under very
permissive MIT license.

The code is available at https://github.com/ddemidov/vexcl.

Doxygen-generated documentation: http://ddemidov.github.io/vexcl.

Slides from VexCL-related talks:

* [Meeting C++ 2012, Dusseldorf](https://speakerdeck.com/ddemidov/vexcl-at-meeting-c-plus-plus-2012)
* [SIAM CSE 2013, Boston](https://speakerdeck.com/ddemidov/vexcl-at-cse13)
* [FOSDEM 2013, Brussels](https://fosdem.org/2013/schedule/event/odes_cuda_opencl)
* [GPU-SMP-2013, Changchun, China](https://speakerdeck.com/ddemidov/converting-generic-c-plus-plus-algorithms-to-opencl-with-vexcl-library)
* [Overview of VexCL interface, UTexas, Austin, USA](https://speakerdeck.com/ddemidov/vexcl-at-pecos-university-of-texas-2013)
* [VexCL implementation, UTexas, Austin, USA](https://speakerdeck.com/ddemidov/vexcl-implementation-university-of-texas-2013)

The paper [Programming CUDA and OpenCL: A Case Study Using Modern C++
Libraries](http://arxiv.org/abs/1212.6326) compares both convenience and
performance of several GPGPU libraries, including VexCL.

### Table of contents

* [Selecting backend](#selecting-backend)
* [Context initialization](#context-initialization)
* [Memory allocation](#memory-allocation)
* [Copies between host and devices](#copies-between-host-and-devices)
* [Vector expressions](#vector-expressions)
    * [Builtin operations](#builtin-operations)
    * [Constants](#constants)
    * [Element indices](#element-indices)
    * [User-defined functions](#user-defined-functions)
    * [Tagged terminals](#tagged-terminals)
    * [Temporary values](#temporary-values)
    * [Random number generation](#random-number-generation)
    * [Permutations](#permutations)
    * [Slicing](#slicing)
    * [Scattered data interpolation with multilevel B-Splines](#mba)
    * [Fast Fourier Transform](#fast-fourier-transform)
* [Reductions](#reductions)
* [Sparse matrix-vector products](#sparse-matrix-vector-products)
* [Stencil convolutions](#stencil-convolutions)
* [Raw pointers](#raw-pointers)
* [Sort, scan, reduce-by-key algorithms](#parallel-primitives)
* [Multivectors](#multivectors)
* [Converting generic C++ algorithms to OpenCL/CUDA](#converting-generic-c-algorithms-to-opencl)
    * [Kernel generator](#kernel-generator)
    * [Function generator](#function-generator)
* [Custom kernels](#custom-kernels)
* [Interoperability with other libraries](#interoperability-with-other-libraries)
* [Supported compilers](#supported-compilers)

## <a name="selecting-backend"></a>Selecting backend

VexCL provides two backends: OpenCL and CUDA. In order to choose either of those,
user has to define `VEXCL_BACKEND_OPENCL` or `VEXCL_BACKEND_CUDA` macros. In
case neither of those are defined, OpenCL backend is chosen by default. One
also has to link to either libOpenCL.so (OpenCL.dll) or libcuda.so (cuda.dll).

For the CUDA backend to work, CUDA Toolkit has to be installed, NVIDIA CUDA
compiler driver `nvcc` has to be in executable PATH and usable at runtime.

## <a name="context-initialization"></a>Context initialization

VexCL transparently works with multiple compute devices that are present in the
system. A VexCL context is initialized with a device filter, which is just a
functor that takes a reference to `vex::device` and returns a `bool`.  Several
[standard filters][filters] are provided, but one can easily add a custom
functor. Filters may be combined with logical operators. All compute devices
that satisfy the provided filter are added to the created context. In the
example below all GPU devices that support double precision arithmetic are
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

## <a name="memory-allocation"></a>Memory allocation

The `vex::vector<T>` class constructor accepts a const reference to
`std::vector<vex::command_queue>`. A `vex::Context` instance may be
conveniently converted to this type, but it is also possible to initialize the
command queues elsewhere (e.g. with the OpenCL backend `vex::command_queue` is
typedefed to `cl::CommandQueue`), thus completely eliminating the need to
create a `vex::Context`.  Each command queue in the list should uniquely
identify a single compute device.

The contents of the created vector will be partitioned across all devices that
were present in the queue list.  The size of each partition will be
proportional to the device bandwidth, which is measured the first time the
device is used. All vectors of the same size are guaranteed to be partitioned
consistently, which minimizes inter-device communication.

In the example below, three device vectors of the same size are allocated.
Vector `A` is copied from host vector `a`, and the other vectors are created
uninitialized:
~~~{.cpp}
const size_t n = 1024 * 1024;
vex::Context ctx( vex::Filter::Any );

std::vector<double> a(n, 1.0);

vex::vector<double> A(ctx, a);
vex::vector<double> B(ctx, n);
vex::vector<double> C(ctx, n);
~~~
Assuming that the current system has an NVIDIA and an AMD GPUs along with an
Intel CPU installed, possible partitioning may look as in the following figure:

![Partitioning](https://raw.github.com/ddemidov/vexcl/master/doc/figures/partitioning.png)

## <a name="copies-between-host-and-devices"></a>Copies between host and devices

The function `vex::copy()` allows one to copy data between host and device
memory spaces.  There are two forms of the function -- a simple one and an
STL-like one:
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

The STL-like variant can copy sub-ranges of the vectors, or copy data from/to
raw host pointers.

Vectors also overload the array subscript operator, `operator[]`, so that users
may directly read or write individual vector elements. This operation is
highly ineffective and should be used with caution. Iterators allow for element
access as well, so that STL algorithms may in principle be used with device
vectors. This would be very slow but may be used as a temporary building
block.

Another option for host-device data transfer is mapping device memory buffer to
a host array. The mapped array then may be transparently read or written.  The
method `vector::map(unsigned d)` maps the d-th partition of the vector and
returns the mapped array:
~~~{.cpp}
vex::vector<double> X(ctx, N);
auto mapped_ptr = X.map(0); // Unmapped automatically when goes out of scope
for(size_t i = 0; i < X.part_size(0); ++i)
    mapped_ptr[i] = host_function(i);
~~~

## <a name="vector-expressions"></a>Vector expressions

VexCL allows the use of convenient and intuitive notation for vector
operations. In order to be used in the same expression, all vectors have to be
_compatible_:
* Have same size;
* Span same set of compute devices.

If these conditions are satisfied, then vectors may be combined with rich set of
available expressions. Vector expressions are processed in parallel across all
devices they were allocated on. One should keep in mind that in case several
command queues are used, then the queues of the vector that is being
assigned to will be employed. Each vector expression results in the launch of a
single compute kernel. The kernel is automatically generated and launched the
first time the expression is encountered in the program. If the
`VEXCL_SHOW_KERNELS` macro is defined, then the sources of all generated
kernels will be dumped to the standard output. For example, the expression:
~~~{.cpp}
X = 2 * Y - sin(Z);
~~~
will lead to the launch of the following compute kernel:
~~~{.c}
kernel void vexcl_vector_kernel(
    ulong n,
    global double * prm_1,
    int prm_2,
    global double * prm_3,
    global double * prm_4
)
{
    for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {
        prm_1[idx] = ( ( prm_2 * prm_3[idx] ) - sin( prm_4[idx] ) );
    }
}
~~~
Here and in the rest of examples `X`, `Y`, and `Z` are compatible instances
of `vex::vector<double>`; it is also assumed that OpenCL backend is selected.

VexCL is able to cache the compiled kernels offline. The compiled binaries are
stored in `$HOME/.vexcl` on Linux and MacOSX, and in `%APPDATA%\vexcl` on
Windows systems. In order to enable this functionality for OpenCL backend, the
user has to define the `VEXCL_CACHE_KERNELS` macro. NVIDIA OpenCL
implementation does the caching already, but on AMD or Intel platforms this may
lead to dramatic decrease of program initialization time (e.g. VexCL tests take
around 20 seconds to complete without kernel caches, and 2 seconds when caches
are available). In case of the CUDA backend the offline caching is always
enabled.

### <a name="builtin-operations"></a>Builtin operations

VexCL expressions may combine device vectors and scalars with arithmetic,
logic, or bitwise operators as well as with builtin OpenCL functions. If some
builtin operator or function is unavailable, it should be considered a bug.
Please do not hesitate to open an issue in this case.

~~~{.cpp}
Z = sqrt(2 * X) + pow(cos(Y), 2.0);
~~~

### <a name="constants"></a>Constants

As you have seen above, `2` in the expression `2 * Y - sin(Z)` is passed to the
generated compute kernel as an `int` parameter (`prm_2`). Sometimes this is
desired behaviour, because the same kernel will be reused for the expressions
`42 * Z - sin(Y)` or `a * Y - sin(Y)` (where `a` is an integer variable). But
this may lead to a slight overhead if an expression involves true constant that
will always have same value. The macro `VEX_CONSTANT` allows one to define such
constants for use in vector expressions. Compare the generated kernel for the
following example with the kernel above:
~~~{.cpp}
VEX_CONSTANT(two, 2);

X = two() * Y - sin(Z);
~~~

~~~{.c}
kernel void vexcl_vector_kernel(
    ulong n,
    global double * prm_1,
    global double * prm_3,
    global double * prm_4
)
{
    for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {
        prm_1[idx] = ( ( ( 2 ) * prm_3[idx] ) - sin( prm_4[idx] ) );
    }
}
~~~

VexCL provides some predefined constants in the `vex::constants` namespace that
correspond to boost::math::constants (e.g. `vex::constants::pi()`).

### <a name="element-indices"></a>Element indices

The function `vex::element_index(size_t offset = 0)` allows one to use the index
of each vector element inside vector expressions. The numbering is continuous
across the compute devices and starts with an optional `offset`.

~~~{.cpp}
// Linear function:
double x0 = 0.0, dx = 1.0 / (X.size() - 1);
X = x0 + dx * vex::element_index();

// Single period of sine function:
Y = sin(vex::constants::two_pi() * vex::element_index() / Y.size());
~~~

### <a name="user-defined-functions"></a>User-defined functions

Users may define custom functions to use in vector expressions. One has to
define the function signature and function body. The body may contain any
number of lines of valid OpenCL or CUDA code, depending on the selected
backend. Function parameters are named `prm1`, `prm2`, etc.  The most
convenient way to define a function is via the `VEX_FUNCTION` macro:

~~~{.cpp}
VEX_FUNCTION(squared_radius, double(double, double), "return prm1 * prm1 + prm2 * prm2;");
Z = sqrt(squared_radius(X, Y));
~~~
The resulting `squared_radius` function object is stateless; only its type is
used for kernel generation. Hence, it is safe to put commonly used functions in
global scope.

Note that any valid vector expression may be passed as a function parameter:
~~~{.cpp}
Z = squared_radius(sin(X + Y), cos(X - Y));
~~~

Custom functions may be used not only for convenience, but also for performance
reasons. The above example could in principle be rewritten as:
~~~{.cpp}
Z = sqrt(X * X + Y * Y);
~~~
The drawback of the latter variant is that `X` and `Y` will be read _twice_.

The convenience macro `VEX_STRINGIZE_SOURCE` may be used to easily enquote the
source code for a user-defined function. It's advantage is that users will have
benefits of syntax highlighting and general readability of their code:
~~~{.cpp}
VEX_FUNCTION(diff_cube, double(double, double),
    VEX_STRINGIZE_SOURCE(
        double d = prm1 - prm2;
        return d * d * d;
        )
    );
~~~

### <a name="tagged-terminals"></a>Tagged terminals

The code snippet from the last paragraph is ineffective because the compiler
cannot tell if any two terminals in an expression tree are actually referring
to the same data. But programmers often have this information. VexCL allows one
to pass this knowledge to compiler by tagging terminals with unique tags.  By
doing this, the programmer guarantees that any two terminals with matching tags
are referencing same data.

Below is a more effective variant of the above example:
~~~{.cpp}
using vex::tag;
Z = sqrt(tag<1>(X) * tag<1>(X) + tag<2>(Y) * tag<2>(Y));
~~~
Here, the generated kernel will have one parameter for each of the vectors `X`
and `Y`.

### <a name="temporary-values"></a>Temporary values

Some expressions may have several occurences of the same subexpression.
Unfortunately, VexCL is not able to determine these cases without the
programmer's help. For example, let's look at the following expression:
~~~{.cpp}
Y = log(X) * (log(X) + Z);
~~~
Here, `log(X)` would be computed twice. One could tag vector `X` as in:
~~~{.cpp}
auto x = vex::tag<1>(X);
Y = log(x) * (log(x) + Z);
~~~

and hope that the backend compiler is smart enough to reuse result of `log(x)`
(e.g. NVIDIA's compiler _is_ smart enough to do this). But it is also possible
to explicitly ask VexCL to store result of a subexpression in a local variable
and reuse it. The `vex::make_temp()` function template serves this purpose:

~~~{.cpp}
auto tmp1 = vex::make_temp<1>( sin(X) );
auto tmp2 = vex::make_temp<2>( cos(X) );
Y = (tmp1 - tmp2) * (tmp1 + tmp2);
~~~

Any valid vector or multivector expression (but not additive expressions, such
as sparse matrix-vector products) may be wrapped into a `make_temp()` call.

### <a name="random-number-generation"></a>Random number generation

VexCL provides a counter-based random number generators from [Random123][]
suite, in which  Nth random number is obtained by applying a stateless mixing
function to N instead of the conventional approach of using N iterations of a
stateful transformation. This technique is easily parallelizable and is well
suited for use in GPGPU applications.

[Random123]: http://www.deshawresearch.com/resources_random123.html

For integral types, the generated values span the complete range; for floating
point types, the generated values lie in the interval [0,1].

In order to use a random number sequence in a vector expression, the user has to
declare an instance of either `vex::Random` or `vex::RandomNormal` class
template as in the following example:
~~~{.cpp}
vex::Random<double, vex::random::threefry> rnd;

// X will contain random numbers from [-1, 1]:
X = 2 * rnd(vex::element_index(), std::rand()) - 1;
~~~
Note that `vex::element_index()` here provides the random number generator with
a sequence position N.

### <a name="permutations"></a>Permutations

`vex::permutation()` allows the use of a permuted vector in a vector
expression. The function accepts a vector expression that returns integral
values (indices).  The following example reverses `X` and assigns it to `Y`:

~~~{.cpp}
vex::vector<size_t> I(ctx, N);
I = N - 1 - vex::element_index();
auto reverse = vex::permutation(I)

Y = reverse(X);
~~~

The drawback of the above approach is that you have to store and access an
index vector. Sometimes this is a necessary evil, but in this simple example we
can do better. In the following snippet a lightweight expression is used to
construct the same permutation:

~~~{.cpp}
auto reverse = vex::permutation( N - 1 - vex::element_index() );
Y = reverse(X);
~~~

Note that any valid vector expression may be used as an index, including
user-defined functions.

_Permutation operations are only supported in single-device contexts._

### <a name="slicing"></a>Slicing

An instance of the `vex::slicer<NDIM>` class allows one to conveniently access
sub-blocks of multi-dimensional arrays that are stored in `vex::vector` in
row-major order. The constructor of the class accepts the dimensions of the
array to be sliced. The following example extracts every other element from
interval `[100, 200)` of a one-dimensional vector X:

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

// vex::extents is a helper object similar to boost::multi_array::extents
vex::slicer<2> slice(vex::extents[n][n]);

Y = slice[42](X);          // Put 42-nd row of X into Y.
Y = slice[range()][42](X); // Put 42-nd column of X into Y.

slice[range()][10](X) = Y; // Put Y into 10-th column of X.

// Assign sub-block [10,20)x[30,40) of X to Z:
vex::vector<double> Z = slice[range(10, 20)][range(30, 40)](X);
assert(Z.size() == 100);
~~~

_Slicing is only supported in single-device contexts._

### <a name="mba"></a>Scattered data interpolation with multilevel B-Splines

VexCL provides an implementation of the MBA algorithm based on paper by Lee,
Wolberg, and Shin ([S. Lee, G. Wolberg, and S. Y. Shin. Scattered data
interpolation with multilevel B-Splines. IEEE Transactions on Visualization and
Computer Graphics, 3:228–244, 1997][bsplines]). This is a fast algorithm for scattered
N-dimensional data interpolation and approximation.  Multilevel B-splines are
used to compute a C2-continuously differentiable surface through a set of irregularly spaced
points. The algorithm makes use of a coarse-to-fine hierarchy of control
lattices to generate a sequence of bicubic B-spline functions whose sum
approaches the desired interpolation function. Large performance gains are
realized by using B-spline refinement to reduce the sum of these functions into
one equivalent B-spline function.  High-fidelity reconstruction is possible
from a selected set of sparse and irregular samples.

[bsplines]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=620490&tag=1

The algorithm is first prepared on a CPU. After that, it may be used in vector
expressions. Here is an example in 2D:
~~~{.cpp}
// Coordinates of data points:
std::vector< std::array<double,2> > coords = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
    {0.4, 0.4},
    {0.6, 0.6}
};

// Data values:
std::vector<double> values = {
    0.2, 0.0, 0.0, -0.2, -1.0, 1.0
};

// Bounding box:
std::array<double, 2> xmin = {-0.01, -0.01};
std::array<double, 2> xmax = { 1.01,  1.01};

// Initial grid size:
std::array<size_t, 2> grid = {5, 5};

// Algorithm setup.
vex::mba<2> surf(ctx, xmin, xmax, coords, values, grid);

// x and y are coordinates of arbitrary 2D points:
// vex::vector<double> x, y, z;

// Get interpolated values:
z = surf(x, y);
~~~

### <a name="fast-fourier-transform"></a>Fast Fourier Transform

VexCL provides an implementation of the Fast Fourier Transform (FFT) that
accepts arbitrary vector expressions as input, allows one to perform
multidimensional transforms (of any number of dimensions), and supports
arbitrary sized vectors:

~~~{.cpp}
vex::FFT<double, cl_double2> fft(ctx, n);
vex::FFT<cl_double2, double> ifft(ctx, n, vex::fft::inverse);

vex::vector<double> rhs(ctx, n), u(ctx, n), K(ctx, n);

// Solve Poisson equation with FFT:
u = ifft( K * fft(rhs) );
~~~

The restriction of the FFT is that it currently only supports contexts with a
single compute device.

## <a name="reductions"></a>Reductions

An instance of `vex::Reductor<T, OP>` allows one to reduce an arbitrary vector
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

## <a name="sparse-matrix-vector-products"></a>Sparse matrix-vector products

One of the most common operations in linear algebra is matrix-vector
multiplication. An instance of `vex::SpMat` class holds a representation of a
sparse matrix. Its constructor accepts a sparse matrix in common [CRS][] format.
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

Matrix-vector products may be used in vector expressions. The only
restriction is that the expressions have to be additive. This is due to the
fact that the matrix representation may span several compute devices. Hence,
a matrix-vector product operation may require several kernel launches and
inter-device communication.

~~~{.cpp}
// Compute residual value for a system of linear equations:
Z = Y - A * X;
~~~

This restriction may be lifted for single-device contexts. In this case VexCL
does not need to worry about inter-device communication. Hence, it is possible
to inline matrix-vector product into a normal vector expression with the help of
`vex::make_inline()`:

~~~{.cpp}
residual = sum(Y - vex::make_inline(A * X));
Z = sin(vex::make_inline(A * X));
~~~

## <a name="stencil-convolutions"></a>Stencil convolutions

Stencil convolution is another common operation that may be used, for example,
to represent a signal filter, or a (one-dimensional) differential operator.
VexCL implements two stencil kinds. The first one is a simple linear stencil
that holds linear combination coefficients. The example below computes the
moving average of a vector with a 5-point window:
~~~{.cpp}
vex::stencil<double> S(ctx, /*coefficients:*/{0.2, 0.2, 0.2, 0.2, 0.2}, /*center:*/2);

Y = X * S;
~~~

Users may also define custom stencil operators. This may be of use if, for
example, the operator is nonlinear. The definition of a stencil operator looks
very similar to a definition of a custom function. The only difference is that
the stencil operator constructor accepts a vector of command queues. The
following example implements the nonlinear operator `y(i) = sin(x(i) - x(i -
1)) + sin(x(i+1) - sin(x(i))`:
~~~{.cpp}
VEX_STENCIL_OPERATOR(S, /*return type:*/double, /*window width:*/3, /*center:*/1,
    "return sin(X[0] - X[-1]) + sin(X[1] - X[0]);", ctx);

Z = S(Y);
~~~

The current window is available inside the body of the operator through the `X`
array, which is indexed relative to the stencil center.

Stencil convolution operations, similar to the matrix-vector products, are only
allowed in additive expressions.

## <a name="raw-pointers"></a>Raw pointers

Unforunately, describing two dimensional stencils (e.g. discretization of the
Laplace operator) would not be effective, because the stencil width would be too
large. One can solve this problem by using a combination of
`raw_pointer(const vector<T>&)` with a derefence operator (essentially doing
pointer arithmetic inside compute kernel). For the sake of simplicity, the
example below implements a simple 3-point laplace operator for a one-dimensional
vector; but this could be easily extended in the two-dimensional case:
~~~{.cpp}
VEX_CONSTANT(zero, 0);
VEX_CONSTANT(one,  1);
VEX_CONSTANT(two,  2);

auto N   = vex::tag<1>( x.size() );
auto ptr = vex::tag<2>( vex::raw_pointer(x) );

auto i     = vex::make_temp<1>( vex::element_index() );
auto left  = vex::make_temp<2>( if_else(i > zero(),    i - one(), i) );
auto right = vex::make_temp<3>( if_else(i + one() < N, i + one(), i) );

y = *(ptr + i) * two() - *(ptr + left) - *(ptr + right);
~~~

This would result in the following compute kernel:
~~~{.c}
kernel void vexcl_vector_kernel(
    ulong n,
    global double * prm_1,
    global double * prm_2,
    ulong prm_3,
    ulong prm_4
)
{
    for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {
        ulong temp_1 = prm_3 + idx;
        ulong temp_2 = temp_1 > 0 ? temp_1 - 1 : temp_1;
        ulong temp_3 = temp_1 + 1 < prm_4 ? temp_1 + 1 : temp_1;
        prm_1[idx] = *(prm_2 + temp_1) * 2 - *(prm_2 + temp_2) - *(prm_2 + temp_3);
    }
}
~~~

The same approach could be used, for example, to implement an N-body problem with a
user-defined function:
~~~{.cpp}
// Takes vector size, current element position, and pointer to a vector to sum:
VEX_FUNCTION(global_interaction, double(size_t, size_t, double*),
    VEX_STRINGIZE_SOURCE(
        double sum = 0;
        double myval = prm3[prm2];
        for(size_t i = 0; i < prm1; ++i)
            if (i != prm2) sum += fabs(prm3[i] - myval);
        return sum;
        )
    );

y = global_interaction(x.size(), vex::element_index(), vex::raw_pointer(x));
~~~

Note that the use of `raw_pointer()` is limited to single-device contexts for
obvious reasons.

## <a name="parallel-primitives"></a>Sort, scan, reduce-by-key algorithms

VexCL provides several standalone parallel primitives that may not be used as
part of a vector expression. These are `inclusive_scan`, `exclusive_scan`,
`sort`, `sort_by_key`, `reduce_by_key`. All of these functions take VexCL
vectors as both input and output parameters.

Sorting and scan functions take an optional function object used for comparison
and summing of elements. The functor should provide the same interface as, e.g.
`std::less` for sorting or `std::plus` for summing; additionally, it should
provide a VexCL function for device-side operations.

Here is an example of such an object comparing integer elements in such a way
that even elements precede odd ones:
~~~{.cpp}
template <typename T>
struct even_first {
    VEX_FUNCTION(device, bool(int, int),
        VEX_STRINGIZE_SOURCE(
            char bit1 = 1 & prm1;
            char bit2 = 1 & prm2;
            if (bit1 == bit2) return prm1 < prm2;
            return bit1 < bit2;
            )
        );
    bool operator()(int a, int b) const {
        char bit1 = 1 & a;
        char bit2 = 1 & b;
        if (bit1 == bit2) return a < b;
        return bit1 < bit2;
    }
};
~~~

Note that VexCL already provides `vex::less<T>`, `vex::less_equal<T>`,
`vex::greater<T>`, `vex::greater_equal<T>`, and `vex::plus<T>`.

The need to provide both host-side and device-side parts of the functor comes
from the fact that multidevice vectors are first sorted partially on each of
the compute devices they are allocated on and then merged on the host.

Sorting algorithms may also take tuples of keys/values (in fact, any
Boost.Fusion sequence will do).  One will have to explicitly specify the
comparison functor in this case. Both host and device variants of the
comparison functor should take `2n` arguments, where `n` is the number of keys.
The first `n` arguments correspond to the left set of keys, and the second `n`
arguments correspond to the right set of keys. Here is an example that sorts
values by a tuple of two keys:

~~~{.cpp}
vex::vector<int>    keys1(ctx, n);
vex::vector<float>  keys2(ctx, n);
vex::vector<double> vals (ctx, n);

struct {
    VEX_FUNCTION(device, bool(int, float, int, float),
            "return (prm1 == prm3) ? (prm2 < prm4) : (prm1 < prm3);"
            );
    bool operator()(int a1, float a2, int b1, float b2) const {
        return std::make_tuple(a1, a2) < std::tuple(b1, b2);
    }
} comp;

vex::sort_by_key(std::tie(keys1, keys2), vals, comp);
~~~

## <a name="multivectors"></a>Multivectors

The class template `vex::multivector<T,N>` allows one to store several equally
sized device vectors and perform computations on all components synchronously.
Each operation is delegated to the underlying vectors, but usually results in
the launch of a single fused kernel. Expressions may include values of
`std::array<T,N>` type, where N is equal to the number of multivector
components. Each component gets the corresponding element of `std::array<>`
when the expression is applied. Similarly, the array subscript operator or
reduction of a multivector returns an `std::array<T,N>`.  In order to access
k-th component of a multivector, one can use the overloaded `operator()`:

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

Some operations can not be expressed with simple multivector arithmetic. For
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
But this would result in two kernel launches. VexCL allows one to assign a tuple of
expressions to a multivector, which will lead to the launch of a single fused
kernel:
~~~
Y = std::tie( X(0) * cos(alpha) - X(1) * sin(alpha),
              X(0) * sin(alpha) + X(1) * cos(alpha) );
~~~

## <a name="converting-generic-c-algorithms-to-opencl"></a>Converting generic C++ algorithms to OpenCL/CUDA

CUDA and OpenCL differ in their handling of compute kernels compilation. In
NVIDIA's framework the compute kernels are compiled to PTX code together with
the host program. In OpenCL the compute kernels are compiled at runtime from
high-level C-like sources, adding an overhead which is particularly noticeable
for smaller sized problems. This distinction leads to higher initialization
cost of OpenCL programs, but at the same time it allows one to generate better
optimized kernels for the problem at hand. VexCL exploits this possibility with
help of its kernel generator mechanism. Moreover, VexCL's CUDA backend uses the
same technique to generate and compile CUDA kernels at runtime.

An instance of `vex::symbolic<T>` dumps to an output stream any arithmetic
operations it is being subjected to. For example, this code snippet:
~~~{.cpp}
vex::symbolic<double> x = 6, y = 7;
x = sin(x * y);
~~~
results in the following output:
~~~
double var1 = 6;
double var2 = 7;
var1 = sin( ( var1 * var2 ) );
~~~

### <a name="kernel-generator"></a>Kernel generator

The symbolic type allows one to record a sequence of arithmetic operations made by
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

The following code snippet makes one hundred RK4 iterations for a single
`double` value on a CPU:
~~~{.cpp}
double x = 1, dt = 0.01;

for(int step = 0; step < 100; ++step)
    runge_kutta_4(sys_func<double>, x, dt);
~~~

Let's now generate the kernel for a single RK4 step and apply the kernel to a
`vex::vector<double>` (by doing this we essentially simultaneously solve a large
number of identical ODEs with different initial conditions).
~~~{.cpp}
// Set recorder for expression sequence.
std::ostringstream body;
vex::generator::set_recorder(body);

// Create symbolic variable.
typedef vex::symbolic<double> sym_state;
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
a substantial amount of both human and machine time when applicable.

### <a name="function-generator"></a>Function generator

VexCL also provides a user-defined function generator which takes a function
signature and generic function object, and returns custom VexCL function ready
to be used in vector expressions. Let's rewrite the above example using an
autogenerated function for a Runge-Kutta stepper. First, we need to implement
generic functor:

~~~{.cpp}
struct rk4_stepper {
    double dt;

    rk4_stepper(double dt) : dt(dt) {}

    template <class state_type>
    state_type operator()(const state_type &x) const {
        state_type new_x = x;
        runge_kutta_4(sys_func<state_type>, new_x, dt);
        return new_x;
    }
};
~~~

Now we can generate and apply the custom function:
~~~{.cpp}
double dt = 0.01;
rk4_stepper stepper(dt);

// Generate custom VexCL function:
auto rk4 = vex::generator::make_function<double(double)>(stepper);

// Create initial state.
const size_t n = 1024 * 1024;
vex::vector<double> x(ctx, n);
x = 10.0 * vex::element_index() / n;

// Use the function to advance initial state:
for(int i = 0; i < 100; i++) x = rk4(x);
~~~

Note that both `runge_kutta_4()` and `rk4_stepper` may be reused for host-side
computations.

It is very easy to generate a VexCL function from a Boost.Phoenix lambda
expression (since Boost.Phoenix lambdas are themselves generic functors):

~~~{.cpp}
using namespace boost::phoenix::arg_names;
using vex::generator::make_function;

auto squared_radius = make_function<double(double, double)>(arg1 * arg1 + arg2 * arg2);

Z = squared_radius(X, Y);
~~~

## <a name="custom-kernels"></a>Custom kernels

As [Kozma Prutkov](http://en.wikipedia.org/wiki/Kozma_Prutkov) repeatedly said,
"One cannot embrace the unembraceable". So in order to be usable, VexCL has to
support custom kernels. `vex::vector::operator()(uint k)` returns a `cl::Buffer`
that holds vector data on the k-th compute device.  If the result depends on the
neighboring points, one has to keep in mind that these points are possibly located
on a different compute device.  In this case the exchange of these halo points
has to be addressed manually.

The following example builds and launches a custom kernel for each device in
the context:
~~~{.cpp}
std::vector<vex::backend::kernel> kernel;

// Compile and store the kernels for later use.
for(uint d = 0; d < ctx.size(); d++) {
    kernel.emplace_back(ctx.queue(d),
        VEX_STRINGIZE_SOURCE(
            kernel void dummy(ulong n, global float *x) {
                for(size_t i = get_global_id(0); i < n; i += get_global_size(0))
                    x[i] = 4.2;
            }
            ),
        "dummy"
        );
}

// Apply the kernels to the vector partitions on each device.
for(uint d = 0; d < ctx.size(); d++) {
    kernel[d].push_arg<cl_ulong>(x.part_size());
    kernel[d].push_arg(x(d));

    kernel[d](ctx.queue(d));
}
~~~

## <a name="interoperability-with-other-libraries"></a>Interoperability with other libraries

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

It is possible to use ViennaCL's generic solvers with VexCL types. See
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

## <a name="supported-compilers"></a>Supported compilers

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
partially supported by RFBR grants No 12-07-0007 and 12-01-00333a._

[jscc]: http://www.jscc.ru/eng/index.shtml
[kpfu]: http://www.kpfu.ru
