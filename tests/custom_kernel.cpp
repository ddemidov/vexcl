#define BOOST_TEST_MODULE CustomKernel
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/function.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(custom_kernel)
{
    const cl_ulong n = 1024;

    std::vector<vex::command_queue> queue(1, ctx.queue(0));

    vex::vector<int> x(queue, n);

    // Single kernel per program
    vex::backend::kernel zeros(queue[0],
#ifdef VEXCL_BACKEND_CUDA
            VEX_STRINGIZE_SOURCE(
                extern "C" __global__ void zeros(size_t n, int *x) {
                    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                    size_t m = blockDim.x * gridDim.x;

                    for(; i < n; i += m) x[i] = 0;
                }
                ),
#else
            VEX_STRINGIZE_SOURCE(
                kernel void zeros(ulong n, global int *x) {
                    for(ulong i = get_global_id(0); i < n; i += get_global_size(0))
                        x[i] = 0;
                }
                ),
#endif
                "zeros"
            );

#ifdef BOOST_NO_VARIADIC_TEMPLATES
    zeros.push_arg(n);
    zeros.push_arg(x(0));
    zeros(queue[0]);
#else
    zeros(queue[0], n, x(0));
#endif

    check_sample(x, [](size_t, int v) { BOOST_CHECK_EQUAL(v, 0); });

    // A couple of kernels per program
    auto program = vex::backend::build_sources(queue[0],
#ifdef VEXCL_BACKEND_CUDA
            VEX_STRINGIZE_SOURCE(
                extern "C" __global__ void ones(size_t n, int *x) {
                    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                    size_t m = blockDim.x * gridDim.x;

                    for(; i < n; i += m) x[i] = 1;
                }
                extern "C" __global__ void twos(size_t n, int *x) {
                    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                    size_t m = blockDim.x * gridDim.x;

                    for(; i < n; i += m) x[i] = 2;
                }
                )
#else
            VEX_STRINGIZE_SOURCE(
                kernel void ones(ulong n, global int *x) {
                    for(ulong i = get_global_id(0); i < n; i += get_global_size(0))
                        x[i] = 1;
                }
                kernel void twos(ulong n, global int *x) {
                    for(ulong i = get_global_id(0); i < n; i += get_global_size(0))
                        x[i] = 2;
                }
                )
#endif
            );

    vex::backend::kernel ones(queue[0], program, "ones");
    vex::backend::kernel twos(queue[0], program, "twos");

#ifdef BOOST_NO_VARIADIC_TEMPLATES
    ones.push_arg(n);
    ones.push_arg(x(0));
    ones(queue[0]);
#else
    ones(queue[0], n, x(0));
#endif

    check_sample(x, [](size_t, int v) { BOOST_CHECK_EQUAL(v, 1); });

#ifdef BOOST_NO_VARIADIC_TEMPLATES
    twos.push_arg(n);
    twos.push_arg(x(0));
    twos(queue[0]);
#else
    twos(queue[0], n, x(0));
#endif

    check_sample(x, [](size_t, int v) { BOOST_CHECK_EQUAL(v, 2); });
}

BOOST_AUTO_TEST_SUITE_END()
