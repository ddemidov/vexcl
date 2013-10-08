#define BOOST_TEST_MODULE VectorPointer
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_pointer.hpp>
#include <vexcl/element_index.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(nbody)
{
    const size_t n = 1024;

    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    std::vector<double> X = random_vector<double>(n);

    vex::vector<double> x(queue, X);
    vex::vector<double> y(queue, n);

    VEX_FUNCTION(nbody, double(size_t, size_t, double*),
            "double sum = 0;\n"
            "for(size_t i = 0; i < prm1; ++i)\n"
            "    if (i != prm2) sum += prm3[i];\n"
            "return sum;"
            );

    y = nbody(n, vex::element_index(), vex::raw_pointer(x));

    check_sample(y, [&](size_t idx, double v) {
            double sum = 0;
            for(size_t i = 0; i < n; ++i)
                if (i != idx) sum += X[i];
            BOOST_CHECK_CLOSE(v, sum, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()

