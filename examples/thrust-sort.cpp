#include <vexcl/vexcl.hpp>
#include "thrust-sort.hpp"

int main() {
    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    vex::vector<float> x(ctx, 32);
    vex::Random<float> rnd;

    x = rnd(vex::element_index(), 0);
    std::cout << "Before: " << x << std::endl;

    // Get raw pointers to the device memory.
    float *x_begin = x(0).raw_ptr();
    float *x_end   = x_begin + x.size();

    // Apply thrust algorithm.
    thrust_sort(x_begin, x_end);
    std::cout << "\nAfter: " << x << std::endl;
}
