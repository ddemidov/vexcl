#include <vexcl/vexcl.hpp>
#include <vexcl/external/boost_compute.hpp>

int main() {
    const size_t n = 1 << 10;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    vex::vector<float> x(ctx, n);

    vex::Random<float, vex::random::philox> rnd;
    x = rnd(vex::element_index(), rand());

    vex::sort(x);
    for(size_t i = 0; i < 10; ++i)
        std::cout << x[i] << std::endl;
}


