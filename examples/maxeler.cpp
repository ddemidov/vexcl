#include <vexcl/devlist.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/function.hpp>
#include <vexcl/backend.hpp>

int main(int argc, char *argv[]) {
    vex::Context ctx(vex::Filter::Any);
    std::cout << ctx << std::endl;

    int n = 32;
    vex::vector<float> x(ctx, n), y(ctx, n);

    for(int i = 0; i < n; ++i) x[i] = i;

    // 1. An expression:
    VEX_FUNCTION(float, square, (float, v),
            return v * v;
            );

    y = square(x + 3.0f);
    std::cout << "y = " << y << std::endl;

    // 2. A custom kernel:
    // Generate the sources:
    vex::backend::command_queue q = ctx.queue(0);
    vex::backend::source_generator src{q};

    src.begin_function<float>("mul2");
    src.begin_function_parameters();
    src.parameter<float>("x");
    src.end_function_parameters();
    src.new_line() << "return 2 * x;";
    src.end_function();

    src.begin_kernel("simple");
    src.begin_kernel_parameters();
    src.input_parameters();
    src.parameter<int>("n");
    src.parameter<const float*>("x");
    src.output_parameters();
    src.parameter<float*>("y");
    src.end_kernel_parameters();
    src.new_line() << "y = mul2.apply(x);";
    src.end_kernel();

    // Compile the kernel:
    auto K = vex::backend::kernel(q, src.str(), "simple");

    // Launch the kernel:
    K(q, n, x(), y());

    std::cout << "y = " << y << std::endl;
}
