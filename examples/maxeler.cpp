#include <vexcl/backend.hpp>

int main() {
    vex::backend::command_queue q;
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

    auto K = vex::backend::kernel(q, src.str(), "simple");

    int n = 16;
    std::vector<float> x(n), y(n);

    for(int i = 0; i < 16; ++i) x[i] = i;
    K(q, n, x.data(), y.data());

    for(int i = 0; i < 16; ++i)
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
}
