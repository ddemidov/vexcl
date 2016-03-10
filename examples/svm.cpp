#include <iostream>

#include <vexcl/vexcl.hpp>

#if defined(VEXCL_BACKEND_CUDA) || defined(CL_VERSION_2_0)
#include <vexcl/svm.hpp>

int main() {
    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

#if !defined(VEXCL_BACKEND_CUDA)
    vex::push_compile_options(ctx.queue(0), "-cl-std=CL2.0");
#endif

    const int n = 16;

    vex::vector<int> y(ctx, n);
    vex::svm<int> x(ctx.queue(0), n);

    {
        auto p = x.map(vex::backend::MAP_WRITE);
        for(int i = 0; i < n; ++i) p[i] = i;
    }

    y = x;

    std::cout << y << std::endl;

    VEX_FUNCTION(int, take_svm, (int, x),
            return 2 * x;
            );

    y = take_svm(x);

    std::cout << y << std::endl;
}
#else
int main() {
    std::cout << "SVM is only supported for OpenCL v2.0" << std::endl;
}
#endif
