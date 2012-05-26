#include <vector>
#include <tuple>
#include <oclutil/oclutil.hpp>

using namespace clu;

int main() {
    std::vector<cl::Context>      cpu_context;
    std::vector<cl::CommandQueue> cpu_queue;
    std::tie(cpu_context, cpu_queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_CPU), true);

    std::vector<cl::Context>      gpu_context;
    std::vector<cl::CommandQueue> gpu_queue;
    std::tie(gpu_context, gpu_queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU), true);

    clu::vector<double> xcpu(cpu_queue, CL_MEM_READ_WRITE, 1024);
    xcpu = Const(1);
    std::cout << xcpu[0] << std::endl;

    clu::vector<double> xgpu(gpu_queue, CL_MEM_READ_WRITE, 1024);
    xgpu = Const(1);
    std::cout << xgpu[0] << std::endl;

    Reductor<double,SUM> sum_cpu(cpu_queue);
    std::cout << sum_cpu(xcpu) << std::endl;

    Reductor<double,SUM> sum_gpu(gpu_queue);
    std::cout << sum_gpu(xgpu) << std::endl;
}
