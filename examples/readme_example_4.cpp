#include <vector>
#include <cassert>
#include <oclutil/oclutil.hpp>
using namespace clu;
int main() {
    cl::Context context;
    std::vector<cl::CommandQueue> queue;
    std::tie(context, queue) = queue_list(Filter::Type(CL_DEVICE_TYPE_GPU));

    const uint n = 1 << 20;
    clu::vector<float> x(queue, CL_MEM_WRITE_ONLY, n);

    auto program = build_sources(context, std::string(
		"kernel void dummy(uint size, global float *x)\n"
		"{\n"
		"    uint i = get_global_id(0);\n"
		"    if (i < size) x[i] = 4.2;\n"
		"}\n"
		));

    for(uint d = 0; d < queue.size(); d++) {
	auto dummy = cl::Kernel(program, "dummy").bind(queue[d], alignup(n, 256), 256);
	dummy((uint)x.part_size(d), x(d));
    }

    std::cout << sum(x) << std::endl;
}

