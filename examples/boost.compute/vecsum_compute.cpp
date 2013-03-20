#include <iostream>
#include <boost/compute.hpp>

int main() {
    const size_t n = 1 << 10;
    const size_t m = 1 << 12;

    boost::compute::device gpu = boost::compute::system::default_gpu_device();
    std::cout << gpu.name() << std::endl;

    boost::compute::context       context(gpu);
    boost::compute::command_queue queue(context, gpu);

    boost::compute::vector<float> x(n, context);
    boost::compute::vector<float> y(n, context);

    boost::compute::fill(x.begin(), x.end(), 0.0f, queue);
    boost::compute::fill(y.begin(), y.end(), 1.0f, queue);

    for(size_t i = 0; i < m; ++i) {
        boost::compute::transform(
                x.begin(), x.end(), y.begin(), x.begin(),
                boost::compute::plus<float>(), queue
                );
    }

    queue.finish();
}

