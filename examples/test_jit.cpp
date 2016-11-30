#include <iostream>
#include <string>
#include <vector>

#include <vexcl/devlist.hpp>
#include <vexcl/backend.hpp>

const std::string source = R"(
#include <boost/config.hpp>
#include <cmath>

struct ndrange {
    size_t x,y,z;
};

struct grid_info {
    ndrange grid_dim;
    ndrange block_dim;
};

struct thread_info {
    ndrange block_id;
    ndrange thread_id;
};

struct kernel_api {
    virtual void execute(const grid_info *_g, const thread_info *_t, char *_p) const = 0;
};

#define KERNEL_PARAMETER(type, name) \
    type name = *reinterpret_cast<type*>(_p); _p+= sizeof(type)

struct sum_t : public kernel_api {
    void execute(const grid_info *_g, const thread_info *_t, char *_params) const;
};

extern "C" BOOST_SYMBOL_EXPORT sum_t sum;
sum_t sum;

void sum_t::execute(const grid_info *_g, const thread_info *_t, char *_p) const {
    KERNEL_PARAMETER(size_t,  n);
    KERNEL_PARAMETER(double*, x);
    KERNEL_PARAMETER(double*, y);

    size_t global_size = _g->grid_dim.x * _g->block_dim.x;
    size_t global_id   = _t->block_id.x * _g->block_dim.x + _t->thread_id.x;

    size_t chunk_size = (n + global_size - 1) / global_size;
    size_t chunk_start = chunk_size * global_id;
    size_t chunk_end = chunk_start + chunk_size;
    if (n < chunk_end) chunk_end = n;

    for(size_t idx = chunk_start; idx < chunk_end; ++idx) {
        y[idx] += x[idx];
    }
}
)";

int main() {
    vex::Context ctx(vex::Filter::Any);
    std::cout << ctx << std::endl;

    vex::backend::kernel sum(ctx.queue(0), source, "sum");

    const size_t n = 1024 * 1024;
    std::vector<double> x(n, 1.0), y(n, 1.0);

    sum(ctx.queue(0), n, x.data(), y.data());

    for(size_t i = 0; i < n; ++i) {
        std::cout << y[i] << " ";
        if (i == 2) {
            std::cout << "... ";
            i = n - 4;
        }
    }
    std::cout << std::endl;
}
