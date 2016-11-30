#ifndef VEXCL_BACKEND_JIT_KERNEL_HPP
#define VEXCL_BACKEND_JIT_KERNEL_HPP

#include <string>
#include <boost/dll/import.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <vexcl/backend/jit/compiler.hpp>

namespace vex {
namespace backend {
namespace jit {

namespace detail {

struct grid_info {
    ndrange grid_dim;
    ndrange block_dim;

    grid_info() :
#ifdef _OPENMP
          grid_dim(omp_get_num_threads() * 8)
#endif
        , block_dim(1024)
    {
    }
};

struct thread_info {
    ndrange block_id;
    ndrange thread_id;
};

struct kernel_api {
    virtual void execute(const grid_info*, const thread_info*, char *) const = 0;
};

} // namespace detail

class kernel {
    public:
        kernel(
                const command_queue &q,
                const std::string &src, const std::string &name,
                size_t smem_per_thread = 0,
                const std::string &options = ""
              )
            : K(boost::dll::import<detail::kernel_api>(build_sources(q, src, options), name))
        {
            stack.reserve(256);
        }

        template <class Arg>
        void push_arg(const Arg &arg) {
            char *c = (char*)&arg;
            stack.insert(stack.end(), c, c + sizeof(arg));
        }

        void operator()(const command_queue&) {
            // All parameters have been pushed; time to call the kernel:
#pragma omp parallel for collapse(3)
            for(size_t g_id_z = 0; g_id_z < grid.grid_dim.z; ++g_id_z) {
                for(size_t g_id_y = 0; g_id_y < grid.grid_dim.y; ++g_id_y) {
                    for(size_t g_id_x = 0; g_id_x < grid.grid_dim.x; ++g_id_x) {

                        detail::thread_info t{ndrange(g_id_x, g_id_y, g_id_z), ndrange()};

                        for(t.thread_id.z = 0; t.thread_id.z < grid.block_dim.z; ++t.thread_id.z)
                            for(t.thread_id.y = 0; t.thread_id.y < grid.block_dim.y; ++t.thread_id.y)
                                for(t.thread_id.x = 0; t.thread_id.x < grid.block_dim.x; ++t.thread_id.x)
                                    K->execute(&grid, &t, stack.data());
                    }
                }
            }

            // Reset parameter stack:
            stack.clear();
        }

        template <class Head, class... Tail>
        void operator()(const command_queue &q, const Head &head, const Tail&... tail) {
            push_arg(head);
            (*this)(q, tail...);
        }

        void reset() {
            stack.clear();
        }
    private:
        boost::shared_ptr<detail::kernel_api> K;
        detail::grid_info grid;
        std::vector<char> stack;
};

} // namespace jit
} // namespace backend
} // namespace vex

#endif
