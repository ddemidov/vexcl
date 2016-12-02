#ifndef VEXCL_BACKEND_JIT_KERNEL_HPP
#define VEXCL_BACKEND_JIT_KERNEL_HPP

#include <string>
#include <boost/dll/import.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <vexcl/util.hpp>
#include <vexcl/backend/jit/compiler.hpp>

namespace vex {
namespace backend {
namespace jit {

namespace detail {

struct kernel_api {
    virtual void execute(
            const ndrange *dim, const ndrange *id, char *smem, char *prm
            ) const = 0;
};

} // namespace detail

class kernel {
    public:
        kernel() : smem(0) {}

        kernel(
                const command_queue &q,
                const std::string &src, const std::string &name,
                size_t smem_per_thread = 0,
                const std::string &options = ""
              )
            : K(boost::dll::import<detail::kernel_api>(build_sources(q, src, options), name)),
              grid(num_workgroups(q)), smem(smem_per_thread)
        {
            stack.reserve(256);
        }

        kernel(const command_queue &q,
               const std::string &src, const std::string &name,
               std::function<size_t(size_t)> smem,
               const std::string &options = ""
               )
            : K(boost::dll::import<detail::kernel_api>(build_sources(q, src, options), name)),
              grid(num_workgroups(q)), smem(smem(1))
        {
            stack.reserve(256);
        }

        kernel(const command_queue &q,
               const program &P,
               const std::string &name,
               size_t smem_per_thread = 0
               )
            : K(boost::dll::import<detail::kernel_api>(P, name)),
              grid(num_workgroups(q)), smem(smem_per_thread)
        {
            stack.reserve(256);
        }

        /// Constructor. Extracts a backend::kernel instance from backend::program.
        kernel(const command_queue &q, const program &P,
               const std::string &name,
               std::function<size_t(size_t)> smem
               )
            : K(boost::dll::import<detail::kernel_api>(P, name)),
              grid(num_workgroups(q)), smem(smem(1))
        {
            stack.reserve(256);
        }

        template <class Arg>
        void push_arg(const Arg &arg) {
            char *c = (char*)&arg;
            stack.insert(stack.end(), c, c + sizeof(arg));
        }

        template <typename T>
        void push_arg(const device_vector<T> &arg) {
            push_arg(arg.raw());
        }

        void set_smem(size_t smem_per_thread) {
            smem.resize(smem_per_thread);
        }

        template <class F>
        void set_smem(F &&f) {
            smem.resize(f(1));
        }

        void operator()(const command_queue&) {
            // All parameters have been pushed; time to call the kernel:
#pragma omp parallel for collapse(3) firstprivate(smem)
            for(size_t z = 0; z < grid.z; ++z) {
                for(size_t y = 0; y < grid.y; ++y) {
                    for(size_t x = 0; x < grid.x; ++x) {
                        ndrange id(x, y, z);
                        K->execute(&grid, &id, smem.data(), stack.data());
                    }
                }
            }

            // Reset parameter stack:
            stack.clear();
        }

#ifndef BOOST_NO_VARIADIC_TEMPLATES
        template <class Head, class... Tail>
        void operator()(const command_queue &q, const Head &head, const Tail&... tail) {
            push_arg(head);
            (*this)(q, tail...);
        }
#endif
        size_t workgroup_size() const {
            return 1UL;
        }

        static inline size_t num_workgroups(const command_queue&) {
#ifdef _OPENMP
            return omp_get_num_procs() * 8;
#else
            return 1UL;
#endif
        }

        size_t max_threads_per_block(const command_queue&) const {
            return 1UL;
        }

        size_t max_shared_memory_per_block(const command_queue&) const {
            return 32768UL;
        }

        size_t preferred_work_group_size_multiple(const backend::command_queue &q) const {
            return 1;
        }

        void config(const command_queue &q, std::function<size_t(size_t)> smem) {
            config(num_workgroups(q), 1);
        }

        void config(ndrange blocks, ndrange threads) {
            precondition(threads == ndrange(), "Maximum workgroup size for the JIT backend is 1");
            grid = blocks;
        }

        void config(size_t blocks, size_t threads) {
            precondition(threads == 1, "Maximum workgroup size for the JIT backend is 1");
            config(ndrange(blocks), ndrange(threads));
        }

        void reset() {
            stack.clear();
        }
    private:
        boost::shared_ptr<detail::kernel_api> K;
        ndrange grid;
        std::vector<char> stack;
        std::vector<char> smem;
};

} // namespace jit
} // namespace backend
} // namespace vex

#endif
