#ifndef VEXCL_BACKEND_JIT_DEVICE_VECTOR_HPP
#define VEXCL_BACKEND_JIT_DEVICE_VECTOR_HPP

#include <vector>

namespace vex {
namespace backend {
namespace jit {

typedef unsigned mem_flags;

static const mem_flags MEM_READ_ONLY  = 1;
static const mem_flags MEM_WRITE_ONLY = 2;
static const mem_flags MEM_READ_WRITE = 4;

template <typename T>
class device_vector {
    public:
        typedef T  value_type;
        typedef T* raw_type;
        typedef std::shared_ptr<std::vector<T>> buffer_type;

        device_vector() {}

        device_vector(const command_queue &q, size_t n, const T *host = 0, mem_flags = MEM_READ_WRITE)
            : buffer(std::make_shared<std::vector<T>>(n))
        {
            if (host) std::copy(host, host + n, buffer->begin());
        }

        device_vector(buffer_type buffer) : buffer(buffer) {}

        void write(const command_queue&, size_t offset, size_t size, const T *host, bool /*blocking*/ = false) const
        {
            std::copy(host, host + size, buffer->begin() + offset);
        }

        void read(const command_queue&, size_t offset, size_t size, T *host, bool /*blocking*/ = false) const
        {
            std::copy_n(buffer->begin() + offset, size, host);
        }

        size_t size() const {
            return buffer->size();
        }

        typedef T* mapped_array;

        T* map(const command_queue&) {
            return buffer ? buffer->data() : nullptr;
        }

        const T* map(const command_queue&) const {
            return buffer ? buffer->data() : nullptr;
        }

        const T* raw() const {
            return buffer ? buffer->data() : nullptr;
        }

        T* raw() {
            return buffer ? buffer->data() : nullptr;
        }

        const buffer_type raw_buffer() const {
            return buffer;
        }
    private:
        mutable buffer_type buffer;
};

} // namespace jit
} // namespace backend
} // namespace vex

#endif
