#ifndef VEXCL_BACKEND_CUDA_DEVICE_VECTOR_HPP
#define VEXCL_BACKEND_CUDA_DEVICE_VECTOR_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl/backend/cuda/device_vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  CUDA device vector.
 */

#include <cuda.h>

#include <vexcl/backend/cuda/context.hpp>

namespace vex {
namespace backend {

typedef unsigned mem_flags;

static const mem_flags MEM_READ_ONLY  = 1;
static const mem_flags MEM_WRITE_ONLY = 2;
static const mem_flags MEM_READ_WRITE = 4;

template <typename T>
class device_vector {
    public:
        typedef CUdeviceptr raw_type;

        device_vector() {}

        device_vector(const command_queue &q, size_t n,
                const T *host = 0, mem_flags flags = MEM_READ_WRITE)
            : n(n)
        {
            (void)flags;

            if (n) {
                q.context().set_current();

                CUdeviceptr ptr;
                cuda_check( cuMemAlloc(&ptr, n * sizeof(T)) );

                buffer.reset(reinterpret_cast<char*>(static_cast<size_t>(ptr)), detail::deleter() );

                if (host) write(q, 0, n, host, true);
            }
        }

        void write(const command_queue &q, size_t offset, size_t size, const T *host,
                bool blocking = false) const
        {
            (void)blocking;

            if (size) {
                q.context().set_current();
                cuda_check( cuMemcpyHtoD(raw() + offset * sizeof(T), host, size * sizeof(T)) );
            }
        }

        void read(const command_queue &q, size_t offset, size_t size, T *host,
                bool blocking = false) const
        {
            (void)blocking;

            if (size) {
                q.context().set_current();
                cuda_check( cuMemcpyDtoH(host, raw() + offset * sizeof(T), size * sizeof(T)) );
            }
        }

        size_t size() const {
            return n;
        }

        struct buffer_unmapper {
            const command_queue &queue;
            const device_vector &buffer;

            buffer_unmapper(const command_queue &q, const device_vector &b)
                : queue(q), buffer(b)
            {}

            void operator()(T* ptr) const {
                buffer.write(queue, 0, buffer.size(), ptr, true);
                delete[] ptr;
            }
        };

        typedef std::unique_ptr<T[], buffer_unmapper> mapped_array;

        mapped_array map(const command_queue &q) {
            T *ptr = new T[n];
            read(q, 0, n, ptr, true);
            return mapped_array(ptr, buffer_unmapper(q, *this));
        }

        CUdeviceptr raw() const {
            return static_cast<CUdeviceptr>(reinterpret_cast<size_t>(buffer.get()));
        }
    private:
        std::shared_ptr<char> buffer;
        size_t n;
};

} // namespace backend
} // namespace vex

#endif
