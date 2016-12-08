#ifndef VEXCL_BACKEND_MAXELER_DEVICE_VECTOR_HPP
#define VEXCL_BACKEND_MAXELER_DEVICE_VECTOR_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   vexcl/backend/maxeler/device_vector.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Wrapper around std::vector for the Maxeler backend.
 */

#include <vector>

namespace vex {
namespace backend {
namespace maxeler {

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

} // namespace maxeler
} // namespace backend
} // namespace vex

#endif
