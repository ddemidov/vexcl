#ifndef VEXCL_GATHER_HPP
#define VEXCL_GATHER_HPP

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
 * \file   vexcl/gather.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Gather scattered points from OpenCL device vector.
 */

#include <vector>
#include <numeric>
#include <cassert>

#include <vexcl/operations.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>

namespace vex {

template <typename T>
class gather {
    public:
        gather(
                const std::vector<backend::command_queue> &queue,
                size_t src_size, std::vector<size_t> indices
              )
            : queue(queue), ptr(queue.size() + 1, 0),
              idx(queue.size()), val(queue.size())
        {
            assert(std::is_sorted(indices.begin(), indices.end()));

            std::vector<size_t> part = partition(src_size, queue);
            column_owner owner(part);

            for(auto i = indices.begin(); i != indices.end(); ++i) {
                size_t d = owner(*i);
                *i -= part[d];
                ++ptr[d + 1];
            }

            std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

            for(unsigned d = 0; d < queue.size(); d++) {
                if (size_t n = ptr[d + 1] - ptr[d]) {
                    val[d] = backend::device_vector<T>(queue[d], n, static_cast<const T*>(0));
                    idx[d] = backend::device_vector<size_t>(
                            queue[d], n, &indices[ptr[d]], backend::MEM_READ_ONLY);
                }
            }

            for(unsigned d = 0; d < queue.size(); d++)
                if (ptr[d + 1] - ptr[d]) queue[d].finish();
        }

        template <class HostVector>
        void operator()(const vex::vector<T> &src, HostVector &dst) {
            using namespace detail;

            static kernel_cache cache;

            for(unsigned d = 0; d < queue.size(); d++) {
                if (size_t n = ptr[d + 1] - ptr[d]) {
                    vector<T>      v(queue[d], val[d]);
                    vector<T>      s(queue[d], src(d));
                    vector<size_t> i(queue[d], idx[d]);

                    v = permutation(i)(s);

                    val[d].read(queue[d], 0, n, &dst[ptr[d]]);
                }
            }

            for(unsigned d = 0; d < queue.size(); d++)
                if (ptr[d + 1] - ptr[d]) queue[d].finish();
        }
    private:
        std::vector<backend::command_queue> queue;
        std::vector<size_t>                           ptr;
        std::vector< backend::device_vector<size_t> > idx;
        std::vector< backend::device_vector<T> >      val;
};

} // namespace vex

#endif
