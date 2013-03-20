#ifndef VEXCL_EXTERNAL_BOOST_COMPUTE_HPP
#define VEXCL_EXTERNAL_BOOST_COMPUTE_HPP

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
 * \file   external/boost_compute.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Provides wrappers for some of Boost.Compute (https://github.com/kylelutz/compute) algorithms.
 */

#include <algorithm>
#include <vexcl/vector.hpp>
#include <boost/compute.hpp>

namespace vex {

template <typename T>
void scan(const vex::vector<T> &src, vex::vector<T> &dst, bool exclusive = false) {
    auto queue = src.queue_list();

    // Scan partitions separately.
    for(unsigned d = 0; d < queue.size(); ++d) {
        if (src.part_size(d)) {
            boost::compute::command_queue q( queue[d]() );
            
            cl::Buffer sbuf = src(d);
            cl::Buffer dbuf = dst(d);

            boost::compute::detail::scan(
                    boost::compute::make_buffer_iterator<T>(sbuf(), 0),
                    boost::compute::make_buffer_iterator<T>(sbuf(), src.part_size(d)),
                    boost::compute::make_buffer_iterator<T>(dbuf(), 0),
                    exclusive && (d == 0), q
                    );
        }
    }

    // If there are more than one partition,
    // update all of them except for the first.
    if (queue.size() > 1) {
        std::vector<T> tail(queue.size() - 1, T());

        for(unsigned d = 0; d < tail.size(); ++d) {
            if (src.part_size(d))
                tail[d] = dst[src.part_start(d + 1) - 1];
        }

        std::partial_sum(tail.begin(), tail.end(), tail.begin());

        for(unsigned d = 1; d < queue.size(); ++d) {
            if (src.part_size(d)) {
                // Wrap partition into vector for ease of use:
                vex::vector<T> part(queue[d], dst(d));
                part += tail[d - 1];
            }
        }
    }
}

/// Inclusive scan.
template <typename T>
void inclusive_scan(const vex::vector<T> &src, vex::vector<T> &dst) {
    scan(src, dst, false);
}

/// Exclusive scan.
template <typename T>
void exclusive_scan(const vex::vector<T> &src, vex::vector<T> &dst) {
    scan(src, dst, true);
}

}

#endif
