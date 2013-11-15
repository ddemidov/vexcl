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

#ifdef VEXCL_BACKEND_CUDA
#  error Boost.Compute interoperation is not supported for the CUDA backend!
#endif

namespace vex {

/// Inclusive scan.
template <typename T>
void inclusive_scan(const vex::vector<T> &src, vex::vector<T> &dst) {
    auto queue = src.queue_list();

    // Scan partitions separately.
    for(unsigned d = 0; d < queue.size(); ++d) {
        if (src.part_size(d)) {
            boost::compute::command_queue q( queue[d]() );

            boost::compute::buffer sbuf( src(d)() );
            boost::compute::buffer dbuf( dst(d)() );

            boost::compute::detail::scan(
                    boost::compute::make_buffer_iterator<T>(sbuf, 0),
                    boost::compute::make_buffer_iterator<T>(sbuf, src.part_size(d)),
                    boost::compute::make_buffer_iterator<T>(dbuf, 0),
                    false, q
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

/// Sort.
/**
 * If there are more than one device in vector's queue list, then all
 * partitions are sorted individually on GPUs and then merged on CPU.
 */
template <typename T>
void sort(vex::vector<T> &x) {
    auto queue = x.queue_list();

    for(unsigned d = 0; d < queue.size(); ++d) {
        if (x.part_size(d)) {
            boost::compute::command_queue q( queue[d]() );
            boost::compute::buffer buf( x(d)() );

            boost::compute::sort(
                    boost::compute::make_buffer_iterator<T>(buf, 0),
                    boost::compute::make_buffer_iterator<T>(buf, x.part_size(d)),
                    q
                    );
        }
    }

    if (queue.size() > 1) {
        // Get sorted partitions to host side and do multiway merge sort.

        std::vector<T> src(x.size()), dst(x.size());
        vex::copy(x, src);

        std::vector< typename std::vector<T>::const_iterator > begin(queue.size());
        std::vector< typename std::vector<T>::const_iterator > end  (queue.size());

        for(unsigned d = 0; d < queue.size(); ++d) {
            begin[d] = src.begin() + x.part_start(d);
            end  [d] = src.begin() + x.part_start(d + 1);
        }


        for(auto pos = dst.begin(); pos != dst.end(); ++pos) {
            int winner = -1;
            for(unsigned d = 0; d < queue.size(); ++d) {
                if (begin[d] == end[d])
                    continue;

                if (winner < 0 || *begin[d] < *begin[winner])
                    winner = d;
            }

            *pos = *begin[winner]++;
        }

        vex::copy(dst, x);
    }
}

}

#endif
