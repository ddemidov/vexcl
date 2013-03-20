#ifndef VEXCL_EXTERNAL_BOOST_COMPUTE_HPP
#define VEXCL_EXTERNAL_BOOST_COMPUTE_HPP

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

}

#endif
