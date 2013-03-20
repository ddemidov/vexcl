#ifndef VEXCL_EXTERNAL_BOOST_COMPUTE_HPP
#define VEXCL_EXTERNAL_BOOST_COMPUTE_HPP

#include <algorithm>
#include <vexcl/vector.hpp>
#include <boost/compute.hpp>

namespace vex {

namespace compute {

template <typename T>
boost::compute::buffer_iterator<T>
begin(const vex::vector<T> x, unsigned d) {
    return boost::compute::make_buffer_iterator<T>( x(d)(), 0 );
}

template <typename T>
boost::compute::buffer_iterator<T>
end(const vex::vector<T> x, unsigned d) {
    return boost::compute::make_buffer_iterator<T>( x(d)(), x.part_size(d) );
}

}

template <typename T>
void scan(const vex::vector<T> &src, vex::vector<T> &dst, bool exclusive = false) {
    auto queue = src.queue_list();

    // Scan partitions separately.
    for(unsigned d = 0; d < queue.size(); ++d) {
        if (src.part_size(d)) {
            boost::compute::command_queue q( queue[d]() );

            boost::compute::detail::scan(
                    compute::begin(src, d), compute::end(src, d),
                    compute::begin(dst, d),
                    exclusive && (d == 0), q
                    );
            q.finish();
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
