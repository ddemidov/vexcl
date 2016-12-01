#ifndef VEXCL_BACKEND_JIT_EVENT_HPP
#define VEXCL_BACKEND_JIT_EVENT_HPP

#include <vexcl/backend/jit/context.hpp>

namespace vex {
namespace backend {
namespace jit {

struct event {
    event() {}
    event(const command_queue&) {}
    void wait() const {}
    vex::backend::context context() const {
        return vex::backend::context();
    }
};

struct wait_list {
    template <class... T>
    wait_list(T&&... t) {}
};

/// Append event to wait list
inline void wait_list_append(wait_list&, const event&) { }

/// Append wait list to wait list
inline void wait_list_append(wait_list&, const wait_list&) {}

/// Get id of the context the event was submitted into
inline context_id get_context_id(const event&) {
    return 0;
}

/// Enqueue marker (with wait list) into the queue
inline event enqueue_marker(const command_queue&, const wait_list& = wait_list()) {
    return event();
}

/// Enqueue barrier (with wait list) into the queue
inline event enqueue_barrier(command_queue&, const wait_list& = wait_list()) {
    return event();
}

/// Wait for events in the list
inline void wait_for_events(const wait_list&) {}

} // namespace jit
} // namespace backend
} // namespace vex

#endif
