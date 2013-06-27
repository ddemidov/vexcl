/* Dummy file to test for multiple definition problems */
#include <vexcl/devlist.hpp>

bool empty_context() {
    return vex::current_context().empty();
}
