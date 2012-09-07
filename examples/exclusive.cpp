#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>
using namespace vex;

int main() {
    // Get exclusive access to single compute device.
    Context ctx(Filter::Exclusive(1));

    if (ctx.size()) {
	std::cout
	    << "Locked devices:" << std::endl
	    << ctx << std::endl
	    << "Press any key to exit: " << std::endl;
	std::cin.get();
    } else {
	std::cout << "No available devices found." << std::endl;
    }
}
