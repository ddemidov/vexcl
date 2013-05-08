#include <iostream>
#include <sstream>
#include <iterator>
#include <vexcl/devlist.hpp>
using namespace vex;

int main() {
    using namespace std;

    cout << "OpenCL devices:" << endl << endl;
    auto dev = device_list(Filter::All);
    for (auto d = dev.begin(); d != dev.end(); d++) {
        cout << "  " << d->getInfo<CL_DEVICE_NAME>() << endl
             << "    " << left << setw(32) << "CL_PLATFORM_NAME" << " = "
             << cl::Platform(d->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>()
             << endl;

#define SHOW_DEVPROP(name) \
        cout << "    " << left << setw(32) << #name << " = " \
                  << d->getInfo< name >() << endl

        SHOW_DEVPROP(CL_DEVICE_VENDOR);
        SHOW_DEVPROP(CL_DEVICE_MAX_COMPUTE_UNITS);
        SHOW_DEVPROP(CL_DEVICE_HOST_UNIFIED_MEMORY);
        SHOW_DEVPROP(CL_DEVICE_GLOBAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_LOCAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        SHOW_DEVPROP(CL_DEVICE_MAX_CLOCK_FREQUENCY);

#undef SHOW_DEVPROP

        cout << "    " << left << setw(32) << "CL_DEVICE_EXTENSIONS" << " = ";
        {
            istringstream iss(d->getInfo<CL_DEVICE_EXTENSIONS>());
            size_t w = 40;
            for(auto s = istream_iterator<string>(iss); s != istream_iterator<string>(); ++s) {
                w += s->length() + 1;
                if (w > 80) {
                    cout << endl << setw(w = 8) << "";
                    w += s->length() + 1;
                }
                cout << *s << " ";
            }
        }
        cout << endl;
    }
}

// vim: et
