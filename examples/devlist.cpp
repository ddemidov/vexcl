#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <set>
#include <algorithm>
#include <vexcl/devlist.hpp>

int main() {
    using namespace std;

    auto dev = vex::backend::device_list(vex::Filter::Any);

#ifdef VEXCL_BACKEND_OPENCL
    cout << "OpenCL devices:" << endl << endl;
    for (auto d = dev.begin(); d != dev.end(); d++) {
        cout << "  " << d->getInfo<CL_DEVICE_NAME>() << endl
             << "    " << left << setw(32) << "CL_PLATFORM_NAME" << " = "
             << cl::Platform(d->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>()
             << endl;

#define SHOW_DEVPROP(name) \
        cout << "    " << left << setw(32) << #name << " = " \
                  << d->getInfo< name >() << endl

        SHOW_DEVPROP(CL_DEVICE_VENDOR);
        SHOW_DEVPROP(CL_DEVICE_VERSION);
        SHOW_DEVPROP(CL_DEVICE_MAX_COMPUTE_UNITS);
        SHOW_DEVPROP(CL_DEVICE_HOST_UNIFIED_MEMORY);
        SHOW_DEVPROP(CL_DEVICE_GLOBAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_LOCAL_MEM_SIZE);
        SHOW_DEVPROP(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        SHOW_DEVPROP(CL_DEVICE_ADDRESS_BITS);
        SHOW_DEVPROP(CL_DEVICE_MAX_CLOCK_FREQUENCY);

#undef SHOW_DEVPROP

        cout << "    " << left << setw(32) << "CL_DEVICE_EXTENSIONS" << " = ";
        {
            istringstream iss(d->getInfo<CL_DEVICE_EXTENSIONS>());
            set<string> extensions;

            extensions.insert(
                    istream_iterator<string>(iss),
                    istream_iterator<string>()
                    );

            size_t w = 40;
            for(auto s = extensions.begin(); s != extensions.end(); ++s) {
                w += s->length() + 1;
                if (w > 80) {
                    cout << endl << setw(w = 8) << "";
                    w += s->length() + 1;
                }
                cout << *s << " ";
            }
        }
        cout << endl << endl;
    }
#elif VEXCL_BACKEND_CUDA
    cout << "CUDA devices:" << endl << endl;
    unsigned pos = 0;
    for(auto d = dev.begin(); d != dev.end(); d++)
        cout << ++pos << ". " << *d << endl;
#else
#error Unsupported backend
#endif
}

// vim: et
