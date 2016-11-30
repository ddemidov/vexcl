#ifndef VEXCL_BACKEND_JIT_FILTER_HPP
#define VEXCL_BACKEND_JIT_FILTER_HPP

#include <string>
#include <vector>
#include <functional>
#include <cstdlib>

namespace vex {
namespace Filter {

struct DummyFilter {
    bool v;
    DummyFilter(bool v) : v(v) {}
    bool operator()(const backend::device &d) const { return v; }
};

const DummyFilter GPU(false);
const DummyFilter CPU(true);
const DummyFilter Accelerator(false);

struct Name {
    explicit Name(std::string name) : devname(std::move(name)) {}

    bool operator()(const backend::device &d) const {
        return d.name().find(devname) != std::string::npos;
    }

    private:
        std::string devname;
};

inline std::vector< std::function<bool(const backend::device&)> >
backend_env_filters()
{
    std::vector< std::function<bool(const backend::device&)> > filter;

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4996)
#endif
    const char *name = getenv("OCL_DEVICE");
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

    if (name) filter.push_back(Name(name));

    return filter;
}

} // namespace Filter
} // namespace vex


#endif
