#ifndef VEXCL_BACKEND_CUDA_FILTER_HPP
#define VEXCL_BACKEND_CUDA_FILTER_HPP

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
 * \file   vexcl/backend/cuda/filter.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Device filters for CUDA backend.
 */

#include <string>
#include <vector>
#include <functional>
#include <cstdlib>

#include <cuda.h>


namespace vex {

/// Device filters.
namespace Filter {

    /// Selects devices whose names match given value.
    struct Name {
        explicit Name(std::string name) : devname(std::move(name)) {}

        bool operator()(const backend::device &d) const {
            return d.name().find(devname) != std::string::npos;
        }

        private:
            std::string devname;
    };

    /// Compute capability filter.
    struct CC {
        CC(int major, int minor) : cc(std::make_tuple(major, minor)) {}

        bool operator()(const backend::device &d) const {
            return d.compute_capability() >= cc;
        }

        private:
            std::tuple<int, int> cc;
    };

    /// Selects devices supporting double precision.
    const CC DoublePrecision(1,3);

    /// List of device filters based on environment variables.
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
