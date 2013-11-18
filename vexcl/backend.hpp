#ifndef VEXCL_BACKEND_HPP
#define VEXCL_BACKEND_HPP

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
 * \file   vexcl/backend.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Compile-time selection of backend (OpenCL/CUDA).
 */

namespace vex {
    /// Backend-specific functionality.
    /**
     * \note Definitions from either vex::backend::opencl or vex::backend::cuda
     * are directly brought into vex::backend namespace. Define either
     * VEXCL_BACKEND_OPENCL or VEXCL_BACKEND_CUDA macro in order to select
     * backend. You will also need to link to libOpenCL or libcuda accordingly.
     */
    namespace backend {
        namespace cuda {}
        using namespace cuda;
    }
}

#if defined(VEXCL_BACKEND_CUDA)

namespace vex {
    namespace backend {
        namespace cuda {}
        using namespace cuda;
    }
}

#include <vexcl/backend/cuda.hpp>

#else // defined(VEXCL_BACKEND_OPENCL)

namespace vex {
    namespace backend {
        namespace opencl {}
        using namespace opencl;
    }
}

#include <vexcl/backend/opencl.hpp>

#endif

#if defined(VEXCL_BACKEND_OPENCL) && defined(VEXCL_BACKEND_CUDA)
#  error Both OpenCL and CUDA backends are selected. Make your mind!
#endif

namespace vex {
    using backend::device;
    using backend::command_queue;
    using backend::command_queue_properties;
    using backend::error;
    using backend::device_vector;
    using backend::is_cpu;
} // namespace vex

#endif
