#ifndef VEXCL_BACKEND_CUDA_CONTEXT_HPP
#define VEXCL_BACKEND_CUDA_CONTEXT_HPP

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
 * \file   vexcl/backend/cuda/context.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  CUDA device enumeration and context initialization.
 */

#include <vector>
#include <iostream>
#include <memory>

#include <cuda.h>

namespace vex {
namespace backend {

inline CUresult do_init() {
    static CUresult rc = cuInit(0);
    return rc;
}

namespace detail {

// Knows how to dispose of various CUDA handles.
struct deleter {
    void operator()(CUcontext context) const {
        cuda_check( cuCtxSetCurrent(context) );
        cuda_check( cuCtxSynchronize()       );
        cuda_check( cuCtxDestroy(context)    );
    }

    void operator()(CUmodule module) const {
        cuda_check( cuModuleUnload(module) );
    }

    void operator()(CUstream stream) const {
        cuda_check( cuStreamDestroy(stream) );
    }

    void operator()(char *ptr) const {
        cuda_check( cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<size_t>(ptr))) );
    }
};

}

class device {
    public:
        device(CUdevice d) : d(d) {}

        CUdevice raw() const { return d; }

        std::string name() const {
            char name[256];
            cuda_check( cuDeviceGetName(name, 256, d) );
            return name;
        }

        std::tuple<int, int> compute_capability() const {
            int major, minor;

            cuda_check( cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d) );
            cuda_check( cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d) );

            return std::make_tuple(major, minor);
        }

        size_t multiprocessor_count() const {
            int n;
            cuda_check( cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d) );
            return n;
        }

        size_t max_threads_per_block() const {
            int n;
            cuda_check( cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, d) );
            return n;
        }

        size_t max_shared_memory_per_block() const {
            int n;
            cuda_check( cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, d) );
            return n;
        }

    private:
        CUdevice d;
};

class context {
    public:
        context() {}

        context(device dev, unsigned flags = 0)
            : c( create(dev, flags), detail::deleter() )
        {
            cuda_check( do_init() );
        }

        CUcontext raw() const {
            return c.get();
        }

        void set_current() const {
            cuda_check( cuCtxSetCurrent( c.get() ) );
        }

    private:
        std::shared_ptr<std::remove_pointer<CUcontext>::type> c;

        static CUcontext create(device dev, unsigned flags) {
            CUcontext h = 0;
            cuda_check( cuCtxCreate(&h, flags, dev.raw()) );
            return h;
        }
};

typedef unsigned command_queue_properties;

class command_queue {
    public:
        command_queue(const vex::backend::context &ctx, vex::backend::device dev, unsigned flags)
            : ctx(ctx), dev(dev), s( create(ctx, flags), detail::deleter() )
        { }

        void finish() const {
            cuda_check( cuStreamSynchronize( s.get() ) );
        }

        vex::backend::context context() const {
            return ctx;
        }

        vex::backend::device device() const {
            return dev;
        }

        unsigned flags() const {
            unsigned f;
            cuda_check( cuStreamGetFlags(s.get(), &f) );
            return f;
        }

        CUstream raw() const {
            return s.get();
        }
    private:
        vex::backend::context  ctx;
        vex::backend::device   dev;
        std::shared_ptr<std::remove_pointer<CUstream>::type> s;

        static CUstream create(const vex::backend::context &ctx, unsigned flags = 0) {
            ctx.set_current();

            CUstream s;
            cuda_check( cuStreamCreate(&s, flags) );

            return s;
        }
};

inline void select_context(const command_queue &q) {
    q.context().set_current();
}

typedef CUdevice  device_id;
typedef CUcontext kernel_cache_key;

inline CUdevice get_device_id(const command_queue &q) {
    return q.device().raw();
}

inline CUcontext cache_key(const command_queue &q) {
    return q.context().raw();
}

inline command_queue duplicate_queue(const command_queue &q) {
    return command_queue(q.context(), q.device(), q.flags());
}

/// Checks if the compute device is CPU.
inline bool is_cpu(const command_queue&) {
    return false;
}

/// Select devices by given criteria.
/**
 * \param filter  Device filter functor. Functors may be combined with logical
 *                operators.
 * \returns list of devices satisfying the provided filter.
 *
 * This example selects any GPU which supports double precision arithmetic:
 * \code
 * auto devices = device_list(
 *          Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision
 *          );
 * \endcode
 */
template<class DevFilter>
std::vector<device> device_list(DevFilter&& filter) {
    cuda_check( do_init() );

    std::vector<device> device;

    int ndev;
    cuda_check( cuDeviceGetCount(&ndev) );

    for(int d = 0; d < ndev; ++d) {
        try {
            CUdevice dev;
            cuda_check( cuDeviceGet(&dev, d) );
            if (!filter(dev)) continue;
            device.push_back(dev);
        } catch(const error&) { }
    }

    return device;
}

/// Create command queues on devices by given criteria.
/**
 * \param filter  Device filter functor. Functors may be combined with logical
 *                operators.
 * \param properties Command queue properties.
 *
 * \returns list of queues accociated with selected devices.
 * \see device_list
 */
template<class DevFilter>
std::pair< std::vector<context>, std::vector<command_queue> >
queue_list(DevFilter &&filter, unsigned queue_flags = 0)
{
    cuda_check( do_init() );

    std::vector<context>       ctx;
    std::vector<command_queue> queue;

    int ndev;
    cuda_check( cuDeviceGetCount(&ndev) );

    for(int d = 0; d < ndev; ++d) {
        try {
            CUdevice dev;
            cuda_check( cuDeviceGet(&dev, d) );
            if (!filter(dev)) continue;

            context       c(dev);
            command_queue q(c, dev, queue_flags);

            ctx.push_back(c);
            queue.push_back(q);
        } catch(const error&) { }
    }

    return std::make_pair(ctx, queue);
}

} // namespace backend
} // namespace vex

namespace std {

/// Output device name to stream.
inline std::ostream& operator<<(std::ostream &os, const vex::backend::device &d)
{
    return os << d.name();
}

/// Output device name to stream.
inline std::ostream& operator<<(std::ostream &os, const vex::backend::command_queue &q)
{
    return os << q.device();
}

} // namespace std

#endif
