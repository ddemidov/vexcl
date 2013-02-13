#ifndef VEXCL_DEVLIST_HPP
#define VEXCL_DEVLIST_HPP

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
 * \file   devlist.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device enumeration.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4290 4715 4996)
#  define NOMINMAX
#endif

#include <vector>
#include <string>
#include <fstream>
#include <tuple>
#include <cstdlib>
#include <vexcl/util.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#ifdef __GNUC__
#  ifndef _GLIBCXX_USE_NANOSLEEP
#    define _GLIBCXX_USE_NANOSLEEP
#  endif
#endif

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include <random>

namespace vex {

/// Device filters.
namespace Filter {
    /// Selects any device.
    struct AllFilter {
        AllFilter() {}

        bool operator()(const cl::Device &) const {
            return true;
        }
    };

    const AllFilter All;

    /// Selects devices whose vendor name match given value.
    struct Vendor {
        explicit Vendor(const std::string &name) : vendor(name) {}

        bool operator()(const cl::Device &d) const {
            return d.getInfo<CL_DEVICE_VENDOR>().find(vendor) != std::string::npos;
        }

        private:
            const std::string &vendor;
    };

    /// Selects devices whose platform name match given value.
    struct Platform {
        explicit Platform(const std::string &name) : platform(name) {}

        bool operator()(const cl::Device &d) const {
            return cl::Platform(d.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>().find(platform) != std::string::npos;
        }

        private:
            const std::string &platform;
    };

    /// Selects devices whose names match given value.
    struct Name {
        explicit Name(const std::string &name) : devname(name) {}

        bool operator()(const cl::Device &d) const {
            return d.getInfo<CL_DEVICE_NAME>().find(devname) != std::string::npos;
        }

        private:
            const std::string &devname;
    };

    /// Selects devices by type.
    struct Type {
        explicit Type(cl_device_type t) : type(t) {}

        bool operator()(const cl::Device &d) const {
            return d.getInfo<CL_DEVICE_TYPE>() == type;
        }

        private:
            cl_device_type type;
    };

    /// Selects devices supporting double precision.
    struct DoublePrecisionFilter {
        DoublePrecisionFilter() {}

        bool operator()(const cl::Device &d) const {
            if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
                return true;

            std::string ext = d.getInfo<CL_DEVICE_EXTENSIONS>();

            return (
                    ext.find("cl_khr_fp64") != std::string::npos ||
                    ext.find("cl_amd_fp64") != std::string::npos
                   );
        }
    };

    const DoublePrecisionFilter DoublePrecision;

    /// Selects no more than given number of devices.
    /**
     * \note This filter should be the last in filter expression. In this case,
     * it will be applied only to devices which passed all other filters.
     * Otherwise, you could get less devices than planned (every time this
     * filter is applied, internal counter is decremented).
     */
    struct Count {
        explicit Count(int c) : count(c) {}

        bool operator()(const cl::Device &) const {
            return --count >= 0;
        }

        private:
            mutable int count;
    };

    /// Selects one device at the given position.
    /**
     * Select one device at the given position in the list of devices
     * satisfying previously applied filters.
     */
    struct Position {
        explicit Position(int p) : pos(p) {}

        bool operator()(const cl::Device &) const {
            return 0 == pos--;
        }

        private:
            mutable int pos;
    };

    /// Environment filter
    /**
     * Selects devices with respect to environment variables. Recognized
     * variables are:
     *
     * \li OCL_PLATFORM -- platform name;
     * \li OCL_VENDOR   -- device vendor;
     * \li OCL_DEVICE   -- device name;
     * \li OCL_MAX_DEVICES -- maximum number of devices to use.
     *
     * \note Since this filter possibly counts passed devices, it should be the
     * last in filter expression. Same reasoning applies as in case of
     * Filter::Count.
     */
    struct EnvFilter {
        EnvFilter()
            : platform(getenv("OCL_PLATFORM")),
              vendor  (getenv("OCL_VENDOR")),
              name    (getenv("OCL_DEVICE")),
              maxdev  (getenv("OCL_MAX_DEVICES")),
              count(maxdev ? atoi(maxdev) : std::numeric_limits<int>::max())
        {}

        bool operator()(const cl::Device &d) const {
            if (platform &&
                    cl::Platform(
                        d.getInfo<CL_DEVICE_PLATFORM>()
                        ).getInfo<CL_PLATFORM_NAME>().find(platform) == std::string::npos
               ) return false;

            if (vendor &&
                    d.getInfo<CL_DEVICE_VENDOR>().find(vendor) == std::string::npos
               ) return false;

            if (name &&
                    d.getInfo<CL_DEVICE_NAME>().find(name) == std::string::npos
               ) return false;

            if (maxdev) return --count >= 0;

            return true;
        }

        private:
            const char *platform;
            const char *vendor;
            const char *name;
            const char *maxdev;
            mutable int count;
    };

    const EnvFilter Env;

    /// \internal Exclusive access to selected devices.
    template <class Filter>
    class ExclusiveFilter {
        private:
            const Filter &filter;

            static std::map<cl_device_id, std::string> get_uids() {
                std::map<cl_device_id, std::string> uids;

                std::vector<cl::Platform> platform;
                cl::Platform::get(&platform);

                const char *lock_dir = getenv("VEXCL_LOCK_DIR");

                for(size_t p_id = 0; p_id < platform.size(); p_id++) {
                    std::vector<cl::Device> device;

                    platform[p_id].getDevices(CL_DEVICE_TYPE_ALL, &device);

                    for(size_t d_id = 0; d_id < device.size(); d_id++) {
                        std::ostringstream id;
#ifdef WIN32
                        id << (lock_dir ? lock_dir : getenv("TEMP")) << "\\";
#else
                        id << (lock_dir ? lock_dir : "/tmp") << "/";
#endif
                        id << "vexcl_device_" << p_id << "_" << d_id << ".lock";

                        uids[device[d_id]()] = id.str();
                    }
                }

                return uids;
            }

            struct locker {
                locker(std::string fname) : file(fname)
                {
                    if (!file.is_open() || file.fail()) {
                        std::cerr
                            << "WARNING: failed to open file \"" << fname << "\"\n"
                            << "  Check that target directory is exists and is writable.\n"
                            << "  Exclusive mode is off.\n"
                            << std::endl;
                    } else {
                        flock.reset(new boost::interprocess::file_lock(fname.c_str()));
                    }
                }

                bool try_lock() {
                    if (flock) {
                        // Try and lock the file related to compute device.
                        // If the file is locked already, it could mean two
                        // things:
                        // 1. Somebody locked the file, and uses the device.
                        // 2. Somebody locked the file, and is in process of
                        //    checking the device. If device is not good (for
                        //    them) they will release the lock in a few
                        //    moments.
                        // To process case 2 correctly, we try to lock the
                        // device a couple of times with a random pause.

                        std::mt19937 rng(reinterpret_cast<size_t>(this));
                        std::uniform_int_distribution<uint> rnd(0, 30);

                        for(int try_num = 0; try_num < 3; ++try_num) {
                            if (flock->try_lock())
                                return true;

                            boost::this_thread::sleep_for(
                                    boost::chrono::milliseconds( rnd(rng) ) );
                        }
                        return false;
                    }
                    else
                        return true;
                }

                std::ofstream file;
                std::unique_ptr<boost::interprocess::file_lock> flock;
            };
        public:
            ExclusiveFilter(const Filter &filter) : filter(filter) {}

            bool operator()(const cl::Device &d) const {
                static std::map<cl_device_id, std::string> dev_uids = get_uids();
                static std::vector<std::unique_ptr<locker>> locks;

                std::unique_ptr<locker> lck(new locker(dev_uids[d()]));

                if (lck->try_lock() && filter(d)) {
                    locks.push_back(std::move(lck));
                    return true;
                }

                return false;
            }

    };

    /// Allows exclusive access to compute devices across several processes.
    /**
     * Returns devices that pass through provided device filter and are not
     * locked.
     *
     * \param filter Compute device filter
     *
     * \note Depends on boost::interprocess library.
     *
     * lock files are created in directory specified in VEXCL_LOCK_DIR
     * environment variable. If the variable does not exist, /tmp is
     * used on Linux and %TMPDIR% on Windows. The lock directory should exist
     * and be writable by the running user.
     */
    template <class Filter>
    ExclusiveFilter<Filter> Exclusive(const Filter &filter) {
        return ExclusiveFilter<Filter>(filter);
    }

    /// \cond INTERNAL

    /// Negation of a filter.
    template <class Flt>
        struct NegateFilter {
            NegateFilter(const Flt &flt) : flt(flt) {}

            bool operator()(const cl::Device &d) const {
                return !flt(d);
            }

            private:
            const Flt &flt;
        };

    /// Filter join operators.
    enum FilterOp {
        FilterAnd, FilterOr
    };

    /// Filter join expression template.
    template <class LeftFilter, class RightFilter, FilterOp op>
        struct FilterBinaryOp {
            FilterBinaryOp(const LeftFilter &l, const RightFilter &r)
                : left(l), right(r) {}

            bool operator()(const cl::Device &d) const {
                switch (op) {
                    case FilterAnd:
                        return left(d) && right(d);
                    case FilterOr:
                        return left(d) || right(d);
                }
            }

            private:
            const LeftFilter &left;
            const RightFilter &right;
        };

    /// \endcond

    /// Join two filters with AND operator.
    template <class LeftFilter, class RightFilter>
        FilterBinaryOp<LeftFilter, RightFilter, FilterAnd> operator&&(
                const LeftFilter &left, const RightFilter &right)
        {
            return FilterBinaryOp<LeftFilter, RightFilter, FilterAnd>(left, right);
        }

    /// Join two filters with OR operator.
    template <class LeftFilter, class RightFilter>
        FilterBinaryOp<LeftFilter, RightFilter, FilterOr> operator||(
                const LeftFilter &left, const RightFilter &right)
        {
            return FilterBinaryOp<LeftFilter, RightFilter, FilterOr>(left, right);
        }

    /// Negate a filter.
    template <class Flt>
        NegateFilter<Flt> operator!(const Flt &flt) {
            return NegateFilter<Flt>(flt);
        }

} // namespace Filter

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
template<class DevFilter
#ifndef WIN32
    = Filter::AllFilter
#endif
    >
std::vector<cl::Device> device_list(DevFilter filter = Filter::All)
{
    std::vector<cl::Device> device;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for(auto p = platforms.begin(); p != platforms.end(); p++) {
        std::vector<cl::Device> dev_list;

        p->getDevices(CL_DEVICE_TYPE_ALL, &dev_list);

        for(auto d = dev_list.begin(); d != dev_list.end(); d++) {
            if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
            if (!filter(*d)) continue;

            device.push_back(*d);
        }
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
template<class DevFilter
#ifndef WIN32
    = Filter::AllFilter
#endif
    >
std::pair<std::vector<cl::Context>, std::vector<cl::CommandQueue>>
queue_list(
        DevFilter filter = Filter::All,
        cl_command_queue_properties properties = 0
        )
{
    std::vector<cl::Context>      context;
    std::vector<cl::CommandQueue> queue;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for(auto p = platforms.begin(); p != platforms.end(); p++) {
        std::vector<cl::Device> device;
        std::vector<cl::Device> dev_list;

        p->getDevices(CL_DEVICE_TYPE_ALL, &dev_list);

        for(auto d = dev_list.begin(); d != dev_list.end(); d++) {
            if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
            if (!filter(*d)) continue;

            device.push_back(*d);
        }

        if (device.empty()) continue;

        for(auto d = device.begin(); d != device.end(); d++)
            try {
                context.push_back(cl::Context(std::vector<cl::Device>(1, *d)));
                queue.push_back(cl::CommandQueue(context.back(), *d, properties));
            } catch(const cl::Error&) {
                // Something bad happened. Better skip this device.
            }
    }

    return std::make_pair(context, queue);
}

class Context;

template <bool dummy = true>
class StaticContext {
    static_assert(dummy, "dummy parameter should be true");

    public:
        static void set(Context &ctx) {
            instance = &ctx;
        }

        static const Context& get() {
            if (!instance) throw std::logic_error("Uninitialized static context");
            return *instance;
        }
    private:
        static Context *instance;
};

template <bool dummy>
Context* StaticContext<dummy>::instance = 0;

inline const Context& current_context() {
    return StaticContext<>::get();
}

/// VexCL context holder.
/**
 * Holds vectors of cl::Contexts and cl::CommandQueues returned by queue_list.
 */
class Context {
    public:
        /// Initialize context from a device filter.
        template <class DevFilter>
        explicit Context(
                DevFilter filter, cl_command_queue_properties properties = 0
                )
        {
            std::tie(c, q) = queue_list(filter, properties);

#ifdef VEXCL_THROW_ON_EMPTY_CONTEXT
            if (q.empty()) throw std::logic_error("No compute devices found");
#endif

            StaticContext<>::set(*this);
        }

        /// Initializes context from user-supplied list of cl::Contexts and cl::CommandQueues.
        Context(const std::vector<std::pair<cl::Context, cl::CommandQueue>> &user_ctx) {
            c.reserve(user_ctx.size());
            q.reserve(user_ctx.size());
            for(auto u = user_ctx.begin(); u != user_ctx.end(); u++) {
                c.push_back(u->first);
                q.push_back(u->second);
            }

            StaticContext<>::set(*this);
        }

        const std::vector<cl::Context>& context() const {
            return c;
        }

        const cl::Context& context(uint d) const {
            return c[d];
        }

        const std::vector<cl::CommandQueue>& queue() const {
            return q;
        }

        operator const std::vector<cl::CommandQueue>&() const {
            return q;
        }

        const cl::CommandQueue& queue(uint d) const {
            return q[d];
        }

        cl::Device device(uint d) const {
            return qdev(q[d]);
        }

        size_t size() const {
            return q.size();
        }

        bool empty() const {
            return q.empty();
        }

        operator bool() const {
            return !empty();
        }
    private:
        std::vector<cl::Context>      c;
        std::vector<cl::CommandQueue> q;
};

} // namespace vex

/// Output device name to stream.
inline std::ostream& operator<<(std::ostream &os, const cl::Device &device) {
    return os << device.getInfo<CL_DEVICE_NAME>();
}

/// Output list of devices to stream.
inline std::ostream& operator<<(std::ostream &os, const std::vector<cl::Device> &device) {
    uint p = 1;

    for(auto d = device.begin(); d != device.end(); d++)
        os << p++ << ". " << *d << std::endl;

    return os;
}

/// Output list of devices to stream.
inline std::ostream& operator<<(std::ostream &os, const std::vector<cl::CommandQueue> &queue) {
    uint p = 1;

    for(auto q = queue.begin(); q != queue.end(); q++)
        os << p++ << ". " << vex::qdev(*q).getInfo<CL_DEVICE_NAME>() << std::endl;

    return os;
}

/// Output list of devices to stream.
inline std::ostream& operator<<(std::ostream &os, const vex::Context &ctx) {
    return os << ctx.queue();
}

#ifdef WIN32
#  pragma warning(pop)
#endif

// vim: et
#endif
