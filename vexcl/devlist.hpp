#ifndef VEXCL_DEVLIST_HPP
#define VEXCL_DEVLIST_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
#  pragma warning(disable : 4290 4715)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <string>
#include <cstdlib>
#include <CL/cl.hpp>

/// OpenCL convenience utilities.
namespace vex {

/// Device filters.
namespace Filter {
    /// Selects any device.
    struct AllFilter {
	bool operator()(const cl::Device &d) const {
	    return true;
	}
    } All;

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
	bool operator()(const cl::Device &d) const {
	    if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
		return true;

	    std::string ext = d.getInfo<CL_DEVICE_EXTENSIONS>();

	    return (
		    ext.find("cl_khr_fp64") != std::string::npos ||
		    ext.find("cl_amd_fp64") != std::string::npos
		   );
	}
    } DoublePrecision;

    /// Selects no more than given number of devices.
    /**
     * \note This filter should be the last in filter expression. In this case,
     * it will be applied only to devices which passed all other filters.
     * Otherwise, you could get less devices than planned (every time this
     * filter is applied, internal counter is decremented).
     */
    struct Count {
	explicit Count(int c) : count(c) {}

	bool operator()(const cl::Device &d) const {
	    return --count >= 0;
	}

	private:
	    mutable int count;
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
    } Env;

    /// \internal Filter join operators.
    enum FilterOp {
	FilterAnd, FilterOr
    };

    /// \internal Filter join expression template.
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

    /// \internal Negation of a filter.
    template <class Flt>
	struct NegateFilter {
	    NegateFilter(const Flt &flt) : flt(flt) {}

	    bool operator()(const cl::Device &d) const {
		return !flt(d);
	    }

	    private:
	    const Flt &flt;
	};

    /// Negate a filter.
    template <class Flt>
	NegateFilter<Flt> operator!(const Flt &flt) {
	    return NegateFilter<Flt>(flt);
	}

} // namespace Filter

/// Select devices by given criteria.
/**
 * \param filter  Device filter functor. Functors may be combined with logical
 *		  operators.
 * \returns list of devices satisfying the provided filter. 
 *
 * This example selects any GPU which supports double precision arithmetic:
 * \code
 * auto devices = device_list(
 *	    Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision
 *	    );
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
 *		  operators.
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

/// VexCL context holder.
/**
 * Holds vectors of cl::Contexts and cl::CommandQueues returned by queue_list.
 */
class Context {
    public:
	template <class DevFilter>
	explicit Context(
		DevFilter filter, cl_command_queue_properties properties = 0
		)
	{
	    std::tie(c, q) = queue_list(filter, properties);
	}

	const std::vector<cl::Context>& context() const {
	    return c;
	}

	const std::vector<cl::CommandQueue>& queue() const {
	    return q;
	}

	size_t size() const {
	    return q.size();
	}
    private:
	std::vector<cl::Context>      c;
	std::vector<cl::CommandQueue> q;
};


std::ostream& operator<<(std::ostream &os, const std::vector<cl::Device> &device) {
    uint p = 1;

    for(auto d = device.begin(); d != device.end(); d++)
	os << p++ << ". " << d->getInfo<CL_DEVICE_NAME>() << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream &os, const std::vector<cl::CommandQueue> &queue) {
    uint p = 1;

    for(auto q = queue.begin(); q != queue.end(); q++)
	os << p++ << ". "
	   << q->getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>()
	   << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream &os, const vex::Context &ctx) {
    return os << ctx.queue();
}

} // namespace vex

#endif
