#ifndef VEXCL_DEVLIST_HPP
#define VEXCL_DEVLIST_HPP

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
#include <CL/cl.hpp>

/// OpenCL convenience utilities.
namespace vex {

/// Device filters.
namespace Filter {
    /// Selects any device.
    struct All {
	bool operator()(const cl::Device &d) const {
	    return true;
	}
    };

    /// Selects devices whose vendor name match given value.
    struct Vendor {
	Vendor(const std::string &name) : vendor(name) {}

	bool operator()(const cl::Device &d) const {
	    return d.getInfo<CL_DEVICE_VENDOR>().find(vendor) != std::string::npos;
	}

	private:
	    const std::string &vendor;
    };

    /// Selects devices whose platform name match given value.
    struct Platform {
	Platform(const std::string &name) : platform(name) {}

	bool operator()(const cl::Device &d) const {
	    return cl::Platform(d.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>().find(platform) != std::string::npos;
	}

	private:
	    const std::string &platform;
    };

    /// Selects devices whose names match given value.
    struct Name {
	Name(const std::string &name) : devname(name) {}

	bool operator()(const cl::Device &d) const {
	    return d.getInfo<CL_DEVICE_NAME>().find(devname) != std::string::npos;
	}

	private:
	    const std::string &devname;
    };

    /// Selects devices by type.
    struct Type {
	Type(cl_device_type t) : type(t) {}

	bool operator()(const cl::Device &d) const {
	    return d.getInfo<CL_DEVICE_TYPE>() == type;
	}

	private:
	    cl_device_type type;
    };

    /// Selects devices supporting double precision.
    struct DoublePrecision {
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

    /// Selects no more than given number of devices.
    struct Count {
	Count(int c) : count(c) {}

	bool operator()(const cl::Device &d) const {
	    return --count >= 0;
	}

	private:
	    mutable int count;
    };

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
 *	    Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision()
 *	    );
 * \endcode
 */
template<class DevFilter = Filter::All>
std::vector<cl::Device> device_list(DevFilter filter = Filter::All())
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
template<class DevFilter = Filter::All>
std::pair<std::vector<cl::Context>, std::vector<cl::CommandQueue>>
queue_list(
	DevFilter filter = Filter::All(),
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

	context.push_back(cl::Context(device));
	for(auto d = device.begin(); d != device.end(); d++)
	    queue.push_back(cl::CommandQueue(context.back(), *d, properties));
    }

    return std::make_pair(context, queue);
}

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

} // namespace vex

#ifdef VEXCL_SMART_PARTITION
#include <vexcl/vector.hpp>
#include <vexcl/profiler.hpp>

namespace vex {

/// Returns relative device weights after simple bandwidth test
template <typename real>
std::vector<double> device_weights(
	const std::vector<cl::CommandQueue> &queue, uint test_size = 1048576)
{
    double max_time = 0;
    std::vector<double> weights(queue.size());

    for(uint d = 0; d < queue.size(); d++) {
	std::vector<cl::CommandQueue> local_queue(1, queue[d]);

	// Allocate test vectors on current device and measure execution
	// time of a simple kernel.
	vex::vector<real> a(local_queue, CL_MEM_READ_WRITE, test_size);
	vex::vector<real> b(local_queue, CL_MEM_READ_WRITE, test_size);
	vex::vector<real> c(local_queue, CL_MEM_READ_WRITE, test_size);

	b = Const(1);
	c = Const(2);

	// Skip the first run.
	a = b + c;

	// Measure the second run.
	profiler prof(local_queue);
	prof.tic_cl("test");
	a = b + c;
	max_time = std::max(max_time, (weights[d] = prof.toc("test")));
    }

    for(auto w = weights.begin(); w != weights.end(); w++)
	*w = max_time / *w;

    return weights;
}

} // namespace vex
#endif

#endif
