#ifndef OCLUTIL_DEVLIST_HPP
#define OCLUTIL_DEVLIST_HPP

/**
 * \file   devlist.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device enumeration.
 */

#ifdef WIN32
#  pragma warning(disable : 4290 4715)
#  define NOMINMAX
#endif

#include <vector>
#include <string>
#include <CL/cl.hpp>

/// OpenCL convenience utilities.
namespace clu {

/// Device filters.
namespace Filter {
    /// Selects any device.
    struct All {
	bool operator()(const cl::Device &d) const {
	    return true;
	}
    };

    /// Selects devices whose platform vendor name match given value.
    struct Vendor {
	Vendor(const std::string &name) : vendor(name) {}

	bool operator()(const cl::Device &d) const {
	    return d.getInfo<CL_DEVICE_VENDOR>().find(vendor) != std::string::npos;
	}

	private:
	    const std::string &vendor;
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
 * \param verbose If set, outputs list of selected devices to stdout.
 * \returns list of devices satisfying the provided filter. 
 *
 * This example selects any GPU which supports double precision arithmetic:
 * \code
 * auto devices = device_list(
 *	    Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision()
 *	    );
 * \endcode
 */
template<class DevFilter>
std::vector<cl::Device> device_list(DevFilter filter = Filter::All(),
	bool verbose = false
	)
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

    if (verbose) {
	uint p = 1;
	for(auto d = device.begin(); d != device.end(); d++)
	    std::cout << p++ << ". "
		      << d->getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    return device;
}

/// Create command queues on devices by given criteria.
/**
 * \param filter  Device filter functor. Functors may be combined with logical
 *		  operators.
 * \param verbose If set, output list of selected devices to stdout.
 * \returns list of queues accociated with selected devices.
 * \see device_list
 */
template<class DevFilter>
std::pair<std::vector<cl::Context>, std::vector<cl::CommandQueue>>
queue_list(DevFilter filter = Filter::All(), bool verbose = false)
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
	    queue.push_back(cl::CommandQueue(context.back(), *d));
    }

    if (verbose) {
	uint p = 1;
	for(auto q = queue.begin(); q != queue.end(); q++)
	    std::cout << p++ << ". "
		      << q->getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>()
		      << std::endl;
    }

    return std::make_pair(context, queue);
}

} // namespace clu

#endif
