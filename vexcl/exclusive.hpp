#ifndef VEXCL_EXCLUSIVE_HPP
#define VEXCLEXCLUSIVE_HPP

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
 * \file   exclusive.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Device filter allowing exclusive access
 *
 * \note This file should be included manually since it depends on
 * boost::interprocess library. 
 */

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <boost/interprocess/sync/file_lock.hpp>
#include <CL/cl.hpp>

namespace vex {

/// Device filters.
namespace Filter {

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
			id << (lock_dir ? lock_dir : getenv("TMPDIR")) << "\\";
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
		    if (flock)
			return flock->try_lock();
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

		if (!filter(d)) return false;

		std::unique_ptr<locker> lck(new locker(dev_uids[d()]));
		if (lck->try_lock()) {
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
}

}


#endif
