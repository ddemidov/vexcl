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
#include <boost/interprocess/sync/file_lock.hpp>
#include <CL/cl.hpp>

namespace vex {

/// Device filters.
namespace Filter {

    /// Allows exclusive access to given number of devices across several processes.
    /**
     * \note This filter should be the last in filter expression.
     */
    class Exclusive {
	private:
	    static std::map<cl_device_id, std::string> get_uids() {
		std::map<cl_device_id, std::string> uids;

		std::vector<cl::Platform> platform;
		cl::Platform::get(&platform);

		for(size_t p_id = 0; p_id < platform.size(); p_id++) {
		    std::vector<cl::Device> device;

		    platform[p_id].getDevices(CL_DEVICE_TYPE_ALL, &device);

		    for(size_t d_id = 0; d_id < device.size(); d_id++) {
			std::ostringstream id;
#ifdef WIN32
			id << getenv("TMPDIR") << "\\vexcl_device_";
#else
			id << "/var/lock/vexcl/device_";
#endif
			id << p_id << "_" << d_id << ".lock";

			uids[device[d_id]()] = id.str();
		    }
		}

		return uids;
	    }

	    struct locker {
		locker(std::string fname) : file(fname)
		{
		    if (!file.is_open() || file.fail()) {
			std::cout
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

	    mutable int count;
	public:
	    Exclusive(int count) : count(count) {}

	    bool operator()(const cl::Device &d) const {
		static std::map<cl_device_id, std::string> dev_uids = get_uids();
		static std::vector<std::unique_ptr<locker>> locks;

		if (count > 0) {
		    try {
			std::unique_ptr<locker> lck(new locker(dev_uids[d()]));
			if (lck->try_lock()) {
			    locks.push_back(std::move(lck));
			    count--;
			    return true;
			}
		    } catch (const std::exception &e) {
			std::cout << e.what() << std::endl;
		    }
		}

		return false;
	    }

    };
}

}


#endif
