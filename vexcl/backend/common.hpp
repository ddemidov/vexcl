#ifndef VEXCL_BACKEND_COMMON_HPP
#define VEXCL_BACKEND_COMMON_HPP

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
 * \file   vexcl/backend/common.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Common backend utilities.
 */

#include <vector>
#include <map>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <boost/uuid/sha1.hpp>
#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

namespace vex {

/// \cond INTERNAL
enum device_options_kind {
    compile_options,    ///< Options sent to the compute kernel compiler.
    program_header      ///< Program header prepended to all compute kernel source.
};

/// Global program options holder
template <device_options_kind kind>
struct device_options {
    static const std::string& get(const backend::command_queue &q) {
        auto dev = backend::get_device_id(q);
        if (options[dev].empty()) options[dev].push_back("");
        return options[dev].back();
    }

    static void push(const backend::command_queue &q, const std::string &str) {
        auto dev = backend::get_device_id(q);
        options[dev].push_back(str);
    }

    static void pop(const backend::command_queue &q) {
        auto dev = backend::get_device_id(q);
        if (!options[dev].empty()) options[dev].pop_back();
    }

    private:
        static std::map<backend::device_id, std::vector<std::string> > options;
};

template <device_options_kind kind>
std::map<backend::device_id, std::vector<std::string> > device_options<kind>::options;

inline std::string get_compile_options(const backend::command_queue &q) {
    return device_options<compile_options>::get(q);
}

inline std::string get_program_header(const backend::command_queue &q) {
    return device_options<program_header>::get(q);
}

/// \endcond

/// Set global compute kernel compilation options for a given device.
/**
 * This replaces any previously set options. To roll back, call
 * pop_compile_options().
 */
inline void push_compile_options(const backend::command_queue &q, const std::string &str) {
    device_options<compile_options>::push(q, str);
}

/// Rolls back changes to compile options.
inline void pop_compile_options(const backend::command_queue &q) {
    device_options<compile_options>::pop(q);
}

/// Set global compute kernel header for a given device.
/**
 * This replaces any previously set header. To roll back, call
 * pop_program_header().
 */
inline void push_program_header(const backend::command_queue &q, const std::string &str) {
    device_options<program_header>::push(q, str);
}

/// Rolls back changes to compile options.
inline void pop_program_header(const backend::command_queue &q) {
    device_options<program_header>::pop(q);
}

/// Set global compute kernel compilation options for each device in queue list.
inline void push_compile_options(const std::vector<backend::command_queue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::push(*q, str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_compile_options(const std::vector<backend::command_queue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<compile_options>::pop(*q);
}

/// Set global OpenCL program header for each device in queue list.
inline void push_program_header(const std::vector<backend::command_queue> &queue, const std::string &str) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::push(*q, str);
}

/// Rolls back changes to compile options for each device in queue list.
inline void pop_program_header(const std::vector<backend::command_queue> &queue) {
    for(auto q = queue.begin(); q != queue.end(); ++q)
        device_options<program_header>::pop(*q);
}

/// Path delimiter symbol.
inline const std::string& path_delim() {
    static const std::string delim = boost::filesystem::path("/").make_preferred().string();
    return delim;
}

/// Path to appdata folder.
inline const std::string& appdata_path() {
#ifdef WIN32
#  ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable: 4996)
#  endif
    static const std::string appdata = getenv("APPDATA") + path_delim() + "vexcl";
#  ifdef _MSC_VER
#    pragma warning(pop)
#  endif
#else
    static const std::string appdata = getenv("HOME") + path_delim() + ".vexcl";
#endif
    return appdata;
}

/// Path to cached binaries.
inline std::string program_binaries_path(const std::string &hash, bool create = false)
{
    std::string dir = appdata_path()    + path_delim()
                    + hash.substr(0, 2) + path_delim()
                    + hash.substr(2);
    if (create) boost::filesystem::create_directories(dir);
    return dir + path_delim();
}

/// Returns SHA1 hash of the string parameter.
inline std::string sha1(const std::string &src) {
    boost::uuids::detail::sha1 sha1;
    sha1.process_bytes(src.c_str(), src.size());

    unsigned int hash[5];
    sha1.get_digest(hash);

    std::ostringstream buf;
    for(int i = 0; i < 5; ++i)
        buf << std::hex << std::setfill('0') << std::setw(8) << hash[i];

    return buf.str();
}

} // namespace vex


#endif
