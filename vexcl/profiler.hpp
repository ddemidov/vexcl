#ifndef VEXCL_PROFILER_HPP
#define VEXCL_PROFILER_HPP

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
 * \file   profiler.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Simple OpenCL/Host profiler.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <memory>
#include <stack>
#include <vector>
#include <cassert>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/global_fun.hpp>

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>

#ifdef WIN32
#  include <sys/timeb.h>
#else
#  include <chrono>
#endif

namespace vex {

/// Class for gathering and printing OpenCL and Host profiling info.
class profiler {
    private:
        class profile_unit {
            public:
                profile_unit(std::string name) : length(0), hit(0), name(name) {}
                virtual ~profile_unit() {}
                virtual void tic() = 0;
                virtual double toc() = 0;

                double length;
                size_t hit;
                std::string name;

                static std::string _name(const std::shared_ptr<profile_unit> &u) {
                    return u->name;
                }
                boost::multi_index_container<
                    std::shared_ptr<profile_unit>,
                    boost::multi_index::indexed_by<
                        boost::multi_index::sequenced<>,
                        boost::multi_index::ordered_unique<boost::multi_index::global_fun<
                                const std::shared_ptr<profile_unit> &, std::string, _name>>
                    >
                > children;

                double children_time() const {
                    double tm = 0;

                    for(auto c = children.begin(); c != children.end(); c++)
                        tm += (*c)->length;

                    return tm;
                }

                uint max_line_width(uint level) const {
                    uint w = name.size() + level;

                    for(auto c = children.begin(); c != children.end(); c++)
                        w = std::max(w, (*c)->max_line_width(level + shift_width));

                    return w;
                }

                void print(std::ostream &out,
                        uint level, double total, uint width) const
                {
                    using namespace std;
                    out << "[" << setw(level) << "";
                    print_line(out, name, length, 100 * length / total, width - level);

                    if (!children.empty()) {
                        double sec = length - children_time();
                        double perc = 100 * sec / total;

                        if (perc > 1e-1) {
                            out << "[" << setw(level + 1) << "";
                            print_line(out, "self", sec, perc, width - level - 1);
                        }
                    }

                    for(auto c = children.begin(); c != children.end(); c++)
                        (*c)->print(out, level + shift_width, total, width);
                }

                void print_line(std::ostream &out, const std::string &name,
                        double time, double perc, uint width) const
                {
                    using namespace std;
                    out << name << ":";
                    out << setw(width - name.size()) << "";
                    out << setiosflags(ios::fixed);
                    out << setw(10) << setprecision(3) << time << " sec.";
                    out << "] (" << setprecision(2) << setw(6) << perc << "%)";
                    if(hit > 1) out << " (" << hit << "x)";
                    out << endl;
                }
            private:
                static const uint shift_width = 2U;
        };

        class cpu_profile_unit : public profile_unit {
            public:
                cpu_profile_unit(const std::string &name) : profile_unit(name) {}

                void tic() {
#ifdef WIN32
                    ftime(&start);
#else
                    start = std::chrono::high_resolution_clock::now();
#endif
                }

                double toc() {
#ifdef WIN32
                    timeb now;
                    ftime(&now);
                    double delta = now.time - start.time + 1e-3 * (now.millitm - start.millitm);
#else
                    double delta = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - start).count();
#endif

                    length += delta;
                    hit++;

                    return delta;
                }

            private:
#ifdef WIN32
                timeb start;
#else
                std::chrono::time_point<std::chrono::high_resolution_clock> start;
#endif
        };

        class cl_profile_unit : public profile_unit {
            public:
                cl_profile_unit(const std::string &name, const std::vector<cl::CommandQueue> &queue)
                    : profile_unit(name), queue(queue), start(queue.size()), stop(queue.size()),
                      dbuf(queue.size()), hbuf(queue.size())
                {
                    for(uint d = 0; d < queue.size(); d++) {
                        dbuf[d] = cl::Buffer(qctx(queue[d]), CL_MEM_READ_WRITE, 1);
                    }
                }

                void tic() {
                    for(uint d = 0; d < queue.size(); d++)
                        queue[d].enqueueReadBuffer(dbuf[d], CL_FALSE, 0, 1, &hbuf[d], 0, &start[d]);
                }

                double toc() {
                    for(uint d = 0; d < queue.size(); d++)
                        queue[d].enqueueReadBuffer(dbuf[d], CL_FALSE, 0, 1, &hbuf[d], 0, &stop[d]);

                    // Measured time ends before marker is in the queue.
                    cl_long max_delta = 0;
                    for(uint d = 0; d < queue.size(); d++) {
                        stop[d].wait();
                        max_delta = std::max<cl_long>(max_delta,
                                stop [d].getProfilingInfo<CL_PROFILING_COMMAND_START>() -
                                start[d].getProfilingInfo<CL_PROFILING_COMMAND_END>()
                                );
                    }

                    double delta = max_delta * 1.0e-9;

                    length += delta;
                    hit++;

                    return delta;
                }
            private:
                const std::vector<cl::CommandQueue> &queue;
                std::vector<cl::Event> start;
                std::vector<cl::Event> stop;
                std::vector<cl::Buffer> dbuf;
                std::vector<char> hbuf;
        };

    public:
        /// Constructor.
        /**
         * \param queue vector of command queues. Each queue should have been
         *              initialized with CL_QUEUE_PROFILING_ENABLE property.
         * \param name  Optional name to be used when profiling info is printed.
         */
        profiler(
                const std::vector<cl::CommandQueue> &queue,
                const std::string &name = "Profile"
                ) : queue(queue)
        {
            auto root = std::shared_ptr<profile_unit>(new cpu_profile_unit(name));
            root->tic();
            stack.push_back(root);
        }

    private:
        void tic(profile_unit *u) {
            assert(!stack.empty());
            auto top = stack.back();
            auto new_unit = std::shared_ptr<profile_unit>(u);
            auto unit = *top->children.push_back(new_unit).first;
            unit->tic();
            stack.push_back(unit);
        }

    public:
        /// Starts a CPU timer.
        /**
         * Also pushes named interval to the top of the profiler hierarchy.
         * \param name name of the measured interval.
         */
        void tic_cpu(const std::string &name) {
            tic(new cpu_profile_unit(name));
        }

        /// Enqueues a marker into each of the provided queues.
        /**
         * Also pushes named interval to the top of the profiler hierarchy.
         * \param name name of the measured interval.
         */
        void tic_cl(const std::string &name) {
            tic(new cl_profile_unit(name, queue));
        }

        /// Returns time since last tic.
        /**
         * Also removes interval from the top of the profiler hierarchy.
         */
        double toc(const std::string &) {
            assert(stack.size() > 1);

            double delta = stack.back()->toc();
            stack.pop_back();

            return delta;
        }

        /// Outputs profile to the provided stream.
        void print(std::ostream &out) {
            if(stack.size() != 1)
                out << "Warning! Profile is incomplete." << std::endl;

            auto root = stack.front();
            double length = root->toc();
            out << std::endl;
            root->print(out, 0, length, root->max_line_width(0));
        }

    private:
        const std::vector<cl::CommandQueue> &queue;
        std::deque<std::shared_ptr<profile_unit>> stack;
};

} // namespace vex

inline std::ostream& operator<<(std::ostream &os, vex::profiler &prof) {
    prof.print(os);
    return os;
}

#ifdef WIN32
#  pragma warning(pop)
#endif

// vim: et
#endif
