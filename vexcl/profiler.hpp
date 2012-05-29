#ifndef VEXCL_PROFILER_HPP
#define VEXCL_PROFILER_HPP

/**
 * \file   profiler.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Simple OpenCL/Host profiler.
 */

#ifdef WIN32
#  pragma warning(disable : 4267 4290)
#  define NOMINMAX
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <map>
#include <memory>
#include <stack>
#include <vector>
#include <CL/cl.hpp>
#include <cassert>

namespace vex {

/// Class for gathering and printing OpenCL and Host profiling info.
class profiler {
    private:
	class profile_unit {
	    public:
		profile_unit() : length(0) {}
		virtual ~profile_unit() {}
		virtual void tic() = 0;
		virtual double toc() = 0;

		double length;
		std::map<std::string, std::unique_ptr<profile_unit>> children;

		double children_time() const {
		    double tm = 0;

		    for(auto c = children.begin(); c != children.end(); c++)
			tm += c->second->length;

		    return tm;
		}

		uint max_line_width(const std::string &name, uint level) const {
		    uint w = name.size() + level;

		    for(auto c = children.begin(); c != children.end(); c++)
			w = std::max(w, c->second->max_line_width(c->first, level + shift_width));

		    return w;
		}

		void print(std::ostream &out, const std::string &name,
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
			c->second->print(out, c->first, level + shift_width, total, width);
		}

		void print_line(std::ostream &out, const std::string &name,
			double time, double perc, uint width) const
		{
		    using namespace std;
		    out << name << ":";
		    out << setw(width - name.size()) << "";
		    out << setiosflags(ios::fixed);
		    out << setw(10) << setprecision(3) << time << " sec.";
		    out << "] (" << setprecision(2) << setw(6) << perc << "%)" << endl;
		}
	    private:
		static const uint shift_width = 2U;
	};

	class cpu_profile_unit : public profile_unit {
	    public:
		void tic() {
		    start = std::chrono::high_resolution_clock::now();
		}

		double toc() {
		    double delta = std::chrono::duration<double>(
			    std::chrono::high_resolution_clock::now() - start).count();

		    length += delta;

		    return delta;
		}

	    private:
		std::chrono::time_point<std::chrono::high_resolution_clock> start;
	};

	class cl_profile_unit : public profile_unit {
	    public:
		cl_profile_unit(std::vector<cl::CommandQueue> &queue)
		    : queue(queue), start(queue.size()), stop(queue.size())
		{}

		void tic() {
		    auto e = start.begin();
		    for(auto q = queue.begin(); q != queue.end(); q++, e++)
			q->enqueueMarker(&e[0]);
		}

		double toc() {
		    for(uint d = 0; d < queue.size(); d++)
			queue[d].enqueueMarker(&stop[d]);

		    // Measured time ends before marker is in the queue.
		    cl_long max_delta = 0;
		    for(uint d = 0; d < queue.size(); d++) {
			stop[d].wait();
			max_delta = std::max<cl_long>(max_delta,
				stop [d].getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				start[d].getProfilingInfo<CL_PROFILING_COMMAND_END>()
				);
		    }

		    double delta = max_delta * 1.0e-9;

		    length += delta;

		    return delta;
		}
	    private:
		std::vector<cl::CommandQueue> &queue;
		std::vector<cl::Event> start;
		std::vector<cl::Event> stop;
	};

    public:
	/// Constructor.
	/**
	 * \param queue vector of command queues. Each queue should have been
	 *              initialized with CL_QUEUE_PROFILING_ENABLE property.
	 * \param name  Opional name to be used when profiling info is printed.
	 */
	profiler(
		std::vector<cl::CommandQueue> &queue,
		const std::string &name = "Profile"
		) : name(name), queue(queue)
	{
	    root.tic();
	    stack.push(&root);
	}

	/// Starts a CPU timer.
	/**
	 * Also pushes named interval to the top of the profiler hierarchy.
	 * \param name name of the measured interval.
	 */
	void tic_cpu(const std::string &name) {
	    assert(!stack.empty());

	    profile_unit *top  = stack.top();
	    cpu_profile_unit *unit = new cpu_profile_unit();
	    unit->tic();

	    top->children[name].reset(unit);

	    stack.push(unit);
	}

	/// Enqueues a marker into each of the provided queues.
	/**
	 * Also pushes named interval to the top of the profiler hierarchy.
	 * \param name name of the measured interval.
	 */
	void tic_cl(const std::string &name) {
	    assert(!stack.empty());

	    profile_unit *top  = stack.top();
	    cl_profile_unit *unit = new cl_profile_unit(queue);
	    unit->tic();

	    top->children[name].reset(unit);

	    stack.push(unit);
	}

	/// Returns time since last tic.
	/**
	 * Also removes interval from the top of the profiler hierarchy.
	 * \param name name of the measured interval. 
	 */
	double toc(const std::string &name) {
	    assert(!stack.empty());
	    assert(stack.top() != &root);

	    profile_unit *top = stack.top();
	    double delta = top->toc();
	    stack.pop();

	    return delta;
	}

    private:
	std::string name;
	std::vector<cl::CommandQueue> &queue;
	cpu_profile_unit root;
	std::stack<profile_unit*> stack;

	void print(std::ostream &out) {
	    if (stack.top() != &root)
		out << "Warning! Profile is incomplete." << std::endl;

	    double length = root.toc();

	    out << std::endl;
	    root.print(out, name, 0, length, root.max_line_width(name, 0));
	}

	friend std::ostream& operator<<(std::ostream &out, profiler &prof);

};

std::ostream& operator<<(std::ostream &os, profiler &prof) {
    prof.print(os);
    return os;
}

} // namespace vex

#endif
