#include <iomanip>
#include <algorithm>
#include "profiler.h"

using namespace std;
#define SHIFT_WIDTH 2

double profile_unit::children_time() const {
    double s = 0.0;

    for(
	    map<string, profile_unit>::const_iterator c = children.begin();
	    c != children.end(); c++
       ) s += c->second.length;

    return s;
}

size_t profile_unit::total_width(const string &name, int level) const {
    size_t w = name.size() + level;

    for(
	    map<string, profile_unit>::const_iterator c = children.begin();
	    c != children.end(); c++
       ) w = max(w, c->second.total_width(c->first, level + SHIFT_WIDTH));

    return w;
}

void profile_unit::print_line(std::ostream &out, const std::string &name,
	double time, double perc, size_t width) const
{
    out << name << ":";
    out << setw(width - name.size()) << "";
    out << setiosflags(ios::fixed);
    out << setw(10) << setprecision(3) << time << " sec.";
    out << "] (" << setprecision(2) << setw(6) << perc << "%)" << endl;
}

void profile_unit::print(ostream &out, const string &name,
	int level, double total, size_t width) const
{
    out << "[" << setw(level) << "";
    print_line(out, name, length, 100 * length / total, width - level);

    if (children.size()) {
	double sec = length - children_time();
	double perc = 100 * sec / total;

	if (perc > 1e-1) {
	    out << "[" << setw(level + 1) << "";
	    print_line(out, "self", sec, perc, width - level - 1);
	}
    }

    for(
	    map<string, profile_unit>::const_iterator c = children.begin();
	    c != children.end(); c++
       ) c->second.print(out, c->first, level + SHIFT_WIDTH, total, width);

}

profiler::profiler(const string &n) : name(n) {
    stack.push(&root);

#ifdef WIN32
    timeb now;
    ftime(&now);
#else
    auto now = std::chrono::monotonic_clock::now();
#endif

    root.start_time = now;
}

void profiler::print(std::ostream &out) {
    if (stack.top() != &root)
	out << "Warning! Profile is incomplete." << endl;

#ifdef WIN32
    timeb now;
    ftime(&now);
    root.length += now - root.start_time;
#else
    auto now = std::chrono::monotonic_clock::now();
    root.length += std::chrono::duration<double>(now - root.start_time).count();
#endif

    root.print(out, name, 0, root.length, root.total_width(name, 0));
}

std::ostream& operator<<(std::ostream &out, profiler &prof) {
    out << endl;
    prof.print(out);
    out << endl;
    return out;
}
