#ifndef PROFILER_H
#define PROFILER_H

/// \cond internal_docs

/**
 * \file   profiler.h
 * \author Денис Демидов ddemidov@ksu.ru
 * \brief  Класс для сбора и вывода профилирующей информации.
 */

#include <map>
#include <string>
#include <stack>
#include <iostream>
#include <cassert>

#ifdef WIN32
#include <sys/timeb.h>

/// Разница в секундах между двумя моментами.
inline double operator-(const timeb &a, const timeb &b) {
    return a.time - b.time + 1e-3 * (a.millitm - b.millitm);
}
#else
#include <chrono>
#endif

class profiler;

/// Элемент профиля.
class profile_unit {
    public:
	profile_unit() : length(0) {}

	double children_time() const;
	size_t total_width(const std::string &name, int level) const;

	void print(std::ostream &out, const std::string &name,
		int level, double total, size_t width) const;

	void print_line(std::ostream &out, const std::string &name,
		double time, double perc, size_t width) const;

#ifdef WIN32
	/// Время последнего отсчета.
	timeb start_time;
#else
	/// Время последнего отсчета.
	std::chrono::time_point<std::chrono::monotonic_clock>  start_time;
#endif

	/// Общее время выполнения.
	double length;

	/// Дочерние элементы.
	std::map<std::string, profile_unit> children;
};

/// Класс для сбора и вывода профилирующей информации.
class profiler {
    public:
	/// Очистка данных профиля.
	profiler(const std::string &name = "Profile");

	/// Засекает время (устанавливает отсчет) для элемента профиля.
	/**
	 * \param name Ключ элемента профиля.
	 */
	void tic(const std::string &name) {
	    assert(!stack.empty());

#ifdef WIN32
	    timeb now;
	    ftime(&now);
#else
	    auto now = std::chrono::monotonic_clock::now();
#endif

	    profile_unit *top = stack.top();

	    top->children[name].start_time = now;

	    stack.push(&top->children[name]);
	}

	/// Возвращает время, прошедшее с последнего отсчета.
	/**
	 * Также увеличивает общее время выполнения элемента профиля на
	 * соответствующее значение.
	 * \param name Ключ элемента профиля.
	 * \return Время, прошедшее с последнего отсчета.
	 */
	double toc(const std::string &name) {
	    assert(!stack.empty());
	    assert(stack.top() != &root);

#ifdef WIN32
	    timeb now;
	    ftime(&now);
#else
	    auto now = std::chrono::monotonic_clock::now();
#endif

	    profile_unit *top = stack.top();
	    stack.pop();

#ifdef WIN32
	    double delta = now - top->start_time;
#else
	    double delta = std::chrono::duration<double>(
		    now - top->start_time).count();
#endif
	    top->length += delta;

	    return delta;
	}

    private:
	std::string name;
	profile_unit root;
	std::stack<profile_unit*> stack;

	void print(std::ostream &out);

	friend std::ostream& operator<<(std::ostream &out, profiler &prof);
};

std::ostream& operator<<(std::ostream &out, profiler &prof);

#endif
