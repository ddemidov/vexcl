#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include <cmath>
#include <vexcl/vexcl.hpp>

using namespace vex;

bool run_test(const std::string &name, std::function<bool()> test) {
    char fc = std::cout.fill('.');
    std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc = test();
    std::cout << (rc ? " success." : " failed.") << std::endl;
    return rc;
}

int main() {
    try {
	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	std::tie(context, queue) = queue_list(Filter::Env());
	std::cout << queue << std::endl;

	if (queue.empty()) {
	    std::cerr << "No OpenCL devices found." << std::endl;
	    return 1;
	}

	run_test("Empty vector construction", [&]() {
		bool rc = true;
		vex::vector<float> x;
		rc = rc && (x.size() == 0);
		rc = rc && (x.end() - x.begin() == 0);
		return rc;
		});

	run_test("Vector construction from size", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		rc = rc && (x.size() == N);
		rc = rc && (x.end() - x.begin() == N);
		return rc;
		});

	run_test("Vector construction from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N, 42);
		std::vector<float> y(N);
		vex::vector<float> X(queue, x);
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() - X.begin() == x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](float a, float b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](float a) {return a == 0; });
		return rc;
		});

	run_test("Vector construction from size and host pointer", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N, 42);
		std::vector<float> y(N);
		vex::vector<float> X(queue, N, x.data());
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() - X.begin() == x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](float a, float b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](float a) {return a == 0; });
		return rc;
		});

	run_test("Vector move construction from vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		x = 42;
		vex::vector<float> y = std::move(x);
		rc = rc && (y.size() == N);
		rc = rc && (y.end() - y.begin() == N);
		Reductor<float,MIN> min(queue);
		Reductor<float,MAX> max(queue);
		rc = rc && min(y) == 42;
		rc = rc && max(y) == 42;
		return rc;
		});

	run_test("Vector move assignment", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N, 42);
		vex::vector<float> X(queue, x);
		vex::vector<float> Y = std::move(X);
		rc = rc && (Y.size() == x.size());
		Reductor<float,MIN> min(queue);
		Reductor<float,MAX> max(queue);
		rc = rc && min(Y) == x[0];
		rc = rc && max(Y) == x[0];
		return rc;
		});

	run_test("Vector swap", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		vex::vector<float> y(queue, N/2);
		x = 42;
		y = 67;
		swap(x, y);
		rc = rc && (y.size() == N);
		rc = rc && (x.size() == N/2);
		Reductor<float,MIN> min(queue);
		Reductor<float,MAX> max(queue);
		rc = rc && min(y) == 42 && max(y) == 42;
		rc = rc && max(x) == 67 && max(x) == 67;
		return rc;
		});

	run_test("Vector resize from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N, 42);
		vex::vector<float> X;
		X.resize(queue, x);
		rc = rc && (X.size() == x.size());
		Reductor<float,MIN> min(queue);
		Reductor<float,MAX> max(queue);
		rc = rc && min(X) == 42 && max(X) == 42;
		return rc;
		});

	run_test("Vector resize vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		x = 42;
		vex::vector<float> y;
		y.resize(x);
		rc = rc && (y.size() == x.size());
		Reductor<float,MAX> max(queue);
		rc = rc && max((x - y) * (x - y)) == 0;
		return rc;
		});

	run_test("Iterate over vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		x = 42;
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](float a) { return a == 42; });
		return rc;
		});

	run_test("Access vex::vector elements", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		for(int i = 0; i < N; i++)
		    x[i] = 42;
		for(int i = 0; i < N; i++)
		    rc == rc && (x[i] == 42);
		return rc;
		});

	run_test("Copy vex::vector to std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N);
		vex::vector<float> X(queue, N);
		X = 42;
		copy(X, x);
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](float a) { return a == 42; });
		X = 67;
		vex::copy(X.begin(), X.end(), x.begin());
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](float a) { return a == 67; });
		return rc;
		});

	run_test("Copy std::vector to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<float> x(N, 42);
		vex::vector<float> X(queue, N);
		copy(x, X);
		Reductor<float,MIN> min(queue);
		Reductor<float,MAX> max(queue);
		rc = rc && min(X) == 42 && max(X) == 42;
		std::fill(x.begin(), x.end(), 67);
		vex::copy(x.begin(), x.end(), X.begin());
		rc = rc && min(X) == 67 && max(X) == 67;
		return rc;
		});

	run_test("Assign expression to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<float> x(queue, N);
		vex::vector<float> y(queue, N);
		vex::vector<float> z(queue, N);
		y = 42;
		z = 67;
		x = 5 * Sin(y) + z;
		Reductor<float,MAX> max(queue);
		rc = rc && max(Abs(x - static_cast<float>(5 * sin(42) + 67))) < 1e-12;
		return rc;
		});


    } catch (const cl::Error &err) {
	std::cerr << "OpenCL error: " << err << std::endl;
	return 1;
    } catch (const std::exception &err) {
	std::cerr << "Error: " << err.what() << std::endl;
	return 1;
    } catch (...) {
	std::cerr << "Unknown error" << std::endl;
	return 1;
    }

    return 0;
}
