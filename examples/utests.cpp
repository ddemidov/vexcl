#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include <numeric>
#include <cmath>

//#define VEX_SHOW_KERNELS
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

extern const char chk_if_gr_body[] = "return prm1 > prm2 ? 1 : 0;";

int main() {
    try {
	std::vector<cl::Context>      context;
	std::vector<cl::CommandQueue> queue;

	std::tie(context, queue) = queue_list(Filter::DoublePrecision() && Filter::Env());
	std::cout << queue << std::endl;

	if (queue.empty()) {
	    std::cerr << "No OpenCL devices found." << std::endl;
	    return 1;
	}

	run_test("Empty vector construction", [&]() {
		bool rc = true;
		vex::vector<double> x;
		rc = rc && (x.size() == 0);
		rc = rc && (x.end() - x.begin() == 0);
		return rc;
		});

	run_test("Vector construction from size", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		rc = rc && (x.size() == N);
		rc = rc && (x.end() - x.begin() == N);
		return rc;
		});

	run_test("Vector construction from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		std::vector<double> y(N);
		vex::vector<double> X(queue, x);
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() - X.begin() == x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](double a, double b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](double a) {return a == 0; });
		return rc;
		});

	run_test("Vector construction from size and host pointer", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		std::vector<double> y(N);
		vex::vector<double> X(queue, N, x.data());
		rc = rc && (X.size() == x.size());
		rc = rc && (X.end() - X.begin() == x.size());
		copy(X, y);
		std::transform(x.begin(), x.end(), y.begin(), y.begin(),
		    [](double a, double b) { return a - b; });
		rc = rc && std::all_of(y.begin(), y.end(),
		    [](double a) {return a == 0; });
		return rc;
		});

	run_test("Vector move construction from vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		x = 42;
		vex::vector<double> y = std::move(x);
		rc = rc && (y.size() == N);
		rc = rc && (y.end() - y.begin() == N);
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && min(y) == 42;
		rc = rc && max(y) == 42;
		return rc;
		});

	run_test("Vector move assignment", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X(queue, x);
		vex::vector<double> Y = std::move(X);
		rc = rc && (Y.size() == x.size());
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && min(Y) == x[0];
		rc = rc && max(Y) == x[0];
		return rc;
		});

	run_test("Vector swap", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		vex::vector<double> y(queue, N/2);
		x = 42;
		y = 67;
		swap(x, y);
		rc = rc && (y.size() == N);
		rc = rc && (x.size() == N/2);
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && min(y) == 42 && max(y) == 42;
		rc = rc && max(x) == 67 && max(x) == 67;
		return rc;
		});

	run_test("Vector resize from std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X;
		X.resize(queue, x);
		rc = rc && (X.size() == x.size());
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && min(X) == 42 && max(X) == 42;
		return rc;
		});

	run_test("Vector resize vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		x = 42;
		vex::vector<double> y;
		y.resize(x);
		rc = rc && (y.size() == x.size());
		Reductor<double,MAX> max(queue);
		rc = rc && max((x - y) * (x - y)) == 0;
		return rc;
		});

	run_test("Iterate over vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		x = 42;
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 42; });
		return rc;
		});

	run_test("Access vex::vector elements", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		for(int i = 0; i < N; i++)
		    x[i] = 42;
		for(int i = 0; i < N; i++)
		    rc == rc && (x[i] == 42);
		return rc;
		});

	run_test("Copy vex::vector to std::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		vex::vector<double> X(queue, N);
		X = 42;
		copy(X, x);
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 42; });
		X = 67;
		vex::copy(X.begin(), X.end(), x.begin());
		rc = rc && std::all_of(x.begin(), x.end(),
		    [](double a) { return a == 67; });
		return rc;
		});

	run_test("Copy std::vector to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N, 42);
		vex::vector<double> X(queue, N);
		copy(x, X);
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && min(X) == 42 && max(X) == 42;
		std::fill(x.begin(), x.end(), 67);
		vex::copy(x.begin(), x.end(), X.begin());
		rc = rc && min(X) == 67 && max(X) == 67;
		return rc;
		});

	run_test("Assign expression to vex::vector", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		vex::vector<double> y(queue, N);
		vex::vector<double> z(queue, N);
		y = 42;
		z = 67;
		x = 5 * sin(y) + z;
		Reductor<double,MAX> max(queue);
		rc = rc && max(fabs(x - (5 * sin(42) + 67))) < 1e-12;
		return rc;
		});


	run_test("Reduction", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });
		vex::vector<double> X(queue, x);
		Reductor<double,SUM> sum(queue);
		Reductor<double,MIN> min(queue);
		Reductor<double,MAX> max(queue);
		rc = rc && fabs(sum(X) - std::accumulate(x.begin(), x.end(), 0.0)) < 1e-6;
		rc = rc && fabs(min(X) - *std::min_element(x.begin(), x.end())) < 1e-4;
		rc = rc && fabs(max(X) - *std::max_element(x.begin(), x.end())) < 1e-4;
		return rc;
		});


	run_test("Sparse matrix-vector product", [&]() {
		bool rc = true;
		const size_t n   = 32;
		const double  h   = 1.0 / (n - 1);
		const double  h2i = (n - 1) * (n - 1);

		std::vector<size_t> row;
		std::vector<size_t> col;
		std::vector<double> val;

		row.reserve(n * n * n + 1);
		col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
		val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

		row.push_back(0);
		for(size_t k = 0, idx = 0; k < n; k++) {
		    double z = k * h;
		    for(size_t j = 0; j < n; j++) {
			double y = j * h;
			for(size_t i = 0; i < n; i++, idx++) {
			    double x = i * h;
			    if (
				i == 0 || i == (n - 1) ||
				j == 0 || j == (n - 1) ||
				k == 0 || k == (n - 1)
			       )
			    {
				col.push_back(idx);
				val.push_back(1);
				row.push_back(row.back() + 1);
			    } else {
				col.push_back(idx - n * n);
				val.push_back(-h2i);

				col.push_back(idx - n);
				val.push_back(-h2i);

				col.push_back(idx - 1);
				val.push_back(-h2i);

				col.push_back(idx);
				val.push_back(6 * h2i);

				col.push_back(idx + 1);
				val.push_back(-h2i);

				col.push_back(idx + n);
				val.push_back(-h2i);

				col.push_back(idx + n * n);
				val.push_back(-h2i);

				row.push_back(row.back() + 7);
			    }
			}
		    }
		}

		std::vector<double> x(n * n * n);
		std::vector<double> y(n * n * n);
		std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

		vex::SpMat<double>  A(queue, x.size(), row.data(), col.data(), val.data());
		vex::vector<double> X(queue, x);
		vex::vector<double> Y(queue, x.size());

		Y = A * X;
		copy(Y, y);

		double res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[i]; j < row[i + 1]; j++)
			sum += val[j] * x[col[j]];
		    res = std::max(res, fabs(sum - y[i]));
		}

		rc = rc && res < 1e-8;

		Y = X + A * X;
		copy(Y, y);

		res = 0;
		for(size_t i = 0; i < x.size(); i++) {
		    double sum = 0;
		    for(size_t j = row[i]; j < row[i + 1]; j++)
			sum += val[j] * x[col[j]];
		    res = std::max(res, fabs(sum + x[i] - y[i]));
		}

		rc = rc && res < 1e-8;

		return rc;
	});

	run_test("Builtin function with one argument", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
		vex::vector<double> X(queue, x);
		Reductor<double,SUM> sum(queue);
		rc = rc && 1e-8 > fabs(sum(sin(X)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + sin(v);
			}));
		rc = rc && 1e-8 > fabs(sum(cos(X)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + cos(v);
			}));
		return rc;
		});

#ifdef VEXCL_VARIADIC_TEMPLATES
	run_test("Builtin function with two arguments", [&]() {
		const size_t N = 1024;
		bool rc = true;
		std::vector<double> x(N);
		std::generate(x.begin(), x.end(), [](){ return (double)rand() / RAND_MAX; });
		vex::vector<double> X(queue, x);
		Reductor<double,SUM> sum(queue);
		rc = rc && 1e-8 > fabs(sum(pow(X, 2.0)) -
		    std::accumulate(x.begin(), x.end(), 0.0, [](double s, double v) {
			return s + pow(v, 2.0);
			}));
		return rc;
		});
#endif

	run_test("Custom function", [&]() {
		const size_t N = 1024;
		bool rc = true;
		vex::vector<double> x(queue, N);
		vex::vector<double> y(queue, N);
		x = 1;
		y = 2;
		UserFunction<chk_if_gr_body, size_t, double, double> chk_if_greater;
		Reductor<size_t,SUM> sum(queue);
		rc = rc && sum(chk_if_greater(x, y)) == 0;
		rc = rc && sum(chk_if_greater(y, x)) == N;
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
