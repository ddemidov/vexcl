#include <iostream>
#include <cmath>

//#define VEXCL_SHOW_KERNELS
#include <boost/mpl/max.hpp>
#include <vexcl/vexcl.hpp>

extern const char greater_body[] = "return prm1 > prm2;";
vex::UserFunction<greater_body, bool(double, double)> greater;

using boost::proto::_;
namespace proto = boost::proto;
namespace mpl = boost::mpl;

struct show {
    template <class Expr>
    void operator()(const Expr &expr) const {
	boost::proto::display_expr(expr);
    }
};

int main() {
    try {
	vex::Context ctx( vex::Filter::Env );
	std::cout << ctx << std::endl;

	const size_t n = 16;

	const double h2i = (n - 1) * (n - 1);

	std::vector<size_t> row;
	std::vector<size_t> col;
	std::vector<double> val;

	row.reserve(n * n * n + 1);
	col.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);
	val.reserve(6 * (n - 2) * (n - 2) * (n - 2) + n * n * n);

	row.push_back(0);
	for(size_t k = 0, idx = 0; k < n; k++) {
	    for(size_t j = 0; j < n; j++) {
		for(size_t i = 0; i < n; i++, idx++) {
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
	std::vector<double> y(n * n * n, 0);
	std::generate(x.begin(), x.end(), []() { return (double)rand() / RAND_MAX; });

	vex::vector<double> X(ctx.queue(), x);
	vex::vector<double> Y(ctx.queue(), y);

	vex::SpMat <double> A(ctx.queue(), x.size(), x.size(), row.data(), col.data(), val.data());

	Y = X;
	Y = A * X;
	Y = X + A * X;
	Y = X * Y + A * X;
    } catch (const cl::Error &e) {
	std::cout << e << std::endl;
    }
}
