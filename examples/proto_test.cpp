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

	const size_t n = 1024 * 1024;

	vex::vector<double> a(ctx.queue(), n);
	vex::vector<double> b(ctx.queue(), n);
	vex::vector<double> c(ctx.queue(), n);

	a = 1;
	b = 2;
	c = sin(M_PI/4 * a) + b;

	std::cout << c[42] << std::endl;

	vex::Reductor<double, vex::SUM> sum(ctx.queue());
	vex::Reductor<double, vex::MAX> max(ctx.queue());

	std::cout << sum(a) / n << std::endl;
	std::cout << max(sin(M_PI/4 * a) + b) << std::endl;

	std::array<float, 3> arr = {1, 2, 3};
	auto tup = std::make_tuple(10, 20.0, 30.0f);

	vex::multivector<double, 3> ma(ctx.queue(), n);
	vex::multivector<double, 3> mb(ctx.queue(), n);
	vex::multivector<double, 3> mc(ctx.queue(), n);

	ma = 1;
	mb = 2;
	mc = 3;

	std::array<double,3> t = ma[0];
	std::cout << t[0] << " " << t[1] << " " << t[2] << " " << c[0] << std::endl;

	ma = sin(arr * ma) + greater(mb, tup + mc);

	std::cout << sin(arr[0] * 1) + (2 > std::get<0>(tup) + 3)
		  << " = " << ma(0)[0] << std::endl;

	ma = std::tie(2 * a, b, c);

    } catch (const cl::Error &e) {
	std::cout << e << std::endl;
    }
}
