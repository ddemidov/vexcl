#define BOOST_TEST_MODULE ExpressionTypeDeduction
#include <boost/test/unit_test.hpp>
#include <vexcl/vector.hpp>
#include "context_setup.hpp"

template <class Result, class Expr>
void check(const Expr &expr) {
    typedef typename boost::proto::result_of::as_expr<Expr>::type ExprType;

    typedef
        typename std::decay<
                typename boost::result_of<
                    vex::detail::deduce_value_type(ExprType)
                >::type
            >::type
        Deduced;

    boost::proto::display_expr( boost::proto::as_child(expr) );
    std::cout << vex::type_name<Deduced>() << std::endl << std::endl;

    BOOST_CHECK( (std::is_same<Deduced, Result>::value) );
}

BOOST_AUTO_TEST_CASE(terminals)
{
    vex::vector<double> x;
    vex::vector<int> y;

    check<int>   (5);
    check<double>(4.2);
    check<double>(x);
    check<int>   (y);
}

BOOST_AUTO_TEST_CASE(logical_expr)
{
    vex::vector<double> x;
    vex::vector<int> y;

    check<bool>(x < y);
    check<bool>(5 > pow(x, 2.0 * y));
    check<bool>(!x);
}

BOOST_AUTO_TEST_CASE(nary_expr)
{
    vex::vector<double> x;
    vex::vector<int> y;

    check<double>(x + y);
    check<double>(x + 2 * y);
    check<int>   (-y);
}

BOOST_AUTO_TEST_CASE(user_functions)
{
    vex::vector<double> x;
    vex::vector<int> y;

    VEX_FUNCTION(f1, double(double),      "return 42;");
    VEX_FUNCTION(f2, int(double, double), "return 42;");

    check<double>( f1(x) );
    check<int>   ( f2(x, y) );
    check<int>   ( f2(x + y, x - y) );
}

BOOST_AUTO_TEST_CASE(ternary_operator)
{
    vex::vector<double> x;
    vex::vector<int> y;

    check<int>( if_else(x < 0, 1, y) );
}

BOOST_AUTO_TEST_CASE(builtin_functions)
{
    vex::vector<double> x;
    vex::vector<int> y;

    check<double>( cos(x) - sin(y) );
    check<double>( pow(x, 2.0 * y) );
}

BOOST_AUTO_TEST_SUITE_END()
