#include <vexcl/vexcl.hpp>

namespace vex {

namespace traits {

template <class T, class Enable = void>
struct value_type {
    typedef T type;
};

template <class T>
struct value_type< vex::vector<T> > {
    typedef T type;
};

struct get_value_type : boost::proto::callable {
    template <class T> struct result;

    template <class This, class T>
    struct result< This(T) > {
        typedef
            typename value_type< typename std::decay<T>::type >::type
            type;
    };
};

}

namespace detail {

//---------------------------------------------------------------------------
struct deduce_value_type
    : boost::proto::or_<
        boost::proto::when <
            boost::proto::and_<
                boost::proto::terminal< boost::proto::_ >,
                boost::proto::if_< traits::proto_terminal_is_value< boost::proto::_value >() >
            >,
            traits::get_value_type( boost::proto::_ )
        > ,
        boost::proto::when <
            boost::proto::terminal< boost::proto::_ >,
            traits::get_value_type( boost::proto::_value )
        >,
        boost::proto::when <
            boost::proto::or_<
                boost::proto::or_<
                    boost::proto::less          < boost::proto::_, boost::proto::_ >,
                    boost::proto::greater       < boost::proto::_, boost::proto::_ >,
                    boost::proto::less_equal    < boost::proto::_, boost::proto::_ >,
                    boost::proto::greater_equal < boost::proto::_, boost::proto::_ >,
                    boost::proto::equal_to      < boost::proto::_, boost::proto::_ >,
                    boost::proto::not_equal_to  < boost::proto::_, boost::proto::_ >
                >,
                boost::proto::or_<
                    boost::proto::logical_and   < boost::proto::_, boost::proto::_ >,
                    boost::proto::logical_or    < boost::proto::_, boost::proto::_ >,
                    boost::proto::logical_not   < boost::proto::_ >
                >
            >,
            bool()
        >
      >
{};

}
}

//---------------------------------------------------------------------------
template <class Expr>
void deduce(const Expr &expr) {
    using namespace vex;
    namespace proto = boost::proto;

    auto ex = proto::as_child(expr);

    typedef
        typename std::decay<
            typename boost::result_of<
                detail::deduce_value_type(decltype(ex))
            >::type
        >::type
        T;

    proto::display_expr(ex);
    std::cout << "Result type: " << vex::type_name<T>() << std::endl;
}

//---------------------------------------------------------------------------
int main() {
    vex::Context ctx( vex::Filter::Name("K20") );
    vex::vector<double> x(ctx, 1024);
    vex::vector<int> y(ctx, 1024);

    deduce(5);
    deduce(4.2);
    deduce(x);
    deduce(y);
    deduce(x < y);
}
