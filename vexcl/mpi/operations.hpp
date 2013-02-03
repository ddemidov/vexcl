#ifndef VEXCL_MPI_OPERATIONS_HPP
#define VEXCL_MPI_OPERATIONS_HPP

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
 * \file   mpi/operations.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Set of operations to be used in mpi::vector expressions.
 */

#include <boost/proto/proto.hpp>
#include <vexcl/operations.hpp>

namespace vex {
namespace mpi {

/// \cond INTERNAL

// mpi::vector operations.
struct mpi_vector_terminal{};

struct mpi_vector_expr_grammar
    : boost::proto::or_<
          boost::proto::or_<
              boost::proto::or_< boost::proto::terminal< elem_index > >, \
              boost::proto::terminal< mpi_vector_terminal >,
              boost::proto::and_<
                  boost::proto::terminal< boost::proto::_ >,
                  boost::proto::if_< is_cl_native< boost::proto::_value >() >
              >
          >,
          BUILTIN_OPERATIONS(mpi_vector_expr_grammar),
          USER_FUNCTIONS(mpi_vector_expr_grammar)
      >
{};

struct mpi_additive_vector_transform{};

struct mpi_additive_vector_transform_grammar
    : boost::proto::or_<
        boost::proto::terminal< mpi_additive_vector_transform >,
        boost::proto::plus<
            mpi_additive_vector_transform_grammar,
            mpi_additive_vector_transform_grammar
        >,
        boost::proto::minus<
            mpi_additive_vector_transform_grammar,
            mpi_additive_vector_transform_grammar
        >,
        boost::proto::negate<
            mpi_additive_vector_transform_grammar
        >
      >
{};

struct mpi_vector_full_grammar
    : boost::proto::or_<
        mpi_vector_expr_grammar,
        boost::proto::terminal< mpi_additive_vector_transform >,
        boost::proto::plus< mpi_vector_full_grammar, mpi_vector_full_grammar >,
        boost::proto::minus< mpi_vector_full_grammar, mpi_vector_full_grammar >,
        boost::proto::negate< mpi_vector_full_grammar >
      >
{};

template <class Expr>
struct mpi_vector_expression;

struct mpi_vector_domain
    : boost::proto::domain<
        boost::proto::generator<mpi_vector_expression>,
        mpi_vector_full_grammar
        >
{ };

template <class Expr>
struct mpi_vector_expression
    : boost::proto::extends< Expr, mpi_vector_expression<Expr>, mpi_vector_domain>
{
    mpi_vector_expression(const Expr &expr = Expr())
        : boost::proto::extends< Expr, mpi_vector_expression<Expr>, mpi_vector_domain>(expr) {}
};


VEXCL_VECTOR_EXPR_EXTRACTOR(mpi_extract_vector_expressions,
        mpi_vector_expr_grammar,
        mpi_additive_vector_transform_grammar,
        mpi_vector_full_grammar
        );

VEXCL_ADDITIVE_EXPR_EXTRACTOR(mpi_extract_additive_vector_transforms,
        mpi_vector_expr_grammar,
        mpi_additive_vector_transform_grammar,
        mpi_vector_full_grammar
        );

// mpi::multivector operations.
struct mpi_multivector_terminal{};

struct mpi_multivector_expr_grammar
    : boost::proto::or_<
          boost::proto::or_<
              boost::proto::or_< boost::proto::terminal< elem_index > >, \
              boost::proto::terminal< mpi_multivector_terminal >,
              boost::proto::and_<
                  boost::proto::terminal< boost::proto::_ >,
                  boost::proto::if_< is_multiscalar< boost::proto::_value >() >
              >
          >,
          BUILTIN_OPERATIONS(mpi_multivector_expr_grammar),
          USER_FUNCTIONS(mpi_multivector_expr_grammar)
      >
{};

struct mpi_additive_multivector_transform {};

struct mpi_additive_multivector_transform_grammar
    : boost::proto::or_<
        boost::proto::terminal< mpi_additive_multivector_transform >,
        boost::proto::plus<
            mpi_additive_multivector_transform_grammar,
            mpi_additive_multivector_transform_grammar
        >,
        boost::proto::minus<
            mpi_additive_multivector_transform_grammar,
            mpi_additive_multivector_transform_grammar
        >,
        boost::proto::negate<
            mpi_additive_multivector_transform_grammar
        >
      >
{};

struct mpi_multivector_full_grammar
    : boost::proto::or_<
        mpi_multivector_expr_grammar,
        boost::proto::terminal< mpi_additive_multivector_transform >,
        boost::proto::plus< mpi_multivector_full_grammar, mpi_multivector_full_grammar >,
        boost::proto::minus< mpi_multivector_full_grammar, mpi_multivector_full_grammar >,
        boost::proto::negate< mpi_multivector_full_grammar >
      >
{};

template <class Expr>
struct mpi_multivector_expression;

struct mpi_multivector_domain
    : boost::proto::domain<
        boost::proto::generator<mpi_multivector_expression>,
        mpi_multivector_full_grammar
      >
{ };

template <class Expr>
struct mpi_multivector_expression
    : boost::proto::extends< Expr, mpi_multivector_expression<Expr>, mpi_multivector_domain>
{
    mpi_multivector_expression(const Expr &expr = Expr())
        : boost::proto::extends< Expr, mpi_multivector_expression<Expr>, mpi_multivector_domain>(expr) {}
};

VEXCL_VECTOR_EXPR_EXTRACTOR(mpi_extract_multivector_expressions,
        mpi_multivector_expr_grammar,
        mpi_additive_multivector_transform_grammar,
        mpi_multivector_full_grammar
        );

VEXCL_ADDITIVE_EXPR_EXTRACTOR(mpi_extract_additive_multivector_transforms,
        mpi_multivector_expr_grammar,
        mpi_additive_multivector_transform_grammar,
        mpi_multivector_full_grammar
        );

// Local expression extractor
template <typename T, bool own = true> class vector;
template <typename T, size_t N, bool own = true> class multivector;

struct extract_local_terminal : boost::proto::callable {
    template <class T>
    struct result;

    template <class This, class T, bool own>
    struct result< This( vex::mpi::vector<T, own>& ) > {
        typedef const vex::vector<T>& type;
    };

    template <class This, class T, bool own>
    struct result< This( const vex::mpi::vector<T, own>& ) > {
        typedef const vex::vector<T>& type;
    };

    template <class This, class T, size_t N, bool own>
    struct result< This( vex::mpi::multivector<T, N, own>& ) > {
        typedef const vex::multivector<T, N, own>& type;
    };

    template <class This, class T, size_t N, bool own>
    struct result< This( const vex::mpi::multivector<T, N, own>& ) > {
        typedef const vex::multivector<T, N, own>& type;
    };

    template <class T, bool own>
    const vex::vector<T>& operator()(const vex::mpi::vector<T, own> &v) const {
        return v.data();
    }

    template <class T, size_t N, bool own>
    const vex::multivector<T, N, own>& operator()(
            const vex::mpi::multivector<T, N, own> &v) const
    {
        return v.data();
    }
};

struct extract_local_expression
    : boost::proto::or_ <
        boost::proto::when <
            boost::proto::terminal< mpi_vector_terminal >,
            extract_local_terminal( boost::proto::_ )
        > ,
        boost::proto::when <
            boost::proto::terminal< mpi_multivector_terminal >,
            extract_local_terminal( boost::proto::_ )
        > ,
        boost::proto::when <
            boost::proto::terminal<boost::proto::_>,
            boost::proto::_
        > ,
        boost::proto::function<
            boost::proto::_,
            boost::proto::vararg< extract_local_expression >
        > ,
        boost::proto::when <
            boost::proto::nary_expr<
                boost::proto::_,
                boost::proto::vararg< extract_local_expression >
            >
        >
    >
{};

/// \endcond

} // namespace mpi
} // namespace vex

#endif
