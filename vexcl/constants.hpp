#ifndef VEXCL_CONSTANTS_HPP
#define VEXCL_CONSTANTS_HPP

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
 * \file   vexcl/constants.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Constants for use in vector expressions.
 */

#include <sstream>
#include <string>
#include <type_traits>

#include <vexcl/types.hpp>
#include <vexcl/operations.hpp>

#if BOOST_VERSION >= 105000
#  include <boost/math/constants/constants.hpp>
#endif

namespace vex {

/// \cond INTERNAL

//---------------------------------------------------------------------------
// std::integral_constant
//---------------------------------------------------------------------------
template <class T, T v>
struct is_cl_native< std::integral_constant<T, v> > : std::true_type {};

namespace traits {

template <class T, T v>
struct kernel_param_declaration< std::integral_constant<T, v> >
{
    static std::string get(const std::integral_constant<T, v>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

template <class T, T v>
struct partial_vector_expr< std::integral_constant<T, v> >
{
    static std::string get(const std::integral_constant<T, v>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return std::to_string(v);
    }
};

template <class T, T v>
struct kernel_arg_setter< std::integral_constant<T, v> >
{
    static void set(const std::integral_constant<T, v>&,
            cl::Kernel&, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &/*position*/, detail::kernel_generator_state_ptr)
    {
    }
};

} // namespace traits

#if (BOOST_VERSION >= 105000) || defined(DOXYGEN)
//---------------------------------------------------------------------------
// boost::math::constants wrappers
//---------------------------------------------------------------------------
template <class Impl>
struct boost_math_constant { };

template <class Impl>
struct is_cl_native< boost_math_constant<Impl> > : std::true_type {};

namespace traits {

template <class Impl>
struct kernel_param_declaration< boost_math_constant<Impl> >
{
    static std::string get(const boost_math_constant<Impl>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

template <class Impl>
struct partial_vector_expr< boost_math_constant<Impl> >
{
    static std::string get(const boost_math_constant<Impl>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        std::ostringstream s;
        s << std::scientific << std::setprecision(16)
          << Impl::get(
                    typename boost::math::constants::construction_traits<
                    double, boost::math::policies::policy<>
                    >::type()
                    );
        return s.str();
    }
};

template <class Impl>
struct kernel_arg_setter< boost_math_constant<Impl> >
{
    static void set(const boost_math_constant<Impl>&,
            cl::Kernel&, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &/*position*/, detail::kernel_generator_state_ptr)
    {
    }
};

} // namespace traits

/// \endcond

/// Mathematical constants imported from Boost.
namespace constants { }

/// Register boost::math::constant for use in VexCL expressions
#define VEX_REGISTER_BOOST_MATH_CONSTANT(name)                                 \
  namespace constants {                                                        \
  inline typename boost::proto::result_of::as_expr<                            \
      boost_math_constant<                                                     \
          boost::math::constants::detail::constant_##name<double> >,           \
      vector_domain>::type name() {                                            \
    return boost::proto::as_expr<vector_domain>(boost_math_constant<           \
        boost::math::constants::detail::constant_##name<double> >());          \
  }                                                                            \
  }

VEX_REGISTER_BOOST_MATH_CONSTANT( catalan )
VEX_REGISTER_BOOST_MATH_CONSTANT( cbrt_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( cosh_one )
VEX_REGISTER_BOOST_MATH_CONSTANT( cos_one )
VEX_REGISTER_BOOST_MATH_CONSTANT( degree )
VEX_REGISTER_BOOST_MATH_CONSTANT( e )
VEX_REGISTER_BOOST_MATH_CONSTANT( e_pow_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( euler )
VEX_REGISTER_BOOST_MATH_CONSTANT( euler_sqr )
VEX_REGISTER_BOOST_MATH_CONSTANT( exp_minus_half )
VEX_REGISTER_BOOST_MATH_CONSTANT( extreme_value_skewness )
VEX_REGISTER_BOOST_MATH_CONSTANT( four_minus_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( four_thirds_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( glaisher )
VEX_REGISTER_BOOST_MATH_CONSTANT( half )
VEX_REGISTER_BOOST_MATH_CONSTANT( half_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( half_root_two )
VEX_REGISTER_BOOST_MATH_CONSTANT( khinchin )
VEX_REGISTER_BOOST_MATH_CONSTANT( ln_ln_two )
VEX_REGISTER_BOOST_MATH_CONSTANT( ln_phi )
VEX_REGISTER_BOOST_MATH_CONSTANT( ln_ten )
VEX_REGISTER_BOOST_MATH_CONSTANT( ln_two )
VEX_REGISTER_BOOST_MATH_CONSTANT( log10_e )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_cbrt_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_euler )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_ln_phi )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_log10_e )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_root_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_root_two )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_root_two_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( one_div_two_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( phi )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi_cubed )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi_minus_three )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi_pow_e )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi_sqr )
VEX_REGISTER_BOOST_MATH_CONSTANT( pi_sqr_div_six )
VEX_REGISTER_BOOST_MATH_CONSTANT( pow23_four_minus_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( radian )
VEX_REGISTER_BOOST_MATH_CONSTANT( rayleigh_kurtosis )
VEX_REGISTER_BOOST_MATH_CONSTANT( rayleigh_kurtosis_excess )
VEX_REGISTER_BOOST_MATH_CONSTANT( rayleigh_skewness )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_e )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_half_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_ln_four )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_one_div_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_three )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_two )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_two_div_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( root_two_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( sinh_one )
VEX_REGISTER_BOOST_MATH_CONSTANT( sin_one )
VEX_REGISTER_BOOST_MATH_CONSTANT( sixth_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( third )
VEX_REGISTER_BOOST_MATH_CONSTANT( third_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( three_quarters )
VEX_REGISTER_BOOST_MATH_CONSTANT( three_quarters_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( two_div_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( two_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( two_thirds )
VEX_REGISTER_BOOST_MATH_CONSTANT( twothirds )
VEX_REGISTER_BOOST_MATH_CONSTANT( two_thirds_pi )
VEX_REGISTER_BOOST_MATH_CONSTANT( zeta_three )
VEX_REGISTER_BOOST_MATH_CONSTANT( zeta_two )

#endif

} // namespace vex

#endif
