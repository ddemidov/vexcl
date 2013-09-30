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

#include <boost/math/constants/constants.hpp>

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

//---------------------------------------------------------------------------
// boost::math::constants wrappers
//---------------------------------------------------------------------------
template <class Impl>
struct user_constant { };

template <class Impl>
struct is_cl_native< user_constant<Impl> > : std::true_type {};

namespace traits {

template <class Impl>
struct kernel_param_declaration< user_constant<Impl> >
{
    static std::string get(const user_constant<Impl>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return "";
    }
};

template <class Impl>
struct partial_vector_expr< user_constant<Impl> >
{
    static std::string get(const user_constant<Impl>&,
            const cl::Device&, const std::string &/*prm_name*/,
            detail::kernel_generator_state_ptr)
    {
        return Impl::get();
    }
};

template <class Impl>
struct kernel_arg_setter< user_constant<Impl> >
{
    static void set(const user_constant<Impl>&,
            cl::Kernel&, unsigned/*device*/, size_t/*index_offset*/,
            unsigned &/*position*/, detail::kernel_generator_state_ptr)
    {
    }
};

} // namespace traits

/// \endcond

/// Create user-defined constant for use in VexCL expressions
#define VEX_CONSTANT(name, value)                                              \
  struct constant_##name {                                                     \
    static std::string get() {                                                 \
      std::ostringstream s;                                                    \
      s << "( " << std::scientific << std::setprecision(16) << value << " )";  \
      return s.str();                                                          \
    }                                                                          \
    boost::proto::result_of::as_expr<                                          \
        vex::user_constant<constant_##name>,                                   \
        vex::vector_domain>::type                                              \
    operator()() const {                                                       \
      return boost::proto::as_expr<vex::vector_domain>(                        \
          vex::user_constant<constant_##name>());                              \
    }                                                                          \
  } name

/// Mathematical constants.
namespace constants {

// Can not implement these as functors, could end up in multiple object files.
#define VEX_GLOBAL_CONSTANT(name, value)                                       \
  struct constant_##name {                                                     \
    static std::string get() {                                                 \
      std::ostringstream s;                                                    \
      s << "( " << std::scientific << std::setprecision(16) << value << " )";  \
      return s.str();                                                          \
    }                                                                          \
  };                                                                           \
  inline boost::proto::result_of::as_expr<                                     \
      vex::user_constant<constant_##name>, vex::vector_domain>::type           \
  name() {                                                                     \
    return boost::proto::as_expr<vex::vector_domain>(                          \
        vex::user_constant<constant_##name>());                                \
  }

VEX_GLOBAL_CONSTANT( pi, boost::math::constants::pi<double>() );
VEX_GLOBAL_CONSTANT( root_pi, boost::math::constants::root_pi<double>() );
VEX_GLOBAL_CONSTANT( root_half_pi, boost::math::constants::root_half_pi<double>() );
VEX_GLOBAL_CONSTANT( root_two_pi, boost::math::constants::root_two_pi<double>() );
VEX_GLOBAL_CONSTANT( root_ln_four, boost::math::constants::root_ln_four<double>() );
VEX_GLOBAL_CONSTANT( e, boost::math::constants::e<double>() );
VEX_GLOBAL_CONSTANT( half, boost::math::constants::half<double>() );
VEX_GLOBAL_CONSTANT( euler, boost::math::constants::euler<double>() );
VEX_GLOBAL_CONSTANT( root_two, boost::math::constants::root_two<double>() );
VEX_GLOBAL_CONSTANT( ln_two, boost::math::constants::ln_two<double>() );
VEX_GLOBAL_CONSTANT( ln_ln_two, boost::math::constants::ln_ln_two<double>() );
VEX_GLOBAL_CONSTANT( third, boost::math::constants::third<double>() );
VEX_GLOBAL_CONSTANT( twothirds, boost::math::constants::twothirds<double>() );
VEX_GLOBAL_CONSTANT( pi_minus_three, boost::math::constants::pi_minus_three<double>() );
VEX_GLOBAL_CONSTANT( four_minus_pi, boost::math::constants::four_minus_pi<double>() );
VEX_GLOBAL_CONSTANT( two_pi, boost::math::constants::two_pi<double>() );
VEX_GLOBAL_CONSTANT( one_div_two_pi, boost::math::constants::one_div_two_pi<double>() );
VEX_GLOBAL_CONSTANT( half_root_two, boost::math::constants::half_root_two<double>() );
VEX_GLOBAL_CONSTANT( pow23_four_minus_pi, boost::math::constants::pow23_four_minus_pi<double>() );
VEX_GLOBAL_CONSTANT( exp_minus_half, boost::math::constants::exp_minus_half<double>() );

#if (BOOST_VERSION >= 105000) || defined(DOXYGEN)
VEX_GLOBAL_CONSTANT( catalan, boost::math::constants::catalan<double>() );
VEX_GLOBAL_CONSTANT( cbrt_pi, boost::math::constants::cbrt_pi<double>() );
VEX_GLOBAL_CONSTANT( cosh_one, boost::math::constants::cosh_one<double>() );
VEX_GLOBAL_CONSTANT( cos_one, boost::math::constants::cos_one<double>() );
VEX_GLOBAL_CONSTANT( degree, boost::math::constants::degree<double>() );
VEX_GLOBAL_CONSTANT( e_pow_pi, boost::math::constants::e_pow_pi<double>() );
VEX_GLOBAL_CONSTANT( euler_sqr, boost::math::constants::euler_sqr<double>() );
VEX_GLOBAL_CONSTANT( extreme_value_skewness, boost::math::constants::extreme_value_skewness<double>() );
VEX_GLOBAL_CONSTANT( four_thirds_pi, boost::math::constants::four_thirds_pi<double>() );
VEX_GLOBAL_CONSTANT( glaisher, boost::math::constants::glaisher<double>() );
VEX_GLOBAL_CONSTANT( half_pi, boost::math::constants::half_pi<double>() );
VEX_GLOBAL_CONSTANT( khinchin, boost::math::constants::khinchin<double>() );
VEX_GLOBAL_CONSTANT( ln_phi, boost::math::constants::ln_phi<double>() );
VEX_GLOBAL_CONSTANT( ln_ten, boost::math::constants::ln_ten<double>() );
VEX_GLOBAL_CONSTANT( log10_e, boost::math::constants::log10_e<double>() );
VEX_GLOBAL_CONSTANT( one_div_cbrt_pi, boost::math::constants::one_div_cbrt_pi<double>() );
VEX_GLOBAL_CONSTANT( one_div_euler, boost::math::constants::one_div_euler<double>() );
VEX_GLOBAL_CONSTANT( one_div_ln_phi, boost::math::constants::one_div_ln_phi<double>() );
VEX_GLOBAL_CONSTANT( one_div_log10_e, boost::math::constants::one_div_log10_e<double>() );
VEX_GLOBAL_CONSTANT( one_div_root_pi, boost::math::constants::one_div_root_pi<double>() );
VEX_GLOBAL_CONSTANT( one_div_root_two, boost::math::constants::one_div_root_two<double>() );
VEX_GLOBAL_CONSTANT( one_div_root_two_pi, boost::math::constants::one_div_root_two_pi<double>() );
VEX_GLOBAL_CONSTANT( phi, boost::math::constants::phi<double>() );
VEX_GLOBAL_CONSTANT( pi_cubed, boost::math::constants::pi_cubed<double>() );
VEX_GLOBAL_CONSTANT( pi_pow_e, boost::math::constants::pi_pow_e<double>() );
VEX_GLOBAL_CONSTANT( pi_sqr, boost::math::constants::pi_sqr<double>() );
VEX_GLOBAL_CONSTANT( pi_sqr_div_six, boost::math::constants::pi_sqr_div_six<double>() );
VEX_GLOBAL_CONSTANT( radian, boost::math::constants::radian<double>() );
VEX_GLOBAL_CONSTANT( rayleigh_kurtosis, boost::math::constants::rayleigh_kurtosis<double>() );
VEX_GLOBAL_CONSTANT( rayleigh_kurtosis_excess, boost::math::constants::rayleigh_kurtosis_excess<double>() );
VEX_GLOBAL_CONSTANT( rayleigh_skewness, boost::math::constants::rayleigh_skewness<double>() );
VEX_GLOBAL_CONSTANT( root_e, boost::math::constants::root_e<double>() );
VEX_GLOBAL_CONSTANT( root_one_div_pi, boost::math::constants::root_one_div_pi<double>() );
VEX_GLOBAL_CONSTANT( root_three, boost::math::constants::root_three<double>() );
VEX_GLOBAL_CONSTANT( root_two_div_pi, boost::math::constants::root_two_div_pi<double>() );
VEX_GLOBAL_CONSTANT( sinh_one, boost::math::constants::sinh_one<double>() );
VEX_GLOBAL_CONSTANT( sin_one, boost::math::constants::sin_one<double>() );
VEX_GLOBAL_CONSTANT( sixth_pi, boost::math::constants::sixth_pi<double>() );
VEX_GLOBAL_CONSTANT( third_pi, boost::math::constants::third_pi<double>() );
VEX_GLOBAL_CONSTANT( three_quarters, boost::math::constants::three_quarters<double>() );
VEX_GLOBAL_CONSTANT( three_quarters_pi, boost::math::constants::three_quarters_pi<double>() );
VEX_GLOBAL_CONSTANT( two_div_pi, boost::math::constants::two_div_pi<double>() );
VEX_GLOBAL_CONSTANT( two_thirds, boost::math::constants::two_thirds<double>() );
VEX_GLOBAL_CONSTANT( two_thirds_pi, boost::math::constants::two_thirds_pi<double>() );
VEX_GLOBAL_CONSTANT( zeta_three, boost::math::constants::zeta_three<double>() );
VEX_GLOBAL_CONSTANT( zeta_two, boost::math::constants::zeta_two<double>() );
#endif


} //namespace constants

} // namespace vex

#endif
