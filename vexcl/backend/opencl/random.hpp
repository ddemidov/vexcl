#ifndef VEXCL_BACKEND_OPENCL_RANDOM_HPP
#define VEXCL_BACKEND_OPENCL_RANDOM_HPP

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
 * \file   vexcl/backend/opencl/random.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Random generators for OpenCL.
 */

#include <vexcl/operations.hpp>
#include <boost/math/constants/constants.hpp>
#include <vexcl/backend/opencl/random/philox.hpp>
#include <vexcl/backend/opencl/random/threefry.hpp>


namespace vex {

/// A random generator.
/**
 * For integral types, generated values span the complete range.
 * For floating point types, generated values are >= 0 and <= 1.
 *
 * Uses Random123 generators which provide 64(2x32), 128(4x32, 2x64)
 * and 256(4x64) random bits, this limits the supported output types,
 * which means `cl_double8` (512bit) is not supported, but `cl_uchar2` is.
 *
 * \code
 * Random<cl_int> rand();
 * // Generate numbers from the same sequence
 * output1 = rand(element_index(), seed1);
 * output2 = rand(element_index(output1.size()), seed1);
 * // Generate a new sequence
 * output3 = rand(element_index(), seed2);
 * \endcode
 */
template <class T, class Generator = random::philox>
struct Random : UserFunction<Random<T, Generator>, T(cl_ulong, cl_ulong)> {
    // TODO: parameter should be same size as ctr_t
    // to allow using full range of the generator.
    typedef typename cl_scalar_of<T>::type Ts;

    static std::string body() {
        const size_t N = cl_vector_length<T>::value;

        std::ostringstream o;
        switch(sizeof(T)) {
            case 1: // 8bit = 2x32[0],lo,lo
            case 2: // 16bit = 2x32[0],lo
            case 4: // 32bit = 2x32[0]
            case 8: // 64bit = 2x32
                Generator::template macro<cl_uint2>(o, "rand");
                break;
            case 16: // 2x64bit = 4x32
                Generator::template macro<cl_uint4>(o, "rand");
                break;
            case 32: // 4x64bit = 4x64
                Generator::template macro<cl_ulong4>(o, "rand");
                break;
            default:
                precondition(false, "Unsupported random output type.");
        }
        o << "ctr_t ctr; ctr.even = prm1; ctr.odd = prm2;\n"
            "key_t key = 0x12345678;\n"
            "rand(ctr, key);\n"
            "#undef rand\n";

        if(std::is_same<Ts, cl_float>::value) {
            o << "return convert_" << type_name<T>() << "(as_"
              << type_name<typename cl_vector_of<cl_uint, N>::type>()
              << "(ctr";
            if(N == 1) o << ".s0";
            o << ")) / "
              << std::numeric_limits<cl_uint>::max()
              << ".0f;";
        } else if(std::is_same<Ts, cl_double>::value) {
            o << "return convert_" << type_name<T>() << "(as_"
              << type_name<typename cl_vector_of<cl_ulong, N>::type>()
              << "(ctr)) / "
              << std::numeric_limits<cl_ulong>::max()
              << ".0;";
        } else {
            o << "return *(" << type_name<T>() << "*)(void*)&ctr;";
        }
        return o.str();
    }
};


/// Returns normal distributed random numbers.
/**
 * \code
 * RandomNormal<cl_double2> rand();
 * output = mean + std_deviation * rand(element_index(), seed);
 * \endcode
 */
template <class T, class Generator = random::philox>
struct RandomNormal : UserFunction<RandomNormal<T,Generator>, T(cl_ulong, cl_ulong)> {
    typedef typename cl_scalar_of<T>::type Ts;
    static_assert(
            std::is_same<Ts, cl_float>::value ||
            std::is_same<Ts, cl_double>::value,
            "Must use float or double vector or scalar."
            );
    typedef typename cl_vector_of<Ts,2>::type T2;

    static std::string body() {
        const size_t N = cl_vector_length<T>::value;
        const bool is_float = std::is_same<Ts, cl_float>::value;

        std::ostringstream o;
        if(is_float)
            Generator::template macro<cl_uint2>(o, "rand");
        else
            Generator::template macro<cl_uint4>(o, "rand");

        o << "ctr_t ctr; ctr.even = prm1; ctr.odd = prm2;\n"
            "key_t key = 0x12345678;\n";
        o << type_name<T>() << " z;\n";

        for(size_t i = 0 ; i < N ; i += 2) {
            o << "rand(ctr, key); {\n";
            if(is_float)
                o << "float2 u = convert_float2(as_uint2(ctr)) / "
                    << std::numeric_limits<cl_uint>::max()
                    << ".0f;\n";
            else
                o << "double2 u = convert_double2(as_ulong2(ctr)) / "
                    << std::numeric_limits<cl_ulong>::max()
                    << ".0;\n";
            if(N == 1)
                o << "z = sqrt(-2 * log(u.s0)) * cospi(2 * u.s1);\n";
            else
                o << type_name<Ts>() << " l = sqrt(-2 * log(u.s0)),\n"
                    << "cs, sn = sincos(" << boost::math::constants::two_pi<double>()
                    << " * u.s1, &cs);\n"
                    << "z.s" << i << " = l * cs;\n"
                    << "z.s" << (i + 1) << " = l * sn;\n";

            o << "}\n";

        }
        o << "#undef rand\n"
            "return z;";
        return o.str();
    }
};




} // namespace vex



#endif
