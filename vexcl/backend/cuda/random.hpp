#ifndef VEXCL_BACKEND_CUDA_RANDOM_HPP
#define VEXCL_BACKEND_CUDA_RANDOM_HPP

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
 * \file   vexcl/backend/cuda/random.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Random generators for CUDA.
 */

#include <vexcl/operations.hpp>
#include <boost/math/constants/constants.hpp>
#include <vexcl/backend/cuda/random/philox.hpp>
#include <vexcl/backend/cuda/random/threefry.hpp>


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
        static_assert(N <= 4, "Unsupported vector type dimension");

        size_t ctr_size;
        size_t key_size;

        std::ostringstream o;
        switch(sizeof(T)) {
            case 1: // 8bit = 2x32[0],lo,lo
            case 2: // 16bit = 2x32[0],lo
            case 4: // 32bit = 2x32[0]
            case 8: // 64bit = 2x32
                Generator::template macro<cl_uint2>(o, "rand");
                ctr_size = Generator::template ctr_size<cl_uint2>();
                key_size = Generator::template key_size<cl_uint2>();
                break;
            case 16: // 2x64bit = 4x32
                Generator::template macro<cl_uint4>(o, "rand");
                ctr_size = Generator::template ctr_size<cl_uint4>();
                key_size = Generator::template key_size<cl_uint4>();
                break;
            case 32: // 4x64bit = 4x64
                Generator::template macro<cl_ulong4>(o, "rand");
                ctr_size = Generator::template ctr_size<cl_ulong4>();
                key_size = Generator::template key_size<cl_ulong4>();
                break;
            default:
                precondition(false, "Unsupported random output type.");
        }
        o << "ctr_t ctr;\n";
        for(unsigned i = 0; i < ctr_size; i += 2)
            o << "ctr[" << i << "] = prm1;\n"
                 "ctr[" << i + 1 << "] = prm2;\n";

        o << "key_t key;\n";
        for(unsigned i = 0; i < key_size; ++i)
            o << "key[" << i << "] = 0x12345678;\n";

        o << "rand(ctr, key);\n"
            "#undef rand\n";

        if(std::is_same<Ts, cl_float>::value) {
            if (N == 1) {
                o << "return (" << type_name<T>() << ")(*(uint*)(void*)ctr) / " << std::numeric_limits<cl_uint>::max() << ".0f;";
            } else {
                char dim[] = {'x', 'y', 'z', 'w'};
                o << type_name<T>() << " res;\n";
                for(size_t i = 0; i < N; ++i) {
                    o << "res." << dim[i] << " = (" << type_name<Ts>() << ")(((uint*)(void*)ctr)[" << i << "]) / " << std::numeric_limits<cl_uint>::max() << ".0f;\n";
                }
                o << "return res;";
            }
        } else if(std::is_same<Ts, cl_double>::value) {
            if (N == 1) {
                o << "return (" << type_name<T>() << ")(*(ulong*)(void*)ctr) / " << std::numeric_limits<cl_ulong>::max() << ".0;";
            } else {
                const char dim[] = {'x', 'y', 'z', 'w'};
                o << type_name<T>() << " res;\n";
                for(size_t i = 0; i < N; ++i) {
                    o << "res." << dim[i] << " = (" << type_name<Ts>() << ")(((ulong*)(void*)ctr)[" << i << "]) / " << std::numeric_limits<cl_ulong>::max() << ".0;\n";
                }
                o << "return res;";
            }
        } else {
            o << "return *(" << type_name<T>() << "*)(void*)ctr;";
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

        size_t ctr_size;
        size_t key_size;

        std::ostringstream o;
        if(is_float) {
            Generator::template macro<cl_uint2>(o, "rand");
            ctr_size = Generator::template ctr_size<cl_uint2>();
            key_size = Generator::template key_size<cl_uint2>();
        } else {
            Generator::template macro<cl_uint4>(o, "rand");
            ctr_size = Generator::template ctr_size<cl_uint4>();
            key_size = Generator::template key_size<cl_uint4>();
        }

        o << "ctr_t ctr;\n";
        for(unsigned i = 0; i < ctr_size; i += 2)
            o << "ctr[" << i << "] = prm1;\n"
                 "ctr[" << i + 1 << "] = prm2;\n";

        o << "key_t key;\n";
        for(unsigned i = 0; i < key_size; ++i)
            o << "key[" << i << "] = 0x12345678;\n";

        o << type_name<T>() << " z;\n";

        for(size_t i = 0 ; i < N ; i += 2) {
            o << "rand(ctr, key); {\n";
            if(is_float) {
                o << "float u0 = (float)(((uint*)(void*)ctr)[0]) / "
                    << std::numeric_limits<cl_uint>::max() << ".0f;\n";
                o << "float u1 = (float)(((uint*)(void*)ctr)[1]) / "
                    << std::numeric_limits<cl_uint>::max() << ".0f;\n";
            } else {
                o << "double u0 = (double)(((ulong*)(void*)ctr)[0]) / "
                    << std::numeric_limits<cl_ulong>::max() << ".0;\n";
                o << "double u1 = (double)(((ulong*)(void*)ctr)[1]) / "
                    << std::numeric_limits<cl_ulong>::max() << ".0;\n";
            }
            if(N == 1)
                o << "z = sqrt(-2 * log(u0)) * cospi(2 * u1);\n";
            else {
                const char dim[] = {'x', 'y', 'z', 'w'};
                o << type_name<Ts>() << " l = sqrt(-2 * log(u0)),\n"
                    << "cs, sn;\nsincos(" << boost::math::constants::two_pi<double>()
                    << " * u1, &sn, &cs);\n"
                    << "z." << dim[i] << " = l * cs;\n"
                    << "z." << dim[i+1] << " = l * sn;\n";
            }

            o << "}\n";

        }
        o << "#undef rand\n"
            "return z;";
        return o.str();
    }
};




} // namespace vex



#endif
