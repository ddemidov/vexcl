#ifndef VEXCL_BACKEND_OPENCL_RANDOM_THREEFRY_HPP
#define VEXCL_BACKEND_OPENCL_RANDOM_THREEFRY_HPP

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
 * \file   vexcl/backend/opencl/random/threefry.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Threefry random generator.

Threefry, based on the Threefish cipher, is a non cryptographic algorithm
for pseudorandom number generation from the Random123 suite,
see <http://www.deshawresearch.com/resources_random123.html>

The original code came with the following copyright notice:

\verbatim
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\endverbatim
*/

namespace vex {
namespace random {


/// Threefry random number generator.
/**
 * Threefry, based on the Threefish cipher, is a non cryptographic algorithm
 * for pseudorandom number generation from the Random123 suite.
 * \see http://www.deshawresearch.com/resources_random123.html
 * \sa vex::Random
 * \sa vex::RandomNormal
 */
struct threefry {
    // print the rotation table for WxN
    template <size_t n>
    static void table(std::ostream &o, const int r[n]) {
        for(size_t i = 0 ; i < n ; i++) {
            if(i > 0) o << ", ";
            o << "R" << i << " = " << r[i];
        }
    }

    template <size_t bits, size_t w>
    static void rotation_table(std::ostream &o) {
        static const int r2x32[8] = {13, 15, 26,  6, 17, 29, 16, 24};
        static const int r4x32[16] = {10, 26, 11, 21, 13, 27, 23,  5,
             6, 20, 17, 11, 25, 10, 18, 20};
        static const int r2x64[8] = {16, 42, 12, 31, 16, 32, 24, 21};
        static const int r4x64[16] = {14, 16, 52, 57, 23, 40,  5, 37,
            25, 33, 46, 12, 58, 22, 32, 32};
        if(w == 2) table<8>(o, bits == 32 ? r2x32 : r2x64);
        else table<16>(o, bits == 32 ? r4x32 : r4x64);
    }

    // Generates a `name(ctr, key)` macro.
    // `ctr` will be modified, containing the random output.
    // `key` will be preserved.
    template <class T>
    static void macro(std::ostream &o, std::string name, size_t rounds = 20) {
        const size_t w = cl_vector_length<T>::value;
        static_assert(w == 2 || w == 4, "Only supports 2- and 4-vectors.");
        typedef typename cl_scalar_of<T>::type Ts;
        static_assert(std::is_same<Ts, cl_uint>::value || std::is_same<Ts, cl_ulong>::value,
            "Only supports 32 or 64 bit integers.");
        const size_t bits = sizeof(Ts) * 8;

        o << "typedef " << type_name<T>() << " ctr_t;\n";
        o << "typedef ctr_t key_t;\n";

        o << "#define " << name << "(ctr, key) {\\\n";
        o << "const " << type_name<Ts>() << " ";
        rotation_table<bits, w>(o);
        o << ", p = "
          << (bits == 32 ? "0x1BD11BDA" : "0x1BD11BDAA9FC1A22");
        for(size_t i = 0 ; i < w ; i++)
            o << " ^ key.s" << i;
        o << ";\\\n";
        // Insert initial key before round 0
        for(size_t i = 0 ; i < w ; i++)
            o << "ctr.s" << i << " += key.s" << i << ";\\\n";
        for(size_t round = 0 ; round < rounds ; round++) {
            // do rounds
            if(w == 2)
                o << "ctr.s0 += ctr.s1; "
                  << "ctr.s1 = rotate(ctr.s1, R" << (round % 8) << "); "
                  << "ctr.s1 ^= ctr.s0;\\\n";
            else {
                const size_t r = 2 * (round % 8),
                    r0 = r + (round % 2),
                    r1 = r + ((round + 1) % 2);
                o << "ctr.s0 += ctr.s1; "
                  << "ctr.s1 = rotate(ctr.s1, R" << r0 << "); "
                  << "ctr.s1 ^= ctr.s0;\\\n"
                  << "ctr.s2 += ctr.s3; "
                  << "ctr.s3 = rotate(ctr.s3, R" << r1 << "); "
                  << "ctr.s3 ^= ctr.s2;\\\n";
            }
            // inject key
            if((round + 1) % 4 == 0) {
                const size_t j = round / 4 + 1;
                for(size_t i = 0 ; i < w ; i++) {
                    const size_t ii = ((j + i) % (w + 1));
                    o << "ctr.s" << i << " += ";
                    if(ii == w) o << "p; ";
                    else o << "key.s" << ii << "; ";
                }
                o << "ctr.s" << (w - 1) << " += " << j << ";\\\n";
            }
        }
        o << "}\n";
    }
};


} // namespace random
} // namespace vex

#endif
