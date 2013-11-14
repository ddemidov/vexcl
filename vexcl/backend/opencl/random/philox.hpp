#ifndef VEXCL_BACKEND_OPENCL_RANDOM_PHILOX_HPP
#define VEXCL_BACKEND_OPENCL_RANDOM_PHILOX_HPP

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
 * \file   vexcl/backend/opencl/random/philox.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Philox random generator.

Philox is an integer multiplication based, non cryptographic algorithm
for pseudorandom number generation from the Random123 suite,
see <http://www.deshawresearch.com/resources_random123.html>.

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

/// Random generators.
namespace random {


/// Philox random number generator.
/**
 * Philox is an integer multiplication based, non cryptographic algorithm
 * for pseudorandom number generation from the Random123 suite.
 * \see http://www.deshawresearch.com/resources_random123.html
 * \sa vex::Random
 * \sa vex::RandomNormal
 */
struct philox {
    /// generates a macro `philox(ctr, key)`
    /// modifies both inputs, uses the components of ctr for randomness.
    template <class T>
    static void macro(std::ostream &o, std::string name, size_t rounds = 10) {
        const size_t w = cl_vector_length<T>::value;
        static_assert(w == 2 || w == 4, "Only supports 2- and 4-vectors.");
        typedef typename cl_scalar_of<T>::type Ts;
        static_assert(std::is_same<Ts, cl_uint>::value || std::is_same<Ts, cl_ulong>::value,
            "Only supports 32 or 64 bit integers.");
        typedef T ctr_t;
        typedef typename cl_vector_of<Ts, w/2>::type key_t;

        o << "typedef " << type_name<ctr_t>() << " ctr_t;\n";
        o << "typedef " << type_name<key_t>() << " key_t;\n";

        // Define macro
        o << "#define " << name << "(ctr, key) {\\\n"
          << "ctr_t __mul;\\\n";
        o << type_name<Ts>() << " ";
        // constants
        if(std::is_same<Ts, cl_uint>::value) { // 32
            o << "W0 = 0x9E3779B9, ";
            if(w == 2)
                o << "M0 = 0xD256D193;\\\n";
            else
                o << "W1 = 0xBB67AE85, "
                    "M0 = 0xD2511F53, "
                    "M1 = 0xCD9E8D57;\\\n";
        } else { // 64
            o << "W0 = 0x9E3779B97F4A7C15, "; // golden ratio
            if(w == 2)
                o << "M0 = 0xD2B74407B1CE6E93;\\\n";
            else
                o << "M0 = 0xD2E7470EE14C6C93, "
                    "M1 = 0xCA5A826395121157, "
                    "W1 = 0xBB67AE8584CAA73B;\\\n"; // sqrt(3)-1
        }

        for(size_t round = 0 ; round < rounds ; round++) {
            if(round > 0) { // bump key
                if(w == 2)
                    o << "key += W0;\\\n";
                else
                    o << "key.s0 += W0;"
                        " key.s1 += W1;\\\n";
            }
            // next round
            if(w == 2)
                o << "__mul.s0 = mul_hi(M0, ctr.s0);"
                    " __mul.s1 = M0 * ctr.s0;"
                    " ctr.s0 = __mul.s0 ^ key ^ ctr.s1;"
                    " ctr.s1 = __mul.s1;\\\n";
            else
                o << "__mul.s0 = mul_hi(M0, ctr.s0);"
                    " __mul.s1 = M0 * ctr.s0;"
                    " __mul.s2 = mul_hi(M1, ctr.s2);"
                    " __mul.s3 = M1 * ctr.s2;"
                    " ctr.s0 = __mul.s2 ^ ctr.s1 ^ key.s0;"
                    " ctr.s1 = __mul.s3;"
                    " ctr.s2 = __mul.s0 ^ ctr.s3 ^ key.s1;"
                    " ctr.s3 = __mul.s1;\\\n";
        }
        o << "}\n";
    }
};


} // namespace random
} // namespace vex

#endif
