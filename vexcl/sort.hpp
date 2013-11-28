#ifndef VEXCL_SORT_HPP
#define VEXCL_SORT_HPP

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
 * \file   vexcl/sort.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Sorting algorithms.

Adopted from NVIDIA Modern GPU patterns,
see <http://nvlabs.github.io/moderngpu>.
The original code came with the following copyright notice:

\verbatim
Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\endverbatim
*/

#include <string>
#include <functional>

#include <vexcl/backend.hpp>
#include <vexcl/util.hpp>
#include <vexcl/vector.hpp>

namespace vex {
namespace detail {

//---------------------------------------------------------------------------
// Memory transfer functions
//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string global_to_regstr_pred() {
    std::ostringstream s;
    s << "global_to_regstr_pred_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void global_to_regstr_pred(backend::source_generator &src) {
    src.function<void>( global_to_regstr_pred<NT,VT,T>() )
        .open("(")
            .template parameter< int                 >("count")
            .template parameter< global_ptr<const T> >("data")
            .template parameter< int                 >("tid")
            .template parameter< regstr_ptr<T>       >("reg")
        .close(")").open("{");

    src.new_line() << type_name<int>() << " index;";

    for(int i = 0; i < VT; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < count) reg[" << i << "] = data[index];";
    }
    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string global_to_regstr() {
    std::ostringstream s;
    s << "global_to_regstr_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void global_to_regstr(backend::source_generator &src) {
    global_to_regstr_pred<NT, VT, T>(src);

    src.function<void>( global_to_regstr<NT,VT,T>() )
        .open("(")
            .template parameter< int                 >("count")
            .template parameter< global_ptr<const T> >("data")
            .template parameter< int                 >("tid")
            .template parameter< regstr_ptr<T>       >("reg")
        .close(")").open("{");

    src.new_line() << "if (count >= " << NT * VT << ")";
    src.open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "reg[" << i << "] = data[" << NT * i << " + tid];";

    src.close("}") << " else "
        << global_to_regstr_pred<NT, VT, T>() << "(count, data, tid, reg);";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string regstr_to_global() {
    std::ostringstream s;
    s << "regstr_to_global_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void regstr_to_global(backend::source_generator &src) {
    src.function<void>( regstr_to_global<NT,VT,T>() )
        .open("(")
            .template parameter< int                 >("count")
            .template parameter< regstr_ptr<const T> >("reg")
            .template parameter< int                 >("tid")
            .template parameter< global_ptr<T>       >("dest")
        .close(")").open("{");

    src.new_line() << type_name<int>() << " index;";

    for(int i = 0; i < VT; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < count) dest[index] = reg[" << i << "];";
    }

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string shared_to_regstr() {
    std::ostringstream s;
    s << "shared_to_regstr_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void shared_to_regstr(backend::source_generator &src) {
    src.function<void>( shared_to_regstr<NT,VT,T>() )
        .open("(")
            .template parameter< shared_ptr<const T> >("data")
            .template parameter< int                 >("tid")
            .template parameter< regstr_ptr<T>       >("reg")
        .close(")").open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "reg[" << i << "] = data[" << NT * i << " + tid];";

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string regstr_to_shared() {
    std::ostringstream s;
    s << "regstr_to_shared_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void regstr_to_shared(backend::source_generator &src) {
    src.function<void>( regstr_to_shared<NT,VT,T>() )
        .open("(")
            .template parameter< regstr_ptr<const T> >("reg")
            .template parameter< int                 >("tid")
            .template parameter< shared_ptr<T>       >("dest")
        .close(")").open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "dest[" << NT * i << " + tid] = reg[" << i << "];";

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string global_to_shared() {
    std::ostringstream s;
    s << "global_to_shared_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void global_to_shared(backend::source_generator &src) {
    src.function<void>( global_to_shared<NT,VT,T>() )
        .open("(")
            .template parameter< int                 >("count")
            .template parameter< global_ptr<const T> >("source")
            .template parameter< int                 >("tid")
            .template parameter< shared_ptr<T>       >("dest")
        .close(")").open("{");

    src.new_line() << type_name<T>() << " reg[" << VT << "];";
    src.new_line() << global_to_regstr<NT, VT, T>() << "(count, source, tid, reg);";
    src.new_line() << regstr_to_shared<NT, VT, T>() << "(reg, tid, dest);";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string shared_to_global() {
    std::ostringstream s;
    s << "shared_to_global_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void shared_to_global(backend::source_generator &src) {
    src.function<void>( shared_to_global<NT,VT,T>() )
        .open("(")
            .template parameter< int                 >("count")
            .template parameter< shared_ptr<const T> >("source")
            .template parameter< int                 >("tid")
            .template parameter< global_ptr<T>       >("dest")
        .close(")").open("{");

    src.new_line() << type_name<int>() << " index;";

    for(int i = 0; i < VT; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < count) dest[index] = source[index];";
    }

    src.new_line().barrier();
    src.close("}");
}

//---------------------------------------------------------------------------
template<int VT, typename T>
std::string shared_to_thread() {
    std::ostringstream s;
    s << "shared_to_thread_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int VT, typename T>
void shared_to_thread(backend::source_generator &src) {
    src.function<void>( shared_to_thread<VT,T>() )
        .open("(")
            .template parameter< shared_ptr<const T> >("data")
            .template parameter< int                 >("tid")
            .template parameter< regstr_ptr<T>       >("reg")
        .close(")").open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "reg[" << i << "] = data[" << VT << " * tid + " << i << "];";

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int VT, typename T>
std::string thread_to_shared() {
    std::ostringstream s;
    s << "thread_to_shared_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int VT, typename T>
void thread_to_shared(backend::source_generator &src) {
    src.function<void>( thread_to_shared<VT,T>() )
        .open("(")
            .template parameter< regstr_ptr<const T> >("reg")
            .template parameter< int                 >("tid")
            .template parameter< shared_ptr<T>       >("dest")
        .close(")").open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "dest[" << VT << " * tid + " << i << "] = reg[" << i << "];";

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
void transfer_functions(backend::source_generator &src) {
    global_to_regstr<NT, VT, T>(src);
    regstr_to_global<NT, VT, T>(src);

    shared_to_regstr<NT, VT, T>(src);
    regstr_to_shared<NT, VT, T>(src);

    global_to_shared<NT, VT, T>(src);
    shared_to_global<NT, VT, T>(src);

    shared_to_thread<VT, T>(src);
    thread_to_shared<VT, T>(src);
}

//---------------------------------------------------------------------------
// Block merge sort kernel
//---------------------------------------------------------------------------
template <typename T>
std::string swap_function() {
    std::ostringstream s;
    s << "swap_" << type_name<T>();
    return s.str();
}

template <typename T>
void swap_function(backend::source_generator &src) {
    src.function<void>(swap_function<T>())
        .open("(")
            .template parameter< regstr_ptr<T> >("a")
            .template parameter< regstr_ptr<T> >("b")
        .close(")").open("{");

    src.new_line() << type_name<T>() << " c = *a;";
    src.new_line() << "*a = *b;";
    src.new_line() << "*b = c;";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int VT, typename K, typename V, bool HasValues>
std::string odd_even_transpose_sort() {
    std::ostringstream s;
    s << "odd_even_transpose_sort_" << VT << "_" << type_name<K>();
    if (HasValues) s << "_" << type_name<V>();
    return s.str();
}

template<int VT, typename K, typename V, bool HasValues>
void odd_even_transpose_sort(backend::source_generator &src) {
    swap_function<K>(src);
    if (HasValues && !std::is_same<K, V>::value) swap_function<V>(src);

    src.function<void>(odd_even_transpose_sort<VT,K,V,HasValues>());
    src.open("(");
    src.template parameter< regstr_ptr<K> >("keys");
    if (HasValues)
        src.template parameter< regstr_ptr<V> >("vals");
    src.close(")").open("{");

    for(int I = 0; I < VT; ++I) {
        for(int i = 1 & I; i < VT - 1; i += 2) {
            src.new_line() << "if (comp(keys[" << i + 1 << "], keys[" << i << "]))";
            src.open("{");
            src.new_line() << swap_function<K>() << "(keys + " << i << ", keys + " << i + 1 << ");";
            if (HasValues)
                src.new_line() << swap_function<V>() << "(vals + " << i << ", vals + " << i + 1 << ");";
            src.close("}");
        }
    }

    src.close("}");
}

//---------------------------------------------------------------------------
template<typename T>
std::string merge_path() {
    typedef typename std::decay<typename vex::remove_ptr<T>::type>::type K;

    std::ostringstream s;
    s << "merge_path_" << type_name<K>();
    return s.str();
}

template<typename T>
void merge_path(backend::source_generator &src) {
    src.function<int>(merge_path<T>())
        .open("(")
            .template parameter< T   >("a")
            .template parameter< int >("a_count")
            .template parameter< T   >("b")
            .template parameter< int >("b_count")
            .template parameter< int >("diag")
        .close(")").open("{");

    typedef typename std::decay<typename vex::remove_ptr<T>::type>::type K;

    src.new_line() << "int begin = max(0, diag - b_count);";
    src.new_line() << "int end   = min(diag, a_count);";

    src.new_line() << "while (begin < end)";
    src.open("{");
    src.new_line() << "int mid = (begin + end) >> 1;";
    src.new_line() << type_name<K>() << " a_key = a[mid];";
    src.new_line() << type_name<K>() << " b_key = b[diag - 1 - mid];";
    src.new_line() << "if ( !comp(b_key, a_key) ) begin = mid + 1;";
    src.new_line() << "else end = mid;";
    src.close("}");

    src.new_line() << "return begin;";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int VT, typename T>
std::string serial_merge() {
    std::ostringstream s;
    s << "serial_merge_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int VT, typename T>
void serial_merge(backend::source_generator &src) {
    src.function<void>(serial_merge<VT, T>())
        .open("(")
            .template parameter< shared_ptr<const T> >("keys_shared")
            .template parameter< int                 >("a_begin")
            .template parameter< int                 >("a_end")
            .template parameter< int                 >("b_begin")
            .template parameter< int                 >("b_end")
            .template parameter< regstr_ptr<T>       >("results")
            .template parameter< regstr_ptr<int>     >("indices")
        .close(")").open("{");

    src.new_line() << type_name<T>() << " a_key = keys_shared[a_begin];";
    src.new_line() << type_name<T>() << " b_key = keys_shared[b_begin];";
    src.new_line() << "bool p;";

    for(int i = 0; i < VT; ++i) {
        src.new_line() << "p = (b_begin >= b_end) || ((a_begin < a_end) && !comp(b_key, a_key));";

        src.new_line() << "results[" << i << "] = p ? a_key : b_key;";
        src.new_line() << "indices[" << i << "] = p ? a_begin : b_begin;";

        src.new_line() << "if(p) a_key = keys_shared[++a_begin];";
        src.new_line() << "else  b_key = keys_shared[++b_begin];";
    }

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string block_sort_pass() {
    std::ostringstream s;
    s << "block_sort_pass_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void block_sort_pass(backend::source_generator &src) {
    merge_path< shared_ptr<const T> >(src);

    src.function<void>(block_sort_pass<NT, VT, T>())
        .open("(")
            .template parameter< shared_ptr<const T> >("keys_shared")
            .template parameter< int                 >("tid")
            .template parameter< int                 >("count")
            .template parameter< int                 >("coop")
            .template parameter< regstr_ptr<T>       >("keys")
            .template parameter< regstr_ptr<int>     >("indices")
        .close(")").open("{");

    src.new_line() << "int list = ~(coop - 1) & tid;";
    src.new_line() << "int diag = min(count, " << VT << " * ((coop - 1) & tid));";
    src.new_line() << "int start = " << VT << " * list;";
    src.new_line() << "int a0 = min(count, start);";
    src.new_line() << "int b0 = min(count, start + " << VT << " * (coop / 2));";
    src.new_line() << "int b1 = min(count, start + " << VT << " * coop);";

    src.new_line() << "int p = " << merge_path< shared_ptr<const T> >() << "(keys_shared + a0, b0 - a0, keys_shared + b0, b1 - b0, diag);";
    src.new_line() << serial_merge<VT, T>() << "(keys_shared, a0 + p, b0, b0 + diag - p, b1, keys, indices);";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename T>
std::string gather() {
    std::ostringstream s;
    s << "gather_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT, typename T>
void gather(backend::source_generator &src) {
    src.function<void>(gather<NT, VT, T>())
        .open("(")
            .template parameter< shared_ptr<const T>   >("data")
            .template parameter< regstr_ptr<const int> >("indices")
            .template parameter< int                   >("tid")
            .template parameter< regstr_ptr<T>         >("reg")
        .close(")").open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "reg[" << i << "] = data[indices[" << i << "]];";

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename K, typename V, bool HasValues>
std::string block_sort_loop() {
    std::ostringstream s;
    s << "block_sort_loop_" << NT << "_" << VT << "_" << type_name<K>();
    if (HasValues) s << "_" << type_name<V>();
    return s.str();
}

template<int NT, int VT, typename K, typename V, bool HasValues>
void block_sort_loop(backend::source_generator &src) {
    block_sort_pass<NT, VT, K>(src);
    if (HasValues) gather<NT, VT, V>(src);

    src.function<void>(block_sort_loop<NT, VT, K, V, HasValues>());
    src.open("(");
    src.template parameter< shared_ptr<K> >("keys_shared");
    if (HasValues) {
        src.template parameter< regstr_ptr<V> >("thread_vals");
        src.template parameter< shared_ptr<V> >("vals_shared");
    }
    src.template parameter< int           >("tid");
    src.template parameter< int           >("count");
    src.close(")").open("{");

    src.new_line() << "int indices[" << VT << "];";
    src.new_line() << type_name<K>() << " keys[" << VT << "];";

    for(int coop = 2; coop <= NT; coop *= 2) {
        src.new_line() << block_sort_pass<NT, VT, K>()
            << "(keys_shared, tid, count, " << coop << ", keys, indices);";

        if (HasValues) {
            // Exchange the values through shared memory.
            src.new_line() << thread_to_shared<VT, V>()
                << "(thread_vals, tid, vals_shared);";
            src.new_line() << gather<NT, VT, V>()
                << "(vals_shared, indices, tid, thread_vals);";
        }

        // Store results in shared memory in sorted order.
        src.new_line() << thread_to_shared<VT, K>()
            << "(keys, tid, keys_shared);";
    }

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename K, typename V, bool HasValues>
std::string mergesort() {
    std::ostringstream s;
    s << "mergesort_" << NT << "_" << VT << "_" << type_name<K>();
    if (HasValues) s << "_" << type_name<V>();
    return s.str();
}

template<int NT, int VT, typename K, typename V, bool HasValues>
void mergesort(backend::source_generator &src) {
    odd_even_transpose_sort<VT, K, V, HasValues>(src);
    block_sort_loop<NT, VT, K, V, HasValues>(src);

    src.function<void>(mergesort<NT, VT, K, V, HasValues>());
    src.open("(");
    src.template parameter< regstr_ptr<K> >("thread_keys");
    src.template parameter< shared_ptr<K> >("keys_shared");
    if (HasValues) {
        src.template parameter< regstr_ptr<V> >("thread_vals");
        src.template parameter< shared_ptr<V> >("vals_shared");
    }
    src.template parameter< int           >("count");
    src.template parameter< int           >("tid");
    src.close(")").open("{");

    // Stable sort the keys in the thread.
    src.new_line() << "if(" << VT << " * tid < count) "
        << odd_even_transpose_sort<VT, K, V, HasValues>()
        << "(thread_keys"
        << (HasValues ? ", thread_vals);" : ");");

    // Store the locally sorted keys into shared memory.
    src.new_line() << thread_to_shared<VT, K>()
        << "(thread_keys, tid, keys_shared);";

    // Recursively merge lists until the entire CTA is sorted.
    src.new_line() << block_sort_loop<NT, VT, K, V, HasValues>()
        << "(keys_shared, "
        << (HasValues ? "thread_vals, vals_shared, " : "")
        << "tid, count);";

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, int VT, typename K, typename V, typename Comp, bool HasValues>
backend::kernel& block_sort_kernel(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Comp::define(src, "comp");

        transfer_functions<NT, VT, int>(src);

        if (!std::is_same<K, int>::value)
            transfer_functions<NT, VT, K>(src);

        if (HasValues && !std::is_same<V, K>::value && !std::is_same<V, int>::value)
            transfer_functions<NT, VT, V>(src);

        serial_merge<VT, K>(src);
        mergesort<NT, VT, K, V, HasValues>(src);

        src.kernel("block_sort");
        src.open("(");
        src.template parameter< int                 >("count");
        src.template parameter< global_ptr<const K> >("keys_src");
        src.template parameter< global_ptr<      K> >("keys_dst");
        if (HasValues) {
            src.template parameter< global_ptr<const V> >("vals_src");
            src.template parameter< global_ptr<      V> >("vals_dst");
        }
        src.close(")").open("{");

        const int NV = NT * VT;

        src.new_line() << "union Shared";
        src.open("{");
        src.new_line() << type_name<K>() << " keys[" << NT * (VT + 1) << "];";
        if (HasValues)
            src.new_line() << type_name<V>() << " vals[" << NV << "];";
        src.close("};");

        src.smem_static_var("union Shared", "shared");

        src.new_line() << "int tid    = " << src.local_id(0) << ";";
        src.new_line() << "int block  = " << src.group_id(0) << ";";
        src.new_line() << "int gid    = " << NV << " * block;";
        src.new_line() << "int count2 = min(" << NV << ", count - gid);";

        // Load the values into thread order.
        if (HasValues) {
            src.new_line() << type_name<V>() << " thread_vals[" << VT << "];";
            src.new_line() << global_to_shared<NT, VT, V>()
                << "(count2, vals_src + gid, tid, shared.vals);";
            src.new_line() << shared_to_thread<VT, V>()
                << "(shared.vals, tid, thread_vals);";
        }

        // Load keys into shared memory and transpose into register in thread order.
        src.new_line() << type_name<K>() << " thread_keys[" << VT << "];";
        src.new_line() << global_to_shared<NT, VT, K>()
            << "(count2, keys_src + gid, tid, shared.keys);";
        src.new_line() << shared_to_thread<VT, K>()
            << "(shared.keys, tid, thread_keys);";

        // If we're in the last tile, set the uninitialized keys for the thread with
        // a partial number of keys.
        src.new_line() << "int first = " << VT << " * tid;";
        src.new_line() << "if(first + " << VT << " > count2 && first < count2)";
        src.open("{");

        src.new_line() << type_name<K>() << " max_key = thread_keys[0];";

        for(int i = 1; i < VT; ++i)
            src.new_line()
                << "if(first + " << i << " < count2)"
                << " max_key = comp(max_key, thread_keys[" << i << "])"
                << " ? thread_keys[" << i << "] : max_key;";

        // Fill in the uninitialized elements with max key.
        for(int i = 0; i < VT; ++i)
            src.new_line()
                << "if(first + " << i << " >= count2)"
                << " thread_keys[" << i << "] = max_key;";

        src.close("}");

        src.new_line() << mergesort<NT, VT, K, V, HasValues>()
            << "(thread_keys, shared.keys, "
            << (HasValues ? "thread_vals, shared.vals, " : "")
            << "count2, tid);";

        // Store the sorted keys to global.
        src.new_line() << shared_to_global<NT, VT, K>()
            << "(count2, shared.keys, tid, keys_dst + gid);";

        if (HasValues) {
            src.new_line() << thread_to_shared<VT, V>()
                << "(thread_vals, tid, shared.vals);";
            src.new_line() << shared_to_global<NT, VT, V>()
                << "(count2, shared.vals, tid, vals_dst + gid);";
        }

        src.close("}");

        backend::kernel krn(queue, src.str(), "block_sort");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
// Merge partition kernel
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
inline void find_mergesort_frame(backend::source_generator &src) {
    src.function<cl_int3>("find_mergesort_frame")
        .open("(")
        .parameter<int>("coop")
        .parameter<int>("block")
        .parameter<int>("nv")
        .close(")").open("{");

    src.new_line() << "int start = ~(coop - 1) & block;";
    src.new_line() << "int size = nv * (coop>> 1);";

    src.new_line() << type_name<cl_int3>() << " frame;";
    src.new_line() << "frame.x = nv * start;";
    src.new_line() << "frame.y = nv * start + size;";
    src.new_line() << "frame.z = size;";
    src.new_line() << "return frame;";

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, typename T, typename Comp>
backend::kernel merge_partition_kernel(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Comp::define(src, "comp");
        merge_path< global_ptr<const T> >(src);
        find_mergesort_frame(src);

        src.kernel("merge_partition")
            .open("(")
                .template parameter< global_ptr<const T> >("a_global")
                .template parameter< int                 >("a_count")
                .template parameter< global_ptr<const T> >("b_global")
                .template parameter< int                 >("b_count")
                .template parameter< int                 >("nv")
                .template parameter< int                 >("coop")
                .template parameter< global_ptr<int>     >("mp_global")
                .template parameter< int                 >("num_searches")
            .close(")").open("{");

        src.new_line() << "int partition = " << src.global_id(0) << ";";
        src.new_line() << "if (partition < num_searches)";
        src.open("{");
        src.new_line() << "int a0 = 0, b0 = 0;";
        src.new_line() << "int gid = nv * partition;";

        src.new_line() << "if(coop)";
        src.open("{");
        src.new_line() << type_name<cl_int3>() << " frame = find_mergesort_frame(coop, partition, nv);";
        src.new_line() << "a0 = frame.x;";
        src.new_line() << "b0 = min(a_count, frame.y);";
        src.new_line() << "b_count = min(a_count, frame.y + frame.z) - b0;";
        src.new_line() << "a_count = min(a_count, frame.x + frame.z) - a0;";

        // Put the cross-diagonal into the coordinate system of the input lists.
        src.new_line() << "gid -= a0;";
        src.close("}");

        src.new_line() << "int mp = " << merge_path< global_ptr<const T> >()
            << "(a_global + a0, a_count, b_global + b0, b_count, min(gid, a_count + b_count));";
        src.new_line() << "mp_global[partition] = mp;";

        src.close("}");

        src.close("}");

        backend::kernel krn(queue, src.str(), "merge_partition");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
template <typename Comp, typename T>
backend::device_vector<int> merge_path_partitions(
        const backend::command_queue &queue,
        const device_vector<T> &keys,
        int count, int nv, int coop
        )
{
    const int NT = 64;

    int num_partitions       = (count + nv - 1) / nv;
    int num_partition_blocks = (num_partitions + NT) / NT;

    backend::device_vector<int> partitions(queue, num_partitions + 1);

    auto merge_partition = merge_partition_kernel<NT, T, Comp>(queue);

    int a_count = keys.size();
    int b_count = 0;

    merge_partition.push_arg(keys);
    merge_partition.push_arg(a_count);
    merge_partition.push_arg(keys);
    merge_partition.push_arg(b_count);
    merge_partition.push_arg(nv);
    merge_partition.push_arg(coop);
    merge_partition.push_arg(partitions);
    merge_partition.push_arg(num_partitions + 1);

    merge_partition.config(num_partition_blocks, NT);

    merge_partition(queue);

    return partitions;
}

//---------------------------------------------------------------------------
// Merge kernel
//---------------------------------------------------------------------------
inline void find_mergesort_interval(backend::source_generator &src) {
    src.function<cl_int4>("find_mergesort_interval")
        .open("(")
            .parameter< cl_int3 >("frame")
            .parameter< int     >("coop")
            .parameter< int     >("block")
            .parameter< int     >("nv")
            .parameter< int     >("count")
            .parameter< int     >("mp0")
            .parameter< int     >("mp1")
        .close(")").open("{");

    // Locate diag from the start of the A sublist.
    src.new_line() << "int diag = nv * block - frame.x;";
    src.new_line() << "int4 interval;";
    src.new_line() << "interval.x = frame.x + mp0;";
    src.new_line() << "interval.y = min(count, frame.x + mp1);";
    src.new_line() << "interval.z = min(count, frame.y + diag - mp0);";
    src.new_line() << "interval.w = min(count, frame.y + diag + nv - mp1);";

    // The end partition of the last block for each merge operation is computed
    // and stored as the begin partition for the subsequent merge. i.e. it is
    // the same partition but in the wrong coordinate system, so its 0 when it
    // should be listSize. Correct that by checking if this is the last block
    // in this merge operation.
    src.new_line() << "if(coop - 1 == ((coop - 1) & block))";
    src.open("{");
    src.new_line() << "interval.y = min(count, frame.x + frame.z);";
    src.new_line() << "interval.w = min(count, frame.y + frame.z);";
    src.close("}");

    src.new_line() << "return interval;";

    src.close("}");
}

//---------------------------------------------------------------------------
inline void compute_merge_range(backend::source_generator &src) {
    find_mergesort_frame(src);
    find_mergesort_interval(src);

    src.function<cl_int4>("compute_merge_range")
        .open("(")
            .parameter< int >("a_count")
            .parameter< int >("b_count")
            .parameter< int >("block")
            .parameter< int >("coop")
            .parameter< int >("nv")
            .parameter< global_ptr<const int> >("mp_global")
        .close(")").open("{");

    // Load the merge paths computed by the partitioning kernel.
    src.new_line() << "int mp0 = mp_global[block];";
    src.new_line() << "int mp1 = mp_global[block + 1];";
    src.new_line() << "int gid = nv * block;";

    // Compute the ranges of the sources in global memory.
    src.new_line() << "int4 range;";
    src.new_line() << "if(coop)";
    src.open("{");
    src.new_line() << type_name<cl_int3>() << " frame = find_mergesort_frame(coop, block, nv);";
    src.new_line() << "range = find_mergesort_interval(frame, coop, block, nv, a_count, mp0, mp1);";
    src.close("}");
    src.new_line() << "else";
    src.open("{");
    src.new_line() << "range.x = mp0;";
    src.new_line() << "range.y = mp1;";
    src.new_line() << "range.z = gid - range.x;";
    src.new_line() << "range.w = min(a_count + b_count, gid + nv) - range.y;";
    src.close("}");

    src.new_line() << "return range;";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT0, int VT1, typename T>
std::string load2_to_regstr() {
    std::ostringstream s;
    s << "load2_to_regstr_" << NT << "_" << VT0 << "_" << VT1 << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT0, int VT1, typename T>
void load2_to_regstr(backend::source_generator &src) {
    src.function<void>(load2_to_regstr<NT,VT0,VT1,T>())
        .open("(")
            .template parameter< global_ptr<const T> >("a_global")
            .template parameter< int                 >("a_count")
            .template parameter< global_ptr<const T> >("b_global")
            .template parameter< int                 >("b_count")
            .template parameter< int                 >("tid")
            .template parameter< regstr_ptr<T>       >("reg")
        .close(")").open("{");

    src.new_line() << "b_global -= a_count;";
    src.new_line() << "int total = a_count + b_count;";
    src.new_line() << "int index;";
    src.new_line() << "if (total >= " << NT * VT0 << ")";
    src.open("{");

    for(int i = 0; i < VT0; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < a_count) reg[" << i << "] = a_global[index];";
        src.new_line() << "else reg[" << i << "] = b_global[index];";
    }

    src.close("}");
    src.new_line() << "else";
    src.open("{");

    for(int i = 0; i < VT0; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < a_count) reg[" << i << "] = a_global[index];";
        src.new_line() << "else if (index < total) reg[" << i << "] = b_global[index];";
    }

    src.close("}");

    for(int i = VT0; i < VT1; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() << "if (index < a_count) reg[" << i << "] = a_global[index];";
        src.new_line() << "else if (index < total) reg[" << i << "] = b_global[index];";
    }

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT0, int VT1, typename T>
std::string load2_to_shared() {
    std::ostringstream s;
    s << "load2_to_shared_" << NT << "_" << VT0 << "_" << VT1 << "_" << type_name<T>();
    return s.str();
}

template<int NT, int VT0, int VT1, typename T>
void load2_to_shared(backend::source_generator &src) {
    load2_to_regstr<NT, VT0, VT1, T>(src);

    src.function<void>(load2_to_shared<NT,VT0,VT1,T>())
        .open("(")
            .template parameter< global_ptr<const T> >("a_global")
            .template parameter< int                 >("a_count")
            .template parameter< global_ptr<const T> >("b_global")
            .template parameter< int                 >("b_count")
            .template parameter< int                 >("tid")
            .template parameter< shared_ptr<T>       >("shared")
        .close(")").open("{");

    src.new_line() << type_name<T>() << " reg[" << VT1 << "];";
    src.new_line() << load2_to_regstr<NT, VT0, VT1, T>()
        << "(a_global, a_count, b_global, b_count, tid, reg);";
    src.new_line() << regstr_to_shared<NT, VT1, T>()
        << "(reg, tid, shared);";

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, int VT, typename T>
std::string merge_keys_indices() {
    std::ostringstream s;
    s << "merge_keys_indices_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template <int NT, int VT, typename T>
void merge_keys_indices(backend::source_generator &src) {
    serial_merge<VT, T>(src);
    merge_path< shared_ptr<const T> >(src);
    load2_to_shared<NT, VT, VT, T>(src);

    src.function<void>(merge_keys_indices<NT, VT, T>())
        .open("(")
            .template parameter< global_ptr<const T> >("a_global")
            .template parameter< int                 >("a_count")
            .template parameter< global_ptr<const T> >("b_global")
            .template parameter< int                 >("b_count")
            .template parameter< cl_int4             >("range")
            .template parameter< int                 >("tid")
            .template parameter< shared_ptr<T>       >("keys_shared")
            .template parameter< regstr_ptr<T>       >("results")
            .template parameter< regstr_ptr<int>     >("indices")
        .close(")").open("{");

    src.new_line() << "int a0 = range.x;";
    src.new_line() << "int a1 = range.y;";
    src.new_line() << "int b0 = range.z;";
    src.new_line() << "int b1 = range.w;";

    // Use the input intervals from the ranges between the merge path
    // intersections.
    src.new_line() << "a_count = a1 - a0;";
    src.new_line() << "b_count = b1 - b0;";

    // Load the data into shared memory.
    src.new_line() << load2_to_shared<NT, VT, VT, T>()
        << "(a_global + a0, a_count, b_global + b0, b_count, tid, keys_shared);";

    // Run a merge path to find the start of the serial merge for each
    // thread.
    src.new_line() << "int diag = " << VT << " * tid;";
    src.new_line() << "int mp = " << merge_path< shared_ptr<const T> >()
        << "(keys_shared, a_count, keys_shared + a_count, b_count, diag);";

    // Compute the ranges of the sources in shared memory.
    src.new_line() << "int a0tid = mp;";
    src.new_line() << "int a1tid = a_count;";
    src.new_line() << "int b0tid = a_count + diag - mp;";
    src.new_line() << "int b1tid = a_count + b_count;";

    // Serial merge into register.
    src.new_line() << serial_merge<VT, T>()
        << "(keys_shared, a0tid, a1tid, b0tid, b1tid, results, indices);";

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, int VT, typename T>
std::string transfer_merge_values_regstr() {
    std::ostringstream s;
    s << "transfer_merge_values_regstr_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template <int NT, int VT, typename T>
void transfer_merge_values_regstr(backend::source_generator &src) {
    src.function<void>(transfer_merge_values_regstr<NT, VT, T>())
        .open("(")
            .template parameter< int                   >("count")
            .template parameter< global_ptr<const T>   >("a_global")
            .template parameter< global_ptr<const T>   >("b_global")
            .template parameter< int                   >("b_start")
            .template parameter< regstr_ptr<const int> >("indices")
            .template parameter< int                   >("tid")
            .template parameter< regstr_ptr<T>         >("reg")
        .close(")").open("{");

    src.new_line() << "b_global -= b_start;";
    src.new_line() << "if(count >= " << NT * VT << ")";
    src.open("{");

    for(int i = 0; i < VT; ++i)
        src.new_line() << "reg[" << i << "] = (indices[" << i << "] < b_start) "
            "? a_global[indices[" << i << "]] : b_global[indices[" << i << "]];";

    src.close("}");
    src.new_line() << "else";
    src.open("{");

    src.new_line() << "int index;";

    for(int i = 0; i < VT; ++i) {
        src.new_line() << "index = " << NT * i << " + tid;";
        src.new_line() <<
            "if(index < count) "
            "reg[" << i << "] = (indices[" << i << "] < b_start) ? "
            "a_global[indices[" << i << "]] : b_global[indices[" << i << "]];";
    }
    src.close("}");

    src.new_line().barrier();

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, int VT, typename T>
std::string transfer_merge_values_shared() {
    std::ostringstream s;
    s << "transfer_merge_values_shared_" << NT << "_" << VT << "_" << type_name<T>();
    return s.str();
}

template <int NT, int VT, typename T>
void transfer_merge_values_shared(backend::source_generator &src) {
    transfer_merge_values_regstr<NT, VT, T>(src);

    src.function<void>(transfer_merge_values_shared<NT, VT, T>())
        .open("(")
            .template parameter< int                   >("count")
            .template parameter< global_ptr<const T>   >("a_global")
            .template parameter< global_ptr<const T>   >("b_global")
            .template parameter< int                   >("b_start")
            .template parameter< shared_ptr<const int> >("indices_shared")
            .template parameter< int                   >("tid")
            .template parameter< global_ptr<T>         >("dest_global")
        .close(")").open("{");

    src.new_line() << "int indices[" << VT << "];";
    src.new_line() << shared_to_regstr<NT, VT, int>()
        << "(indices_shared, tid, indices);";

    src.new_line() << type_name<T>() << " reg[" << VT << "];";
    src.new_line() << transfer_merge_values_regstr<NT, VT, T>()
        << "(count, a_global, b_global, b_start, indices, tid, reg);";
    src.new_line() << regstr_to_global<NT, VT, T>()
        << "(count, reg, tid, dest_global);";

    src.close("}");
}

//---------------------------------------------------------------------------
template<int NT, int VT, typename K, typename V, bool HasValues>
std::string device_merge() {
    std::ostringstream s;
    s << "device_merge_" << NT << "_" << VT << "_" << type_name<K>();
    if (HasValues) s << "_" << type_name<V>();
    return s.str();
}

template<int NT, int VT, typename K, typename V, bool HasValues>
void device_merge(backend::source_generator &src) {
    merge_keys_indices<NT, VT, K>(src);
    transfer_merge_values_shared<NT, VT, V>(src);

    src.function<void>(device_merge<NT, VT, K, V, HasValues>());
    src.open("(");
    src.template parameter< int                   >("a_count");
    src.template parameter< int                   >("b_count");
    src.template parameter< global_ptr<const K>   >("a_keys_global");
    src.template parameter< global_ptr<const K>   >("b_keys_global");
    src.template parameter< global_ptr<K>         >("keys_global");
    src.template parameter< shared_ptr<K>         >("keys_shared");
    if (HasValues) {
        src.template parameter< global_ptr<const V>   >("a_vals_global");
        src.template parameter< global_ptr<const V>   >("b_vals_global");
        src.template parameter< global_ptr<V>         >("vals_global");
    }
    src.template parameter< int                   >("tid");
    src.template parameter< int                   >("block");
    src.template parameter< cl_int4               >("range");
    src.template parameter< shared_ptr<int>       >("indices_shared");
    src.close(")").open("{");

    src.new_line() << type_name<K>() << " results[" << VT << "];";
    src.new_line() << "int indices[" << VT << "];";

    src.new_line() << merge_keys_indices<NT, VT, K>()
        << "(a_keys_global, a_count, b_keys_global, b_count, range, tid, "
        << "keys_shared, results, indices);";

    // Store merge results back to shared memory.
    src.new_line() << thread_to_shared<VT, K>()
        << "(results, tid, keys_shared);";


    // Store merged keys to global memory.
    src.new_line() << "a_count = range.y - range.x;";
    src.new_line() << "b_count = range.w - range.z;";
    src.new_line() << shared_to_global<NT, VT, K>()
        << "(a_count + b_count, keys_shared, tid, keys_global + "
        << NT * VT << " * block);";

    // Copy the values.
    if (HasValues) {
        src.new_line() << thread_to_shared<VT, int>()
            << "(indices, tid, indices_shared);";
        src.new_line() << transfer_merge_values_shared<NT, VT, V>()
            << "(a_count + b_count, a_vals_global + range.x, "
            "b_vals_global + range.z, a_count, indices_shared, tid, "
            "vals_global + " << NT * VT << " * block);";
    }

    src.close("}");
}

//---------------------------------------------------------------------------
template <int NT, int VT, typename K, typename V, typename Comp, bool HasValues>
backend::kernel merge_kernel(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Comp::define(src, "comp");
        compute_merge_range(src);

        transfer_functions<NT, VT, int>(src);

        if (!std::is_same<K, int>::value)
            transfer_functions<NT, VT, K>(src);

        if (HasValues && !std::is_same<V, K>::value && !std::is_same<V, int>::value)
            transfer_functions<NT, VT, V>(src);

        device_merge<NT, VT, K, V, HasValues>(src);

        src.kernel("merge");
        src.open("(");
        src.template parameter< int                   >("a_count");
        src.template parameter< int                   >("b_count");
        src.template parameter< global_ptr<const K>   >("a_keys_global");
        src.template parameter< global_ptr<const K>   >("b_keys_global");
        src.template parameter< global_ptr<K>         >("keys_global");
        if (HasValues) {
            src.template parameter< global_ptr<const V>   >("a_vals_global");
            src.template parameter< global_ptr<const V>   >("b_vals_global");
            src.template parameter< global_ptr<V>         >("vals_global");
        }
        src.template parameter< global_ptr<const int> >("mp_global");
        src.template parameter< int                   >("coop");
        src.close(")").open("{");

        const int NV = NT * VT;

        src.new_line() << "union Shared";
        src.open("{");
        src.new_line() << type_name<K>() << " keys[" << NT * (VT + 1) << "];";
        src.new_line() << "int indices[" << NV << "];";
        src.close("};");

        src.smem_static_var("union Shared", "shared");

        src.new_line() << "int tid    = " << src.local_id(0) << ";";
        src.new_line() << "int block  = " << src.group_id(0) << ";";

        src.new_line() << "int4 range = compute_merge_range("
            "a_count, b_count, block, coop, " << NV << ", mp_global);";

        src.new_line() << device_merge<NT, VT, K, V, HasValues>()
            << "(a_count, b_count, a_keys_global, b_keys_global, keys_global, shared.keys, "
            << (HasValues ? "a_vals_global, b_vals_global, vals_global, " : "")
            << "tid, block, range, shared.indices);";

        src.close("}");

        backend::kernel krn(queue, src.str(), "merge");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
inline int clz(int x) {
    for(int i = 31; i >= 0; --i)
        if((1 << i) & x) return 31 - i;
    return 32;
}

//---------------------------------------------------------------------------
inline int find_log2(int x, bool round_up = false) {
    int a = 31 - clz(x);

    if(round_up) {
        bool is_pow_2 = (0 == (x & (x - 1)));
        a += !is_pow_2;
    }

    return a;
}

/// Sorts single partition of a vector.
template <class K, class Comp>
void sort(const backend::command_queue &queue,
        backend::device_vector<K> &keys, Comp)
{
    backend::select_context(queue);

    const int NT_cpu = 1;
    const int NT_gpu = 256;
    const int NT = is_cpu(queue) ? NT_cpu : NT_gpu;
    const int VT = (sizeof(K) > 4) ? 7 : 11;
    const int NV = NT * VT;

    const int count = keys.size();
    const int num_blocks = (count + NV - 1) / NV;
    const int num_passes = detail::find_log2(num_blocks, true);


    device_vector<K> keys_dst(queue, count);

    auto block_sort = is_cpu(queue) ?
        detail::block_sort_kernel<NT_cpu, VT, K, int, Comp, false>(queue) :
        detail::block_sort_kernel<NT_gpu, VT, K, int, Comp, false>(queue);

    block_sort.push_arg(count);
    block_sort.push_arg(keys);
    block_sort.push_arg(1 & num_passes ? keys_dst : keys);

    block_sort.config(num_blocks, NT);

    block_sort(queue);

    if (1 & num_passes) {
        std::swap(keys, keys_dst);
    }

    auto merge = is_cpu(queue) ?
        detail::merge_kernel<NT_cpu, VT, K, int, Comp, false>(queue) :
        detail::merge_kernel<NT_gpu, VT, K, int, Comp, false>(queue);

    for(int pass = 0; pass < num_passes; ++pass) {
        int coop = 2 << pass;

        auto partitions = detail::merge_path_partitions<Comp>(queue, keys, count, NV, coop);

        merge.push_arg(count);
        merge.push_arg(0);
        merge.push_arg(keys);
        merge.push_arg(keys);
        merge.push_arg(keys_dst);
        merge.push_arg(partitions);
        merge.push_arg(coop);

        merge.config(num_blocks, NT);
        merge(queue);

        std::swap(keys, keys_dst);
    }
}

/// Sorts single partition of a vector.
template <class K, class V, class Comp>
void sort_by_key(const backend::command_queue &queue,
        backend::device_vector<K> &keys,
        backend::device_vector<V> &vals,
        Comp
        )
{
    precondition(keys.size() == vals.size(),
            "keys and values should have same size"
            );

    backend::select_context(queue);

    const int NT_cpu = 1;
    const int NT_gpu = 256;
    const int NT = is_cpu(queue) ? NT_cpu : NT_gpu;
    const int VT = (sizeof(K) > 4) ? 7 : 11;
    const int NV = NT * VT;

    const int count = keys.size();
    const int num_blocks = (count + NV - 1) / NV;
    const int num_passes = detail::find_log2(num_blocks, true);

    device_vector<K> keys_dst(queue, count);
    device_vector<V> vals_dst(queue, count);

    auto block_sort = is_cpu(queue) ?
        detail::block_sort_kernel<NT_cpu, VT, K, V, Comp, true>(queue) :
        detail::block_sort_kernel<NT_gpu, VT, K, V, Comp, true>(queue);

    block_sort.push_arg(count);
    block_sort.push_arg(keys);
    block_sort.push_arg(1 & num_passes ? keys_dst : keys);
    block_sort.push_arg(vals);
    block_sort.push_arg(1 & num_passes ? vals_dst : vals);

    block_sort.config(num_blocks, NT);

    block_sort(queue);

    if (1 & num_passes) {
        std::swap(keys, keys_dst);
        std::swap(vals, vals_dst);
    }

    auto merge = is_cpu(queue) ?
        detail::merge_kernel<NT_cpu, VT, K, V, Comp, true>(queue) :
        detail::merge_kernel<NT_gpu, VT, K, V, Comp, true>(queue);

    for(int pass = 0; pass < num_passes; ++pass) {
        int coop = 2 << pass;

        auto partitions = detail::merge_path_partitions<Comp>(queue, keys, count, NV, coop);

        merge.push_arg(count);
        merge.push_arg(0);
        merge.push_arg(keys);
        merge.push_arg(keys);
        merge.push_arg(keys_dst);
        merge.push_arg(vals);
        merge.push_arg(vals);
        merge.push_arg(vals_dst);
        merge.push_arg(partitions);
        merge.push_arg(coop);

        merge.config(num_blocks, NT);
        merge(queue);

        std::swap(keys, keys_dst);
        std::swap(vals, vals_dst);
    }
}

/// Merges partially sorted vector partitions into host vector
template <typename K, class Comp>
std::vector<K> merge(const vector<K> &keys, Comp comp) {
    const auto &queue = keys.queue_list();

    std::vector<K> dst(keys.size());
    vex::copy(keys, dst);

    if (queue.size() > 1) {
        std::vector<K> src(keys.size());
        std::swap(src, dst);

        typedef typename std::vector<K>::const_iterator iterator;
        std::vector<iterator> begin(queue.size());
        std::vector<iterator> end(queue.size());

        for(unsigned d = 0; d < queue.size(); ++d) {
            begin[d] = src.begin() + keys.part_start(d);
            end  [d] = src.begin() + keys.part_start(d + 1);
        }

        for(auto pos = dst.begin(); pos != dst.end(); ++pos) {
            int winner = -1;
            for(unsigned d = 0; d < queue.size(); ++d) {
                if (begin[d] == end[d]) continue;

                if (winner < 0 || comp(*begin[d], *begin[winner]))
                    winner = d;
            }

            *pos = *begin[winner]++;
        }
    }

    return dst;
}

/// Merges partially sorted vector partitions into host vector
template <typename K, typename V, class Comp>
void merge(const vector<K> &keys, const vector<V> &vals, Comp comp,
        std::vector<K> &dst_keys, std::vector<V> &dst_vals)
{
    const auto &queue = keys.queue_list();

    dst_keys.resize(keys.size());
    dst_vals.resize(keys.size());

    vex::copy(keys, dst_keys);
    vex::copy(vals, dst_vals);

    if (queue.size() > 1) {
        std::vector<K> src_keys(keys.size());
        std::vector<V> src_vals(keys.size());

        std::swap(src_keys, dst_keys);
        std::swap(src_vals, dst_vals);

        typedef typename std::vector<K>::const_iterator key_iterator;
        typedef typename std::vector<V>::const_iterator val_iterator;

        std::vector<key_iterator> key_begin(queue.size()), key_end(queue.size());
        std::vector<val_iterator> val_begin(queue.size()), val_end(queue.size());

        for(unsigned d = 0; d < queue.size(); ++d) {
            key_begin[d] = src_keys.begin() + keys.part_start(d);
            key_end  [d] = src_keys.begin() + keys.part_start(d + 1);

            val_begin[d] = src_vals.begin() + keys.part_start(d);
            val_end  [d] = src_vals.begin() + keys.part_start(d + 1);
        }

        auto key_pos = dst_keys.begin();
        auto val_pos = dst_vals.begin();

        while(key_pos != dst_keys.end()) {
            int winner = -1;
            for(unsigned d = 0; d < queue.size(); ++d) {
                if (key_begin[d] == key_end[d]) continue;

                if (winner < 0 || comp(*key_begin[d], *key_begin[winner]))
                    winner = d;
            }

            *key_pos++ = *key_begin[winner]++;
            *val_pos++ = *val_begin[winner]++;
        }
    }
}

} // namespace detail

/// Function object class for less-than inequality comparison.
/**
 * The need for host-side and device-side parts comes from the fact that
 * vectors are partially sorted on device and then final merge step is done on
 * host.
 */
template <typename T>
struct less : std::less<T> {
    VEX_FUNCTION(device, bool(T, T), "return prm1 < prm2;");
};

/// Function object class for less-than-or-equal inequality comparison.
/**
 * The need for host-side and device-side parts comes from the fact that
 * vectors are partially sorted on device and then final merge step is done on
 * host.
 */
template <typename T>
struct less_equal : std::less_equal<T> {
    VEX_FUNCTION(device, bool(T, T), "return prm1 <= prm2;");
};

/// Function object class for greater-than inequality comparison.
/**
 * The need for host-side and device-side parts comes from the fact that
 * vectors are partially sorted on device and then final merge step is done on
 * host.
 */
template <typename T>
struct greater : std::greater<T> {
    VEX_FUNCTION(device, bool(T, T), "return prm1 > prm2;");
};

/// Function object class for greater-than-or-equal inequality comparison.
/**
 * The need for host-side and device-side parts comes from the fact that
 * vectors are partially sorted on device and then final merge step is done on
 * host.
 */
template <typename T>
struct greater_equal : std::greater_equal<T> {
    VEX_FUNCTION(device, bool(T, T), "return prm1 >= prm2;");
};

/// Sorts the vector into ascending order.
/**
 * \param comp comparison function.
 */
template <class K, class Comp>
void sort(vector<K> &keys, Comp comp) {
    const auto &queue = keys.queue_list();

    for(unsigned d = 0; d < queue.size(); ++d)
        if (keys.part_size(d)) detail::sort(queue[d], keys(d), comp.device);

    if (queue.size() <= 1) return;

    // Vector partitions have been sorted on compute devices.
    // Now we need to merge them on a CPU. This is a linear time operation,
    // so total performance should be good enough.
    vex::copy(detail::merge(keys, comp), keys);
}

/// Sorts the elements in keys and values into ascending key order.
template <class K>
void sort(vector<K> &keys) {
    sort(keys, less<K>());
}

/// Sorts the elements in keys and values into ascending key order.
/**
 * \param comp comparison function.
 */
template <class K, class V, class Comp>
void sort_by_key(vector<K> &keys, vector<V> &vals, Comp comp) {
    precondition(
            keys.queue_list().size() == vals.queue_list().size(),
            "Keys and values span different devices"
            );

    auto &queue = keys.queue_list();

    for(unsigned d = 0; d < queue.size(); ++d)
        if (keys.part_size(d))
            detail::sort_by_key(queue[d], keys(d), vals(d), comp.device);

    if (queue.size() <= 1) return;

    // Vector partitions have been sorted on compute devices.
    // Now we need to merge them on a CPU. This is a linear time operation,
    // so total performance should be good enough.
    std::vector<K> k;
    std::vector<V> v;

    detail::merge(keys, vals, comp, k, v);

    vex::copy(k, keys);
    vex::copy(v, vals);
}

/// Sorts the elements in keys and values into ascending key order.
template <class K, class V>
void sort_by_key(vector<K> &keys, vector<V> &vals) {
    sort_by_key(keys, vals, less<K>());
}

} // namespace vex

#endif
