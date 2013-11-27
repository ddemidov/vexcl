#ifndef VEXCL_REDUCE_BY_KEY_HPP
#define VEXCL_REDUCE_BY_KEY_HPP

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
 * \file   vexcl/reduce_by_key.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Reduce by key algortihm.

Adopted from Bolt code, see <https://github.com/HSA-Libraries/Bolt>.
The original code came with the following copyright notice:

\verbatim
Copyright 2012 - 2013 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
\endverbatim
*/

#include <string>

#include <vexcl/vector.hpp>
#include <vexcl/scan.hpp>

namespace vex {
namespace detail {

//---------------------------------------------------------------------------
template <typename T, class Comp>
backend::kernel offset_calculation(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Comp::define(src, "comp");

        src.kernel("offset_calculation")
            .open("(")
                .template parameter< size_t              >("n")
                .template parameter< global_ptr<const T> >("keys")
                .template parameter< global_ptr<int>     >("offsets")
            .close(")").open("{");

        src.new_line().grid_stride_loop().open("{");
        src.new_line()
            << "if (idx > 0)"
            << " offsets[idx] = !comp(keys[idx - 1], keys[idx]);";
        src.new_line() << "else offsets[idx] = 0;";
        src.close("}");
        src.close("}");

        backend::kernel krn(queue, src.str(), "offset_calculation");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
template <int NT, typename T, class Oper>
backend::kernel block_scan_by_key(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Oper::define(src, "oper");

        src.kernel("block_scan_by_key")
            .open("(")
                .template parameter< size_t                >("n")
                .template parameter< global_ptr<const int> >("keys")
                .template parameter< global_ptr<const T>   >("vals")
                .template parameter< global_ptr<T>         >("output")
                .template parameter< global_ptr<int>       >("key_buf")
                .template parameter< global_ptr<T>         >("val_buf")
            .close(")").open("{");

        src.new_line() << "size_t l_id  = " << src.local_id(0)   << ";";
        src.new_line() << "size_t g_id  = " << src.global_id(0)  << ";";
        src.new_line() << "size_t block = " << src.group_id(0)   << ";";
        src.new_line() << "size_t wgsz  = " << src.local_size(0) << ";";

        src.new_line() << "struct Shared";
        src.open("{");
            src.new_line() << "int keys[" << NT << "];";
            src.new_line() << type_name<T>() << " vals[" << NT << "];";
        src.close("};");

        src.smem_static_var("struct Shared", "shared");

        src.new_line() << "int key;";
        src.new_line() << type_name<T>() << " val;";

        src.new_line() << "if (g_id < n)";
        src.open("{");
        src.new_line() << "key = keys[g_id];";
        src.new_line() << "val = vals[g_id];";
        src.new_line() << "shared.keys[l_id] = key;";
        src.new_line() << "shared.vals[l_id] = val;";
        src.close("}");

        // Computes a scan within a workgroup updates vals in lds but not keys
        src.new_line() << type_name<T>() << " sum = val;";
        src.new_line() << "for(size_t offset = 1; offset < wgsz; offset *= 2)";
        src.open("{");
        src.new_line().barrier();
        src.new_line() << "if (l_id >= offset && shared.keys[l_id - offset] == key)";
        src.open("{");
        src.new_line() << "sum = oper(sum, shared.vals[l_id - offset]);";
        src.close("}");
        src.new_line().barrier();
        src.new_line() << "shared.vals[l_id] = sum;";
        src.close("}");
        src.new_line().barrier();

        src.new_line() << "if (g_id >= n) return;";

        // Each work item writes out its calculated scan result, relative to the
        // beginning of each work group
        src.new_line() << "int key2 = -1;";
        src.new_line() << "if (g_id < n - 1) key2 = keys[g_id + 1];";
        src.new_line() << "if (key != key2) output[g_id] = sum;";

        src.new_line() << "if (l_id == 0)";
        src.open("{");
        src.new_line() << "key_buf[block] = shared.keys[wgsz - 1];";
        src.new_line() << "val_buf[block] = shared.vals[wgsz - 1];";
        src.close("}");

        src.close("}");

        backend::kernel krn(queue, src.str(), "block_scan_by_key");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
template <int NT, typename T, class Oper>
backend::kernel block_inclusive_scan_by_key(const backend::command_queue &queue)
{
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Oper::define(src, "oper");

        src.kernel("block_inclusive_scan_by_key")
            .open("(")
                .template parameter< size_t                >("n")
                .template parameter< global_ptr<const int> >("key_sum")
                .template parameter< global_ptr<const T>   >("pre_sum")
                .template parameter< global_ptr<T>         >("post_sum")
                .template parameter< cl_uint               >("work_per_thread")
            .close(")").open("{");

        src.new_line() << "size_t l_id   = " << src.local_id(0)   << ";";
        src.new_line() << "size_t g_id   = " << src.global_id(0)  << ";";
        src.new_line() << "size_t wgsz   = " << src.local_size(0) << ";";
        src.new_line() << "size_t map_id = g_id * work_per_thread;";

        src.new_line() << "struct Shared";
        src.open("{");
            src.new_line() << "int keys[" << NT << "];";
            src.new_line() << type_name<T>() << " vals[" << NT << "];";
        src.close("};");

        src.smem_static_var("struct Shared", "shared");

        src.new_line() << "uint offset;";
        src.new_line() << "int  key;";
        src.new_line() << type_name<T>() << " work_sum;";

        src.new_line() << "if (map_id < n)";
        src.open("{");
        src.new_line() << "int prev_key;";

        // accumulate zeroth value manually
        src.new_line() << "offset   = 0;";
        src.new_line() << "key      = key_sum[map_id];";
        src.new_line() << "work_sum = pre_sum[map_id];";

        src.new_line() << "post_sum[map_id] = work_sum;";

        //  Serial accumulation
        src.new_line() << "for( offset = offset + 1; offset < work_per_thread; ++offset )";
        src.open("{");
        src.new_line() << "prev_key = key;";
        src.new_line() << "key      = key_sum[ map_id + offset ];";

        src.new_line() << "if ( map_id + offset < n )";
        src.open("{");
        src.new_line() << type_name<T>() << " y = pre_sum[ map_id + offset ];";

        src.new_line() << "if ( key == prev_key ) work_sum = oper( work_sum, y );";
        src.new_line() << "else work_sum = y;";

        src.new_line() << "post_sum[ map_id + offset ] = work_sum;";
        src.close("}");
        src.close("}");
        src.close("}");
        src.new_line().barrier();

        // load LDS with register sums
        src.new_line() << "shared.vals[ l_id ] = work_sum;";
        src.new_line() << "shared.keys[ l_id ] = key;";

        // scan in lds
        src.new_line() << type_name<T>() << " scan_sum = work_sum;";

        src.new_line() << "for( offset = 1; offset < wgsz; offset *= 2 )";
        src.open("{");
        src.new_line().barrier();

        src.new_line() << "if (map_id < n)";
        src.open("{");
        src.new_line() << "if (l_id >= offset)";
        src.open("{");
        src.new_line() << "int key1 = shared.keys[ l_id ];";
        src.new_line() << "int key2 = shared.keys[ l_id - offset ];";

        src.new_line() << "if ( key1 == key2 ) scan_sum = oper( scan_sum, shared.vals[ l_id - offset ] );";
        src.new_line() << "else scan_sum = shared.vals[ l_id ];";
        src.close("}");

        src.close("}");
        src.new_line().barrier();

        src.new_line() << "shared.vals[ l_id ] = scan_sum;";
        src.close("}");

        src.new_line().barrier();

        // write final scan from pre-scan and lds scan
        src.new_line() << "for( offset = 0; offset < work_per_thread; ++offset )";
        src.open("{");
        src.new_line().barrier(true);

        src.new_line() << "if (map_id < n && l_id > 0)";
        src.open("{");
        src.new_line() << type_name<T>() << " y = post_sum[ map_id + offset ];";
        src.new_line() << "int key1 = key_sum    [ map_id + offset ];";
        src.new_line() << "int key2 = shared.keys[ l_id - 1 ];";

        src.new_line() << "if ( key1 == key2 ) y = oper( y, shared.vals[l_id - 1] );";

        src.new_line() << "post_sum[ map_id + offset ] = y;";
        src.close("}");
        src.close("}");

        src.close("}");

        backend::kernel krn(queue, src.str(), "block_inclusive_scan_by_key");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
template <typename T, class Oper>
backend::kernel block_sum_by_key(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        Oper::define(src, "oper");

        src.kernel("block_sum_by_key")
            .open("(")
                .template parameter< size_t                >("n")
                .template parameter< global_ptr<const int> >("key_sum")
                .template parameter< global_ptr<const T>   >("post_sum")
                .template parameter< global_ptr<const int> >("keys")
                .template parameter< global_ptr<T>         >("output")
            .close(")").open("{");

        src.new_line() << "size_t g_id  = " << src.global_id(0)  << ";";
        src.new_line() << "size_t block = " << src.group_id(0)   << ";";

        src.new_line() << "if (g_id >= n) return;";

        // accumulate prefix
        src.new_line() << "int key2 = keys[ g_id ];";
        src.new_line() << "int key1 = (block > 0    ) ? key_sum[ block - 1 ] : key2 - 1;";
        src.new_line() << "int key3 = (g_id  < n - 1) ? keys   [ g_id  + 1 ] : key2 - 1;";

        src.new_line() << "if (block > 0 && key1 == key2 && key2 != key3)";
        src.open("{");
        src.new_line() << type_name<T>() << " scan_result    = output  [ g_id      ];";
        src.new_line() << type_name<T>() << " post_block_sum = post_sum[ block - 1 ];";
        src.new_line() << "output[ g_id ] = oper( scan_result, post_block_sum );";
        src.close("}");

        src.close("}");

        backend::kernel krn(queue, src.str(), "block_sum_by_key");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}

//---------------------------------------------------------------------------
template <typename K, typename V>
backend::kernel key_value_mapping(const backend::command_queue &queue) {
    static detail::kernel_cache cache;

    auto cache_key = backend::cache_key(queue);
    auto kernel    = cache.find(cache_key);

    if (kernel == cache.end()) {
        backend::source_generator src(queue);

        src.kernel("key_value_mapping")
            .open("(")
                .template parameter< size_t              >("n")
                .template parameter< global_ptr<const K> >("ikeys")
                .template parameter< global_ptr<K>       >("okeys")
                .template parameter< global_ptr<V>       >("ovals")
                .template parameter< global_ptr<int>     >("offset")
                .template parameter< global_ptr<const V> >("ivals")
            .close(")").open("{");

        src.new_line().grid_stride_loop().open("{");

        src.new_line() << "int num_sections = offset[n - 1] + 1;";

        src.new_line() << "int off = offset[idx];";
        src.new_line() << "if (idx < (n - 1) && off != offset[idx + 1])";
        src.open("{");
        src.new_line() << "okeys[off] = ikeys[ idx ];";
        src.new_line() << "ovals[off] = ivals[ idx ];";
        src.close("}");

        src.new_line() << "if (idx == (n - 1))";
        src.open("{");
        src.new_line() << "okeys[num_sections - 1] = ikeys[idx];";
        src.new_line() << "ovals[num_sections - 1] = ivals[idx];";
        src.close("}");

        src.close("}");

        src.close("}");

        backend::kernel krn(queue, src.str(), "key_value_mapping");
        kernel = cache.insert(std::make_pair(cache_key, krn)).first;
    }

    return kernel->second;
}


}

/// Reduce by key algorithm.
template <typename K, typename V, class Comp, class Oper>
int reduce_by_key(
        vector<K> const &ikeys,
        vector<V> const &ivals,
        vector<K>       &okeys,
        vector<V>       &ovals,
        Comp, Oper
        )
{
    precondition(
            ikeys.queue_list().size() == 1 &&
            ivals.queue_list().size() == 1,
            "Sorting is only supported for single device contexts"
            );

    precondition(ikeys.size() == ivals.size(),
            "keys and values should have same size"
            );

    auto &queue = ikeys.queue_list()[0];
    backend::select_context(queue);

    const int NT_cpu = 1;
    const int NT_gpu = 256;
    const int NT = is_cpu(queue) ? NT_cpu : NT_gpu;

    size_t count         = ikeys.size();
    size_t num_blocks    = (count + NT - 1) / NT;
    size_t scan_buf_size = alignup(num_blocks, NT);

    backend::device_vector<int> key_sum   (queue, scan_buf_size);
    backend::device_vector<V>   pre_sum   (queue, scan_buf_size);
    backend::device_vector<V>   post_sum  (queue, scan_buf_size);
    backend::device_vector<V>   offset_val(queue, count);
    backend::device_vector<int> offset    (queue, count);

    /***** Kernel 0 *****/
    auto krn0 = detail::offset_calculation<K, Comp>(queue);

    krn0.push_arg(count);
    krn0.push_arg(ikeys(0));
    krn0.push_arg(offset);

    krn0(queue);

    VEX_FUNCTION(plus, int(int, int), "return prm1 + prm2;");
    detail::scan(queue, offset, offset, 0, false, plus);

    /***** Kernel 1 *****/
    auto krn1 = is_cpu(queue) ?
        detail::block_scan_by_key<NT_cpu, V, Oper>(queue) :
        detail::block_scan_by_key<NT_gpu, V, Oper>(queue);

    krn1.push_arg(count);
    krn1.push_arg(offset);
    krn1.push_arg(ivals(0));
    krn1.push_arg(offset_val);
    krn1.push_arg(key_sum);
    krn1.push_arg(pre_sum);

    krn1.config(num_blocks, NT);
    krn1(queue);

    /***** Kernel 2 *****/
    uint work_per_thread = std::max<uint>(1U, scan_buf_size / NT);

    auto krn2 = is_cpu(queue) ?
        detail::block_inclusive_scan_by_key<NT_cpu, V, Oper>(queue) :
        detail::block_inclusive_scan_by_key<NT_gpu, V, Oper>(queue);

    krn2.push_arg(num_blocks);
    krn2.push_arg(key_sum);
    krn2.push_arg(pre_sum);
    krn2.push_arg(post_sum);
    krn2.push_arg(work_per_thread);

    krn2.config(1, NT);
    krn2(queue);

    /***** Kernel 3 *****/
    auto krn3 = detail::block_sum_by_key<V, Oper>(queue);

    krn3.push_arg(count);
    krn3.push_arg(key_sum);
    krn3.push_arg(post_sum);
    krn3.push_arg(offset);
    krn3.push_arg(offset_val);

    krn3.config(num_blocks, NT);
    krn3(queue);

    /***** resize okeys and ovals *****/
    int out_elements;
    offset.read(queue, count - 1, 1, &out_elements, true);
    ++out_elements;

    okeys.resize(ikeys.queue_list(), out_elements);
    ovals.resize(ivals.queue_list(), out_elements);

    /***** Kernel 4 *****/
    auto krn4 = detail::key_value_mapping<K, V>(queue);

    krn4.push_arg(count);
    krn4.push_arg(ikeys(0));
    krn4.push_arg(okeys(0));
    krn4.push_arg(ovals(0));
    krn4.push_arg(offset);
    krn4.push_arg(offset_val);

    krn4(queue);

    return out_elements;
}

/// Reduce by key algorithm.
template <typename K, typename V>
int reduce_by_key(
        vector<K> const &ikeys,
        vector<V> const &ivals,
        vector<K>       &okeys,
        vector<V>       &ovals
        )
{
    VEX_FUNCTION(equal, bool(K, K), "return prm1 == prm2;");
    VEX_FUNCTION(plus,  V(V, V), "return prm1 + prm2;");

    return reduce_by_key(ikeys, ivals, okeys, ovals, equal, plus);
}

}

#endif
