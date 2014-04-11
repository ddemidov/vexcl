#ifndef VEXCL_CLOGS_HPP
#define VEXCL_CLOGS_HPP

/*
The MIT License

Copyright (c) 2014 Bruce Merry <bmerry@users.sourceforge.net>

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
 * \file   external/clogs.hpp
 * \author Bruce Merry <bmerry@users.sourceforge.net>
 * \brief  Enables use of CLOGS (http://clogs.sourceforge.net) primitives.
 */

#include <type_traits>
#include <vector>
#include <memory>
#include <utility>
#include <iterator>
#include <vexcl/vector.hpp>
#include <vexcl/types.hpp>
#include <vexcl/operations.hpp> // for cache
#include <vexcl/sort.hpp> // for merging
#include <clogs/scan.h>
#include <clogs/radixsort.h>

#ifdef VEXCL_BACKEND_CUDA
#  error "clogs interoperation is not supported for the CUDA backend!"
#endif

namespace vex {

/// Wrappers for the clogs library
namespace clogs {

namespace detail {

struct compare_queue
{
    bool operator()(const cl::CommandQueue &a, const cl::CommandQueue &b) const
    {
        return a() < b();
    }
};

template<typename T>
struct queue_cache : public vex::detail::object_cache_base {
    typedef std::map<cl::CommandQueue, T, compare_queue> store_type;

    store_type store;

    queue_cache() {
        vex::detail::cache_register<>::add(this);
    }

    template <typename S>
    std::pair<typename store_type::iterator, bool> insert(S&& item) {
        return store.insert(std::forward<S>(item));
    }

    typename store_type::const_iterator end() const {
        return store.end();
    }

    template <typename S>
    typename store_type::iterator find(S&& key) {
        return store.find( std::forward<S>(key) );
    }

    virtual void clear() {
        store.clear();
    }

    virtual void erase(backend::kernel_cache_key key) {
        auto p = store.begin();
        while (p != store.end()) {
            auto next = std::next(p);
            if (p->first.template getInfo<CL_QUEUE_CONTEXT>()() == key)
                store.erase(p);
            p = next;
        }
    }
};

} // namespace detail

/// Maps a compile-time C type to a clogs type structure
template<typename T, typename Enable = void>
struct clogs_type {};

#define VEXCL_REGISTER_CLOGS_TYPE(cltype, token) \
    template<> struct clogs_type<cltype, void> { \
        static inline ::clogs::Type type() { return ::clogs::token; } \
    }

VEXCL_REGISTER_CLOGS_TYPE(cl_char, TYPE_CHAR);
VEXCL_REGISTER_CLOGS_TYPE(cl_short, TYPE_SHORT);
VEXCL_REGISTER_CLOGS_TYPE(cl_int, TYPE_INT);
VEXCL_REGISTER_CLOGS_TYPE(cl_long, TYPE_LONG);

VEXCL_REGISTER_CLOGS_TYPE(cl_uchar, TYPE_UCHAR);
VEXCL_REGISTER_CLOGS_TYPE(cl_ushort, TYPE_USHORT);
VEXCL_REGISTER_CLOGS_TYPE(cl_uint, TYPE_UINT);
VEXCL_REGISTER_CLOGS_TYPE(cl_ulong, TYPE_ULONG);

VEXCL_REGISTER_CLOGS_TYPE(cl_float, TYPE_FLOAT);
VEXCL_REGISTER_CLOGS_TYPE(cl_double, TYPE_DOUBLE);

// Generates clogs_type for vector types, using vex::cl_scalar_of
template<typename T>
struct clogs_type<T, typename std::enable_if<vex::is_cl_vector<T>::value>::type> {
    static inline ::clogs::Type type()
    {
        return ::clogs::Type(
            clogs_type<typename vex::cl_scalar_of<T>::type>::type().getBaseType(),
            vex::cl_vector_length<T>::value);
    }
};

#undef VEXCL_REGISTER_CLOGS_TYPE

/// Whether T can be mapped to a clogs type
template<typename T, typename Enable = void>
struct is_clogs_type : public std::false_type {};

template<typename T>
struct is_clogs_type<T, typename std::enable_if<sizeof(clogs_type<T>::type())>::type>
    : public std::true_type
{};

/// Whether T can be used for @ref exclusive_scan
template<typename T, typename Enable = void>
struct is_scannable : public std::false_type {};

template<typename T>
struct is_scannable<T, typename std::enable_if<
        is_clogs_type<T>::value
        && std::is_integral<typename vex::cl_scalar_of<T>::type>::value>::type>
    : public std::true_type {};

/// Whether T can be used as a key type for @ref sort or @ref stable_sort_by_key
template<typename T, typename Enable = void>
struct is_sort_key : public std::false_type {};

template<typename T>
struct is_sort_key<T, typename std::enable_if<
        is_clogs_type<T>::value
        && std::is_integral<T>::value
        && std::is_unsigned<T>::value>::type>
    : public std::true_type
{};


/// Perform exclusive scan using clogs
/**
 * It is legal for @a src and @a dst to be the same vector, in which case an
 * in-place scan is performed.
 */
template<typename T>
void exclusive_scan(const vex::vector<T> &src, vex::vector<T> &dst,
        const T &init = T())
{
    static_assert(
            is_scannable<T>::value,
            "Unsupported type for clogs::exclusive_scan"
            );

    static detail::queue_cache<std::unique_ptr< ::clogs::Scan> > cache;

    const std::vector<backend::command_queue> &queue = src.queue_list();

    std::vector<T> tail;
    /* If there is more than one partition, we need to take a copy the last
     * element in each partition (except the last) as otherwise information
     * about it is lost.
     *
     * This must be captured here rather than later, in case the input and
     * output alias.
     */
    if (queue.size() > 1) {
        tail.resize(queue.size() - 1);
        for (unsigned d = 0; d < tail.size(); ++d) {
            if (src.part_size(d))
                tail[d] = src[src.part_start(d + 1) - 1];
        }
    }

    for (unsigned d = 0; d < queue.size(); ++d) {
        if (src.part_size(d)) {
            auto entry = cache.find(queue[d]);
            if (entry == cache.end()) {
                std::unique_ptr< ::clogs::Scan> scanner(new ::clogs::Scan(
                    queue[d].getInfo<CL_QUEUE_CONTEXT>(),
                    queue[d].getInfo<CL_QUEUE_DEVICE>(),
                    clogs_type<T>::type()));
                entry = cache.insert(std::make_pair(queue[d], std::move(scanner))).first;
            }
            entry->second->enqueue(
                    queue[d],
                    src(d).raw_buffer(), dst(d).raw_buffer(), src.part_size(d),
                    d == 0 ? &init : nullptr);
        }
    }

    /* If there is more than one partition, update all of them except for the
     * first. This is not very efficient: it would be better to have deeper
     * hooks into clogs so that the block sums could be prefix summed across
     * partitions.
     */
    if (queue.size() > 1) {
        T sum{};
        for (unsigned d = 0; d < tail.size(); ++d) {
            if (src.part_size(d)) {
                sum += tail[d];
                sum += dst[src.part_start(d + 1) - 1];
                // Wrap partition into vector for ease of use:
                vex::vector<T> part(queue[d + 1], dst(d + 1));
                part = sum + part;
            }
        }
    }
}

/// Perform sort using clogs
/**
 * If there are more than one device in vector's queue list, then all
 * partitions are sorted individually on devices and then merged on the host.
 */
template<typename K>
void sort(vex::vector<K> &keys)
{
    static_assert(is_sort_key<K>::value, "Unsupported type for clogs::sort");

    static detail::queue_cache<std::unique_ptr< ::clogs::Radixsort> > cache;

    const std::vector<backend::command_queue> &queue = keys.queue_list();

    for (unsigned d = 0; d < queue.size(); ++d) {
        if (keys.part_size(d)) {
            auto entry = cache.find(queue[d]);
            if (entry == cache.end()) {
                std::unique_ptr< ::clogs::Radixsort> sorter(new ::clogs::Radixsort(
                    queue[d].getInfo<CL_QUEUE_CONTEXT>(),
                    queue[d].getInfo<CL_QUEUE_DEVICE>(),
                    clogs_type<K>::type(), ::clogs::Type()));
                entry = cache.insert(std::make_pair(queue[d], std::move(sorter))).first;
            }
            entry->second->enqueue(
                queue[d], keys(d).raw_buffer(), cl::Buffer(), keys.part_size(d));
        }
    }

    // If there are multiple queues, merge the results on the CPU
    if (queue.size() > 1) {
        namespace fusion = boost::fusion;

        auto key_vectors = fusion::vector_tie(keys);
        auto host_vectors = vex::detail::merge(key_vectors, vex::less<K>());
        fusion::for_each( vex::detail::make_zip_view(host_vectors, key_vectors), vex::detail::do_copy() );
    }
}

/// Perform stable sort of keys and values using clogs
/**
 * If there are more than one device in vector's queue list, then all
 * partitions are sorted individually on devices and then merged on the host.
 */
template<typename K, typename V>
void stable_sort_by_key(vex::vector<K> &keys, vex::vector<V> &values)
{
    static_assert(
            is_sort_key<K>::value && is_clogs_type<V>::value,
            "Unsupported types for clogs::stable_sort_by_key"
            );

    static detail::queue_cache<std::unique_ptr< ::clogs::Radixsort> > cache;

    const std::vector<backend::command_queue> &queue = keys.queue_list();

    for (unsigned d = 0; d < queue.size(); ++d) {
        if (keys.part_size(d)) {
            auto entry = cache.find(queue[d]);
            if (entry == cache.end()) {
                std::unique_ptr< ::clogs::Radixsort> sorter(new ::clogs::Radixsort(
                    queue[d].getInfo<CL_QUEUE_CONTEXT>(),
                    queue[d].getInfo<CL_QUEUE_DEVICE>(),
                    clogs_type<K>::type(),
                    clogs_type<V>::type()));
                entry = cache.insert(std::make_pair(queue[d], std::move(sorter))).first;
            }
            entry->second->enqueue(
                queue[d], keys(d).raw_buffer(), values(d).raw_buffer(), keys.part_size(d));
        }
    }

    // If there are multiple queues, merge the results on the CPU
    if (queue.size() > 1) {
        namespace fusion = boost::fusion;

        auto key_vectors = fusion::vector_tie(keys);
        auto value_vectors = fusion::vector_tie(values);
        auto host_vectors = vex::detail::merge(key_vectors, value_vectors, vex::less<K>());
        auto dev_vectors = fusion::join(key_vectors, value_vectors);
        boost::fusion::for_each( vex::detail::make_zip_view(host_vectors, dev_vectors), vex::detail::do_copy() );
    }
}

} // namespace clogs
} // namespace vex

#endif
