#ifndef VEXCL_VECTOR_HPP
#define VEXCL_VECTOR_HPP

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
 * \file   vexcl/vector.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL device vector.
 */

#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <functional>

#include <boost/proto/proto.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <vexcl/backend.hpp>
#include <vexcl/util.hpp>
#include <vexcl/operations.hpp>
#include <vexcl/profiler.hpp>
#include <vexcl/devlist.hpp>

#ifdef BOOST_NO_NOEXCEPT
#  define noexcept throw()
#endif

/// Vector expression template library for OpenCL.
namespace vex {

//--- Partitioning ----------------------------------------------------------

/// Weights device wrt to vector performance.
/**
 * Launches the following kernel on each device:
 * \code
 * a = b + c;
 * \endcode
 * where a, b and c are device vectors. Each device gets portion of the vector
 * proportional to the performance of this operation.
 */
inline double device_vector_perf(const backend::command_queue&);

/// Assigns equal weight to each device.
/**
 * This results in equal partitioning.
 */
inline double equal_weights(const backend::command_queue&) {
    return 1;
}

/// \cond INTERNAL

template <bool dummy = true>
struct partitioning_scheme {
    static_assert(dummy, "dummy parameter should be true");

    typedef std::function< double(const backend::command_queue&) > weight_function;

    static void set(weight_function f) {
        if (!is_set) {
            weight = f;
            is_set = true;
        } else {
            std::cerr <<
                "Warning: "
                "device weighting function is already set and will be left as is."
                << std::endl;
        }
    }

    static std::vector<size_t> get(size_t n, const std::vector<backend::command_queue> &queue);

    private:
        static bool is_set;
        static weight_function weight;
        static std::map<backend::device_id, double> device_weight;
};

template <bool dummy>
bool partitioning_scheme<dummy>::is_set = false;

template <bool dummy>
std::map<backend::device_id, double> partitioning_scheme<dummy>::device_weight;

template <bool dummy>
std::vector<size_t> partitioning_scheme<dummy>::get(size_t n,
        const std::vector<backend::command_queue> &queue)
{
    if (!is_set) {
        weight = device_vector_perf;
        is_set = true;
    }

    std::vector<size_t> part;
    part.reserve(queue.size() + 1);
    part.push_back(0);

    if (queue.size() > 1) {
        std::vector<double> cumsum;
        cumsum.reserve(queue.size() + 1);
        cumsum.push_back(0);

        for(auto q = queue.begin(); q != queue.end(); q++) {
            auto dev_id = backend::get_device_id(*q);
            auto dw = device_weight.find(dev_id);

            double w = (dw == device_weight.end()) ?
                (device_weight[dev_id] = weight(*q)) :
                dw->second;

            cumsum.push_back(cumsum.back() + w);
        }

        for(unsigned d = 1; d < queue.size(); d++)
            part.push_back(
                    std::min(n,
                        alignup(static_cast<size_t>(n * cumsum[d] / cumsum.back()))
                        )
                    );
    }

    part.push_back(n);
    return part;
}

template <bool dummy>
typename partitioning_scheme<dummy>::weight_function partitioning_scheme<dummy>::weight;

/// \endcond

/// Partitioning scheme for vectors and matrices.
/**
 * Should be set once before any object of vector or matrix type is declared.
 * Otherwise default parttioning function (partition_by_vector_perf) is
 * selected.
 */
inline void set_partitioning(
        std::function< double(const backend::command_queue&) > f
        )
{
    partitioning_scheme<>::set(f);
}

/// Returns partitioning for the specified vector size on a given set of queues.
inline std::vector<size_t> partition(size_t n,
            const std::vector<backend::command_queue> &queue)
{
    return partitioning_scheme<>::get(n, queue);
}

/// \cond INTERNAL

//--- Vector Type -----------------------------------------------------------
struct vector_terminal {};

typedef vector_expression<
    typename boost::proto::terminal< vector_terminal >::type
    > vector_terminal_expression;

namespace traits {

// Hold vector terminals by reference:
template <class T>
struct hold_terminal_by_reference< T,
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr< T >::type,
                boost::proto::terminal< vector_terminal >
            >::value
        >::type
    >
    : std::true_type
{ };

} // namespace traits

/// \endcond

/// Device vector.
template <typename T>
class vector : public vector_terminal_expression {
    public:
        typedef T      value_type;
        typedef size_t size_type;

        /// Proxy class.
        /**
         * Instances of this class are returned from vector::operator[]. These
         * may be used to read or write single element of a vector, although
         * this operations are too expensive to be used extensively and should
         * be reserved for debugging purposes.
         */
        class element {
            public:
                /// Read associated element of a vector.
                operator T() const {
                    T val;
                    buf.read(queue, index, 1, &val, true);
                    return val;
                }

                /// Write associated element of a vector.
                T operator=(T val) {
                    buf.write(queue, index, 1, &val, true);
                    return val;
                }

                T operator=(const element &other) {
                    return (*this) = static_cast<T>(other);
                }

                friend void swap(element &&a, element &&b) {
                    T tmp = static_cast<T>(a);
                    a     = static_cast<T>(b);
                    b     = tmp;
                }

            private:
                element(const backend::command_queue &q,
                        const backend::device_vector<T> &b,
                        size_t i
                        ) : queue(q), buf(b), index(i)
                {}

                const backend::command_queue    &queue;
                const backend::device_vector<T> &buf;

                size_t index;

                friend class vector;
        };

        /// Iterator class.
        /**
         * This class may in principle be used with standard template library,
         * although its main purpose is range specification for vector copy
         * operations.
         */
        template <class vector_type, class element_type>
        class iterator_type
            : public boost::iterator_facade<
                        iterator_type<vector_type, element_type>,
                        T,
                        std::random_access_iterator_tag,
                        element_type
                     >
        {
            public:
                typedef boost::iterator_facade<
                            iterator_type<vector_type, element_type>,
                            T,
                            std::random_access_iterator_tag,
                            element_type
                         > super_type;
                typedef typename super_type::reference       reference;
                typedef typename super_type::difference_type difference_type;

                static const bool device_iterator = true;

                vector_type *vec;
                size_t  pos;
                size_t  part;

            private:
                friend class ::boost::iterator_core_access;
                friend class vector;

                iterator_type(vector_type &vec, size_t pos)
                    : vec(&vec), pos(pos), part(0)
                {
                    if (!vec.part.empty()) {
                        part = std::upper_bound(
                                vec.part.begin(), vec.part.end(), pos
                                ) - vec.part.begin() - 1;
                    }
                }

                reference dereference() const {
                    return element_type(
                            vec->queue[part], vec->buf[part],
                            pos - vec->part[part]
                            );
                }

                bool equal(const iterator_type &it) const {
                    return pos == it.pos;
                }

                void increment() {
                    ++pos;
                    while (part < vec->nparts() && pos >= vec->part[part + 1])
                        ++part;
                }

                void decrement() {
                    --pos;
                    while (part > 0 && pos < vec->part[part])
                        --part;
                }

                void advance(difference_type n) {
                    pos += n;
                    if (n > 0) {
                        while (part < vec->nparts() && pos >= vec->part[part + 1])
                            ++part;
                    } else if (n < 0) {
                        while (part > 0 && pos < vec->part[part])
                            --part;
                    }
                }

                difference_type distance_to(const iterator_type &it) const {
                    return static_cast<difference_type>(it.pos - pos);
                }
        };

        typedef iterator_type<vector, element> iterator;
        typedef iterator_type<const vector, const element> const_iterator;

        /// Empty constructor.
        vector() {}

        /// Copy constructor.
        vector(const vector &v) : queue(v.queue), part(v.part)
        {
#ifdef VEXCL_SHOW_COPIES
            std::cout << "Copying vex::vector<" << type_name<T>()
                      << "> of size " << size() << std::endl;
#endif
            if (size()) allocate_buffers(backend::MEM_READ_WRITE, 0);
            *this = v;
        }

        /// Wrap a native buffer
        vector(const backend::command_queue &q,
               const backend::device_vector<T> &buffer,
               size_t size = 0
               ) : queue(1, q), part(2), buf(1, buffer)
        {
            part[0] = 0;
            part[1] = size ? size : buffer.size();
        }

        /// Copy host data to the new buffer.
        vector(const std::vector<backend::command_queue> &queue,
                size_t size, const T *host = 0,
                backend::mem_flags flags = backend::MEM_READ_WRITE
              ) : queue(queue), part(vex::partition(size, queue))
        {
            if (size) allocate_buffers(flags, host);
        }

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
        /// Copy host data to the new buffer, use static context.
        vector(size_t size, const T *host = 0,
                backend::mem_flags flags = backend::MEM_READ_WRITE
              ) : queue(current_context().queue()), part(vex::partition(size, queue))
        {
            if (size) allocate_buffers(flags, host);
        }
#endif

        /// Copy host data to the new buffer.
        vector(const std::vector<backend::command_queue> &queue,
                const std::vector<T> &host,
                backend::mem_flags flags = backend::MEM_READ_WRITE
              ) : queue(queue), part(vex::partition(host.size(), queue))
        {
            if (!host.empty()) allocate_buffers(flags, host.data());
        }

#ifndef VEXCL_NO_STATIC_CONTEXT_CONSTRUCTORS
        /// Copy host data to the new buffer, use static context.
        vector(const std::vector<T> &host,
                backend::mem_flags flags = backend::MEM_READ_WRITE
              ) : queue(current_context().queue()), part(vex::partition(host.size(), queue))
        {
            if (!host.empty()) allocate_buffers(flags, host.data());
        }
#endif

        /// Move constructor
        vector(vector &&v) noexcept {
            swap(v);
        }

        /// Construct new vector from vector expression.
        /**
         * Vector expression should contain at least one vector for the
         * constructor to be able to determine queues and size to use.
         */
        template <class Expr
#ifndef BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
            , class Enable = typename std::enable_if<
            !std::is_integral<Expr>::value &&
                boost::proto::matches<
                    typename boost::proto::result_of::as_expr<Expr>::type,
                    vector_expr_grammar
                >::value
            >::type
#endif
        >
        vector(const Expr &expr) {
#ifdef BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
            static_assert(
                boost::proto::matches<
                    typename boost::proto::result_of::as_expr<Expr>::type,
                    vector_expr_grammar
                >::value,
                "Only vector expressions can be used to initialize a vector"
                );
#endif
            detail::get_expression_properties prop;
            detail::extract_terminals()(expr, prop);

            precondition(!prop.queue.empty() && !prop.part.empty(),
                    "Can not determine expression size and queue list"
                    );

            queue = prop.queue;
            part  = prop.part;

            allocate_buffers(backend::MEM_READ_WRITE, 0);

            *this = expr;
        }

        /// Move assignment
        const vector& operator=(vector &&v) {
            swap(v);
            return *this;
        }

        /// Swap function.
        void swap(vector &v) {
            std::swap(queue,   v.queue);
            std::swap(part,    v.part);
            std::swap(buf,     v.buf);
        }

        /// Resize vector.
        void resize(const vector &v, backend::mem_flags flags = backend::MEM_READ_WRITE)
        {
            // Reallocate bufers
            *this = std::move(vector(v.queue, v.size(), 0, flags));

            // Copy data
            *this = v;
        }

        /// Resize vector.
        void resize(const std::vector<backend::command_queue> &queue,
                size_t size, const T *host = 0,
                backend::mem_flags flags = backend::MEM_READ_WRITE
                )
        {
            *this = std::move(vector(queue, size, host, flags));
        }

        /// Resize vector.
        void resize(const std::vector<backend::command_queue> &queue,
                const std::vector<T> &host,
                backend::mem_flags flags = backend::MEM_READ_WRITE
              )
        {
            *this = std::move(vector(queue, host, flags));
        }

        /// Resize vector with static context.
        void resize(size_t size, const T *host = 0, backend::mem_flags flags = backend::MEM_READ_WRITE)
        {
            vector(size, host, flags).swap(*this);
        }

        /// Fills vector with zeros.
        void clear() {
            *this = static_cast<T>(0);
        }

        /// Return memory buffer located on a given device.
        const backend::device_vector<T>& operator()(unsigned d = 0) const {
            return buf[d];
        }

        /// Return memory buffer located on a given device.
        backend::device_vector<T>& operator()(unsigned d = 0) {
            return buf[d];
        }

        /// Const iterator to beginning.
        const_iterator begin() const {
            return const_iterator(*this, 0);
        }

        /// Const iterator to end.
        const_iterator end() const {
            return const_iterator(*this, size());
        }

        /// Iterator to beginning.
        iterator begin() {
            return iterator(*this, 0);
        }

        /// Iterator to end.
        iterator end() {
            return iterator(*this, size());
        }

        /// Access element.
        const element operator[](size_t index) const {
            size_t d = std::upper_bound(
                    part.begin(), part.end(), index) - part.begin() - 1;
            return element(queue[d], buf[d], index - part[d]);
        }

        /// Access element.
        element operator[](size_t index) {
            unsigned d = static_cast<unsigned>(
                std::upper_bound(part.begin(), part.end(), index) - part.begin() - 1
                );
            return element(queue[d], buf[d], index - part[d]);
        }

        /// Return size .
        size_t size() const {
            return part.empty() ? 0 : part.back();
        }

        /// Return number of parts (devices).
        size_t nparts() const {
            return queue.size();
        }

        /// Return size of part on a given device.
        size_t part_size(unsigned d) const {
            return part[d + 1] - part[d];
        }

        /// Return part start for a given device.
        size_t part_start(unsigned d) const {
            return part[d];
        }

        /// Return reference to vector's queue list
        const std::vector<backend::command_queue>& queue_list() const {
            return queue;
        }

        /// Return reference to vector's partition.
        const std::vector<size_t>& partition() const {
            return part;
        }

        const vector& operator=(const vector &x) {
            if (&x != this)
                detail::assign_expression<assign::SET>(*this, x, queue, part);
            return *this;
        }

        /// Maps device buffer to host array.
        typename backend::device_vector<T>::mapped_array
        map(unsigned d = 0) {
            return buf[d].map(queue[d]);
        }

#ifdef DOXYGEN
#define VEXCL_ASSIGNMENT(cop, op)                                              \
  /** \brief Vector expression assignment.
   * \details The appropriate kernel is compiled first time the assignment is
   * made. Vectors participating in expression should have same number of
   * parts; corresponding parts of the vectors should reside on the same
   * compute devices.
   */                                                                          \
  template <class Expr> const vector &operator cop(const Expr & expr);
#else
#define VEXCL_ASSIGNMENT(cop, op)                                              \
  template <class Expr>                                                        \
  typename std::enable_if<                                                     \
      boost::proto::matches<                                                   \
          typename boost::proto::result_of::as_expr<Expr>::type,               \
          vector_expr_grammar>::value,                                         \
      const vector &>::type operator cop(const Expr & expr) {                  \
    detail::assign_expression<op>(*this, expr, queue, part);                   \
    return *this;                                                              \
  }
#endif

        VEXCL_ASSIGNMENT(=,   assign::SET)
        VEXCL_ASSIGNMENT(+=,  assign::ADD)
        VEXCL_ASSIGNMENT(-=,  assign::SUB)
        VEXCL_ASSIGNMENT(*=,  assign::MUL)
        VEXCL_ASSIGNMENT(/=,  assign::DIV)
        VEXCL_ASSIGNMENT(%=,  assign::MOD)
        VEXCL_ASSIGNMENT(&=,  assign::AND)
        VEXCL_ASSIGNMENT(|=,  assign::OR)
        VEXCL_ASSIGNMENT(^=,  assign::XOR)
        VEXCL_ASSIGNMENT(<<=, assign::LSH)
        VEXCL_ASSIGNMENT(>>=, assign::RSH)

#undef VEXCL_ASSIGNMENT

#ifndef DOXYGEN
        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/false>(
                    *this, detail::simplify_additive_transform()( expr )
                    );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator+=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/true>(
                    *this, detail::simplify_additive_transform()( expr )
                    );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator-=(const Expr &expr) {
            detail::apply_additive_transform</*append=*/true>(
                    *this, detail::simplify_additive_transform()( -expr )
                    );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                vector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator=(const Expr &expr) {
            *this  = detail::extract_vector_expressions()( expr );
            *this += detail::extract_additive_vector_transforms()( expr );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                vector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator+=(const Expr &expr) {
            *this += detail::extract_vector_expressions()( expr );
            *this += detail::extract_additive_vector_transforms()( expr );

            return *this;
        }

        template <class Expr>
        typename std::enable_if<
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                vector_expr_grammar
            >::value &&
            !boost::proto::matches<
                typename boost::proto::result_of::as_expr<Expr>::type,
                additive_vector_transform_grammar
            >::value,
            const vector&
        >::type
        operator-=(const Expr &expr) {
            *this -= detail::extract_vector_expressions()( expr );
            *this -= detail::extract_additive_vector_transforms()( expr );

            return *this;
        }
#endif

        /// Copy data from host buffer to device(s).
        void write_data(size_t offset, size_t size, const T *hostptr, bool blocking)
        {
            if (!size) return;

            for(unsigned d = 0; d < queue.size(); d++) {
                size_t start = std::max(offset,        part[d]);
                size_t stop  = std::min(offset + size, part[d + 1]);

                if (stop <= start) continue;

                buf[d].write(queue[d], start - part[d], stop - start, hostptr + start - offset);
            }

            if (blocking)
                for(size_t d = 0; d < queue.size(); d++) {
                    size_t start = std::max(offset,        part[d]);
                    size_t stop  = std::min(offset + size, part[d + 1]);

                    if (start < stop) queue[d].finish();
                }
        }

        /// Copy data from device(s) to host buffer .
        void read_data(size_t offset, size_t size, T *hostptr, bool blocking) const
        {
            if (!size) return;

            for(unsigned d = 0; d < queue.size(); d++) {
                size_t start = std::max(offset,        part[d]);
                size_t stop  = std::min(offset + size, part[d + 1]);

                if (stop <= start) continue;

                buf[d].read(queue[d], start - part[d], stop - start, hostptr + start - offset);
            }

            if (blocking)
                for(unsigned d = 0; d < queue.size(); d++) {
                    size_t start = std::max(offset,        part[d]);
                    size_t stop  = std::min(offset + size, part[d + 1]);

                    if (start < stop) queue[d].finish();
                }
        }

    private:
        std::vector<backend::command_queue>      queue;
        std::vector<size_t>                      part;
        std::vector< backend::device_vector<T> > buf;

        void allocate_buffers(backend::mem_flags flags, const T *hostptr) {
            buf.clear();
            buf.reserve(queue.size());

            for(unsigned d = 0; d < queue.size(); d++)
                buf.push_back(
                        backend::device_vector<T>(
                            queue[d], part[d + 1] - part[d],
                            hostptr ? hostptr + part[d] : 0, flags)
                        );
        }

        template <typename S, size_t N>
        friend class multivector;
};

//---------------------------------------------------------------------------
// Support for vector expressions
//---------------------------------------------------------------------------
namespace traits {

template <>
struct is_vector_expr_terminal< vector_terminal > : std::true_type {};

template <>
struct proto_terminal_is_value< vector_terminal > : std::true_type {};

template <typename T>
struct kernel_param_declaration< vector<T> > {
    static void get(backend::source_generator &src,
            const vector<T>&,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        src.parameter< global_ptr<T> >(prm_name);
    }
};

template <typename T>
struct partial_vector_expr< vector<T> > {
    static void get(backend::source_generator &src,
            const vector<T>&,
            const backend::command_queue&, const std::string &prm_name,
            detail::kernel_generator_state_ptr)
    {
        src << prm_name << "[idx]";
    }
};

template <typename T>
struct kernel_arg_setter< vector<T> > {
    static void set(const vector<T> &term,
            backend::kernel &kernel, unsigned device, size_t/*index_offset*/,
            detail::kernel_generator_state_ptr)
    {
        kernel.push_arg(term(device));
    }
};

template <class T>
struct expression_properties< vector<T> > {
    static void get(const vector<T> &term,
            std::vector<backend::command_queue> &queue_list,
            std::vector<size_t> &partition,
            size_t &size
            )
    {
        queue_list = term.queue_list();
        partition  = term.partition();
        size       = term.size();
    }
};

} // namespace traits

//---------------------------------------------------------------------------
/// Copy device vector to host vector.
template <class T>
void copy(const vex::vector<T> &dv, std::vector<T> &hv, bool blocking = true) {
    dv.read_data(0, dv.size(), hv.data(), blocking);
}

/// Copy device vector to host pointer.
template <class T>
void copy(const vex::vector<T> &dv, T *hv, bool blocking = true) {
    dv.read_data(0, dv.size(), hv, blocking);
}

/// Copy host vector to device vector.
template <class T>
void copy(const std::vector<T> &hv, vex::vector<T> &dv, bool blocking = true) {
    dv.write_data(0, dv.size(), hv.data(), blocking);
}

/// Copy host pointer to device vector.
template <class T>
void copy(const T *hv, vex::vector<T> &dv, bool blocking = true) {
    dv.write_data(0, dv.size(), hv, blocking);
}

/// \cond INTERNAL

template<class Iterator, class Enable = void>
struct stored_on_device : std::false_type {};

template<class Iterator>
struct stored_on_device<Iterator,
    typename std::enable_if<Iterator::device_iterator>::type
    > : std::true_type {};

/// \endcond

/// Copy range from device vector to host vector.
template<class InputIterator, class OutputIterator>
#ifdef DOXYGEN
OutputIterator
#else
typename std::enable_if<
    std::is_same<
        typename std::iterator_traits<InputIterator>::value_type,
        typename std::iterator_traits<OutputIterator>::value_type
        >::value &&
    stored_on_device<InputIterator>::value &&
    !stored_on_device<OutputIterator>::value,
    OutputIterator
    >::type
#endif
copy(InputIterator first, InputIterator last,
        OutputIterator result, bool blocking = true)
{
    first.vec->read_data(first.pos, last - first, &result[0], blocking);
    return result + (last - first);
}

/// Copy range from host vector to device vector.
template<class InputIterator, class OutputIterator>
#ifdef DOXYGEN
OutputIterator
#else
typename std::enable_if<
    std::is_same<
        typename std::iterator_traits<InputIterator>::value_type,
        typename std::iterator_traits<OutputIterator>::value_type
        >::value &&
    !stored_on_device<InputIterator>::value &&
    stored_on_device<OutputIterator>::value,
    OutputIterator
    >::type
#endif
copy(InputIterator first, InputIterator last,
        OutputIterator result, bool blocking = true)
{
    result.vec->write_data(result.pos, last - first, &first[0], blocking);
    return result + (last - first);
}

/// Swap two vectors.
template <typename T>
void swap(vector<T> &x, vector<T> &y) {
    x.swap(y);
}

/// Returns device weight after simple bandwidth test
inline double device_vector_perf(const backend::command_queue &q) {
    static const size_t test_size = 1024U * 1024U;
    std::vector<backend::command_queue> queue(1, q);

    // Allocate test vectors on current device and measure execution
    // time of a simple kernel.
    vex::vector<float> a(queue, test_size);
    vex::vector<float> b(queue, test_size);
    vex::vector<float> c(queue, test_size);

    // Skip the first run.
    a = b + c;

    // Measure the second run.
    profiler<> prof(queue);
    prof.tic_cl("");
    a = b + c;
    return 1.0 / prof.toc("");
}


/// Download and print the vector elements.
template<class T>
std::ostream &operator<<(std::ostream &o, const vex::vector<T> &t) {
    boost::io::ios_all_saver stream_state(o);
    const size_t chunk = std::is_integral<T>::value ? 10 : 5;

    std::vector<T> data(t.size());
    copy(t, data);

    o << "{" << std::setprecision(6);
    for(size_t i = 0 ; i < data.size() ; i++) {
        if (i % chunk == 0) o << "\n" << std::setw(6) << i << ":";

        if (std::is_integral<T>::value)
            o << " " << std::setw(6) << data[i];
        else if (std::is_arithmetic<T>::value)
            o << std::scientific << std::setw(14) << data[i];
        else
            o << " " << data[i];
    }
    return o << "\n}\n";
}

} // namespace vex

namespace boost { namespace fusion { namespace traits {

template <class T>
struct is_sequence< vex::vector<T> > : std::false_type
{};

} } }


#endif
