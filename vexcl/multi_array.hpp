#ifndef VEXCL_MULTI_ARRAY_HPP
#define VEXCL_MULTI_ARRAY_HPP

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
 * \file   vexcl/multi_array.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  An analog of boost::multi_array
 */

#include <vector>

#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>

namespace vex {

template <typename T, size_t NR, class Dimensions>
class multi_array_view
{
    public:
        typedef boost::mpl::size_t<boost::fusion::result_of::size<Dimensions>::type::value> ndim;
        typedef vector_view< T, gslice<NR> > base_type;

        multi_array_view(vector<T> &data, const gslice<NR> &slice)
            : data(data), slice(slice)
        {}

        const vector_view< T, gslice<NR> > vec() const {
            return slice(data);
        }

        vector_view< T, gslice<NR> > vec() {
            return slice(data);
        }

        template <size_t d>
        size_t size() const {
            static_assert(d < ndim::value, "Wrong dimension!");

            return slice.length[boost::fusion::result_of::value_at_c<Dimensions, d>::type::value];
        }
    private:
        vector<T>  &data;
        gslice<NR> slice;
};

template <typename T, size_t NR>
class multi_array
{
    public:
        typedef boost::mpl::size_t<NR> ndim;
        typedef vector<T> base_type;

        multi_array(
                const std::vector<cl::CommandQueue> &queue,
                const extent_gen<NR> &ext
                )
            : data(queue, ext.size()), slice(ext)
        {
            precondition(
                    queue.size() == 1,
                    "Multi-arrays are restricted to single-device contexts"
                    );
        }

        const vector<T>& vec() const {
            return data;
        }

        vector<T>& vec() {
            return data;
        }

        template <class Dim>
        const multi_array_view<T, NR, Dim> operator()(const index_gen<NR, Dim> &idx) const {
            return multi_array_view<T, NR, Dim>(const_cast<vector<T>&>(data), slice(idx));
        }

        template <class Dim>
        multi_array_view<T, NR, Dim> operator()(const index_gen<NR, Dim> &idx) {
            return multi_array_view<T, NR, Dim>(data, slice(idx));
        }

        template<size_t d>
        size_t size() const {
            static_assert(d < NR, "Wrong dimension!");

            return slice.dim[d];
        }

    private:
        vector<T>  data;
    public:
        slicer<NR> slice;
};

template <class RDC, typename T, size_t NDIM, size_t NR>
reduced_vector_view<vector<T>, NDIM, NR, RDC> reduce(
        const multi_array<T, NDIM> &m,
        const std::array<size_t, NR> &reduce_dims
        )
{
    return reduced_vector_view<vector<T>, NDIM, NR, RDC>(m.vec(), m.slice[_], reduce_dims);
}

template <class RDC, typename T, size_t NDIM>
reduced_vector_view<vector<T>, NDIM, 1, RDC> reduce(
        const multi_array<T, NDIM> &m,
        size_t reduce_dim
        )
{
    std::array<size_t, 1> dim = {{reduce_dim}};
    return reduced_vector_view<vector<T>, NDIM, 1, RDC>(m.vec(), m.slice[_], dim);
}

} // namespace vex


#endif
