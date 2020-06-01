#ifndef VEXCL_BACKEND_HIP_CUSPARSE_HPP
#define VEXCL_BACKEND_HIP_CUSPARSE_HPP

#include <map>
#include <memory>

#include <hipsparse.h>

#include <vexcl/vector.hpp>
#include <vexcl/cache.hpp>
#include <vexcl/backend/hip/error.hpp>
#include <vexcl/backend/hip/context.hpp>
#include <vexcl/detail/backtrace.hpp>

namespace vex {
namespace backend {
namespace hip {

/// Send human-readable representation of hipError_t to the output stream.
inline std::ostream& operator<<(std::ostream &os, hipsparseStatus_t rc) {
    os << "CUSPARSE Error (";
#define VEXCL_HIP_ERR2TXT(e) case e: os << static_cast<int>(e) << " - " << #e; break
    switch(rc) {
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_SUCCESS);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_NOT_INITIALIZED);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_ALLOC_FAILED);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_INVALID_VALUE);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_ARCH_MISMATCH);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_MAPPING_ERROR);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_EXECUTION_FAILED);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_INTERNAL_ERROR);
        VEXCL_HIP_ERR2TXT(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        default:
            os << "Unknown error";
    }
#undef VEXCL_HIP_ERR2TXT
    return os << ")";
}

inline void check(hipsparseStatus_t rc, const char *file, int line) {
    if (rc != HIPSPARSE_STATUS_SUCCESS) {
        vex::detail::print_backtrace();
        throw error(rc, file, line);
    }
}

namespace detail {

template <>
struct deleter_impl<hipsparseHandle_t> {
    static void dispose(hipsparseHandle_t handle) {
        hip_check( hipsparseDestroy(handle) );
    }
};

template <>
struct deleter_impl<hipsparseMatDescr_t> {
    static void dispose(hipsparseMatDescr_t handle) {
        hip_check( hipsparseDestroyMatDescr(handle) );
    }
};

template <>
struct deleter_impl<hipsparseHybMat_t> {
    static void dispose(hipsparseHybMat_t handle) {
        hip_check( hipsparseDestroyHybMat(handle) );
    }
};

} // namespace detail

inline hipsparseHandle_t cusparse_handle(const command_queue &q) {
    typedef std::shared_ptr<std::remove_pointer<hipsparseHandle_t>::type> smart_handle;
    typedef vex::detail::object_cache<vex::detail::index_by_context, smart_handle> cache_type;

    static cache_type cache;

    auto h = cache.find(q);

    if (h == cache.end()) {
        select_context(q);
        hipsparseHandle_t handle;
        hip_check( hipsparseCreate(&handle) );
        hip_check( hipsparseSetStream(handle, q.raw()) );

        h = cache.insert(q, smart_handle(handle, detail::deleter(q.context().raw())));
    }

    return h->second.get();
}

template <typename val_t>
class spmat_hyb {
    static_assert(
            std::is_same<val_t, float>::value ||
            std::is_same<val_t, double>::value,
            "Unsupported value type for spmat_cusparse"
            );

    public:
        template <typename row_t, typename col_t>
        spmat_hyb(
                const command_queue &queue,
                int n, int m,
                const row_t *row_begin,
                const col_t *col_begin,
                const val_t *val_begin
                )
            : handle( cusparse_handle(queue) ),
              desc  ( create_description(), detail::deleter(queue.context().raw()) ),
              mat   ( create_matrix(),      detail::deleter(queue.context().raw()) )
        {
            hip_check( hipsparseSetMatType(desc.get(), HIPSPARSE_MATRIX_TYPE_GENERAL) );
            hip_check( hipsparseSetMatIndexBase(desc.get(), HIPSPARSE_INDEX_BASE_ZERO) );

            fill_matrix(queue, n, m, row_begin, col_begin, val_begin);
        }

        void apply(const vex::vector<val_t> &x, vex::vector<val_t> &y,
                 val_t alpha = 1, bool append = false) const
        {
            precondition(x.nparts() == 1 && y.nparts() == 1,
                    "Incompatible vectors");

            mul(x(0), y(0), alpha, append);
        }

        void mul(const device_vector<float> &x, device_vector<float> &y,
                 float alpha = 1, bool append = false) const
        {
            float beta = append ? 1.0f : 0.0f;

            hip_check(
                    hipsparseShybmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, desc.get(), mat.get(),
                        x.raw_ptr(), &beta, y.raw_ptr()
                        )
                    );
        }

        void mul(const device_vector<double> &x, device_vector<double> &y,
                 double alpha = 1, bool append = false) const
        {
            double beta = append ? 1.0 : 0.0;

            hip_check(
                    hipsparseDhybmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, desc.get(), mat.get(),
                        x.raw_ptr(), &beta, y.raw_ptr()
                        )
                    );
        }
    private:
        hipsparseHandle_t handle;

        std::shared_ptr<std::remove_pointer<hipsparseMatDescr_t>::type> desc;
        std::shared_ptr<std::remove_pointer<hipsparseHybMat_t>::type>   mat;

        static hipsparseMatDescr_t create_description() {
            hipsparseMatDescr_t desc;
            hip_check( hipsparseCreateMatDescr(&desc) );
            return desc;
        }

        static hipsparseHybMat_t create_matrix() {
            hipsparseHybMat_t mat;
            hip_check( hipsparseCreateHybMat(&mat) );
            return mat;
        }

        template <typename row_t, typename col_t>
        void fill_matrix(const command_queue &q,
                int n, int m, const row_t *row, const col_t *col, const float *val)
        {
            device_vector<int>   r(q, n + 1,  row);
            device_vector<int>   c(q, row[n], col + row[0]);
            device_vector<float> v(q, row[n], val + row[0]);

            if (row[0] != 0) vector<int>(q, r) -= row[0];

            hip_check(
                    hipsparseScsr2hyb(handle, n, m, desc.get(),
                        v.raw_ptr(), r.raw_ptr(), c.raw_ptr(), mat.get(), 0,
                        HIPSPARSE_HYB_PARTITION_AUTO
                        )
                    );
        }

        template <typename row_t, typename col_t>
        void fill_matrix(const command_queue &q,
                int n, int m, const row_t *row, const col_t *col, const double *val)
        {
            device_vector<int>    r(q, n + 1,  row);
            device_vector<int>    c(q, row[n], col + row[0]);
            device_vector<double> v(q, row[n], val + row[0]);

            if (row[0] != 0) vector<int>(q, r) -= row[0];

            hip_check(
                    hipsparseDcsr2hyb(handle, n, m, desc.get(),
                        v.raw_ptr(), r.raw_ptr(), c.raw_ptr(), mat.get(), 0,
                        HIPSPARSE_HYB_PARTITION_AUTO
                        )
                    );
        }

};

template <typename T>
additive_operator< spmat_hyb<T>, vector<T> >
operator*(const spmat_hyb<T> &A, const vector<T> &x) {
    return additive_operator< spmat_hyb<T>, vector<T> >(A, x);
}

template <typename val_t>
class spmat_crs {
    static_assert(
            std::is_same<val_t, float>::value ||
            std::is_same<val_t, double>::value,
            "Unsupported value type for spmat_cusparse"
            );

    public:
        template <typename row_t, typename col_t>
        spmat_crs(
                const command_queue &queue,
                int n, int m,
                const row_t *row_begin,
                const col_t *col_begin,
                const val_t *val_begin
                )
            : n(n), m(m), nnz(static_cast<unsigned>(row_begin[n] - row_begin[0])),
              handle( cusparse_handle(queue) ),
              desc  ( create_description(), detail::deleter(queue.context().raw()) ),
              row(queue, n+1, row_begin),
              col(queue, nnz, col_begin + row_begin[0]),
              val(queue, nnz, val_begin + row_begin[0])
        {
            if (row_begin[0] != 0)
                vector<int>(queue, row) -= row_begin[0];

            hip_check( hipsparseSetMatType(desc.get(), HIPSPARSE_MATRIX_TYPE_GENERAL) );
            hip_check( hipsparseSetMatIndexBase(desc.get(), HIPSPARSE_INDEX_BASE_ZERO) );
        }

        void apply(const vex::vector<val_t> &x, vex::vector<val_t> &y,
                 val_t alpha = 1, bool append = false) const
        {
            precondition(x.nparts() == 1 && y.nparts() == 1,
                    "Incompatible vectors");

            mul(x(0), y(0), alpha, append);
        }

        void mul(const device_vector<float> &x, device_vector<float> &y,
                 float alpha = 1, bool append = false) const
        {
            float beta = append ? 1.0f : 0.0f;

            hip_check(
                    hipsparseScsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        n, m, nnz, &alpha, desc.get(),
                        val.raw_ptr(), row.raw_ptr(), col.raw_ptr(),
                        x.raw_ptr(), &beta, y.raw_ptr()
                        )
                 );
        }

        void mul(const device_vector<double> &x, device_vector<double> &y,
                 double alpha = 1, bool append = false) const
        {
            double beta = append ? 1.0 : 0.0;

            hip_check(
                    hipsparseDcsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        n, m, nnz, &alpha, desc.get(),
                        val.raw_ptr(), row.raw_ptr(), col.raw_ptr(),
                        x.raw_ptr(), &beta, y.raw_ptr()
                        )
                 );
        }
    private:
        unsigned n, m, nnz;

        hipsparseHandle_t handle;

        std::shared_ptr<std::remove_pointer<hipsparseMatDescr_t>::type> desc;

        device_vector<int>   row;
        device_vector<int>   col;
        device_vector<val_t> val;

        static hipsparseMatDescr_t create_description() {
            hipsparseMatDescr_t desc;
            hip_check( hipsparseCreateMatDescr(&desc) );
            return desc;
        }
};

template <typename T>
additive_operator< spmat_crs<T>, vector<T> >
operator*(const spmat_crs<T> &A, const vector<T> &x) {
    return additive_operator< spmat_crs<T>, vector<T> >(A, x);
}

} // namespace hip
} // namespace backend
} // namespace vex

#endif
