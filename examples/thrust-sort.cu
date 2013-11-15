#include <thrust/sort.h>
#include <thrust/device_ptr.h>

//---------------------------------------------------------------------------
// NVCC is not yet able to compile C++11 code.
// Hence the need to keep Thrust and VexCL code in separate files.
//---------------------------------------------------------------------------
template <typename T>
void thrust_sort(T *begin, T *end) {
    thrust::sort(
            thrust::device_pointer_cast(begin),
            thrust::device_pointer_cast(end)
            );
}

//---------------------------------------------------------------------------
// Due to the code separation we also need to explicitly instantiate the
// necessary templates.
//---------------------------------------------------------------------------
template void thrust_sort<float>(float *begin, float *end);
template void thrust_sort<double>(double *begin, double *end);
