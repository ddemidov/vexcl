function(add_vexcl_example example)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example}
        VexCL::Backend
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        )
endfunction()

#----------------------------------------------------------------------------
# General examples
#----------------------------------------------------------------------------
add_vexcl_example(devlist)
add_vexcl_example(fft_profile)
add_vexcl_example(benchmark)
add_vexcl_example(mba_benchmark)

if (TARGET compute_target)
    target_link_libraries(benchmark compute_target)
endif()

if (TARGET clogs_target)
    target_link_libraries(benchmark clogs_target)
endif()

if ("${VEXCL_BACKEND}" STREQUAL "Maxeler")
    add_vexcl_example(maxeler)
endif()

if (VEXCL_BACKEND MATCHES "CUDA")
    target_link_libraries(benchmark ${CUDA_cusparse_LIBRARY})
endif()

if (VEXCL_BACKEND MATCHES "OpenCL")
    add_vexcl_example(exclusive)
endif()

#----------------------------------------------------------------------------
# Symbolic example uses Boost.odeint available since Boost v1.53
#----------------------------------------------------------------------------
if (NOT ${Boost_VERSION} STRLESS "105300")
    add_vexcl_example(symbolic)
endif ()

#----------------------------------------------------------------------------
# FFT benchmark
#----------------------------------------------------------------------------
add_vexcl_example(fft_benchmark)

find_package(FFTW QUIET)
if (FFTW_FOUND)
    target_include_directories(fft_benchmark PRIVATE ${FFTW_INCLUDES})
    target_compile_definitions(fft_benchmark PRIVATE VEXCL_HAVE_FFTW)
    target_link_libraries(fft_benchmark ${FFTW_LIBRARIES})
endif ()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_include_directories(fft_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(fft_benchmark PRIVATE VEXCL_HAVE_CUDA)
    target_link_libraries(fft_benchmark ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
endif()

#----------------------------------------------------------------------------
# Thrust interoperation examples
#----------------------------------------------------------------------------
if (VEXCL_BACKEND MATCHES "CUDA")
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
    list(APPEND CUDA_NVCC_FLAGS
        ${CUDA_ARCH_FLAGS}
        -Wno-deprecated-gpu-targets
        $<$<CXX_COMPILER_ID:GNU>:-Xcompiler -std=c++03>
        )
    cuda_add_executable(thrust-sort thrust-sort.cpp thrust-sort.cu)
    target_link_libraries(thrust-sort VexCL::CUDA)
endif()

#----------------------------------------------------------------------------
add_subdirectory(viennacl)

