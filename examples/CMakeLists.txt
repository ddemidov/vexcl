function(add_vexcl_examples example)
    if (VEXCL_BACKEND MATCHES "All")
        vexcl_add_executables(${example} ${example}.cpp)
        target_link_libraries(${example}
            INTERFACE
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            )
    else()
        add_executable(${example} ${example}.cpp)
        target_link_libraries(${example}
            PUBLIC
            VexCL::Backend
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            )
    endif()
endfunction()

#----------------------------------------------------------------------------
# General examples
#----------------------------------------------------------------------------
add_vexcl_examples(devlist)
add_vexcl_examples(fft_profile)
add_vexcl_examples(benchmark)
add_vexcl_examples(mba_benchmark)
add_vexcl_examples(complex_simple)
add_vexcl_examples(complex_spmv)


if (VEXCL_BACKEND MATCHES "All")
    set(LIB_TYPE "INTERFACE")
else()
    set(LIB_TYPE "PUBLIC")
endif()

if (TARGET compute_target)
    target_link_libraries(benchmark ${LIB_TYPE} compute_target)
endif()

if (TARGET clogs_target)
    target_link_libraries(benchmark ${LIB_TYPE} clogs_target)
endif()

if (VEXCL_BACKEND MATCHES "CUDA")
    target_link_libraries(benchmark PUBLIC CUDA::cusparse)
elseif (VEXCL_BACKEND MATCHES "All" AND TARGET VexCL::CUDA)
    target_link_libraries(benchmark_cuda PUBLIC CUDA::cusparse)
endif()

if (VEXCL_BACKEND MATCHES "OpenCL")
    add_vexcl_examples(exclusive)
elseif(VEXCL_BACKEND MATCHES "All" AND TARGET VexCL::OpenCL)
    add_executable(exclusive exclusive.cpp)
    target_link_libraries(exclusive PUBLIC VexCL::OpenCL ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()

#----------------------------------------------------------------------------
# Symbolic example uses Boost.odeint available since Boost v1.53
#----------------------------------------------------------------------------
if (${Boost_VERSION} VERSION_GREATER_EQUAL "1.53.0")
    add_vexcl_examples(symbolic)
endif ()

#----------------------------------------------------------------------------
# FFT benchmark
#----------------------------------------------------------------------------
add_vexcl_examples(fft_benchmark)

find_package(FFTW QUIET)
if (FFTW_FOUND)
    target_include_directories(fft_benchmark ${LIB_TYPE} ${FFTW_INCLUDES})
    target_compile_definitions(fft_benchmark ${LIB_TYPE} VEXCL_HAVE_FFTW)
    target_link_libraries(fft_benchmark ${LIB_TYPE} ${FFTW_LIBRARIES})
endif ()

if (CUDAToolkit_FOUND)
    target_link_libraries(fft_benchmark ${LIB_TYPE} CUDA::cufft CUDA::cudart)
    target_compile_definitions(fft_benchmark ${LIB_TYPE} VEXCL_HAVE_CUDA)
endif()

#----------------------------------------------------------------------------
# Thrust interoperation examples
#----------------------------------------------------------------------------
if (VEXCL_BACKEND MATCHES "CUDA")
    add_executable(thrust-sort thrust-sort.cpp thrust-sort.cu)
    target_link_libraries(thrust-sort VexCL::CUDA)
    set_target_properties(thrust-sort PROPERTIES CUDA_STANDARD 17)
    target_compile_options(thrust-sort PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
endif()

#----------------------------------------------------------------------------
add_subdirectory(viennacl)

