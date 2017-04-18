#------------------------------------------------------------------------
find_package(Boost REQUIRED COMPONENTS program_options)

function(add_vexcl_example example)
    add_executable(${example} ${example}.cpp)

    if (NOT "${VEXCL_EXAMPLES_INCLUDES}" STREQUAL "")
        target_include_directories(${example} PRIVATE ${VEXCL_EXAMPLES_INCLUDES})
    endif()

    if (NOT "${VEXCL_EXAMPLES_DEFINITIONS}" STREQUAL "")
        target_compile_definitions(${example} PRIVATE ${VEXCL_EXAMPLES_DEFINITIONS})
    endif()

    target_link_libraries(${example} VexCL::Backend ${Boost_PROGRAM_OPTIONS_LIBRARY})
endfunction()

#----------------------------------------------------------------------------
# General examples
#----------------------------------------------------------------------------
add_vexcl_example(devlist)
add_vexcl_example(benchmark)
add_vexcl_example(fft_profile)

if ("${VEXCL_BACKEND}" STREQUAL "CUDA")
    target_link_libraries(benchmark ${CUDA_cusparse_LIBRARY})
endif()

if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
    add_vexcl_example(exclusive)
endif()

if (CLOGS)
    target_link_libraries(benchmark ${CLOGS_LIB})
endif()

find_path(MBA_INCLUDE mba/mba.hpp)
if (MBA_INCLUDE)
    add_vexcl_example(mba_benchmark)
    target_include_directories(mba_benchmark ${MBA_INCLUDE})
endif ()

#----------------------------------------------------------------------------
# Symbolic example uses Boost.odeint available since Boost v1.53
#----------------------------------------------------------------------------
if (NOT ${Boost_VERSION} STRLESS "105300")
    add_vexcl_example(symbolic)
endif ()

#----------------------------------------------------------------------------
# FFT benchmark
#----------------------------------------------------------------------------
set(VEXCL_FFT_INCLUDES    "")
set(VEXCL_FFT_DEFINITIONS "")
set(VEXCL_FFT_LIBRARIES   "")

find_package(FFTW QUIET)
if (FFTW_FOUND)
    list(APPEND VEXCL_FFT_INCLUDES    ${FFTW_INCLUDES})
    list(APPEND VEXCL_FFT_DEFINITIONS VEXCL_HAVE_FFTW)
    list(APPEND VEXCL_FFT_LIBRARIES   ${FFTW_LIBRARIES})
endif ()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    list(APPEND VEXCL_FFT_INCLUDES    ${CUDA_INCLUDE_DIRS})
    list(APPEND VEXCL_FFT_DEFINITIONS VEXCL_HAVE_CUDA)
    list(APPEND VEXCL_FFT_LIBRARIES   ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
endif()


add_vexcl_example(fft_benchmark)
if (NOT "${VEXCL_FFT_INCLUDES}" STREQUAL "")
    message(STATUS "FFTW: ${VEXCL_FFT_INCLUDES}")
    target_include_directories(fft_benchmark PRIVATE ${VEXCL_FFT_INCLUDES})
endif()
if (NOT "${VEXCL_FFT_DEFINITIONS}" STREQUAL "")
    target_compile_definitions(fft_benchmark PRIVATE ${VEXCL_FFT_DEFINITIONS})
endif()
if (NOT "${VEXCL_FFT_LIBRARIES}" STREQUAL "")
    target_link_libraries(fft_benchmark ${VEXCL_FFT_LIBRARIES})
endif()

#----------------------------------------------------------------------------
# Thrust interoperation examples
#----------------------------------------------------------------------------
if ("${VEXCL_BACKEND}" STREQUAL "CUDA")
    set(CUDA_OPTIONS "-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    if (NOT WIN32)
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_OPTIONS "${CUDA_OPTIONS} -Xcompiler -std=c++03")
    endif()
    cuda_add_executable(thrust-sort thrust-sort.cpp thrust-sort.cu OPTIONS ${CUDA_OPTIONS})
    target_link_libraries(thrust-sort ${Boost_LIBRARIES} ${BACKEND_LIBS})
endif()

#----------------------------------------------------------------------------
add_subdirectory(viennacl)

