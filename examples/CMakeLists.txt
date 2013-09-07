function(add_vexcl_example example)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})

    list(APPEND binaries "${CMAKE_CURRENT_BINARY_DIR}/${example}")
    list(APPEND sources  "${CMAKE_CURRENT_SOURCE_DIR}/${example}.cpp")
endfunction()

#----------------------------------------------------------------------------
# General examples
#----------------------------------------------------------------------------
add_vexcl_example(devlist)
add_vexcl_example(exclusive)
add_vexcl_example(benchmark)
add_vexcl_example(fft_benchmark)
add_vexcl_example(fft_profile)

find_path(MBA_INCLUDE mba/mba.hpp)
if (MBA_INCLUDE)
    include_directories(${MBA_INCLUDE})
    add_vexcl_example(mba_benchmark)
endif (MBA_INCLUDE)

#----------------------------------------------------------------------------
# Symbolic example uses Boost.odeint available since Boost v1.53
#----------------------------------------------------------------------------
if (NOT ${Boost_VERSION} STRLESS "105300")
    add_vexcl_example(symbolic)
endif (NOT ${Boost_VERSION} STRLESS "105300")

#----------------------------------------------------------------------------
# FFT benchmark
#----------------------------------------------------------------------------
find_package(FFTW QUIET)
if (FFTW_FOUND)
    include_directories(${FFTW_INCLUDES})
    add_definitions(-DHAVE_FFTW)
    set(FFT_BENCHMARK_LIBS ${FFT_BENCHMARK_LIBS} ${FFTW_LIBRARIES})
endif (FFTW_FOUND)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DHAVE_CUDA)
    set(FFT_BENCHMARK_LIBS ${FFT_BENCHMARK_LIBS} ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
endif(CUDA_FOUND)

target_link_libraries(fft_benchmark ${FFT_BENCHMARK_LIBS})

#----------------------------------------------------------------------------
# Install compiled examples and sources
#----------------------------------------------------------------------------
install(PROGRAMS ${binaries} DESTINATION share/vexcl/examples)
install(FILES    ${sources}  DESTINATION share/vexcl/examples/src)

#----------------------------------------------------------------------------
add_subdirectory(viennacl)
add_subdirectory(mpi)

