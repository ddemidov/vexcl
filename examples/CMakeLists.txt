#----------------------------------------------------------------------------
# General examples
#----------------------------------------------------------------------------
set(EXAMPLES
    devlist
    exclusive
    benchmark
    fft_benchmark
    )

find_path(ODEINT boost/numeric/odeint.hpp)
if (ODEINT)
    include_directories(${ODEINT})
    set(EXAMPLES ${EXAMPLES} symbolic)
endif (ODEINT)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach(example)

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
foreach(file ${EXAMPLES})
    list(APPEND binaries "${CMAKE_CURRENT_BINARY_DIR}/${file}")
    list(APPEND source "${CMAKE_CURRENT_SOURCE_DIR}/${file}.cpp")
endforeach()

install(PROGRAMS ${binaries} DESTINATION share/vexcl/examples)
install(FILES ${source} DESTINATION share/vexcl/examples/src)

#----------------------------------------------------------------------------
add_subdirectory(viennacl)
add_subdirectory(mpi)

