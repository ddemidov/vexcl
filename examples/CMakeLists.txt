set(EXAMPLES
    benchmark
    devlist
    exclusive
    utests
    )

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (WIN32)
    # TODO: What to do with AMD platform?
    set(OpenCL_INCPATH "$ENV{CUDA_INC_PATH}")
    set(OpenCL_LIBPATH "$ENV{CUDA_LIB_PATH}")

    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost COMPONENTS system date_time thread chrono)
else (WIN32)
    find_package(Boost COMPONENTS system thread chrono)
endif (WIN32)

find_package(OpenCL REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OPENCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    )

find_path(ODEINT boost/numeric/odeint.hpp)
if (ODEINT)
    include_directories(${ODEINT})
    set(EXAMPLES ${EXAMPLES} symbolic)
endif (ODEINT)

find_package(OpenMP)

if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OpenMP_CXX_FLAGS)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter -Werror")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif (USE_LIBCPP)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach(example)

option(VIENNACL_EXAMPLES "Compile ViennaCL examples" OFF)

if (VIENNACL_EXAMPLES)
    add_subdirectory(viennacl)
endif (VIENNACL_EXAMPLES)

option(MPI_EXAMPLES "Compile MPI examples" ON)

if (MPI_EXAMPLES)
    add_subdirectory(mpi)
endif (MPI_EXAMPLES)

option(FFT_EXAMPLES "Compile FFT examples" OFF)
option(FFT_AMD "Use AMDFFT (env AMDFFTROOT)" OFF)
if (FFT_EXAMPLES)
    set(FFT_LIBS ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})

    # Use AMD FFT instead of ours
    if (FFT_AMD)
        find_library(AMDFFT_LIBRARY "clAmdFft.Runtime"
            HINTS "/opt/AMDAPP" ENV "AMDFFTROOT"
            PATH_SUFFIXES "lib64" "lib32")
        find_path(AMDFFT_INCLUDE_DIR "clAmdFft.h"
            HINTS "/opt/AMDAPP" ENV "AMDFFTROOT"
            PATH_SUFFIXES "include")
        set(FFT_LIBS ${FFT_LIBS} ${AMDFFT_LIBRARY})
        include_directories(${AMDFFT_INCLUDE_DIR} ${FFTW_INCLUDE_DIR})
        add_definitions(-DUSE_AMD_FFT)
    endif()

    # Verification, compare against FFTw
    find_library(FFTW_LIBRARY NAMES "fftw3f")
    find_path(FFTW_INCLUDE_DIR "fftw3.h")
    set(FFT_LIBS ${FFT_LIBS} ${FFTW_LIBRARY})
    include_directories(${FFTW_INCLUDE_DIR})
    add_executable(fft_verify fft_verify.cpp)
    target_link_libraries(fft_verify ${FFT_LIBS})

    # Benchmark, compare against CUDA if available
    # compare against multi threaded FFTw if available
    if(OpenMP_FOUND)
        find_library(FFTW_OMP_LIBRARY NAMES "fftw3f_omp")
        set(FFT_LIBS ${FFT_LIBS} ${FFTW_OMP_LIBRARY})
        add_definitions(-DUSE_OPENMP)
    endif()
    add_executable(fft_benchmark fft_benchmark.cpp)
    find_package(CUDA)
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        set(FFT_LIBS ${FFT_LIBS} ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
        add_definitions(-DUSE_CUDA=1)
    endif()
    target_link_libraries(fft_benchmark ${FFT_LIBS})
endif (FFT_EXAMPLES)
