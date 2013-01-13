set(EXAMPLES
    benchmark
    devlist
    exclusive
    utests
    )

if (WIN32)
    find_library(OPENCL_LIBRARY OpenCL
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/x64"
	)
    find_path(OPENCL_INCLUDE CL/opencl.h
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/include"
	)
    include_directories(${OPENCL_INCLUDE})

    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost COMPONENTS date_time)
else (WIN32)
    find_library(OPENCL_LIBRARY OpenCL
        HINTS "/opt/AMDAPP" ENV "AMDAPPSDKROOT"
        PATH_SUFFIXES "lib/x86_64" "lib/x86")
    find_package(Boost)
endif (WIN32)

include_directories(
    ${CMAKE_SOURCE_DIR}
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})

    if (WIN32)
	target_link_libraries(${example} ${Boost_DATE_TIME_LIBRARY})
    endif(WIN32)
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
    set(FFT_LIBS ${OPENCL_LIBRARY})

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
