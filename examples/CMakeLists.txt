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
    find_library(OPENCL_LIBRARY OpenCL)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
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

find_package(MPI QUIET)
if (MPI_CXX_FOUND)
    add_executable(mpitests mpitests.cpp)
    set_target_properties(mpitests PROPERTIES
        COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
    target_link_libraries(mpitests ${MPI_CXX_LIBRARIES} OpenCL)

    if (ODEINT)
        add_executable(damped_oscillator damped_oscillator.cpp)
        set_target_properties(damped_oscillator PROPERTIES
            COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
            LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
            INCLUDE_DIRECTORIES "${ODEINT};${CMAKE_SOURCE_DIR}"
            )
        target_link_libraries(damped_oscillator ${MPI_CXX_LIBRARIES} OpenCL)
    endif(ODEINT)
endif (MPI_CXX_FOUND)
