set(EXAMPLES
    benchmark
    devlist
    exclusive
    utests
    )

if (WIN32)
    # TODO: What to do with AMD platform?
    find_library(OPENCL_LIBRARY OpenCL "$ENV{CUDA_LIB_PATH}")
    find_path(OPENCL_INCLUDE CL/opencl.h "$ENV{CUDA_INC_PATH}")
    include_directories(${OPENCL_INCLUDE})

    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost COMPONENTS system date_time thread chrono)
else (WIN32)
    find_library(OPENCL_LIBRARY OpenCL)
    find_package(Boost COMPONENTS system thread chrono)
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
    target_link_libraries(${example} ${OPENCL_LIBRARY} ${Boost_LIBRARIES})
endforeach(example)

option(VIENNACL_EXAMPLES "Compile ViennaCL examples" OFF)

if (VIENNACL_EXAMPLES)
    add_subdirectory(viennacl)
endif (VIENNACL_EXAMPLES)

option(MPI_EXAMPLES "Compile MPI examples" ON)

if (MPI_EXAMPLES)
    add_subdirectory(mpi)
endif (MPI_EXAMPLES)
