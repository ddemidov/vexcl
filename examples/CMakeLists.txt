set(EXAMPLES
    benchmark
    cg
    devlist
    utests
    vector
    proto_test
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
    find_package(Boost COMPONENTS date_time QUIET)
    if (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	set(EXAMPLES ${EXAMPLES} exclusive)
    endif (Boost_FOUND)
else (WIN32)
    find_library(OPENCL_LIBRARY OpenCL)
    set(EXAMPLES ${EXAMPLES} exclusive)
endif (WIN32)

include_directories(${CMAKE_SOURCE_DIR})

find_path(ODEINT boost/numeric/odeint.hpp)
if (ODEINT)
    include_directories(${ODEINT})
    set(EXAMPLES ${EXAMPLES} symbolic)
endif (ODEINT)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++")
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    if (WIN32 AND "${example}" STREQUAL "exclusive")
	target_link_libraries(${example} ${Boost_DATE_TIME_LIBRARY})
    endif(WIN32 AND "${example}" STREQUAL "exclusive")
endforeach(example)
