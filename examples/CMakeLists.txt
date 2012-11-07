set(EXAMPLES
    benchmark
    cg
    devlist
    exclusive
    utests
    vector
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

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++")
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    if (WIN32)
	target_link_libraries(${example} ${Boost_DATE_TIME_LIBRARY})
    endif(WIN32)
endforeach(example)
