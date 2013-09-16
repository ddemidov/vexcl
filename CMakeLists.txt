cmake_minimum_required(VERSION 2.8)
project(VexCL)

include_directories( ${CMAKE_SOURCE_DIR} )

#----------------------------------------------------------------------------
# Show generated kernels in tests and examples
#----------------------------------------------------------------------------
option(VEXCL_SHOW_KERNELS "Show generated kernels in tests and examples")
if (VEXCL_SHOW_KERNELS)
    add_definitions(-DVEXCL_SHOW_KERNELS)
endif (VEXCL_SHOW_KERNELS)

option(VEXCL_CACHE_KERNELS "Cache compiled kernels offline")
if (VEXCL_CACHE_KERNELS)
    add_definitions(-DVEXCL_CACHE_KERNELS)
endif (VEXCL_CACHE_KERNELS)

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
    add_definitions(-D_VARIADIC_MAX=10)
else (WIN32)
    add_definitions(-DBOOST_TEST_DYN_LINK)
endif (WIN32)

set(BOOST_COMPONENTS
    date_time
    filesystem
    system
    unit_test_framework
    program_options
    )

if (MSVC10)
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} chrono)
endif (MSVC10)

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})

include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Find OpenCL
#----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)

include_directories( ${OPENCL_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Find OpenMP
#----------------------------------------------------------------------------
find_package(OpenMP)

if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OpenMP_CXX_FLAGS)

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter -Wno-comment")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wno-comment")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif (USE_LIBCPP)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

#----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

add_subdirectory(examples)
add_subdirectory(doc)
add_subdirectory(cmake)

install(DIRECTORY vexcl DESTINATION include)
