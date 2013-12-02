cmake_minimum_required(VERSION 2.8)
project(VexCL)

include_directories( ${CMAKE_SOURCE_DIR} )

#----------------------------------------------------------------------------
# Show generated kernels in tests and examples
#----------------------------------------------------------------------------
option(VEXCL_SHOW_KERNELS "Show generated kernels in tests and examples")
if (VEXCL_SHOW_KERNELS)
    add_definitions(-DVEXCL_SHOW_KERNELS)
endif ()

option(VEXCL_CACHE_KERNELS "Cache compiled kernels offline" ON)
if (VEXCL_CACHE_KERNELS)
    add_definitions(-DVEXCL_CACHE_KERNELS)
endif ()

option(VEXCL_SHOW_COPIES "Log vector copies to stdout for debugging purposes")
if (VEXCL_SHOW_COPIES)
    add_definitions(-DVEXCL_SHOW_COPIES)
endif ()

set(VEXCL_CHECK_SIZES 2 CACHE STRING "Check that expressions have correct sizes")
add_definitions(-DVEXCL_CHECK_SIZES=${VEXCL_CHECK_SIZES})

option(BOOST_TEST_DYN_LINK
    "Link tests against dynamic version of boost unittest library"
    ON)

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
    add_definitions(-D_VARIADIC_MAX=10)
else ()
  if (BOOST_TEST_DYN_LINK)
    add_definitions(-DBOOST_TEST_DYN_LINK)
  endif ()
endif ()

set(BOOST_COMPONENTS
    date_time
    filesystem
    system
    unit_test_framework
    program_options
    )

if (MSVC10)
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} chrono)
endif ()

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})

include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Protect Visual Studio from itself
#----------------------------------------------------------------------------
if (WIN32)
    add_definitions(-DNOMINMAX)
endif ()

set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA)")
set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA")

#----------------------------------------------------------------------------
# Find Backend
#----------------------------------------------------------------------------
if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    find_package(OpenCL REQUIRED)
    include_directories( ${OPENCL_INCLUDE_DIRS} )
    set(BACKEND_LIBS ${OPENCL_LIBRARIES})
    add_definitions(-DVEXCL_BACKEND_OPENCL)
elseif ("${VEXCL_BACKEND}" STREQUAL "CUDA")
    find_package(CUDA REQUIRED)
    include_directories( ${CUDA_INCLUDE_DIRS} )
    set(BACKEND_LIBS ${CUDA_CUDA_LIBRARY})
    add_definitions(-DVEXCL_BACKEND_CUDA)
endif()

#----------------------------------------------------------------------------
# Find OpenMP
#----------------------------------------------------------------------------
find_package(OpenMP)

if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter -Wno-comment -Wno-type-limits")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wno-comment -Wno-tautological-compare")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif ()
endif ()

#----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

add_subdirectory(examples)
add_subdirectory(doc)
add_subdirectory(cmake)

install(DIRECTORY vexcl DESTINATION include)
