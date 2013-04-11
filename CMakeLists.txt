cmake_minimum_required(VERSION 2.8)
project(VexCL CXX)

include_directories( ${CMAKE_SOURCE_DIR} )

#----------------------------------------------------------------------------
# Enable math constants in Win32
#----------------------------------------------------------------------------
if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif (WIN32)

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
else (WIN32)
    add_definitions(-DBOOST_TEST_DYN_LINK)
endif (WIN32)

find_package(Boost COMPONENTS
    chrono
    date_time
    filesystem
    system
    thread
    unit_test_framework
    )

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

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
