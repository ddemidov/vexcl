cmake_minimum_required(VERSION 2.8)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
endif()

project(VexCL)

set(VEXCL_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(VEXCL_MASTER_PROJECT ON)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if (CMAKE_VERSION VERSION_LESS "3.1.0")
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/opencl)
endif()

set(VEXCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(VEXCL_INCLUDES "")
set(VEXCL_LIBRARIES "")
set(VEXCL_DEFINITIONS "")
set(VEXCL_COMPILE_OPTIONS "")

#----------------------------------------------------------------------------
# Compile-time options
#----------------------------------------------------------------------------
option(VEXCL_SHOW_KERNELS "Show generated kernels in tests and examples" OFF)
if (VEXCL_SHOW_KERNELS)
    list(APPEND VEXCL_DEFINITIONS VEXCL_SHOW_KERNELS)
endif ()

option(VEXCL_CACHE_KERNELS "Cache compiled kernels offline" ON)
if (VEXCL_CACHE_KERNELS)
    list(APPEND VEXCL_DEFINITIONS VEXCL_CACHE_KERNELS)
endif ()

option(VEXCL_SHOW_COPIES "Log vector copies to stdout for debugging purposes" OFF)
if (VEXCL_SHOW_COPIES)
    list(APPEND VEXCL_DEFINITIONS VEXCL_SHOW_COPIES)
endif ()

set(VEXCL_CHECK_SIZES 0 CACHE STRING "Check that expressions have correct sizes")
list(APPEND VEXCL_DEFINITIONS VEXCL_CHECK_SIZES=${VEXCL_CHECK_SIZES})

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
set(VEXCL_BOOST_COMPONENTS date_time filesystem system thread
    CACHE STRING "Boost components required by VexCL.")

if (WIN32)
    set(VEXCL_BOOST_COMPONENTS ${VEXCL_BOOST_COMPONENTS} chrono)
    set(Boost_USE_STATIC_LIBS ON)
else ()
    # TODO
    option(BOOST_TEST_DYN_LINK "Link tests against dynamic version of boost unittest library" ON)
    if (BOOST_TEST_DYN_LINK)
        list(APPEND VEXCL_DEFINITIONS BOOST_TEST_DYN_LINK)
    endif ()
endif ()

find_package(Boost REQUIRED COMPONENTS ${VEXCL_BOOST_COMPONENTS})
list(APPEND VEXCL_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND VEXCL_LIBRARIES ${Boost_LIBRARIES})

#----------------------------------------------------------------------------
# Compile options
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND VEXCL_COMPILE_OPTIONS
        -std=c++0x
        -Wno-comment -Wno-type-limits -Wno-strict-aliasing -Wno-missing-braces
        -Wno-deprecated-declarations -Wno-unused-function
        -Wno-ignored-attributes
        )
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(APPEND VEXCL_COMPILE_OPTIONS
        -std=c++0x
        -Wno-comment -Wno-tautological-compare -Wno-missing-braces
        -Wno-deprecated-declarations -Wno-unused-function
        )

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        list(APPEND VEXCL_COMPILE_OPTIONS -stdlib=libc++)
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND VEXCL_DEFINITIONS NOMINMAX _VARIADIC_MAX=10)

    list(APPEND VEXCL_COMPILE_OPTIONS
        /bigobj
        /wd4996 # Deprecated declarations in CL/cl.hpp
        /wd4003 # Not enough actual parameters for a BOOST_PP macro
        )
endif ()

#----------------------------------------------------------------------------
# Find VexCL backends
#----------------------------------------------------------------------------
find_package(OpenCL)
if(OpenCL_FOUND)
    add_library(OpenCL INTERFACE)

    target_include_directories(OpenCL INTERFACE
        $<BUILD_INTERFACE:${VEXCL_INCLUDE_DIR}>
        ${VEXCL_INCLUDES} ${OpenCL_INCLUDE_DIRS})

    target_link_libraries(OpenCL INTERFACE
        ${VEXCL_LIBRARIES} ${OpenCL_LIBRARY})

    target_compile_definitions(OpenCL INTERFACE
        ${VEXCL_DEFINITIONS} VEXCL_BACKEND_OPENCL)

    target_compile_options(OpenCL INTERFACE
        ${VEXCL_COMPILE_OPTIONS})

    add_library(VexCL::OpenCL ALIAS OpenCL)

    message(STATUS "Found VexCL::OpenCL")

    if (NOT "${Boost_VERSION}" STRLESS "106100")
        add_library(Compute INTERFACE)

        target_include_directories(Compute INTERFACE
            $<BUILD_INTERFACE:${VEXCL_INCLUDE_DIR}>
            ${VEXCL_INCLUDES} ${OpenCL_INCLUDE_DIRS})

        target_link_libraries(Compute INTERFACE
            ${VEXCL_LIBRARIES} ${OpenCL_LIBRARY})

        target_compile_definitions(Compute INTERFACE
            ${VEXCL_DEFINITIONS} VEXCL_BACKEND_COMPUTE)

        target_compile_options(Compute INTERFACE
            ${VEXCL_COMPILE_OPTIONS})

        add_library(VexCL::Compute ALIAS Compute)

        message(STATUS "Found VexCL::Compute")
    endif()
endif()

find_package(CUDA)
if(CUDA_FOUND)
    add_library(CUDA INTERFACE)

    target_include_directories(CUDA INTERFACE
        $<BUILD_INTERFACE:${VEXCL_INCLUDE_DIR}>
        ${VEXCL_INCLUDES} ${CUDA_INCLUDE_DIRS})

    target_link_libraries(CUDA INTERFACE
        ${VEXCL_LIBRARIES} ${CUDA_CUDA_LIBRARY})

    target_compile_definitions(CUDA INTERFACE
        ${VEXCL_DEFINITIONS} VEXCL_BACKEND_CUDA)

    target_compile_options(CUDA INTERFACE
        ${VEXCL_COMPILE_OPTIONS})

    add_library(VexCL::CUDA ALIAS CUDA)

    message(STATUS "Found VexCL::CUDA")
endif()

if (NOT "${Boost_VERSION}" STRLESS "106100")
    find_package(OpenMP)
    set(VEXCL_JIT_LIBRARIES ${OpenMP_CXX_FLAGS})

    if (NOT WIN32)
        list(APPEND VEXCL_JIT_LIBRARIES dl)
    endif()

    add_library(JIT INTERFACE)

    target_include_directories(JIT INTERFACE
        $<BUILD_INTERFACE:${VEXCL_INCLUDE_DIR}>
        ${VEXCL_INCLUDES})

    target_link_libraries(JIT INTERFACE
        ${VEXCL_LIBRARIES} ${VEXCL_JIT_LIBRARIES})

    target_compile_definitions(JIT INTERFACE
        ${VEXCL_DEFINITIONS} VEXCL_BACKEND_JIT)

    target_compile_options(JIT INTERFACE
        ${VEXCL_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS})

    add_library(VexCL::JIT ALIAS JIT)

    message(STATUS "Found VexCL::JIT")
endif()

#----------------------------------------------------------------------------
if (VEXCL_MASTER_PROJECT)
    option(VEXCL_BUILD_TESTS    OFF)
    option(VEXCL_BUILD_EXAMPLES OFF)

    set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA/Compute/JIT)")
    set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA" "Compute" "JIT")

    if("${VEXCL_BACKEND}" STREQUAL "OpenCL")
        add_library(VexCL::Backend ALIAS OpenCL)
    elseif("${VEXCL_BACKEND}" STREQUAL "Compute")
        add_library(VexCL::Backend ALIAS Compute)
    elseif("${VEXCL_BACKEND}" STREQUAL "CUDA")
        add_library(VexCL::Backend ALIAS CUDA)
    elseif("${VEXCL_BACKEND}" STREQUAL "JIT")
        add_library(VexCL::Backend ALIAS JIT)
    endif()

    #------------------------------------------------------------------------
    # Interoperation with Boost.compute
    #------------------------------------------------------------------------
    set(VEXCL_EXAMPLES_INCLUDES "")
    set(VEXCL_EXAMPLES_DEFINITIONS "")

    option(VEXCL_HAVE_BOOST_COMPUTE "Use Boost.Compute algorithms" OFF)
    if (VEXCL_HAVE_BOOST_COMPUTE)
        find_path(BOOST_COMPUTE_INCLUDE boost/compute.hpp)
        list(APPEND VEXCL_EXAMPLES_INCLUDES    ${BOOST_COMPUTE_INCLUDE})
        list(APPEND VEXCL_EXAMPLES_DEFINITIONS VEXCL_HAVE_BOOST_COMPUTE)
    endif ()

    #------------------------------------------------------------------------
    # Interoperation with clogs
    #------------------------------------------------------------------------
    option(VEXCL_CLOGS "Use clogs algorithms" OFF)
    if (VEXCL_CLOGS)
        find_path(CLOGS_INCLUDE clogs/clogs.h)
        find_library(CLOGS_LIB clogs)
        list(APPEND VEXCL_EXAMPLES_INCLUDES ${CLOGS_INCLUDE})
        list(APPEND VEXCL_EXAMPLES_DEFINITIONS VEXCL_HAVE_CLOGS)
    endif ()

    if (VEXCL_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()

    if (VEXCL_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()

    add_subdirectory(docs)

    install(DIRECTORY vexcl DESTINATION include)

    if (TARGET VexCL::OpenCL)
        install(TARGETS OpenCL EXPORT VexCLTargets)
    endif()

    if (TARGET VexCL::Compute)
        install(TARGETS Compute EXPORT VexCLTargets)
    endif()

    if (TARGET VexCL::CUDA)
        install(TARGETS CUDA EXPORT VexCLTargets)
    endif()

    if (TARGET VexCL::JIT)
        install(TARGETS JIT EXPORT VexCLTargets)
    endif()

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VexCLConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/VexCLConfig.cmake"
        COPYONLY
        )

    export(EXPORT VexCLTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/VexCLTargets.cmake"
        NAMESPACE VexCL::
        )

    export(PACKAGE VexCL)

    install(EXPORT VexCLTargets
        FILE VexCLTargets.cmake
        NAMESPACE VexCL::
        DESTINATION share/vexcl/cmake
        )

    install(
        FILES cmake/VexCLConfig.cmake
        DESTINATION share/vexcl/cmake
        )
endif()
