cmake_minimum_required(VERSION 3.1)
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

#----------------------------------------------------------------------------
# Compile-time options
#----------------------------------------------------------------------------
option(VEXCL_SHOW_KERNELS "Show generated kernels in tests and examples" OFF)
option(VEXCL_CACHE_KERNELS "Cache compiled kernels offline" ON)
option(VEXCL_SHOW_COPIES "Log vector copies to stdout for debugging purposes" OFF)
set(VEXCL_CHECK_SIZES 0 CACHE STRING "Check that expressions have correct sizes")

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()

find_package(Boost REQUIRED COMPONENTS
    chrono
    date_time
    filesystem
    program_options
    system
    thread
    unit_test_framework
    )

#----------------------------------------------------------------------------
# Generic target
#----------------------------------------------------------------------------
add_library(Common INTERFACE)
add_library(VexCL::Common ALIAS Common)

target_compile_features(Common INTERFACE
    cxx_auto_type
    cxx_nullptr
    cxx_rvalue_references
    cxx_right_angle_brackets
    cxx_static_assert
    cxx_variadic_templates
    cxx_decltype
)

target_include_directories(Common INTERFACE
    ${Boost_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )


target_link_libraries(Common INTERFACE
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    )

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_link_libraries(Common INTERFACE
        ${Boost_CHRONO_LIBRARY}
        ${Boost_DATE_TIME_LIBRARY}
        )
endif()

target_compile_options(Common INTERFACE
    # g++
    $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall>>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-braces>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>
    # Clang
    $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall>>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-missing-braces>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-ignored-attributes>
    # MSVC
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4003>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
    )

target_compile_definitions(Common INTERFACE
    # MSVC
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    $<$<CXX_COMPILER_ID:MSVC>:_VARIADIC_MAX=10>
    )

#----------------------------------------------------------------------------
# Find VexCL backends
#----------------------------------------------------------------------------
find_package(OpenCL)
if(OpenCL_FOUND)
    add_library(OpenCL INTERFACE)
    add_library(VexCL::OpenCL ALIAS OpenCL)

    target_include_directories(OpenCL INTERFACE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(OpenCL INTERFACE Common ${OpenCL_LIBRARIES})
    target_compile_definitions(OpenCL INTERFACE VEXCL_BACKEND_OPENCL)

    message(STATUS "Found VexCL::OpenCL")

    if (NOT "${Boost_VERSION}" STRLESS "106100")
        add_library(Compute INTERFACE)
        add_library(VexCL::Compute ALIAS Compute)

        target_include_directories(Compute INTERFACE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(Compute INTERFACE Common ${OpenCL_LIBRARIES})
        target_compile_definitions(Compute INTERFACE VEXCL_BACKEND_COMPUTE)

        message(STATUS "Found VexCL::Compute")
    endif()
endif()

find_package(CUDA)
if(CUDA_FOUND)
    add_library(CUDA INTERFACE)
    add_library(VexCL::CUDA ALIAS CUDA)

    target_include_directories(CUDA INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(CUDA INTERFACE Common ${CUDA_CUDA_LIBRARY})
    target_compile_definitions(CUDA INTERFACE VEXCL_BACKEND_CUDA)

    message(STATUS "Found VexCL::CUDA")
endif()

if (NOT "${Boost_VERSION}" STRLESS "106100")

    add_library(JIT INTERFACE)
    add_library(VexCL::JIT ALIAS JIT)

    find_package(OpenMP)

    # Have to check several OPENMP_FOUND due to bug in
    # one version of CMake and the docs (fixed in patch release)
    # OpenMP is missing on macOS llvm default, for example
    if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)

        # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
        if(TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(JIT INTERFACE OpenMP::OpenMP_CXX Common)
        else()
            # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
            target_link_libraries(JIT INTERFACE
                $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
                $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
                $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
                )
            target_compile_options(JIT INTERFACE ${OpenMP_CXX_FLAGS})
        endif()
    endif()

    target_link_libraries(JIT INTERFACE Common ${CMAKE_DL_LIBS})
    target_compile_definitions(JIT INTERFACE VEXCL_BACKEND_JIT)

    message(STATUS "Found VexCL::JIT")

endif()

function(vexcl_add_executables targetname)
    add_library(${targetname} INTERFACE)

    if(TARGET VexCL::OpenCL)
        add_executable(${targetname}_cl ${ARGN})
        target_link_libraries(${targetname}_cl VexCL::OpenCL ${targetname})
        set_target_properties(${targetname}_cl PROPERTIES CXX_EXTENSIONS OFF)
    endif()
    if(TARGET VexCL::Compute)
        add_executable(${targetname}_comp ${ARGN})
        target_link_libraries(${targetname}_comp VexCL::Compute ${targetname})
        set_target_properties(${targetname}_comp PROPERTIES CXX_EXTENSIONS OFF)
    endif()
    if(TARGET VexCL::CUDA)
        add_executable(${targetname}_cuda ${ARGN})
        target_link_libraries(${targetname}_cuda VexCL::CUDA ${targetname})
        set_target_properties(${targetname}_cuda PROPERTIES CXX_EXTENSIONS OFF)
    endif()
    if(TARGET VexCL::JIT)
        add_executable(${targetname}_jit ${ARGN})
        target_link_libraries(${targetname}_jit VexCL::JIT ${targetname})
        set_target_properties(${targetname}_jit PROPERTIES CXX_EXTENSIONS OFF)
    endif()
endfunction()

function(vexcl_add_libraries targetname typeoflib)
    add_library(${targetname} INTERFACE)
    add_library(${targetname}::Common ALIAS ${targetname})

    if(TARGET VexCL::OpenCL)
        add_library(${targetname}_cl ${TYPEOFLIB} ${ARGN})
        target_link_libraries(${targetname}_cl PUBLIC VexCL::OpenCL ${targetname})
        set_target_properties(${targetname}_cl PROPERTIES CXX_EXTENSIONS OFF)
        add_library(${targetname}::OpenCl ALIAS ${targetname}_cl)
    endif()
    if(TARGET VexCL::Compute)
        add_library(${targetname}_comp ${TYPEOFLIB} ${ARGN})
        target_link_libraries(${targetname}_comp PUBLIC VexCL::Compute ${targetname})
        set_target_properties(${targetname}_comp PROPERTIES CXX_EXTENSIONS OFF)
        add_library(${targetname}::Compute ALIAS ${targetname}_comp)
    endif()
    if(TARGET VexCL::CUDA)
        add_library(${targetname}_cuda ${TYPEOFLIB} ${ARGN})
        target_link_libraries(${targetname}_cuda PUBLIC VexCL::CUDA ${targetname})
        set_target_properties(${targetname}_cuda PROPERTIES CXX_EXTENSIONS OFF)
        add_library(${targetname}::CUDA ALIAS ${targetname}_cuda)
    endif()
    if(TARGET VexCL::JIT)
        add_library(${targetname}_jit ${TYPEOFLIB} ${ARGN})
        target_link_libraries(${targetname}_jit PUBLIC VexCL::JIT ${targetname})
        set_target_properties(${targetname}_jit PROPERTIES CXX_EXTENSIONS OFF)
        add_library(${targetname}::JIT ALIAS ${targetname}_jit)
    endif()
endfunction()

#----------------------------------------------------------------------------
if (VEXCL_MASTER_PROJECT)
    option(VEXCL_BUILD_TESTS    OFF)
    option(VEXCL_BUILD_EXAMPLES OFF)

    set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA/Compute/JIT/All)")
    set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "ALL" "OpenCL" "CUDA" "Compute" "JIT")

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
    option(VEXCL_HAVE_BOOST_COMPUTE "Use Boost.Compute algorithms" OFF)
    if (VEXCL_HAVE_BOOST_COMPUTE)
        find_path(BOOST_COMPUTE_INCLUDE boost/compute.hpp)

        add_library(compute_target INTERFACE)
        target_include_directories(compute_target INTERFACE ${BOOST_COMPUTE_INCLUDE})
        target_compile_definitions(compute_target INTERFACE VEXCL_HAVE_BOOST_COMPUTE)
    endif ()

    #------------------------------------------------------------------------
    # Interoperation with clogs
    #------------------------------------------------------------------------
    option(VEXCL_CLOGS "Use clogs algorithms" OFF)
    if (VEXCL_CLOGS)
        find_path(CLOGS_INCLUDE clogs/clogs.h)
        find_library(CLOGS_LIBRARY clogs)

        add_library(clogs_target INTERFACE)
        target_include_directories(clogs_target INTERFACE ${CLOGS_INCLUDE})
        target_compile_definitions(clogs_target INTERFACE VEXCL_HAVE_CLOGS)
        target_link_libraries(clogs_target INTERFACE ${CLOGS_LIBRARY})
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

    install(TARGETS Common EXPORT VexCLTargets)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VexCLConfig.cmake.in"
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
        FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/VexCLConfig.cmake
        DESTINATION share/vexcl/cmake
        )
endif()
