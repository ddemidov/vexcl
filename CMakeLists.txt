cmake_minimum_required(VERSION 2.8)
project(oclutil)

if (WIN32)
    find_library(OPENCL_LIBRARY OpenCL
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/x64"
	)
    find_path(OPENCL_INCLUDE CL/opencl.h
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/include"
	)
    include_directories(${OPENCL_INCLUDE})
else (WIN32)
    find_library(OPENCL_LIBRARY OpenCL)
endif (WIN32)

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif (WIN32)
find_package(Boost COMPONENTS regex)

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_definitions(-D__CL_ENABLE_EXCEPTIONS)

add_executable(clcc clcc.cpp)
target_link_libraries(clcc
    ${OPENCL_LIBRARY}
    ${Boost_LIBRARIES}
    )

add_subdirectory(examples)
add_subdirectory(doc)
