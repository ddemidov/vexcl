cmake_minimum_required(VERSION 3.1)
project(hello)

find_package(VexCL)

add_executable(hello hello.cpp)
target_link_libraries(hello VexCL::OpenCL)
