cmake_minimum_required(VERSION 2.8)
project(VexCL CXX)

enable_testing()

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(doc)
add_subdirectory(cmake)

install(DIRECTORY vexcl DESTINATION include)
