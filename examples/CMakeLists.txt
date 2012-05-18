include_directories(${CMAKE_SOURCE_DIR})

add_definitions(-D__CL_ENABLE_EXCEPTIONS)

add_executable(devlist devlist.cpp)
target_link_libraries(devlist ${OPENCL_LIBRARY})

add_executable(vector vector.cpp)
target_link_libraries(vector ${OPENCL_LIBRARY})

add_executable(spmv spmv.cpp)
target_link_libraries(spmv ${OPENCL_LIBRARY})

add_executable(cg cg.cpp)
target_link_libraries(cg ${OPENCL_LIBRARY})

