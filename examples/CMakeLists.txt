include_directories(${CMAKE_SOURCE_DIR})

add_definitions(-D__CL_ENABLE_EXCEPTIONS)

foreach(example
	cg devlist spmv vector readme_example_1 readme_example_2
	readme_example_3 readme_example_4)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})
endforeach(example)

