find_path(VIENNACL_INCLUDE viennacl/vector.hpp)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${VIENNACL_INCLUDE}
    )

set(EXAMPLES cg)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})
    set_target_properties(${example} PROPERTIES COMPILE_FLAGS -Wno-comment)

    if (WIN32)
	target_link_libraries(${example} ${Boost_DATE_TIME_LIBRARY})
    endif(WIN32)
endforeach(example)
