option(VIENNACL_EXAMPLES "Compile ViennaCL examples" OFF)

if (VIENNACL_EXAMPLES)
    find_path(VIENNACL_INCLUDE viennacl/vector.hpp)

    include_directories(
        ${CMAKE_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${VIENNACL_INCLUDE}
        )

    set(EXAMPLES solvers)

    foreach(example ${EXAMPLES})
        add_executable(${example} ${example}.cpp)
        target_link_libraries(${example} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
        set_target_properties(${example} PROPERTIES COMPILE_FLAGS -Wno-comment)
    endforeach(example)
endif (VIENNACL_EXAMPLES)
