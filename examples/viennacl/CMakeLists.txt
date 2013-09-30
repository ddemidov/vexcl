option(VIENNACL_EXAMPLES "Compile ViennaCL examples" OFF)

if (VIENNACL_EXAMPLES)
    find_path(VIENNACL_INCLUDE viennacl/vector.hpp)

    include_directories(
        ${CMAKE_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${VIENNACL_INCLUDE}
        )

    add_executable(solvers solvers.cpp)
    target_link_libraries(solvers ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endif (VIENNACL_EXAMPLES)
