option(VIENNACL_EXAMPLES "Compile ViennaCL examples" OFF)

if (VIENNACL_EXAMPLES)
    find_path(VIENNACL_INCLUDE viennacl/vector.hpp)
    add_executable(viennacl_solvers viennacl_solvers.cpp)
    target_include_directories(viennacl_solvers PRIVATE ${VIENNACL_INCLUDE})
    target_link_libraries(viennacl_solvers VexCL::Backend)
endif (VIENNACL_EXAMPLES)
