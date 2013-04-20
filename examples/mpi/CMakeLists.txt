function(add_vexcl_mpi_example example)
    add_executable(${example} ${example}.cpp)
    set_target_properties(${example} PROPERTIES
        COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        LINK_FLAGS    "${MPI_CXX_LINK_FLAGS}"
        )
    target_link_libraries(${example}
        ${Boost_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        ${OPENCL_LIBRARIES}
        )
endfunction()

find_package(MPI QUIET)
if (MPI_CXX_FOUND)
    option(MPI_EXAMPLES "Compile MPI examples" ON)
    if (MPI_EXAMPLES)
        include_directories(${MPI_CXX_INCLUDE_PATH})

        add_vexcl_mpi_example(mpitests)

        #--------------------------------------------------------------------
        # The two examples below need Boost.odeint available since 1.53
        #--------------------------------------------------------------------
        if (NOT ${Boost_VERSION} STRLESS "105300")
            add_vexcl_mpi_example(lorenz_ensemble)
            add_vexcl_mpi_example(disordered_lattice)
        endif (NOT ${Boost_VERSION} STRLESS "105300")
    endif (MPI_EXAMPLES)
endif (MPI_CXX_FOUND)
