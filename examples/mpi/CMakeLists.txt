option(MPI_EXAMPLES "Compile MPI examples" ON)
if (MPI_EXAMPLES)
    find_package(MPI QUIET)
    if (MPI_CXX_FOUND)
        include_directories(${MPI_CXX_INCLUDE_PATH})

        add_executable(mpitests mpitests.cpp)
        set_target_properties(mpitests PROPERTIES
            COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
            LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
            )
        target_link_libraries(mpitests
            ${Boost_LIBRARIES}
            ${MPI_CXX_LIBRARIES}
            ${OPENCL_LIBRARIES}
            )

        if (ODEINT)
            add_executable(lorenz_ensemble lorenz_ensemble.cpp)
            set_target_properties(lorenz_ensemble PROPERTIES
                COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
                LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
                INCLUDE_DIRECTORIES "${ODEINT};${CMAKE_SOURCE_DIR}"
                )
            target_link_libraries(lorenz_ensemble
                ${Boost_LIBRARIES}
                ${MPI_CXX_LIBRARIES}
                ${OPENCL_LIBRARIES}
                )

            add_executable(disordered_lattice disordered_lattice.cpp)
            set_target_properties(disordered_lattice PROPERTIES
                COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
                LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
                INCLUDE_DIRECTORIES "${ODEINT};${CMAKE_SOURCE_DIR}"
                )
            target_link_libraries(disordered_lattice
                ${Boost_LIBRARIES}
                ${MPI_CXX_LIBRARIES}
                ${OPENCL_LIBRARIES}
                )
        endif(ODEINT)
    endif (MPI_CXX_FOUND)
endif (MPI_EXAMPLES)
