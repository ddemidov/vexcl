find_package(MPI QUIET)
if (MPI_CXX_FOUND)
    add_executable(mpitests mpitests.cpp)
    set_target_properties(mpitests PROPERTIES
        COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
    target_link_libraries(mpitests ${MPI_CXX_LIBRARIES} OpenCL)

    if (ODEINT)
        add_executable(lorenz_ensemble lorenz_ensemble.cpp)
        set_target_properties(lorenz_ensemble PROPERTIES
            COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
            LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
            INCLUDE_DIRECTORIES "${ODEINT};${CMAKE_SOURCE_DIR}"
            )
        target_link_libraries(lorenz_ensemble ${MPI_CXX_LIBRARIES} OpenCL)
    endif(ODEINT)
endif (MPI_CXX_FOUND)
