option(TEST_COVERAGE "Analyze test coverage with gcov/lcov" OFF)

if(TEST_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")

    # Resets coverage statistics
    add_custom_target(coverage_reset
        COMMAND lcov --zerocounters --directory .
        COMMAND lcov --capture --initial --directory . --base-directory ${CMAKE_SOURCE_DIR}/vexcl --no-external --output-file coverage.info
        )

    # Converts accumulated coverage statistics into coverage/index.html
    # Run
    #   make tests
    # for all OpenCL platforms after
    #   make coverage_reset
    # and before
    #   make coverage
    add_custom_target(coverage
        COMMAND lcov --directory . --base-directory ${CMAKE_SOURCE_DIR}/vexcl --no-external --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr*' '*/cl.hpp' -o coverage.info
        COMMAND genhtml coverage.info --output-directory coverage
        )
endif()

function(add_vexcl_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        ${BACKEND_LIBS}
        ${Boost_LIBRARIES}
        )
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

#----------------------------------------------------------------------------
# General tests
#----------------------------------------------------------------------------
add_vexcl_test(types                    types.cpp)
add_vexcl_test(deduce                   deduce.cpp)
add_vexcl_test(context                  context.cpp)
add_vexcl_test(vector_create            vector_create.cpp)
add_vexcl_test(vector_copy              vector_copy.cpp)
add_vexcl_test(vector_arithmetics       vector_arithmetics.cpp)
add_vexcl_test(vector_view              vector_view.cpp)
add_vexcl_test(vector_pointer           vector_pointer.cpp)
add_vexcl_test(tagged_terminal          tagged_terminal.cpp)
add_vexcl_test(temporary                temporary.cpp)
add_vexcl_test(cast                     cast.cpp)
add_vexcl_test(multivector_create       multivector_create.cpp)
add_vexcl_test(multivector_arithmetics  multivector_arithmetics.cpp)
add_vexcl_test(multi_array              multi_array.cpp)
add_vexcl_test(spmv                     spmv.cpp)
add_vexcl_test(stencil                  stencil.cpp)
add_vexcl_test(generator                generator.cpp)
add_vexcl_test(mba                      mba.cpp)
add_vexcl_test(random                   random.cpp)
add_vexcl_test(sort                     sort.cpp)
add_vexcl_test(scan                     scan.cpp)
add_vexcl_test(reduce_by_key            reduce_by_key.cpp)
add_vexcl_test(multiple_objects         "dummy1.cpp;dummy2.cpp")

#----------------------------------------------------------------------------
# Test interoperation with Boost.compute
#----------------------------------------------------------------------------
option(BOOST_COMPUTE "Use Boost.Compute algorithms" OFF)

if (BOOST_COMPUTE)
    find_path(BOOST_COMPUTE_INCLUDE boost/compute.hpp)
    include_directories(${BOOST_COMPUTE_INCLUDE})

    add_vexcl_test(boost_compute_sort boost_compute_sort.cpp)
    add_vexcl_test(boost_compute_scan boost_compute_scan.cpp)
endif ()

#----------------------------------------------------------------------------
# Test Fast Fourier Transform
#----------------------------------------------------------------------------
if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
    find_package(FFTW QUIET)

    if (FFTW_FOUND)
        include_directories(${FFTW_INCLUDES})
        add_definitions(-DHAVE_FFTW)
    endif ()

    add_vexcl_test(fft fft.cpp)

    if (FFTW_FOUND)
        target_link_libraries(fft ${FFTW_LIBRARIES})
    endif ()
endif ()

if ("${VEXCL_BACKEND}" STREQUAL "CUDA")
    add_vexcl_test(cusparse cusparse.cpp)
    target_link_libraries(cusparse ${CUDA_cusparse_LIBRARY})
endif()
