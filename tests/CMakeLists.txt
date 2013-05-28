function(add_vexcl_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        ${OPENCL_LIBRARIES}
        ${Boost_LIBRARIES}
        )
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

#----------------------------------------------------------------------------
# General tests
#----------------------------------------------------------------------------
add_vexcl_test(types                    types.cpp)
add_vexcl_test(vector_create            vector_create.cpp)
add_vexcl_test(vector_copy              vector_copy.cpp)
add_vexcl_test(vector_arithmetics       vector_arithmetics.cpp)
add_vexcl_test(vector_view              vector_view.cpp)
add_vexcl_test(multivector_create       multivector_create.cpp)
add_vexcl_test(multivector_arithmetics  multivector_arithmetics.cpp)
add_vexcl_test(spmv                     spmv.cpp)
add_vexcl_test(stencil                  stencil.cpp)
add_vexcl_test(generator                generator.cpp)
add_vexcl_test(random                   random.cpp)
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
endif (BOOST_COMPUTE)

#----------------------------------------------------------------------------
# Test Fast Fourier Transform
#----------------------------------------------------------------------------
find_package(FFTW QUIET)

if (FFTW_FOUND)
    include_directories(${FFTW_INCLUDES})
    add_definitions(-DHAVE_FFTW)
endif (FFTW_FOUND)

add_vexcl_test(fft fft.cpp)

if (FFTW_FOUND)
    target_link_libraries(fft ${FFTW_LIBRARIES})
endif (FFTW_FOUND)
