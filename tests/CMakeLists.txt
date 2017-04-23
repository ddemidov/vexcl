option(VEXCL_TEST_COVERAGE "Analyze test coverage with gcov/lcov" OFF)

if(VEXCL_TEST_COVERAGE)
    target_compile_options(Common INTERFACE --coverage)
    target_link_libraries(Common INTERFACE --coverage)

    # Resets coverage statistics
    add_custom_target(coverage_reset
        COMMAND lcov --zerocounters --directory .
        COMMAND lcov --capture --initial --directory . --base-directory "${PROJECT_SOURCE_DIR}/vexcl" --no-external --output-file coverage.info
        VERBATIM
        )

    # Converts accumulated coverage statistics into coverage/index.html
    # Run
    #   make tests
    # for all OpenCL platforms after
    #   make coverage_reset
    # and before
    #   make coverage
    add_custom_target(coverage
        COMMAND lcov --directory . --base-directory "${PROJECT_SOURCE_DIR}/vexcl" --no-external --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr*' '*/cl.hpp' -o coverage.info
        COMMAND genhtml coverage.info --output-directory coverage
        VERBATIM
        )
endif()

function(add_vexcl_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    if (NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(${TEST_NAME} PRIVATE BOOST_TEST_DYN_LINK)
    endif ()

    target_link_libraries(${TEST_NAME}
        VexCL::Backend
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        )

    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

#----------------------------------------------------------------------------
# General tests
#----------------------------------------------------------------------------
add_vexcl_test(boost_version            boost_version.cpp)
add_vexcl_test(types                    types.cpp)
add_vexcl_test(deduce                   deduce.cpp)
add_vexcl_test(vector_create            vector_create.cpp)
add_vexcl_test(vector_copy              vector_copy.cpp)
add_vexcl_test(vector_arithmetics       vector_arithmetics.cpp)
add_vexcl_test(vector_view              vector_view.cpp)
add_vexcl_test(tensordot                tensordot.cpp)
add_vexcl_test(vector_pointer           vector_pointer.cpp)
add_vexcl_test(tagged_terminal          tagged_terminal.cpp)
add_vexcl_test(temporary                temporary.cpp)
add_vexcl_test(cast                     cast.cpp)
add_vexcl_test(multivector_create       multivector_create.cpp)
add_vexcl_test(multivector_arithmetics  multivector_arithmetics.cpp)
add_vexcl_test(multi_array              multi_array.cpp)
add_vexcl_test(spmv                     spmv.cpp)
add_vexcl_test(sparse_matrices          sparse_matrices.cpp)
add_vexcl_test(stencil                  stencil.cpp)
add_vexcl_test(generator                generator.cpp)
add_vexcl_test(mba                      mba.cpp)
add_vexcl_test(random                   random.cpp)
add_vexcl_test(sort                     sort.cpp)
add_vexcl_test(scan                     scan.cpp)
add_vexcl_test(scan_by_key              scan_by_key.cpp)
add_vexcl_test(reduce_by_key            reduce_by_key.cpp)
add_vexcl_test(logical                  logical.cpp)
add_vexcl_test(threads                  threads.cpp)
add_vexcl_test(svm                      svm.cpp)
add_vexcl_test(events                   events.cpp)
add_vexcl_test(image                    image.cpp)
add_vexcl_test(custom_kernel            custom_kernel.cpp)
add_vexcl_test(eval                     eval.cpp)
add_vexcl_test(constants                constants.cpp)
add_vexcl_test(vector_io                vector_io.cpp)
add_vexcl_test(reinterpret              reinterpret.cpp)
add_vexcl_test(multiple_objects         "dummy1.cpp;dummy2.cpp")

if (NOT DEFINED ENV{APPVEYOR})
    # This fails on AppVeyor-CI
    add_vexcl_test(context              context.cpp)
endif ()

#----------------------------------------------------------------------------
# Test interoperation with Boost.compute
#----------------------------------------------------------------------------
if (TARGET compute_target)
    add_vexcl_test(boost_compute_sort boost_compute_sort.cpp)
    add_vexcl_test(boost_compute_scan boost_compute_scan.cpp)

    target_link_libraries(boost_compute_sort compute_target)
    target_link_libraries(boost_compute_scan compute_target)
endif ()

#----------------------------------------------------------------------------
# Test interoperation with clogs
#----------------------------------------------------------------------------
if (TARGET clogs_target)
    add_vexcl_test(clogs_scan clogs_scan.cpp)
    add_vexcl_test(clogs_sort clogs_sort.cpp)

    target_link_libraries(clogs_scan clogs_target)
    target_link_libraries(clogs_sort clogs_target)
endif ()

#----------------------------------------------------------------------------
# Test Fast Fourier Transform
#----------------------------------------------------------------------------
add_vexcl_test(fft fft.cpp)
find_package(FFTW QUIET)
if (FFTW_FOUND)
    target_compile_definitions(fft PRIVATE VEXCL_HAVE_FFTW)
    target_include_directories(fft PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(fft ${FFTW_LIBRARIES})
endif ()


if (VEXCL_BACKEND MATCHES "CUDA")
    add_vexcl_test(cusparse cusparse.cpp)
    target_link_libraries(cusparse ${CUDA_cusparse_LIBRARY})
endif()
