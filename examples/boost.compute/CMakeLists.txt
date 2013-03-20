find_path(BOOST_COMPUTE_INCLUDE boost/compute.hpp)
include_directories(${BOOST_COMPUTE_INCLUDE})

add_executable(scan scan.cpp)
target_link_libraries(scan ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(vecsum vecsum.cpp)
target_link_libraries(vecsum ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(vecsum_vexcl vecsum_vexcl.cpp)
target_link_libraries(vecsum_vexcl ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(vecsum_compute vecsum_compute.cpp)
target_link_libraries(vecsum_compute ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(rowapi rowapi.cpp)
target_link_libraries(rowapi ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
