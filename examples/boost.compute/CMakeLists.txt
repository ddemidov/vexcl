find_path(BOOST_COMPUTE_INCLUDE boost/compute.hpp)
include_directories(${BOOST_COMPUTE_INCLUDE})

add_executable(scan scan.cpp)
target_link_libraries(scan ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(vecsum vecsum.cpp)
target_link_libraries(vecsum ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

add_executable(sort sort.cpp)
target_link_libraries(sort ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
