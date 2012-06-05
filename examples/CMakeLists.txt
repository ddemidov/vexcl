if (WIN32)
    find_library(OPENCL_LIBRARY OpenCL
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/x64"
	)
    find_path(OPENCL_INCLUDE CL/opencl.h
	"c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/include"
	)
    include_directories(${OPENCL_INCLUDE})
else (WIN32)
    find_library(OPENCL_LIBRARY OpenCL)
endif (WIN32)

include_directories(${CMAKE_SOURCE_DIR})

set(EXAMPLES
    benchmark
    cg
    devlist
    utests
    vector
    )

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})
endforeach(example)

