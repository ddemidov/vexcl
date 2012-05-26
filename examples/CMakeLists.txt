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
    cg
    devlist
    readme_example_1
    readme_example_2
    readme_example_3
    readme_example_4
    spmv
    vector
    )

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} ${OPENCL_LIBRARY})
endforeach(example)

