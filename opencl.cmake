add_definitions(-D__CL_ENABLE_EXCEPTIONS)

add_subdirectory(oclutil)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/oclutil)

set(script ${CMAKE_SOURCE_DIR}/oclutil/cl2cpp.sed)

function(cl_source basename)
    set(clsrc ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl)
    if (NOT EXISTS ${clsrc})
	set(clsrc ${CMAKE_CURRENT_BINARY_DIR}/${basename}.cl)
    endif(NOT EXISTS ${clsrc})

    add_custom_command(
	OUTPUT ${basename}.cpp
	COMMAND clcc ${clsrc} && sed -f ${script} ${clsrc} > ${basename}.cpp
	MAIN_DEPENDENCY ${clsrc}
	DEPENDS ${script} clcc
	)
endfunction(cl_source basename)

