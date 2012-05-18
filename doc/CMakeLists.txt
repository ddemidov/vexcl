cmake_minimum_required(VERSION 2.6)
project(doc)

find_package(LATEX QUIET)
find_package(Doxygen QUIET)

if (LATEX_COMPILER AND DOXYGEN_FOUND)
    option(DOC_BUILD_CHM "Build documentaion in chm format")

    set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/oclutil)

    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    set(SOURCE_BROWSER YES)

    if (DOC_BUILD_CHM)
	set(GENERATE_HTMLHELP YES)
    endif (DOC_BUILD_CHM)

    configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.cmake
	${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	)

    add_custom_target(doc
	COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	COMMENT "Build Doxygen-generated documentation"
	)
endif(LATEX_COMPILER AND DOXYGEN_FOUND)
