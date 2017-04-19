Building VexCL programs with CMake
==================================

In order to build a VexCL program with the `CMake`_ build system you need just
a couple of lines in your ``CmakeLists.txt``:

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8)
    project(example)

    find_package(VexCL)

    add_executable(example example.cpp)
    target_link_libraries(example VexCL::OpenCL)

VexCL provides interface targets for the backends supported on the current
system. Possible choices are ``VexCL::OpenCL`` for the OpenCL backend,
``VexCL::Compute`` for Boost.Compute, ``VexCL::CUDA`` for CUDA, and
``VexCL::JIT`` for the just-in-time compiled OpenMP kernels.
The targets will take care of the appropriate compiler and linker flags for the
selected backend.

``find_package(VexCL)`` may be used when VexCL was installed system wide. If
that is not the case, you can just copy the VexCL into a subdirectory of your
project and replace the line with

.. code-block:: cmake

    add_subdirectory(vexcl)

.. _`CMake`: https://cmake.org/
