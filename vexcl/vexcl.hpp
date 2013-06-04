#ifndef VEXCL_VEXCL_HPP
#define VEXCL_VEXCL_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   vexcl.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Vector expression template library for OpenCL.
 */

#ifdef WIN32
#  pragma warning(push)
#  pragma warning(disable : 4290 4503)
#  define NOMINMAX
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
#  define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <iostream>

#include <vexcl/devlist.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <vexcl/multivector.hpp>
#include <vexcl/reduce.hpp>
#include <vexcl/spmat.hpp>
#include <vexcl/stencil.hpp>
#include <vexcl/gather.hpp>
#include <vexcl/random.hpp>
#include <vexcl/fft.hpp>
#include <vexcl/generator.hpp>
#include <vexcl/profiler.hpp>

#ifdef WIN32
#  pragma warning(pop)
#endif

// vim: et
#endif
