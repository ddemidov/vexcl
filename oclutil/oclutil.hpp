#ifndef OCLUTIL_HPP
#define OCLUTIL_HPP

/**
 * \file   oclutil.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  OpenCL convenience utilities.
 */

#include <CL/cl.hpp>
#include <iostream>

#include <oclutil/devlist.hpp>
#include <oclutil/vector.hpp>
#include <oclutil/spmat.hpp>
#include <oclutil/reduce.hpp>

/// Output description of an OpenCL error to a stream.
std::ostream& operator<<(std::ostream &os, const cl::Error &e) {
    os << e.what() << "(";

    switch (e.err()) {
	case 0:
	    os << "Success";
	    break;
	case -1:
	    os << "Device not found";
	    break;
	case -2:
	    os << "Device not available";
	    break;
	case -3:
	    os << "Compiler not available";
	    break;
	case -4:
	    os << "Mem object allocation failure";
	    break;
	case -5:
	    os << "Out of resources";
	    break;
	case -6:
	    os << "Out of host memory";
	    break;
	case -7:
	    os << "Profiling info not available";
	    break;
	case -8:
	    os << "Mem copy overlap";
	    break;
	case -9:
	    os << "Image format mismatch";
	    break;
	case -10:
	    os << "Image format not supported";
	    break;
	case -11:
	    os << "Build program failure";
	    break;
	case -12:
	    os << "Map failure";
	    break;
	case -13:
	    os << "Misaligned sub buffer offset";
	    break;
	case -14:
	    os << "Exec status error for events in wait list";
	    break;
	case -30:
	    os << "Invalid value";
	    break;
	case -31:
	    os << "Invalid device type";
	    break;
	case -32:
	    os << "Invalid platform";
	    break;
	case -33:
	    os << "Invalid device";
	    break;
	case -34:
	    os << "Invalid context";
	    break;
	case -35:
	    os << "Invalid queue properties";
	    break;
	case -36:
	    os << "Invalid command queue";
	    break;
	case -37:
	    os << "Invalid host ptr";
	    break;
	case -38:
	    os << "Invalid mem object";
	    break;
	case -39:
	    os << "Invalid image format descriptor";
	    break;
	case -40:
	    os << "Invalid image size";
	    break;
	case -41:
	    os << "Invalid sampler";
	    break;
	case -42:
	    os << "Invalid binary";
	    break;
	case -43:
	    os << "Invalid build options";
	    break;
	case -44:
	    os << "Invalid program";
	    break;
	case -45:
	    os << "Invalid program executable";
	    break;
	case -46:
	    os << "Invalid kernel name";
	    break;
	case -47:
	    os << "Invalid kernel definition";
	    break;
	case -48:
	    os << "Invalid kernel";
	    break;
	case -49:
	    os << "Invalid arg index";
	    break;
	case -50:
	    os << "Invalid arg value";
	    break;
	case -51:
	    os << "Invalid arg size";
	    break;
	case -52:
	    os << "Invalid kernel args";
	    break;
	case -53:
	    os << "Invalid work dimension";
	    break;
	case -54:
	    os << "Invalid work group size";
	    break;
	case -55:
	    os << "Invalid work item size";
	    break;
	case -56:
	    os << "Invalid global offset";
	    break;
	case -57:
	    os << "Invalid event wait list";
	    break;
	case -58:
	    os << "Invalid event";
	    break;
	case -59:
	    os << "Invalid operation";
	    break;
	case -60:
	    os << "Invalid gl object";
	    break;
	case -61:
	    os << "Invalid buffer size";
	    break;
	case -62:
	    os << "Invalid mip level";
	    break;
	case -63:
	    os << "Invalid global work size";
	    break;
	case -64:
	    os << "Invalid property";
	    break;
	default:
	    os << "Unknown error";
	    break;
    }

    return os << ")";
}

#endif
