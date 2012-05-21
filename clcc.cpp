#include <iostream>
#include <fstream>
#include <string>
#include <boost/regex.hpp>
#include <oclutil/oclutil.hpp>
using namespace std;

#ifdef WIN32
#  pragma warning (disable : 4800)
#endif

//---------------------------------------------------------------------------
void precondition(bool cond, const char *fail_msg) {
    if (!cond) {
	std::cerr << "Error: " << fail_msg << std::endl;
	exit(1);
    }
}

//---------------------------------------------------------------------------
size_t fileSize(ifstream &f) {
    ifstream::pos_type current_position = f.tellg();

    f.seekg(0, ios_base::beg);
    ifstream::pos_type begin = f.tellg();

    f.seekg(0, ios_base::end);
    ifstream::pos_type end = f.tellg();

    f.seekg(current_position);

    return end - begin;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 2) {
	cerr << "Usage: " << argv[0] << " <source.cl>" << endl;
	return 1;
    }

    const char *fname = argv[1];

    // Read file into memory.
    ifstream fsrc(fname);
    precondition(fsrc, (string("Failed to open \"") + fname + string("\"")).c_str());
    size_t srcsize = fileSize(fsrc);
    string src(srcsize + 1, 0);
    fsrc.read(&src[0], srcsize);

    // Get any device.
    auto device = clu::device_list(clu::Filter::Count(1));

    precondition(device.size() > 0, "No OpenCL devices found.");

    // Try to build the program.
    cl::Context context(device);
    cl::Program program(context, cl::Program::Sources(
		1, std::make_pair(src.c_str(), srcsize)
		)
	    );

    try {
	program.build(device);
    } catch(const cl::Error&) {
	boost::regex ps("(<program source>)?:(\\d+):(\\d+):");
	std::string  rep = fname + std::string(":\\2:\\3:");

	std::cerr << boost::regex_replace(
		program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0]),
		ps, rep, boost::match_default | boost::format_sed
		) << std::endl;
	throw;
    }
}
