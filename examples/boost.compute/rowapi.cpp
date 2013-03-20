#include <iostream>
#include <vector>
 
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <boost/compute.hpp>
 
int main() {
    const size_t N = 1 << 20;
 
    try {
	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);
 
	if (platform.empty()) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return 1;
	}
 
	// Get first available GPU device which supports double precision.
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
	    std::vector<cl::Device> pldev;
 
	    try {
		p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);
 
		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
		    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
 
		    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();
 
		    if (
			    ext.find("cl_khr_fp64") == std::string::npos &&
			    ext.find("cl_amd_fp64") == std::string::npos
		       ) continue;
 
		    device.push_back(*d);
		    context = cl::Context(device);
		}
	    } catch(...) {
		device.clear();
	    }
	}
 
	if (device.empty()) {
	    std::cerr << "GPUs with double precision not found." << std::endl;
	    return 1;
	}
 
	std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;
 
	// Create command queue.
	cl::CommandQueue queue(context, device[0]);
 
	// Allocate device buffers.
	cl::Buffer A(context, CL_MEM_READ_WRITE, N * sizeof(double));
	cl::Buffer B(context, CL_MEM_READ_WRITE, N * sizeof(double));
	cl::Buffer C(context, CL_MEM_READ_WRITE, N * sizeof(double));
 
        //------------------------------------------------------------------
        // Call boost.compute algorithm for vector copy
        //------------------------------------------------------------------

        // Queue.
        boost::compute::command_queue bcq( queue() );

        // Iterators.
        auto a_begin = boost::compute::make_buffer_iterator<double>(A(), 0);
        auto a_end   = boost::compute::make_buffer_iterator<double>(A(), N);
        auto b_begin = boost::compute::make_buffer_iterator<double>(B(), 0);
        auto b_end   = boost::compute::make_buffer_iterator<double>(B(), N);
        auto c_begin = boost::compute::make_buffer_iterator<double>(C(), 0);
        auto c_end   = boost::compute::make_buffer_iterator<double>(C(), N);

        // Call algorithms from Boost.Compute.
        boost::compute::fill(a_begin, a_end, 1.0, bcq);
        boost::compute::fill(b_begin, b_end, 2.0, bcq);

        boost::compute::transform(
                a_begin, a_end, b_begin, c_begin, boost::compute::plus<double>(), bcq
                );

	// Get result back to host.
	std::vector<double> c(N);
        boost::compute::copy(c_begin, c_end, c.begin(), bcq);
 
	// Should get '3' here.
	std::cout << c[42] << std::endl;
    } catch (const cl::Error &err) {
	std::cerr
	    << "OpenCL error: "
	    << err.what() << "(" << err.err() << ")"
	    << std::endl;
	return 1;
    } catch (const std::exception &err) {
	std::cerr << "Error: " << err.what() << std::endl;
	return 1;
    }
}
