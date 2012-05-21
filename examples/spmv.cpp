#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <oclutil/oclutil.hpp>
using namespace clu;

int main() {
    try {
	// Read matrix.
	std::ifstream f("problem.dat", std::ios::binary);

	uint n;

	f.read((char*)&n, sizeof(n));

	std::cout << "Looks like " << n << "x" << n << " matrix." << std::endl;

	std::vector<uint> row(n + 1);
	f.read((char*)row.data(), row.size() * sizeof(row[0]));

	std::vector<uint>   col(row.back(), 0);
	std::vector<double> val(row.back(), 0);

	f.read((char*)col.data(), row.back() * sizeof(col[0]));
	f.read((char*)val.data(), row.back() * sizeof(val[0]));

	// Select every GPU supporting double precision.
	auto device = device_list(
		Filter::Type(CL_DEVICE_TYPE_GPU) && Filter::DoublePrecision(),
		true);

	cl::Context context(device);
	std::vector<cl::CommandQueue> queue;
	std::vector<cl::CommandQueue> squeue;

	for(auto d = device.begin(); d != device.end(); d++) {
	    queue.push_back(cl::CommandQueue(context, *d));
	    squeue.push_back(cl::CommandQueue(context, *d));
	}

	// Create OpenCL matrix.
	SpMat<double> A(queue, squeue, n, row.data(), col.data(), val.data());

	clu::vector<double> x(queue, CL_MEM_READ_WRITE, n);
	clu::vector<double> y(queue, CL_MEM_READ_WRITE, n);

	x = Const(1.0);
	y = A * x;

	std::vector<double> Y(n);

	copy(y, Y);

	double res = 0;
	for(uint i = 0; i < n; i++) {
	    double sum = 0;
	    for(uint j = row[i]; j < row[i + 1]; j++)
		sum += val[j];
	    res = std::max(res, fabs(Y[i] - sum));
	}

	std::cout << "res = " << res << std::endl;
    } catch (const cl::Error &e) {
	std::cerr << "OpenCL error: " << e << std::endl;
    }
}
