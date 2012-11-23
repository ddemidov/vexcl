#include <vector>

// Prepare problem (1D Poisson equation).
template <typename real>
void genproblem(
	size_t size,
        std::vector<size_t> &row,
        std::vector<size_t> &col,
        std::vector<real>   &val,
        std::vector<real>   &rhs
	)
{
    real h2i = (size - 1) * (size - 1);

    row.clear();
    col.clear();
    val.clear();
    rhs.clear();

    row.reserve(size + 1);
    col.reserve(2 + (size - 2) * 3);
    val.reserve(2 + (size - 2) * 3);
    rhs.reserve(size);

    row.push_back(0);
    for(size_t i = 0; i < size; i++) {
	if (i == 0 || i == size - 1) {
	    col.push_back(i);
	    val.push_back(1);
	    rhs.push_back(0);
	    row.push_back(row.back() + 1);
	} else {
	    col.push_back(i-1);
	    val.push_back(-h2i);

	    col.push_back(i);
	    val.push_back(2 * h2i);

	    col.push_back(i+1);
	    val.push_back(-h2i);

	    rhs.push_back(2);
	    row.push_back(row.back() + 3);
	}
    }
}
