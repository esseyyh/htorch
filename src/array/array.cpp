#include "../../include/array/array.h"

#include <iostream>
#include <stdexcept>
#include <vector>

class Array {
public:
    float* data;   // Pointer to raw data
    int ndim;      // Number of dimensions
    int* shape;    // Array shape
    int* strides;  // Strides for each dimension

    // Constructor
    Array(float* data, int ndim, int* shape, int* strides)
        : data(data), ndim(ndim), shape(shape), strides(strides) {}

    // Variadic operator() for multi-dimensional indexing
    float& operator()(int idx1) {
        return get_offset(idx1);
    }

    float& operator()(int idx1, int idx2) {
        return get_offset(idx1, idx2);
    }

    float& operator()(int idx1, int idx2, int idx3) {
        return get_offset(idx1, idx2, idx3);
    }

    // General template for N-dimensional indexing
    template <typename... Indices>
    float& operator()(Indices... indices) {
        return get_offset(indices...);
    }

    // Helper function to compute offset with multiple indices
    template <typename... Indices>
    float& get_offset(Indices... indices) {
        int offset = 0;
        int dims[] = {indices...};  // Unpack indices into array
        for (int dim = 0; dim < sizeof...(indices); ++dim) {
            if (dims[dim] < 0 || dims[dim] >= shape[dim]) {
                throw std::out_of_range("Index out of bounds.");
            }
            offset += dims[dim] * strides[dim];
        }
        return data[offset];
    }

    void print() {
        std::cout << "ndim: " << ndim << "\n";
        std::cout << "shape: ";
        for (int i = 0; i < ndim; ++i) {
            std::cout << shape[i] << " ";
        }
        std::cout << "\n";
        std::cout << "strides: ";
        for (int i = 0; i < ndim; ++i) {
            std::cout << strides[i] << " ";
        }
        std::cout << "\n";
    }
};
