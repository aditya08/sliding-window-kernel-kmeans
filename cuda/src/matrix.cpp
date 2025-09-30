#include <algorithm>
#include <iostream>

#include "cuda_runtime.h"

#include "matrix.hpp"

template <typename T>
Matrix<T>::Matrix(int rows, int cols) : nrows(rows), ncols(cols) {
    size = rows * cols;
    allocateMemory(size);
    if (!data_ptr) {
        std::cerr << "Memory allocation failed in constructor\n";
    }
}

// copy constructor
template <typename T>
Matrix<T>::Matrix(int rows, int cols, T* data) : nrows(rows), ncols(cols) {
    size = rows * cols;
    allocateMemory(size);
    copy(rows, cols, data);
}

template <typename T>
void Matrix<T>::copy(int rows, int cols, const T* data){
    if (rows != nrows || cols != ncols) {
        freeMemory();
        nrows = rows;
        ncols = cols;
        size = rows * cols;
        allocateMemory(size);
        if (!data_ptr) {
            std::cerr << "Memory allocation failed during copy\n";
            return;
        }
    }
    std::copy(data, data + size, data_ptr);
}

// allocate pinned host memory
template <typename T>
void Matrix<T>::allocateMemory(int len) {
    if (data_ptr) {
        freeMemory();
    }
    size = len;
    cudaError_t err = cudaMallocHost((void**)&data_ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory\n";
        data_ptr = nullptr;
        size = 0;
        nrows = 0;
        ncols = 0;
        return;
    }
    std::fill(data_ptr, data_ptr + size, 0.0);
}

template <typename T>
void Matrix<T>::freeMemory() {
    if (data_ptr) {
        cudaFreeHost(data_ptr);
        data_ptr = nullptr;
        size = 0;
        nrows = 0;
        ncols = 0;
    }
}

template <typename T>
Matrix<T>::~Matrix() {
    freeMemory();
}

template <typename T>
void Matrix<T>::setValue(int row, int col, T value) {
    if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
        data_ptr[row * ncols + col] = value;
    }
}

template <typename T>
T Matrix<T>::getValue(int row, int col) const {
    if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
        return data_ptr[row * ncols + col];
    }
    return T(); // Return default value if out of bounds
}

template <typename T>
int Matrix<T>::getRows() const {
    return nrows;
}

template <typename T>
int Matrix<T>::getCols() const {
    return ncols;
}

template <typename T>
const T* Matrix<T>::getDataPtr(int offset) {
    if (offset >= 0 && offset < size) {
        return data_ptr + offset;
    }
    return nullptr; // Return nullptr if offset is out of bounds
}