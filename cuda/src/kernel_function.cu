// Author: Aditya Devarakonda
// Date: August 28, 2025

#include <cmath>

#include "blaswrapper.cuh"
#include "kernel_function.cuh"

// Specialization for Linear Kernel
template <typename T>
__host__ void LinearKernel<T>::compute_kernel_matrix(cublasOperation_t OpA, cublasOperation_t OpB, const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    // Compute the linear kernel matrix C = A * B^T
    if (diagonal) {
        // Compute only the diagonal elements
    }
    else {
        const T ONE = 1.0f;
        const T ZERO = 0.0f;
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &ONE, A, lda, B, ldb, &ZERO, C, ldc);
    }
}

template <typename T>
__device__ void compute_row_norms(const int rows, const int cols, const T* matrix, T* norms) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row >= rows || col >= cols) return;
    T val = matrix[rows * col + row];
    val *= val;
    // TODO: add code branch to use cub reductions for large norm computations
    atomicAdd(&norms[row], val);

}
// FIXME: rbf kernel and compute norms are only partially implemented
template <typename T>
__global__ void compute_rbf_transform(const int m, const int n, const int k, const T* A, const T* B, T* matrix, const T gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m*n) return;
    T* normA = static_cast<T*>(malloc(m * sizeof(T)));
    compute_row_norms(m, k, A, normA);
    T * normB;
    if (A != B) {
        T* normB = static_cast<T*>(malloc(n * sizeof(T)));
        compute_row_norms(n, k, B, normB);
    }
    else {
        normB = normA;
    }
    int row = idx / n;
    int col = idx % n;
    T val = normA[row] + normB[col];
    atomicAdd(&matrix[idx], val);
    matrix[idx] = expf(-gamma * matrix[idx]);
}

// Specialization for RBF Kernel
template <typename T>
__host__ void RBFKernel<T>::compute_kernel_matrix(cublasOperation_t OpA, cublasOperation_t OpB, const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {

    // Compute the squared Euclidean distance matrix
    const T alpha_dist = -2.0f;
    const T ZERO = 0.0f;
    gemm(OpA, OpB, m, n, k, &alpha_dist, A, lda, B, ldb, &ZERO, C, ldc);
    T *devA, *devB, *devC;
    cudaMalloc(&devA, m * k * sizeof(T));
    cudaMalloc(&devB, n * k * sizeof(T));
    cudaMalloc(&devC, m * n * sizeof(T));
    cudaMemcpy(devA, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    // Apply the RBF kernel transformation
    compute_rbf_transform<<<(m*n + 255)/256, 256>>>(m, n, k, devA, devB, devC, gamma);
    cudaMemcpy(C, devC, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

template <typename T>
__global__ void compute_tanh_transform(const int size, T* matrix, const T alpha, const T beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    matrix[idx] = tanhf(alpha * matrix[idx] + beta);
}
// Specialization for Tanh Kernel
template <typename T>
__host__ void TanhKernel<T>::compute_kernel_matrix(cublasOperation_t OpA, cublasOperation_t OpB, const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    if (diagonal) {
        // // Compute only the diagonal elements
        // for (int i = 0; i < m; ++i) {
        //     float dot_product = 0.0f;
        //     for (int j = 0; j < k; ++j) {
        //         dot_product += A[i * lda + j] * B[i * ldb + j];
        //     }
        //     C[i * ldc + i] = tanhf(dot_product);
        // }
    } else {
        const T ZERO = 0.0f;
        gemm(OpA, OpB, m, n, k, &alpha, A, lda, B, ldb, &ZERO, C, ldc);
        T *devC;
        cudaMalloc(&devC, m * n * sizeof(T));
        cudaMemcpy(devC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
        // Apply the Tanh kernel transformation
        compute_tanh_transform<<<(m*n + 255)/256, 256>>>(m*n, devC, alpha, beta);
        cudaMemcpy(C, devC, m * n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(devC);
        // Alternative CPU implementation
        // for (int i = 0; i < m * n; ++i) {
        //     C[i] = tanhf(C[i] + beta);
        // }
    }
}

template <typename T>
__global__ void compute_pow_transform(const int size, T* matrix, const T beta, const int degree) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    matrix[idx] = powf(matrix[idx] + beta, degree);
}
// Specialization for Polynomial Kernel
template <typename T>
__host__ void PolynomialKernel<T>::compute_kernel_matrix(cublasOperation_t OpA, cublasOperation_t OpB, const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    if (diagonal) {
        // Compute only the diagonal elements
        // for (int i = 0; i < m; ++i) {
        //     T dot_product = 0.0f;
        //     for (int j = 0; j < k; ++j) {
        //         dot_product += A[i * lda + j] * B[i * ldb + j];
        //     }
        //     C[i * ldc + i] = powf(dot_product + beta, degree);
        // }
    } else {
        const T ZERO = 0.0f;
        gemm(OpA, OpB, m, n, k, &alpha, A, lda, B, ldb, &ZERO, C, ldc);
        T *devC;
        cudaMalloc(&devC, m * n * sizeof(T));
        cudaMemcpy(devC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
        // Apply the Polynomial kernel transformation
        compute_pow_transform<<<(m*n + 255)/256, 256>>>(m*n, devC, beta, degree);
        cudaMemcpy(C, devC, m * n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(devC);
        // for (int i = 0; i < m * n; ++i) {
        //     C[i] = powf(C[i] + beta, degree);
        // }
    }
}

// Explicit template instantiations
template class KernelFunction<float>;
template class KernelFunction<double>;

template class LinearKernel<float>;
template class LinearKernel<double>;

template class RBFKernel<float>;
template class RBFKernel<double>;

template class TanhKernel<float>;
template class TanhKernel<double>;

template class PolynomialKernel<float>;
template class PolynomialKernel<double>;
