// Author: Aditya Devarakonda
// Date: August 28, 2025

#include <cmath>

#include "blaswrapper.cuh"
#include "kernel_function.cuh"

/* TODO: convert matrices into row-major format*/

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
        gemm(OpA, OpB, m, n, k, &ONE, A, lda, B, ldb, &ZERO, C, ldc);
    }
}

template <typename T>
__device__ void compute_row_norms(cublasOperation_t transpose, const int rows, const int cols, const T* matrix, T* norms) {
    int row = threadIdx.x;
    if (row >= rows) return;
    norms[row] = 0.0f;
    T val = 0.0f;
    for (int col = 0; col < cols; ++col) {
        val = matrix[row + rows*col];
        norms[row] += val * val;
    }
    // printf("Row: %d, row-norm: %.16f\n", row, norms[row]);
}

template <typename T>
__global__ void compute_rbf_transform(cublasOperation_t transA, cublasOperation_t transB, const int m, const int n, const int k, const T* A, const T* B, T* normA, T* normB, T* matrix, const T gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m*n) return;
    compute_row_norms(transA, m, k, A, normA);
    if (A != B) {
        compute_row_norms(transB, n, k, B, normB);
    }
    int row = idx % m;
    int col = idx / m;
    // printf("Idx: %d Row: %d Col: %d, gamma: %.2f\n", idx, row, col, gamma);

    T val = normA[row] + normB[col] + matrix[idx];
    // if (row == 0 && col == 1){
    //     printf("normA: %.16f, normB: %.16f, dot: %.16f, val: %.16f\n", normA[row], normB[col], matrix[idx], val);
    //     printf("Exp argument: %.16f\n", -gamma * val);
    //     printf("Exp value: %.16f\n", expf(-gamma * val));
    // }
    matrix[idx] = expf(-gamma * val);
}

// Specialization for RBF Kernel
template <typename T>
__host__ void RBFKernel<T>::compute_kernel_matrix(cublasOperation_t OpA, cublasOperation_t OpB, const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {

    // Compute the squared Euclidean distance matrix
    const T alpha_dist = -2.0f;
    const T ZERO = 0.0f;
    gemm(OpA, OpB, m, n, k, &alpha_dist, A, lda, B, ldb, &ZERO, C, ldc);
    T *devA, *devB, *devC, *devNormsA, *devNormsB;
    cudaMalloc(&devA, m * k * sizeof(T));
    cudaMalloc(&devNormsA, m * sizeof(T));
    cudaMemcpy(devA, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
    if (A != B){
        cudaMalloc(&devB, n * k * sizeof(T));
        cudaMemcpy(devB, B, n * k * sizeof(T), cudaMemcpyHostToDevice);
        cudaMalloc(&devNormsB, n * sizeof(T));
    }
    else {
        devB = devA;
        devNormsB = devNormsA;
    }
    cudaMalloc(&devC, m * n * sizeof(T));
    cudaMemcpy(devC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    // Apply the RBF kernel transformation
    compute_rbf_transform<<<(m*n + 255)/256, 256>>>(OpA, OpB, m, n, k, devA, devB, devNormsA, devNormsB, devC, gamma);
    cudaMemcpy(C, devC, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devNormsA);
    if (A != B){
        cudaFree(devB);
        cudaFree(devNormsB);
    }
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
