// Author: Aditya Devarakonda
// Date: August 28, 2025

#include <cmath>

#include "blaswrapper.cuh"
#include "kernel_function.cuh"

// Specialization for Linear Kernel
template <typename T>
__host__ void LinearKernel<T>::compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
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
void compute_row_norms(const int rows, const int cols, const T* matrix, T* norms) {
    T sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            T val = matrix[rows * col + row];
            sum += val * val;
        }
        norms[row] = sum;
    }
}

// Specialization for RBF Kernel
template <typename T>
__host__ void RBFKernel<T>::compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    // Pre-compute norms of A and B
    T* normA;
    T* normB;
    cudaMalloc(&normA, m * sizeof(T));
    cudaMalloc(&normB, n * sizeof(T));

    // compute row-wise norms in parallel
    compute_row_norms(m, k, A, normA);
    compute_row_norms(n, k, B, normB);

    // Compute the squared Euclidean distance matrix
    const T alpha_dist = -2.0f;
    const T ZERO = 0.0f;
    gemm(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha_dist, A, lda, B, ldb, &ZERO, C, ldc);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * ldc + j] += normA[j] + normB[i];
        }
    }

    // Apply the RBF kernel transformation
    for (int i = 0; i < m * n; ++i) {
        C[i] = expf(-gamma * C[i]);
    }

    cudaFree(normA);
    cudaFree(normB);
}

// Specialization for Tanh Kernel
template <typename T>
__host__ void TanhKernel<T>::compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    if (diagonal) {
        // Compute only the diagonal elements
        for (int i = 0; i < m; ++i) {
            float dot_product = 0.0f;
            for (int j = 0; j < k; ++j) {
                dot_product += A[i * lda + j] * B[i * ldb + j];
            }
            C[i * ldc + i] = tanhf(dot_product);
        }
    } else {
        const T ZERO = 0.0f;
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &ZERO, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = tanhf(C[i] + beta);
        }
    }
}

// Specialization for Polynomial Kernel
template <typename T>
__host__ void PolynomialKernel<T>::compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    if (diagonal) {
        // Compute only the diagonal elements
        for (int i = 0; i < m; ++i) {
            T dot_product = 0.0f;
            for (int j = 0; j < k; ++j) {
                dot_product += A[i * lda + j] * B[i * ldb + j];
            }
            C[i * ldc + i] = powf(dot_product + beta, degree);
        }
    } else {
        const T ZERO = 0.0f;
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &ZERO, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = powf(C[i] + beta, degree);
        }
    }
}

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
