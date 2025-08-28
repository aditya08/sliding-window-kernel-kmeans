// Author: Aditya Devarakonda
// Date: August 28, 2025

#include <cmath>
#include <cublas_v2.h>
#include "kernel_functions.cuh"

template <typename T>
__host__ void KernelFunctions<T>::compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal) {
    // Default implementation does nothing
}

// Specialization for Linear Kernel
template <>
__host__ void LinearKernel<float>::compute_kernel_matrix(const int m, const int n, const int k, const float* A, const int lda, const float* B, const int ldb, float* C, const int ldc, bool diagonal) {
    // Compute the linear kernel matrix C = A * B^T
    if (diagonal) {
        // Compute only the diagonal elements
    }
    else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        cublasDestroy(handle);
    }

}

template <>
__host__ void LinearKernel<double>::compute_kernel_matrix(const int m, const int n, const int k, const double* A, const int lda, const double* B, const int ldb, double* C, const int ldc, bool diagonal) {
    // Compute the linear kernel matrix C = A * B^T
    if (diagonal) {
        // Compute only the diagonal elements
    }
    else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        cublasDestroy(handle);
    }
}
void compute_row_norms(const int rows, const int cols, const float* matrix, float* norms) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float val = matrix[row * cols + col];
            sum += val * val;
        }
        norms[row] = sum;
    }
}

void compute_row_norms(const int rows, const int cols, const double* matrix, double* norms) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            double val = matrix[row * cols + col];
            sum += val * val;
        }
        norms[row] = sum;
    }
}

// Specialization for RBF Kernel
template <>
__host__ void RBFKernel<float>::compute_kernel_matrix(const int m, const int n, const int k, const float* A, const int lda, const float* B, const int ldb, float* C, const int ldc, bool diagonal) {
    const float gamma = 0.5f; // RBF kernel parameter

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Pre-compute norms of A and B
    float* normA;
    float* normB;
    cudaMalloc(&normA, m * sizeof(float));
    cudaMalloc(&normB, n * sizeof(float));

    // compute row-wise norms in parallel
    compute_row_norms(m, k, A, normA);
    compute_row_norms(n, k, B, normB);

    // Compute the squared Euclidean distance matrix
    const float alpha_dist = -2.0f;
    const float beta_dist = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha_dist, A, lda, B, ldb, &beta_dist, C, ldc);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] += normA[i] + normB[j];
        }
    }

    // Apply the RBF kernel transformation
    for (int i = 0; i < m * n; ++i) {
        C[i] = expf(-gamma * C[i]);
    }

    cudaFree(normA);
    cudaFree(normB);
    cublasDestroy(handle);
}

template <>
__host__ void RBFKernel<double>::compute_kernel_matrix(const int m, const int n, const int k, const double* A, const int lda, const double* B, const int ldb, double* C, const int ldc, bool diagonal) {
    const double gamma = 0.5; // RBF kernel parameter

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Pre-compute norms of A and B
    double* normA;
    double* normB;
    cudaMalloc(&normA, m * sizeof(double));
    cudaMalloc(&normB, n * sizeof(double));

    // compute row-wise norms in parallel
    compute_row_norms(m, k, A, normA);
    compute_row_norms(n, k, B, normB);

    // Compute the squared Euclidean distance matrix
    const double alpha_dist = -2.0;
    const double beta_dist = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha_dist, A, lda, B, ldb, &beta_dist, C, ldc);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] += normA[i] + normB[j];
        }
    }

    // Apply the RBF kernel transformation
    for (int i = 0; i < m * n; ++i) {
        C[i] = exp(-gamma * C[i]);
    }

    cudaFree(normA);
    cudaFree(normB);
    cublasDestroy(handle);
}

// Specialization for Tanh Kernel
template <>
__host__ void TanhKernel<float>::compute_kernel_matrix(const int m, const int n, const int k, const float* A, const int lda, const float* B, const int ldb, float* C, const int ldc, bool diagonal) {
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
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = tanhf(C[i]);
        }
        cublasDestroy(handle);
    }
}

template <>
__host__ void TanhKernel<double>::compute_kernel_matrix(const int m, const int n, const int k, const double* A, const int lda, const double* B, const int ldb, double* C, const int ldc, bool diagonal) {
    if (diagonal) {
        // Compute only the diagonal elements
        for (int i = 0; i < m; ++i) {
            double dot_product = 0.0;
            for (int j = 0; j < k; ++j) {
                dot_product += A[i * lda + j] * B[i * ldb + j];
            }
            C[i * ldc + i] = tanh(dot_product);
        }
    } else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = tanh(C[i]);
        }
        cublasDestroy(handle);
    }
}

// Specialization for Polynomial Kernel
template <>
__host__ void PolynomialKernel<float>::compute_kernel_matrix(const int m, const int n, const int k, const float* A, const int lda, const float* B, const int ldb, float* C, const int ldc, bool diagonal) {
    const float coef0 = 1.0f; // Coefficient for the constant term
    const int degree = 3;     // Degree of the polynomial
    if (diagonal) {
        // Compute only the diagonal elements
        for (int i = 0; i < m; ++i) {
            float dot_product = 0.0f;
            for (int j = 0; j < k; ++j) {
                dot_product += A[i * lda + j] * B[i * ldb + j];
            }
            C[i * ldc + i] = powf(dot_product + coef0, degree);
        }
    } else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = powf(C[i] + coef0, degree);
        }
        cublasDestroy(handle);
    }
}

template <>
__host__ void PolynomialKernel<double>::compute_kernel_matrix(const int m, const int n, const int k, const double* A, const int lda, const double* B, const int ldb, double* C, const int ldc, bool diagonal) {
    const double coef0 = 1.0; // Coefficient for the constant term
    const int degree = 3;     // Degree of the polynomial
    if (diagonal) {
        // Compute only the diagonal elements
        for (int i = 0; i < m; ++i) {
            double dot_product = 0.0;
            for (int j = 0; j < k; ++j) {
                dot_product += A[i * lda + j] * B[i * ldb + j];
            }
            C[i * ldc + i] = pow(dot_product + coef0, degree);
        }
    } else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        for (int i = 0; i < m * n; ++i) {
            C[i] = pow(C[i] + coef0, degree);
        }
        cublasDestroy(handle);
    }
}

template class KernelFunctions<float>;
template class KernelFunctions<double>;

template class LinearKernel<float>;
template class LinearKernel<double>;

template class RBFKernel<float>;
template class RBFKernel<double>;

template class TanhKernel<float>;
template class TanhKernel<double>;

template class PolynomialKernel<float>;
template class PolynomialKernel<double>;
