#ifndef BLASWRAPPER_CUH
#define BLASWRAPPER_CUH
#include <iostream>

#include "cublas_v2.h"

__host__ void gemm(cublasOperation_t transA, cublasOperation_t transB, const int m, const int n, const int k, const float* alpha, const float* A, const int lda, const float* B, const int ldb, const float* beta, float* C, const int ldc) {
    cublasHandle_t handle;
    cublasStatus_t status;
    cublasCreate(&handle);
    status = cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Handle error
        std::cerr << status <<  " CUBLAS SGEMM failed" << std::endl;
    }
    cublasDestroy(handle);
}

__host__ void gemm(cublasOperation_t transA, cublasOperation_t transB, const int m, const int n, const int k, const double* alpha, const double* A, const int lda, const double* B, const int ldb, const double* beta, double* C, const int ldc) {
    cublasHandle_t handle;
    cublasStatus_t status;
    cublasCreate(&handle);
    status = cublasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Handle error
      std::cerr << status <<  " CUBLAS DGEMM failed" << std::endl;
    }
    cublasDestroy(handle);
}
#endif  // BLASWRAPPER_CUH
