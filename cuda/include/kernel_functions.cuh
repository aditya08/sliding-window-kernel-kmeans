// Author: Aditya Devarakonda
// Date: August 28, 2025

#ifndef KERNEL_FUNCTIONS_CUH
#define KERNEL_FUNCTIONS_CUH
#include <cuda_runtime.h>

template <typename T>
class KernelFunctions {
  public:
    __host__ KernelFunctions(): kernel_name("undefined") {};
    __host__ ~KernelFunctions() {}
    __host__ virtual void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc,  bool diagonal = false);
  protected:
    std::string kernel_name;
};

template <typename T>
class LinearKernel : public KernelFunctions<T> {
  public:
    __host__ LinearKernel(): KernelFunctions<T>() {
        this->kernel_name = "linear";
    }
    __host__ ~LinearKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
};

template <typename T>
class RBFKernel : public KernelFunctions<T> {
  public:
    __host__ RBFKernel(T gamma): KernelFunctions<T>(), gamma(gamma) {
        this->kernel_name = "rbf";
    }
    __host__ ~RBFKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T gamma;
};

template <typename T>
class TanhKernel : public KernelFunctions<T> {
  public:
    __host__ TanhKernel(T alpha, T beta): KernelFunctions<T>(), alpha(alpha), beta(beta) {
        this->kernel_name = "tanh";
    }
    __host__ ~TanhKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T alpha;
    T beta;
};

template <typename T>
class PolynomialKernel : public KernelFunctions<T> {
  public:
    __host__ PolynomialKernel(T alpha, T beta, int degree): KernelFunctions<T>(), alpha(alpha), beta(beta), degree(degree) {
        this->kernel_name = "polynomial";
    }
    __host__ ~PolynomialKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T alpha;
    T beta;
    int degree;
};

#endif  // KERNEL_FUNCTIONS_CUH