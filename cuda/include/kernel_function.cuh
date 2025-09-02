// Author: Aditya Devarakonda
// Date: August 28, 2025

#ifndef KERNEL_FUNCTION_CUH
#define KERNEL_FUNCTION_CUH
#include <cuda_runtime.h>

enum KernelType : int {
    LINEAR = 0,
    RBF = 1,
    TANH = 2,
    POLYNOMIAL = 3,
    UNDEFINED = -1
};

template <typename T>
class KernelFunction {
  public:
  __host__ KernelFunction(): kernel_type(KernelType::UNDEFINED) {};
  __host__ virtual ~KernelFunction() = default;
  __host__ virtual void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc,  bool diagonal = false) = 0;
  __host__ std::string get_kernel_name() { return kernel_name; };
  protected:
    KernelType kernel_type;
    std::string kernel_name = "Undefined";

};

template <typename T>
class LinearKernel : public KernelFunction<T> {
  public:
    __host__ LinearKernel(): KernelFunction<T>() {
        this->kernel_type = KernelType::LINEAR;
        this->kernel_name = "Linear Kernel";
    }
    __host__ ~LinearKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
};

template <typename T>
class RBFKernel : public KernelFunction<T> {
  public:
    __host__ RBFKernel(T gamma): KernelFunction<T>(), gamma(gamma) {
        this->kernel_type = KernelType::RBF;
        this->kernel_name = "RBF Kernel";
    }
    __host__ ~RBFKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T gamma;
};

template <typename T>
class TanhKernel : public KernelFunction<T> {
  public:
    __host__ TanhKernel(T alpha, T beta): KernelFunction<T>(), alpha(alpha), beta(beta) {
        this->kernel_type = KernelType::TANH;
        this->kernel_name = "Tanh Kernel";
    }
    __host__ ~TanhKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T alpha;
    T beta;
};

template <typename T>
class PolynomialKernel : public KernelFunction<T> {
  public:
    __host__ PolynomialKernel(T alpha, T beta, int degree): KernelFunction<T>(), alpha(alpha), beta(beta), degree(degree) {
        this->kernel_type = KernelType::POLYNOMIAL;
        this->kernel_name = "Polynomial Kernel";
    }
    __host__ ~PolynomialKernel() {}
    __host__ void compute_kernel_matrix(const int m, const int n, const int k, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc, bool diagonal = false) override;
  private:
    T alpha;
    T beta;
    int degree;
};

#endif  // KERNEL_FUNCTION_CUH
