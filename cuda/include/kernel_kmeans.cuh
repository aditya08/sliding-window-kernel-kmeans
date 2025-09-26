#ifndef KERNEL_KMEANS_CUH
#define KERNEL_KMEANS_CUH
#include <cuda_runtime.h>
#include "kernel_function.cuh"
template <typename T>
class KernelKMeans {
  public:
    __host__ KernelKMeans(KernelType type, int max_iters=300, T tol=1e-4): kernel_type(type), max_iters(max_iters), tol(tol) {}
    __host__ ~KernelKMeans();
    __host__ void fit(const T* data, const int n_samples, const int n_features, const int n_clusters, const int block_size, int seed=42);
    __host__ void predict(const T* data, int* labels, const int n_samples, const int n_features);
  private:
    int max_iters;
    T tol;
    T* centroids; // In kernel space
    int *labels;
    KernelType kernel_type;
    KernelFunction<T>* kernel;
};

#endif  // KERNEL_KMEANS_CUH
