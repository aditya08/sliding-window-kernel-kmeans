#ifndef KERNEL_KMEANS_CUH
#define KERNEL_KMEANS_CUH
#include <cuda_runtime.h>

#include "matrix.hpp"

#include "kernel_function.cuh"

template <typename T>
class KernelKMeans {
  public:
    __host__ KernelKMeans(KernelFunction<T>* kernel, int max_it=300, T tol=1e-4);
    __host__ ~KernelKMeans();
    __host__ void fit(Matrix<T>& data, const int n_samples, const int n_features, const int n_clusters, const int block_size, int seed=42);
    __host__ void predict(Matrix<T>& data, const int n_samples, const int n_features);
  private:
    KernelFunction<T>* kernel_fn;
    int max_iters;
    T tolerance;
    int *labels = nullptr;
    T *centroids_matrix = nullptr;
    T *dist_matrix = nullptr;
};

template class KernelKMeans<float>;
template class KernelKMeans<double>;

#endif  // KERNEL_KMEANS_CUH
