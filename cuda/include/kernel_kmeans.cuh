#ifndef KERNEL_KMEANS_CUH
#define KERNEL_KMEANS_CUH
#include <cuda_runtime.h>

template <typename T>
class KernelKMeans {
  public:
    __host__ __device__ KernelKMeans();
    __host__ __device__ ~KernelKMeans();
    __global__ void kmeans_kernel(const T* data, const int n_samples, const int n_features, const int n_clusters, int* labels); 
    __host__ __device__ void fit(const T* data, const int n_samples, const int n_features, const int n_clusters);
    __host__ __device__ void predict(const T* data, const int n_samples, const int n_features, int* labels);
};

#endif  // KERNEL_KMEANS_CUH