// Copyright 2025. All rights reserved.
// author: Aditya Devarakonda
#include <limits>
#include <random>
#include <algorithm>

#include "blaswrapper.cuh"
#include "kernel_kmeans.cuh"

template <typename T>
__global__ void kmeans_kernel(const T* kernel_block_matrix, const int block_size, const int n_samples, const T* dist_matrix, const int n_clusters, int* block_labels, T* centroids){

}

__host__ void check_labels(const int* labels, const int* prev_labels, const int n_samples, bool& labels_converged) {
    if (prev_labels == nullptr) {
        labels_converged = false;
        return;
    }
    labels_converged = true;
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] != prev_labels[i]) {
            labels_converged = false;
            break;
        }
    }
}

__host__ void random_label_initialization(int* labels, const int n_samples, const int n_clusters, int seed=42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(0, n_clusters - 1);
    for (int i = 0; i < n_samples; i++) {
        labels[i] = dis(gen);
    }
}

template <typename T>
__host__ void centroids_matrix_initialization(T* centroids_matrix, T* counts, int* labels, const int n_samples, const int n_clusters) {
    // Initialize centroids matrix to zero
    std::fill(centroids_matrix, centroids_matrix + n_samples * n_clusters, 0);
    // Count number of points in each cluster
    for (int i = 0; i < n_samples; i++) {
        counts[labels[i]]++;
    }
    // Avoid division by zero
    for (int i = 0; i < n_clusters; i++) {
        if (counts[i] == 0) counts[i] = 1;
    }
    // Fill centroids matrix, assumed to be row-major
    for (int i = 0; i < n_samples; i++) {
        centroids_matrix[i * n_clusters + labels[i]] = 1.0;
        // TODO: consider folding 1/counts into centroids matrix
        // centroids_matrix[i * n_clusters + labels[i]] += 1.0 / counts[labels[i]];
    }
}

template <typename T>
__host__ void distance_matrix_divide_counts(T* dist_matrix, T* counts, const int n_samples, const int n_clusters) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_clusters; j++) {
            dist_matrix[i * n_clusters + j] /= counts[j];
        }
    }
}


template <typename T>
__global__ void masked_distance_matrix_kernel(const T* distance_matrix, const int n_samples, const int n_clusters, T* centroids, T* masked_distance_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples*n_clusters) {
        masked_distance_matrix[idx] = (-0.5f * distance_matrix[idx]) * centroids[idx];
    }
}

template <typename T>
__host__ void KernelKMeans<T>::fit(const T* data, const int n_samples, const int n_features, const int n_clusters, const int block_size, int seed) {
    bool converged = false;
    bool labels_converged = false;
    int iter = 0;
    int current_block_size = 0;
    const T NEG_TWO = -2.0;
    const T ZERO = 0.0, ONE = 1.0;
    T objective_diff = std::numeric_limits<T>::max();
    // Allocate host memory
    int* prev_labels = nullptr; // To check for label convergence
    int* labels = new int[n_samples];
    T* kernel_block_matrix = new T[block_size * n_samples];
    T* dist_matrix = new T[n_samples * n_clusters];
    T* masked_dist_matrix = new T[n_samples * n_clusters];
    T* centroids_matrix = new T[n_samples * n_clusters];
    T* counts = new T[n_clusters]; // To store counts of points in each cluster
    T* ones_vector = new T[block_size];  // For gemv
    std::fill(ones_vector, ones_vector + block_size, 1.0);
    // Allocate device memory
    int* dev_labels;
    cudaMalloc(&dev_labels, n_samples * sizeof(int));
    T* dev_kernel_block_matrix, *dev_dist_matrix, *dev_centroids_matrix;
    T* dev_masked_dist_matrix, *dev_masked_dist_sum;
    cudaMalloc(&dev_kernel_block_matrix, block_size * n_samples * sizeof(T));
    cudaMalloc(&dev_dist_matrix, n_samples * n_clusters * sizeof(T));
    cudaMalloc(&dev_masked_dist_matrix, n_samples * n_clusters * sizeof(T));
    cudaMalloc(&dev_centroids_matrix, n_samples * n_clusters * sizeof(T));
    cudaMalloc(&dev_masked_dist_sum, n_clusters * sizeof(T));
    // Initialize labels randomly
    // TODO: consider kmeans++ initialization
    // TODO: make this a kernel call?
    random_label_initialization(labels, n_samples, n_clusters, seed);
    // TODO: make this a kernel call?
    centroids_matrix_initialization(centroids_matrix, counts, labels, n_samples, n_clusters);

    while (iter < max_iters) {
        for (int block_start = 0; block_start < n_samples; block_start += block_size) {
            // handle last block which might be smaller than block_size
            current_block_size = std::min(block_size, n_samples - block_start);
            // a row block of the kernel matrix
            kernel->compute_kernel_matrix(CUBLAS_OP_T, CUBLAS_OP_N, current_block_size, n_samples, n_features, data + block_start, current_block_size, data, n_features, kernel_block_matrix, n_samples);

            // call kernel_block_matrix @ centroids
            gemm(CUBLAS_OP_N, CUBLAS_OP_N, current_block_size, n_clusters, n_samples, &NEG_TWO, kernel_block_matrix, n_samples, centroids_matrix, n_samples, &ZERO, dist_matrix, n_clusters);

            // TODO: divide dist_matrix by counts (make this a kernel call?)
            distance_matrix_divide_counts(dist_matrix, counts, n_samples, n_clusters);

            // Copy to device, may be able to do this asynchronously by overlapping with compute_kernel_matrix and following gemm computation.
            // cudaMemcpy(dev_kernel_block_matrix, kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_dist_matrix, dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_centroids_matrix, centroids_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyHostToDevice);

            // perform elementwise dev_masked_dist_matrix = dev_dist_matrix * dev_centroids_matrix on device
            // TODO: tune kernel launch parameters
            int nblocks = (n_samples * n_clusters + 255) / 256;
            masked_distance_matrix_kernel<<<nblocks, 1>>>(dev_dist_matrix, n_samples, n_clusters, dev_centroids_matrix, dev_masked_dist_matrix);
            // TODO: perform gemv to compute distances to centroids
            gemv(CUBLAS_OP_T, current_block_size, n_clusters, &ONE, dev_masked_dist_matrix, current_block_size, ones_vector, 1, &ZERO, dev_masked_dist_sum, 1);

            // TODO: divide masked_dist_sum by counts on host

            // TODO: add masked_dist_sum to dist_matrix on device

            // cudaMemcpy(kernel_block_matrix, dev_kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(masked_dist_matrix, dev_masked_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(dist_matrix, dev_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(labels, dev_labels, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
            // int grid_size = (current_block_size + 255) / 256;
            // kmeans_kernel<<<grid_size, 256>>>(data + block_start * n_features, current_block_size, n_features, n_clusters, labels + block_start, centroids, max_iters, tol);
            cudaDeviceSynchronize();
        }
        check_labels(labels, prev_labels, n_samples, labels_converged);  // make this a kernel call
        converged = (iter == max_iters || objective_diff < tol || labels_converged);
        if (converged){
            break;
        }
        iter++;
    }
}

template class KernelKMeans<float>;
template class KernelKMeans<double>;