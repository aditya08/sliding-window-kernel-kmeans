// Copyright 2025. All rights reserved.
// author: Aditya Devarakonda
#include <limits>
#include <random>
#include <algorithm>

#include "matrix.hpp"

#include "blaswrapper.cuh"
#include "kernel_function.cuh"
#include "kernel_kmeans.cuh"

template <typename T>
KernelKMeans<T>::KernelKMeans(KernelFunction<T>* kernel, int max_iters, T tol) {
    kernel_fn = kernel;
    max_iters = max_iters;
    tol = tol;
}

template <typename T>
__global__ void kmeans_kernel(const T* kernel_block_matrix, const int block_size, const int n_samples, const T* dist_matrix, const int n_clusters, int* block_labels, T* centroids) {

}

__host__ void check_labels(const int* labels, const int* prev_labels, const int n_samples, bool& labels_converged) {
    if (prev_labels == nullptr) {
        labels_converged = false;
        return;
    }
    labels_converged = true;
    #pragma omp parallel for schedule(static) reduction(&:labels_converged)
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] != prev_labels[i]) {
            labels_converged = false;
        }
    }
}

__host__ void random_label_initialization(int* labels, const int n_samples, const int n_clusters, int seed=42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(0, n_clusters - 1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        labels[i] = dis(gen);
    }
}

template <typename T>
__host__ void centroids_matrix_initialization(T* centroids_matrix, const int n_samples, const int n_clusters, int* labels) {
    // Initialize centroids matrix to zero
    std::fill(centroids_matrix, centroids_matrix + n_samples * n_clusters, 0);
    //!! Fill centroids matrix, assumed to be row-major
    for (int i = 0; i < n_samples; i++) {
        centroids_matrix[i * n_clusters + labels[i]] = 1.0;
        // TODO: consider folding 1/counts into centroids matrix
        // centroids_matrix[i * n_clusters + labels[i]] += 1.0 / counts[labels[i]];
    }
}

template <typename T>
__host__ void counts_initialization(int* labels, const int n_samples, T* counts, const int n_clusters) {
    for (int i = 0; i < n_samples; i++) {
        counts[labels[i]]++;
    }
    for (int i = 0; i < n_clusters; i++) {
        if (counts[i] == 0) counts[i] = 1;
    }
}

template <typename T>
__host__ void distance_matrix_divide_counts(T* dist_matrix, T* counts, const int n_samples, const int n_clusters) {
    // #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_clusters; j++) {
            dist_matrix[i * n_clusters + j] /= counts[j];
        }
    }
}


template <typename T>
__global__ void compute_masked_distance_matrix_kernel(const T* distance_matrix, const int n_samples, const int n_clusters, T* centroids, T* counts, T* masked_distance_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples*n_clusters) {
        T count = counts[idx % n_clusters];
        masked_distance_matrix[idx] = (-0.5f * distance_matrix[idx]/count) * centroids[idx];
    }
}

template <typename T>
__host__ void KernelKMeans<T>::fit(Matrix<T>& data, const int n_samples, const int n_features, const int n_clusters, const int block_size, int seed) {
    bool converged = false;
    bool labels_converged = false;
    int iter = 0;
    int current_block_size = 0;
    const T NEG_TWO = -2.0;
    const T ZERO = 0.0, ONE = 1.0;
    T sampled_KV_sum = 0.0;
    T objective_diff = std::numeric_limits<T>::max();
    T previous_objective = std::numeric_limits<T>::max();
    T current_objective = 0.0;
    // Allocate host memory
    int* prev_labels = nullptr; // To check for label convergence
    int* labels = new int[n_samples];
    T* kernel_block_matrix = new T[block_size * n_samples];
    T* dist_matrix = new T[n_samples * n_clusters];
    T* masked_dist_matrix = new T[n_samples * n_clusters];
    T* centroids_matrix = new T[n_samples * n_clusters];
    T* masked_dist_sum = new T[n_clusters];
    T* counts = new T[n_clusters]; // To store counts of points in each cluster
    T* ones_vector = new T[block_size];  // For gemv
    std::fill(ones_vector, ones_vector + block_size, 1.0);
    // Allocate device memory
    int* dev_labels;
    cudaMalloc(&dev_labels, n_samples * sizeof(int));
    T* dev_kernel_block_matrix, *dev_dist_matrix, *dev_centroids_matrix;
    T* dev_masked_dist_matrix, *dev_masked_dist_sum;
    T* dev_counts;
    cudaMalloc(&dev_kernel_block_matrix, block_size * n_samples * sizeof(T));
    cudaMalloc(&dev_dist_matrix, block_size * n_clusters * sizeof(T));
    cudaMalloc(&dev_masked_dist_matrix, block_size * n_clusters * sizeof(T));
    cudaMalloc(&dev_centroids_matrix, block_size * n_clusters * sizeof(T));
    cudaMalloc(&dev_masked_dist_sum, n_clusters * sizeof(T));
    cudaMalloc(&dev_counts, n_clusters * sizeof(T));
    // Initialize labels randomly
    // TODO: consider kmeans++ initialization
    // TODO: make this a kernel call?
    random_label_initialization(labels, n_samples, n_clusters, seed);
    // TODO: make this a kernel call?

    while (iter < max_iters) {
        std::copy(labels, labels + n_samples, prev_labels);
        centroids_matrix_initialization(centroids_matrix, n_samples, n_clusters, labels);
        counts_initialization(labels, n_samples, counts, n_clusters);
        cudaMemcpy(dev_centroids_matrix, centroids_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_counts, counts, n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        sampled_KV_sum = 0.0;
        // Process data in blocks
        for (int block_start = 0; block_start < n_samples; block_start += block_size) {
            // handle last block which might be smaller than block_size
            current_block_size = std::min(block_size, n_samples - block_start);
            // // a row block of the kernel matrix
            // kernel->compute_kernel_matrix(CUBLAS_OP_T, CUBLAS_OP_N, current_block_size, n_samples, n_features, data + block_start, current_block_size, data, n_features, kernel_block_matrix, n_samples);

            //!! computes kernel_block_matrix in column-major order (current_block_size x n_samples), with a block of samples (data + block_start) in row-major order and with all samples (data) in row-major order **//
            kernel_fn->compute_kernel_matrix(CUBLAS_OP_T, CUBLAS_OP_N, current_block_size, n_samples, n_features, data.getDataPtr(block_start), n_features, data.getDataPtr(), n_features, dev_kernel_block_matrix, current_block_size);


            // call dist_matrix_ptr (row_major) = kernel_block_matrix (column-major) @ centroids (row-major)
            gemm(CUBLAS_OP_T, CUBLAS_OP_T, n_clusters, current_block_size, n_samples, &NEG_TWO, dev_centroids_matrix, n_clusters, dev_kernel_block_matrix, n_samples, &ZERO, dev_dist_matrix, n_clusters);
            cudaMemcpy(dist_matrix + block_start * n_clusters, dev_dist_matrix, current_block_size * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // !! following computation was folded into compute_masked_distance_matrix_kernel
            // distance_matrix_divide_counts(dist_matrix_ptr, counts, n_samples, n_clusters);

            // Copy to device, may be able to do this asynchronously by overlapping with compute_kernel_matrix and following gemm computation.
            // cudaMemcpy(dev_kernel_block_matrix, kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyHostToDevice);

            // perform elementwise dev_masked_dist_matrix = dev_dist_matrix * dev_centroids_matrix on device
            // TODO: tune kernel launch parameters
            int nblocks = (current_block_size * n_clusters + 255) / 256;
            // dev_dist_matrix = sampled_KV (row-major)
            compute_masked_distance_matrix_kernel<<<nblocks, 1>>>(dev_dist_matrix, current_block_size, n_clusters, dev_centroids_matrix, dev_counts, dev_masked_dist_matrix);

            // dev_masked_dist_sum = sampled_KV.sum(0)
            gemv(CUBLAS_OP_N, n_clusters, current_block_size, &ONE, dev_masked_dist_matrix, n_clusters, ones_vector, 1, &ZERO, dev_masked_dist_sum, 1);

            cudaMemcpy(masked_dist_sum, dev_masked_dist_sum, n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // masked_dist_sum = sampled_KV_sum0_accum
            for (int j = 0; j < n_clusters; j++) {
                sampled_KV_sum += masked_dist_sum[j];
                masked_dist_sum[j] += 1.0 / counts[j]; // add back the 1/counts term
            }

            // cudaMemcpy(kernel_block_matrix, dev_kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(masked_dist_matrix, dev_masked_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(dist_matrix, dev_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(labels, dev_labels, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
            // int grid_size = (current_block_size + 255) / 256;
            // kmeans_kernel<<<grid_size, 256>>>(data + block_start * n_features, current_block_size, n_features, n_clusters, labels + block_start, centroids, max_iters, tol);
            cudaDeviceSynchronize();
        }
        previous_objective = (iter == 0) ? std::numeric_limits<T>::max() : current_objective;
        current_objective = sampled_KV_sum;
        objective_diff = std::abs(previous_objective - current_objective);
        // TODO: compute new labels
        // add masked_dist_sum[j] to each column j of dist_matrix
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_clusters; j++) {
                dist_matrix[i * n_clusters + j] += masked_dist_sum[j];
            }
        }
        // find argmin along each row
        int min_index;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            T* min_elem = std::min_element(dist_matrix + i * n_clusters, dist_matrix + (i + 1) * n_clusters);
            min_index = std::distance(dist_matrix + i * n_clusters, min_elem);
            labels[i] = min_index;
        }
        check_labels(labels, prev_labels, n_samples, labels_converged);
        converged = (iter == max_iters || objective_diff < tol || labels_converged);
        if (converged){
            break;
        }
        iter++;
    }
    // Free device memory
    cudaFree(dev_kernel_block_matrix);
    cudaFree(dev_dist_matrix);
    cudaFree(dev_centroids_matrix);
    cudaFree(dev_masked_dist_matrix);
    cudaFree(dev_masked_dist_sum);
    cudaFree(dev_counts);
    cudaFree(dev_labels);
    // Free host memory
    delete[] kernel_block_matrix;
    delete[] dist_matrix;
    delete[] masked_dist_matrix;
    delete[] centroids_matrix;
    delete[] masked_dist_sum;
    delete[] counts;
    delete[] ones_vector;
    delete[] prev_labels;
    delete[] labels;
}

template class KernelKMeans<float>;
template class KernelKMeans<double>;
