#include <limits>
#include <random>
#include <algorithm>

#include "curand_kernel.h"
#include <cub/cub.cuh>

#include "blaswrapper.cuh"
#include "kernel_kmeans.cuh"

template <typename T>
KernelKMeans<T>::KernelKMeans(KernelFunction<T>* kernel, int max_it, T tol) {
    kernel_fn = kernel;
    max_iters = max_it;
    tolerance = tol;
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

__global__ void random_label_initialization_kernel(int* labels, const int n_samples, const int n_clusters, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    if (idx < n_samples) {
        labels[idx] = curand(&state) % n_clusters;
    }
    if (idx == 0) {
        printf("Random label initialization done.\n");
        printf("First 10 labels: ");
        for (int i = 0; i < 10 && i < n_samples; i++) {
            printf("%d ", labels[i]);
        }
        printf("\n");
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
__global__ void centroids_matrix_initialization_kernel(T* centroids_matrix, const int n_samples, const int n_clusters, int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * n_clusters) {
        centroids_matrix[idx] = 0.0;
    }
    __syncthreads();
    if (idx < n_samples) {
        centroids_matrix[idx * n_clusters + labels[idx]] = 1.0;
    }
}

template <typename T>
__host__ void centroids_matrix_initialization(T* centroids_matrix, const int n_samples, const int n_clusters, int* labels) {
    // Initialize centroids matrix to zero
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples * n_clusters; i++) {
        centroids_matrix[i] = 0.0;
    }
    //!! Fill centroids matrix, assumed to be row-major
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        centroids_matrix[i * n_clusters + labels[i]] = 1.0;
        // TODO: consider folding 1/counts into centroids matrix
        // centroids_matrix[i * n_clusters + labels[i]] += 1.0 / counts[labels[i]];
    }
}

template <typename T>
__global__ void counts_initialization_kernel(int* labels, const int n_samples, T* counts, const int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        counts[idx] = 0;
    }
    __syncthreads();
    if (idx < n_samples) {
        atomicAdd(&counts[labels[idx]], 1);
    }
    __syncthreads();
    if (idx < n_clusters) {
        if (counts[idx] == 0) counts[idx] = 1;
    }
}

template <typename T>
__host__ void counts_initialization(int* labels, const int n_samples, T* counts, const int n_clusters) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_clusters; i++) {
        counts[i] = 0;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        counts[labels[i]]++;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_clusters; i++) {
        if (counts[i] == 0) counts[i] = 1;
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
__global__ void compute_masked_distance_matrix_kernel(const T* distance_matrix, const int n_samples, const int n_clusters, T* centroids, T* counts, T* masked_distance_matrix) {
    //!! distance_matrix is in column-major order
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples*n_clusters) {
        // printf("Thread %d processing index %d\n", idx, idx);
        int row_major_idx = (idx % n_samples) * n_clusters + (idx / n_samples);
        T count = counts[(idx / n_samples) % n_clusters];
        masked_distance_matrix[row_major_idx] = ( distance_matrix[row_major_idx]/count) * centroids[row_major_idx];
        // printf("masked_distance_matrix[%d] = %f\n", row_major_idx, masked_distance_matrix[row_major_idx]);
    }
}

template <typename T>
__global__ void compute_objective_function_kernel(const T* masked_distance_matrix, const int n_samples, const int n_clusters, T* counts, T* masked_dist_sum, T* masked_dist_sum_accum, T& current_objective) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        masked_dist_sum[idx] = 0.0;
    }
    __syncthreads();
    if (idx < n_samples * n_clusters) {
        atomicAdd(&masked_dist_sum[idx % n_clusters], masked_distance_matrix[idx]);
    }
    __syncthreads();
    if (idx < n_clusters) {
        masked_dist_sum_accum[idx] = masked_dist_sum[idx]/counts[idx];
    }
    __syncthreads();
    // Now compute the total objective function value
    // n_clusters < 256, so we can do a simple reduction in a single block
    __shared__ T shared_sum[256];

    // Each thread processes a subset of clusters
    T thread_sum = 0.0;
    for (int j = threadIdx.x; j < n_clusters; j += blockDim.x) {
        thread_sum += masked_dist_sum_accum[j];
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the result to current_objective
    if (threadIdx.x == 0) {
        atomicAdd(&current_objective, shared_sum[0]);
    }
    if (idx == 0) {
        printf("Current objective: %f\n", current_objective);
    }
}

template <typename T>
__global__ void update_distances_and_labels(T* dist_matrix, const int n_samples, const int n_clusters, T* counts, T* masked_dist_sum, int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * n_clusters) {
        dist_matrix[idx] = (-2. * dist_matrix[idx]/counts[idx % n_clusters] + masked_dist_sum[idx % n_clusters]);
    }
    __syncthreads();
    if (idx < n_samples) {
        // Allocate shared memory for CUB
        extern __shared__ char shared_memory[];

        T thread_min = FLT_MAX;
        int thread_min_idx = -1;

        // Each thread processes a subset of clusters
        for (int j = threadIdx.x; j < n_clusters; j += blockDim.x) {
            T value = dist_matrix[idx * n_clusters + j];
            if (value < thread_min) {
                thread_min = value;
                thread_min_idx = j;
            }
        }

        // Use CUB to perform block-wide reduction
        typedef cub::BlockReduce<cub::KeyValuePair<int, T>, 256> BlockReduceKV;

        __shared__ typename BlockReduceKV::TempStorage temp_storage_kv;

        cub::KeyValuePair<int, T> thread_kv(thread_min_idx, thread_min);
        cub::KeyValuePair<int, T> block_kv = BlockReduceKV(temp_storage_kv).Reduce(thread_kv, cub::ArgMin());

        // Write the result to labels
        if (threadIdx.x == 0) {
            labels[idx] = block_kv.key; // Use the key from the CUB reduction result
            // printf("Updated label for sample %d: %d\n", idx, labels[idx]);
        }
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
    // Allocate pinned host memory
    int* prev_labels; // To check for label convergence
    T *kernel_block_matrix;
    T *masked_dist_sum, *counts, *ones_vector;
    T *masked_dist_sum_accum;
    cudaError_t err;
    err = cudaMallocHost((void**)&labels, n_samples * sizeof(int));
    err = cudaMallocHost((void**)&prev_labels, n_samples * sizeof(int));
    err = cudaMallocHost((void**)&centroids_matrix, n_samples * n_clusters * sizeof(T));
    err = cudaMallocHost((void**)&dist_matrix, n_samples * n_clusters * sizeof(T));
    err = cudaMallocHost((void**)&kernel_block_matrix, block_size * n_samples * sizeof(T));
    err = cudaMallocHost((void**)&masked_dist_sum, n_clusters * sizeof(T));
    err = cudaMallocHost((void**)&masked_dist_sum_accum, n_clusters * sizeof(T));
    err = cudaMallocHost((void**)&counts, n_clusters * sizeof(T));
    err = cudaMallocHost((void**)&ones_vector, block_size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating pinned host memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    std::fill(dist_matrix, dist_matrix + n_samples * n_clusters, 0.0);
    std::fill(ones_vector, ones_vector + block_size, 1.0);
    // labels = new int[n_samples];
    // T* kernel_block_matrix = new T[block_size * n_samples];
    // T* dist_matrix = new T[n_samples * n_clusters];
    // T* masked_dist_matrix = new T[n_samples * n_clusters];
    // T* centroids_matrix = new T[n_samples * n_clusters];
    // T* masked_dist_sum = new T[n_clusters];
    // T* counts = new T[n_clusters]; // To store counts of points in each cluster
    // T* ones_vector = new T[block_size];  // For gemv
    // Allocate device memory
    int* dev_labels;
    T* dev_kernel_block_matrix, *dev_dist_matrix, *dev_centroids_matrix;
    T* dev_masked_dist_matrix, *dev_masked_dist_sum, *dev_masked_dist_sum_accum;
    T* dev_counts, *dev_data_matrix;

    err = cudaMalloc(&dev_labels, n_samples * sizeof(int));
    err = cudaMalloc(&dev_kernel_block_matrix, block_size * n_samples * sizeof(T));
    err = cudaMalloc(&dev_dist_matrix, block_size * n_clusters * sizeof(T));
    err = cudaMalloc(&dev_masked_dist_matrix, block_size * n_clusters * sizeof(T));
    err = cudaMalloc(&dev_centroids_matrix, n_samples * n_clusters * sizeof(T));
    err = cudaMalloc(&dev_masked_dist_sum, n_clusters * sizeof(T));
    err = cudaMalloc(&dev_masked_dist_sum_accum, n_clusters * sizeof(T));
    err = cudaMalloc(&dev_counts, n_clusters * sizeof(T));
    err = cudaMalloc(&dev_data_matrix, n_samples * n_features * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(dev_data_matrix, data.getDataPtr(), n_samples * n_features * sizeof(T), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Main loop
    // Initialize labels randomly
    // TODO: consider kmeans++ initialization
    // TODO: make this a kernel call?
    // random_label_initialization_kernel<<<(n_samples + 255)/256, 256>>>(dev_labels, n_samples, n_clusters, seed);
    random_label_initialization(labels, n_samples, n_clusters, seed);
    // TODO: make this a kernel call?
    // std::cout << "Initial labels: " << "\n";
    // for (int i = 0; i < n_samples; ++i) {
    //     std::cout << labels[i] << " ";
    // }
    // std::cout << "\n";
    while (iter < max_iters) {
        std::cout << "Starting Iteration " << iter << std::endl;
        // std::copy(labels, labels + n_samples, prev_labels);
        // centroids_matrix_initialization_kernel<<<(n_samples * n_clusters + 255)/256, 256>>>(dev_centroids_matrix, n_samples, n_clusters, dev_labels);
        // counts_initialization_kernel<<<(n_samples + 255)/256, 256>>>(dev_labels, n_samples, dev_counts, n_clusters);

        std::copy(labels, labels + n_samples, prev_labels);
        centroids_matrix_initialization(centroids_matrix, n_samples, n_clusters, labels);
        counts_initialization(labels, n_samples, counts, n_clusters);
        cudaMemcpy(dev_counts, counts, n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_centroids_matrix, centroids_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        sampled_KV_sum = 0.0;
        std::fill(masked_dist_sum_accum, masked_dist_sum_accum + n_clusters, 0.0);
        // cudaMemcpy(dev_counts, counts, n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_centroids_matrix, centroids_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        // sampled_KV_sum = 0.0;
        // std::fill(masked_dist_sum_accum, masked_dist_sum_accum + n_clusters, 0.0);
        // std::cout << "counts: \n";
        // for (int j = 0; j < n_clusters; j++) {
        //     std::cout << counts[j] << " ";
        // }
        // std::cout << "\n";
        // Process data in blocks
        for (int block_start = 0; block_start < n_samples; block_start += block_size) {
            // handle last block which might be smaller than block_size
            current_block_size = std::min(block_size, n_samples - block_start);
            // // a row block of the kernel matrix
            // kernel->compute_kernel_matrix(CUBLAS_OP_T, CUBLAS_OP_N, current_block_size, n_samples, n_features, data + block_start, current_block_size, data, n_features, kernel_block_matrix, n_samples);

            //!! computes kernel_block_matrix in column-major order (current_block_size x n_samples), with a block of samples (data + block_start) in row-major order and with all samples (data) in row-major order **//
            // std::cout << "Computing kernel matrix for block starting at " << block_start << " with block size " << current_block_size << "...\n";

            kernel_fn->compute_kernel_matrix(CUBLAS_OP_T, CUBLAS_OP_N, current_block_size, n_samples, n_features, dev_data_matrix + block_start*n_features, n_features, dev_data_matrix, n_features, dev_kernel_block_matrix, current_block_size);
            // cudaMemcpy(kernel_block_matrix, dev_kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyDeviceToHost);
            // for (int i = 0; i < current_block_size * n_samples; i++) {
                //     std::cout << kernel_block_matrix[i] << " ";
                // }
                // std::cout << "\n";

            // std::cout << "Computing distance matrix for block starting at " << block_start << "...\n";
            // dist_matrix (row major) = kernel_block_matrix (column-major) @ centroids (row-major)
            gemm(CUBLAS_OP_N, CUBLAS_OP_T, n_clusters, current_block_size, n_samples, &ONE, dev_centroids_matrix, n_clusters, dev_kernel_block_matrix, current_block_size, &ZERO, dev_dist_matrix, n_clusters);

            cudaMemcpyAsync(dist_matrix + block_start * n_clusters, dev_dist_matrix, current_block_size * n_clusters * sizeof(T), cudaMemcpyDeviceToHost, stream);

            int nblocks = (current_block_size * n_clusters + 255) / 256;
            compute_masked_distance_matrix_kernel<<<nblocks, 256>>>(dev_dist_matrix, current_block_size, n_clusters, dev_centroids_matrix + block_start * n_clusters, dev_counts, dev_masked_dist_matrix);

            // cudaMemcpy(masked_dist_matrix, dev_masked_dist_matrix, current_block_size * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // std::cout << "Masked dist matrix: \n";
            // for (int i = 0; i < current_block_size * n_clusters; ++i) {
            //      std::cout << masked_dist_matrix[i] << " ";
            // }
            // std::cout << "\n";
            // cudaMemcpy(masked_dist_matrix + block_start * n_clusters, dev_masked_dist_matrix, current_block_size * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // dev_masked_dist_sum = sampled_KV.sum(0)
            gemv(CUBLAS_OP_N, n_clusters, current_block_size, &ONE, dev_masked_dist_matrix, n_clusters, ones_vector, 1, &ZERO, dev_masked_dist_sum, 1);

            cudaMemcpy(masked_dist_sum, dev_masked_dist_sum, n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // masked_dist_sum = sampled_KV_sum0_accum
            for (int j = 0; j < n_clusters; j++) {
                sampled_KV_sum += masked_dist_sum[j]; // add to total sampled_KV_sum
                masked_dist_sum_accum[j] += masked_dist_sum[j]/counts[j];
                // std::cout << "masked_dist_sum[" << j << "] before adding to sampled_KV_sum: " << masked_dist_sum[j]  << " masked_dist_sum_accum[" << j << "]: " << masked_dist_sum_accum[j] << "\n";
            }
            // std::cout << "Sampled KV sum after processing block starting at " << block_start << ": " << sampled_KV_sum << "\n";
            // cudaMemcpy(kernel_block_matrix, dev_kernel_block_matrix, current_block_size * n_samples * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(masked_dist_matrix, dev_masked_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(dist_matrix, dev_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(labels, dev_labels, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
            // int grid_size = (current_block_size + 255) / 256;
            // kmeans_kernel<<<grid_size, 256>>>(data + block_start * n_features, current_block_size, n_features, n_clusters, labels + block_start, centroids, max_iters, tol);
            // cudaDeviceSynchronize();
        }
        // cudaMemcpy(masked_dist_sum, dev_masked_dist_sum, n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
        previous_objective = (iter == 0) ? std::numeric_limits<T>::max() : current_objective;
        current_objective = 0.0;
        current_objective = sampled_KV_sum;
        // compute_objective_function_kernel<<<(n_samples + 255)/256, 256>>>(dev_masked_dist_matrix, n_samples, n_clusters, dev_counts, dev_masked_dist_sum, dev_masked_dist_sum_accum, current_objective);
        // for (int j = 0; j < n_clusters; j++) {
        //     sampled_KV_sum += masked_dist_sum[j]; // add to total sampled_KV_sum
        //     masked_dist_sum_accum[j] += masked_dist_sum[j]/counts[j];
        //     masked_dist_sum[j] = 0.0; // reset for next iteration
        //     // std::cout << "masked_dist_sum[" << j << "] before adding to sampled_KV_sum: " << masked_dist_sum[j]  << " masked_dist_sum_accum[" << j << "]: " << masked_dist_sum_accum[j] << "\n";
        // }
        // cudaMemcpy(dev_masked_dist_sum, masked_dist_sum, n_clusters * sizeof(T), cudaMemcpyHostToDevice);
        // current_objective = sampled_KV_sum;
        objective_diff = std::abs(previous_objective - current_objective);
        std::cout << "Objective at iteration " << iter << ": " << current_objective << ", Objective diff: " << objective_diff << std::endl;
        // cudaStreamSynchronize(stream);
        // TODO: compute new labels
        // add masked_dist_sum[j] to each column j of dist_matrix
        // #pragma omp parallel for schedule(static)
        // std::cout << "Distance matrix update: \n";
        // update_distances_and_labels<<<(n_samples * n_clusters + 255)/256, 256>>>(dev_dist_matrix, n_samples, n_clusters, dev_counts, dev_masked_dist_sum, dev_labels);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_clusters; j++) {
                dist_matrix[i * n_clusters + j] = (-2. * dist_matrix[i * n_clusters + j]/counts[j] + masked_dist_sum_accum[j]);
                // std::cout << dist_matrix[i * n_clusters + j] << " ";
            }
            // std::cout << "\n";
        }
        // find argmin along each row
        int min_index;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            T* min_elem = std::min_element(dist_matrix + i * n_clusters, dist_matrix + (i + 1) * n_clusters);
            min_index = std::distance(dist_matrix + i * n_clusters, min_elem);
            labels[i] = min_index;
        }
        // check_labels(labels, prev_labels, n_samples, labels_converged);
        converged = iter == max_iters - 1;
        // converged = (iter == max_iters || objective_diff < tolerance || labels_converged);
        if (converged){
            std::cout << "Converged at iteration " << iter + 1 << "/" << max_iters << " with objective: " << current_objective << " objective diff: " << objective_diff << " labels_converged: " << labels_converged << std::endl;
            // cudaMemcpy(labels, dev_labels, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(dist_matrix, dev_dist_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            // cudaMemcpy(centroids_matrix, dev_centroids_matrix, n_samples * n_clusters * sizeof(T), cudaMemcpyDeviceToHost);
            break;
        }
        std::cout << "Iteration " << iter << " complete. Objective: " << current_objective << ", Objective Diff: " << objective_diff << std::endl;
        iter++;
    }
    // Free device memory
    cudaFree(dev_labels);
    cudaFree(dev_kernel_block_matrix);
    cudaFree(dev_dist_matrix);
    cudaFree(dev_masked_dist_matrix);
    cudaFree(dev_centroids_matrix);
    cudaFree(dev_masked_dist_sum);
    cudaFree(dev_masked_dist_sum_accum);
    cudaFree(dev_counts);
    cudaFree(dev_data_matrix);

    // Free host memory
    cudaFreeHost(labels);
    cudaFreeHost(prev_labels);
    cudaFreeHost(centroids_matrix);
    cudaFreeHost(dist_matrix);
    cudaFreeHost(kernel_block_matrix);
    cudaFreeHost(masked_dist_sum);
    cudaFreeHost(masked_dist_sum_accum);
    cudaFreeHost(counts);
    cudaFreeHost(ones_vector);

}

template <typename T>
KernelKMeans<T>::~KernelKMeans() {
    if (labels) cudaFreeHost(labels);
    if (centroids_matrix) cudaFreeHost(centroids_matrix);
    if (dist_matrix) cudaFreeHost(dist_matrix);
}
