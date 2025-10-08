This repository contains implementations of Kernel K-means clustering that uses a sliding window technique to solve large scale clustering problems on a single GPU.
We include a PyTorch implementation along with other matrix-based and naive Kernel K-means variants.
We also include a CUDA implementation that leverages cuBLAS to perform matrix operations involved in computing the kernel matrix and solving the clustering problem, itself.
