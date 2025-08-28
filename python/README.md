# streaming-kernel-kmeans
A PyTorch implementation of several different algorithmic variants to solve the Kernel K-means clustering problem.

1. Naive implementation (stores full kernel matrix, single cluster update formulation)
2. Matrix implementation (stores full kernel matrix, all clusters update formulation): based on the Popcorn paper: [PPoPP25](https://dl.acm.org/doi/10.1145/3710848.3710887)
3. Streaming implementation (computes blocks of the kernel matrix on the fly, all clusters update formulation)
