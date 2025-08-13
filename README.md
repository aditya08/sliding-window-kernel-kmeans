# streaming-kernel-kmean
A PyTorch implementation of several different algorithmic variants to solve the Kernel K-means clustering problem.

1. Naive implementation (computes full kernel matrix, single cluster update formulation)
2. Matrix implementation (computes full kernel matrix, all clusters update formulation)
3. Streaming implementation (computes blocks of the kernel matrix on the fly, all clusters update formulation)
