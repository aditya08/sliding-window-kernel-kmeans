#include <string>

#include "argparser.hpp"
#include "matrix.hpp"
#include "dataloader.hpp"

#include "kernel_function.cuh"
#include "kernel_kmeans.cuh"

int main(int argc, char** argv) {
    ArgParser args("Kernel KMeans Clustering");
    args.add_argument<std::string>("data", "Path to the input data file", "./data/acoustic_small");
    // args.add_argument<int>("--n_samples", "Number of samples in the dataset", 0);
    // args.add_argument<int>("--n_features", "Number of features in the dataset", 0);
    args.add_argument<int>("n_clusters", "Number of clusters", 0);
    args.add_argument<int>("block_size", "Block size for processing data", 0);
    args.add_argument<std::string>("kernel", "Kernel type (linear, rbf, polynomial)", "linear");
    args.add_argument<int>("max_iters", "Maximum number of iterations", 300);
    args.add_argument<float>("tol", "Tolerance for convergence", 1e-4);
    args.add_argument<int>("seed", "Random seed for initialization", 42);

    try {
        args.parse(argc, argv);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        args.print_help();
        return EXIT_FAILURE;
    }
    // Example usage of KernelKMeans class
    std::string data_path = args.get<std::string>("data");
    // int n_samples = args.get<int>("--n_samples");
    // int n_features = args.get<int>("--n_features");
    int n_clusters = args.get<int>("n_clusters");
    int block_size = args.get<int>("block_size");
    std::string kernel_type = args.get<std::string>("kernel");
    int max_iters = args.get<int>("max_iters");
    float tol = args.get<float>("tol");
    int seed = args.get<int>("seed");

    if (n_clusters <= 0 || block_size <= 0) {
        std::cerr << "Error: --n_clusters and --block_size must be positive integers." << std::endl;
        args.print_help();
        return EXIT_FAILURE;
    }
    if (kernel_type != "linear" && kernel_type != "rbf" && kernel_type != "polynomial") {
        std::cerr << "Error: --kernel must be one of 'linear', 'rbf', or 'polynomial'." << std::endl;
        args.print_help();
        return EXIT_FAILURE;
    }
    // Load data from LIBSVM file
    LIBSVMReader<float> loader(data_path);
    Matrix<float> data(loader.getNumRows(), loader.getNumCols());
    Matrix<int> labels(loader.getNumRows(), 1);
    loader.loadData(data, labels);
    std::cout << "Loaded data with " << loader.getNumRows() << " samples and " << loader.getNumCols() << " features." << std::endl;
    // Initialize KernelFunction
    LinearKernel<float> kernel = LinearKernel<float>();
    std::cout << "Using kernel: " << kernel.get_kernel_name() << "\n";
    // Initialize and fit KernelKMeans
    KernelKMeans<float> kmeans(&kernel, max_iters=max_iters, tol=tol);
    std::cout << "Fitting KernelKMeans with " << n_clusters << " clusters, block size " << block_size << ", max iters " << max_iters << ", tol " << tol << ", seed " << seed << "\n";
    kmeans.fit(data, loader.getNumRows(), loader.getNumCols(), n_clusters, block_size, seed);
    std::cout << "Clustering complete.\n";
    return EXIT_SUCCESS;
}