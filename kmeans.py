from typing import Optional

from kernel_functions import KernelFunction
from dataset import Dataset

class Kmeans:
    def __init__(self, n_clusters: int, dataset: Dataset, kernel: KernelFunction, max_iter: int = 100, device: str = "cpu") -> None:
        """
        Initializes the Kmeans class with parameters for clustering.
        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            dataset (Dataset): Dataset object containing data and ground truth.
            kernel (KernelFunction): Kernel function to use for clustering.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.device = device
        self.dataset = dataset
        self.kernel = kernel

    def fit(self) -> None:
        """
        Fit the Kmeans model to the dataset.
        This method should implement the Kmeans clustering algorithm.
        """
        pass  # Implementation of fit is deferred.

    def predict(self) -> None:
        """
        Predict the cluster labels for the dataset.
        This method should return the cluster labels for each data point.
        """
        pass  # Implementation of predict is deferred.