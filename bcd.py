import torch
import time
from typing import Optional

from kmeans import Kmeans
from dataset import Dataset
from kernel_functions import KernelFunction

class KmeansBCD(Kmeans):
    def __init__(self, n_clusters: int, dataset: Dataset, kernel: KernelFunction, max_iter: int = 100, device: str = "cpu") -> None:
        """
        Initializes the KmeansBCD class with parameters for clustering.
        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            filepath (str): Path to the CSV file containing the dataset.
            device (str): Device to load the tensors onto (default is 'cpu').
        """
        super().__init__(n_clusters, dataset, kernel, max_iter=max_iter, device=device)

    def fit(self) -> None:
        """
        Fit the KmeansBCD model to the dataset using Block Coordinate Descent.
        This method implements the Kmeans clustering algorithm with BCD optimization.
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for fitting.")

        self.X = self.dataset.data
        n_samples, n_features = self.X.shape
        labels = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)

        for _ in range(self.max_iter):
            # Update cluster centers
            centers = torch.zeros((self.n_clusters, n_features), device=self.device)
            counts = torch.zeros(self.n_clusters, device=self.device)

            for i in range(n_samples):
                centers[labels[i]] += self.X[i]
                counts[labels[i]] += 1

            for j in range(self.n_clusters):
                if counts[j] > 0:
                    centers[j] /= counts[j]

            # Update labels based on new centers
            distances = torch.cdist(self.X, centers)
            new_labels = distances.argmin(dim=1)

            if torch.equal(labels, new_labels):
                break
            labels = new_labels

        self.labels_ = labels.cpu().numpy()

    def predict(self) -> None:
        """
        Predict the cluster labels for the dataset.
        This method returns the cluster labels for each data point.
        """
        raise NotImplementedError("Predict method is not implemented in KmeansBCD. Use fit method to obtain labels.")
