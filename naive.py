import torch
import torch.nn.functional as F
import time

from kmeans import Kmeans
from dataset import Dataset
from kernel_functions import KernelFunction, RBF

class KmeansNaive(Kmeans):

    def __init__(self, n_clusters: int, dataset: Dataset, kernel: KernelFunction, max_iter: int = 100, tol: float = 1e-6, device='cpu') -> None:
        """
        Initializes the KernelKMeans class with parameters for clustering.
        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            dataset (Dataset): Dataset object containing data and ground truth.
            kernel (KernelFunction): Kernel function to use for clustering.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        super().__init__(n_clusters, dataset, kernel, max_iter=max_iter, tol=tol, device=device)

    def fit(self):
        X = self.dataset.data
        n_samples = self.dataset.shape[0]
        self.labels = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)
        K = self.kernel(X, X)
        tr_K = torch.trace(K)
        objective = 0.0
        for _ in range(self.max_iter):
            prev_objective = objective
            dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
            for j in range(self.n_clusters):
                mask = (self.labels == j)
                count = torch.clamp(mask.sum(), min=1)  # Avoid division by zero
                K_j = K[mask][:, mask]
                sum_K_j = K[mask].sum(0)
                dist[:, j] = K.diag() - 2 * sum_K_j / count + K_j.sum() / (count ** 2)
                objective += K_j.sum().item() / count
            new_labels = dist.argmin(1)
            objective = tr_K - objective
            print(f"Iteration objective value: {objective:.4f}, objective change: {abs(prev_objective - objective):.4f}")
            if torch.equal(self.labels, new_labels) or abs(prev_objective - objective) < self.tol:
                print("Cluster assignments are stable. Stopping iterations.")
                self.labels = new_labels
                break
            self.labels = new_labels
    def __fit_matrix_form(self):
        raise NotImplementedError("Matrix form from the Popcorn paper is not implemented.")

    def predict(self):
        # For kernel k-means, prediction is not straightforward without refitting
        raise NotImplementedError("Prediction for new data is not supported in kernel k-means.")

# Example usage:
if __name__ == "__main__":
    start_time = time.time()
    dataset = Dataset("./data/acoustic", device='cpu')
    end_time = time.time()
    print(f"Dataset loaded in {end_time - start_time:.4f} seconds")
    kernel = RBF(gamma=0.5)
    n_clusters = 10
    model = KmeansNaive(n_clusters=n_clusters, dataset=dataset, kernel=kernel, max_iter=20, device='cpu')
    start_time = time.time()
    model.fit()
    end_time = time.time()
    print(f"Kernel K-means execution time: {end_time - start_time:.4f} seconds")