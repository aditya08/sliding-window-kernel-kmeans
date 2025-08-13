import torch
import torch.nn.functional as F
import time

from kmeans import Kmeans
from dataset import Dataset
from kernel_functions import KernelFunction, RBF

class KmeansMatrix(Kmeans):

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
        V = F.one_hot(self.labels, num_classes=self.n_clusters).to(dtype=torch.float32, device=self.device)
        V_sum = V.sum(0).unsqueeze(0)  # (1, n_clusters) vector
        K = self.kernel(X, X)  # (n_samples, n_samples) kernel matrix
        K_diag = K.diag().unsqueeze(1)  # (n_samples, 1) diagonal of K
        tr_K = torch.trace(K)
        objective = 0.0
        prev_objective = 0.0
        for _ in range(self.max_iter):
            # Avoid division by zero
            counts = torch.clamp(V_sum, min=1)  # (1, n_clusters) vector
            KV= (K @ V) / counts  # (n_samples, n_clusters) column subsampled kernel matrix
            sampled_KV = (KV * V)  # (n_samples, n_clusters) row sampled kernel matrix
            dist = K_diag - (2  * KV) + (sampled_KV.sum(0) / counts) # (n_samples, n_clusters) distance matrix
            new_labels = dist.argmin(1)  # (n_samples,) new labels
            objective = (tr_K - sampled_KV.sum()).item()
            print(f"Iteration objective value: {objective:.4f}, objective change: {abs(prev_objective - objective):.4f}")
            if torch.equal(self.labels, new_labels) or abs(prev_objective - objective) < self.tol:
                print("Cluster assignments are stable. Stopping iterations.")
                self.labels = new_labels
                break
            self.labels = new_labels
            V = F.one_hot(self.labels, num_classes=self.n_clusters).to(dtype=torch.float32, device=self.device)  # (n_samples, n_clusters) cluster assignment matrix
            V_sum = V.sum(0).unsqueeze(0)  # (1, n_clusters) vector
            prev_objective = objective
            objective = 0.0  # Reset objective for the next iteration

    def predict(self):
        # For kernel k-means, prediction is not straightforward without refitting
        raise NotImplementedError("Prediction for new data is not supported in kernel k-means.")

# Example usage:
if __name__ == "__main__":
    torch.random.manual_seed(42)
    start_time = time.time()
    dataset = Dataset("./data/acoustic", device='cpu')
    end_time = time.time()
    print(f"Dataset loaded in {end_time - start_time:.4f} seconds")
    kernel = RBF(gamma=0.5)
    n_clusters = 10
    model = KmeansMatrix(n_clusters=n_clusters, dataset=dataset, kernel=kernel, max_iter=5, device='cpu')
    start_time = time.time()
    model.fit()
    end_time = time.time()
    print(f"Kernel K-means (matrix form) execution time: {end_time - start_time:.4f} seconds")