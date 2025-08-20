import time
import torch
import torch.nn.functional as F

from kmeans import Kmeans
from dataset import Dataset
from kernel_functions import KernelFunction

class KmeansStreaming(Kmeans):
    def __init__(self, n_clusters: int, dataset: Dataset, kernel: KernelFunction, max_iter: int = 100, tol: float = 1e-6, block_size: int = 16, device: str = "cpu") -> None:
        """
        Initializes the KmeansBCD class with parameters for clustering using Block Coordinate Descent.
        Args:
            dataset (Dataset): Dataset object containing data and ground truth.
            kernel (KernelFunction): Kernel function to use for clustering.
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            block_size (int): Size of the blocks for BCD optimization.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        super().__init__(n_clusters, dataset, kernel, max_iter=max_iter, tol=tol, device=device)
        self.block_size = block_size

    def fit(self) -> None:
        """
        Fit the KmeansBCD model to the dataset using Block Coordinate Descent.
        This method implements the Kmeans clustering algorithm with BCD optimization.
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for fitting.")
        X = self.dataset.data
        n_samples = self.dataset.shape[0]
        dist = torch.zeros(n_samples, self.n_clusters, device=self.device)
        self.labels = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)
        new_labels = self.labels.clone()
        V = F.one_hot(self.labels, num_classes=self.n_clusters).to(dtype=torch.float32, device=self.device)
        V_sum = V.sum(0).unsqueeze(0)
        K_diag = self.kernel(X, X, diag=True).unsqueeze(1)
        tr_K = K_diag.sum()
        prev_objective = 0.0
        for _ in range(self.max_iter):
            objective = tr_K.item()
            counts = torch.clamp(V_sum, min=1)
            sampled_KV_sum = 0
            sampled_KV_sum0_accum = torch.zeros(1, self.n_clusters, device=self.device)
            for start in range(0, n_samples, self.block_size):
                end = min(start + self.block_size, n_samples)
                KV = (self.kernel(X[start:end], X) @ V) / counts
                sampled_KV = (KV * V[start:end])
                dist[start:end] = K_diag[start:end] - (2 * KV)
                sampled_KV_sum0_accum += sampled_KV.sum(0) / counts
                sampled_KV_sum += sampled_KV.sum().item()
            objective -= sampled_KV_sum
            dist += sampled_KV_sum0_accum
            new_labels = dist.argmin(1)
            print(f"Iteration objective value: {objective:.4f}, objective change: {abs(prev_objective - objective):.4f}")
            if torch.equal(self.labels, new_labels) or abs(prev_objective - objective) < self.tol:
                print("Cluster assignments are stable. Stopping iterations.")
                self.labels = new_labels
                break
            self.labels = new_labels.clone()
            # todo: handle the choice of when to update V and V_sum cleanly (maybe with a flag)
            V = F.one_hot(self.labels, num_classes=self.n_clusters).to(dtype=torch.float32, device=self.device)
            V_sum = V.sum(0).unsqueeze(0)
            prev_objective = objective
    def predict(self) -> None:
        """
        Predict the cluster labels for the dataset.
        This method returns the cluster labels for each data point.
        """
        raise NotImplementedError("Predict method is not implemented in KmeansBCD. Use fit method to obtain labels.")

# Example usage:
if __name__ == "__main__":
    from kernel_functions import RBF
    torch.random.manual_seed(42)
    start_time = time.time()
    dataset = Dataset("./data/acoustic", device='cpu')
    end_time = time.time()
    print(dataset.shape)
    print(f"Dataset loaded in {end_time - start_time:.4f} seconds")
    kernel = RBF(gamma=0.5)
    n_clusters = 10
    block_size = 1024
    print(f"Running Kernel K-means (streaming) with block size {block_size}")
    model = KmeansStreaming(n_clusters=n_clusters, dataset=dataset, kernel=kernel, block_size=block_size, max_iter=10, device='cpu')
    start_time = time.time()
    model.fit()
    end_time = time.time()
    print(f"Kernel K-means (streaming) execution time: {end_time - start_time:.4f} seconds")