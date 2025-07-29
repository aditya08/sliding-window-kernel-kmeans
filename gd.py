import torch
import time
from typing import Optional

from kmeans import Kmeans
from dataset import Dataset
from kernel_functions import KernelFunction

class KmeansGD(Kmeans):

    def __init__(self, n_clusters: int, dataset: Dataset, kernel: KernelFunction, max_iter: int = 100, device='cpu') -> None:
        """
        Initializes the KernelKMeans class with parameters for clustering.
        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            dataset (Dataset): Dataset object containing data and ground truth.
            kernel (KernelFunction): Kernel function to use for clustering.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        super().__init__(n_clusters, dataset, kernel, max_iter=max_iter, device=device)

    def fit(self):
        K = self.kernel(self.dataset)
        n_samples = self.dataset.shape[0]
        labels = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)
        for _ in range(self.max_iter):
            dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
            for j in range(self.n_clusters):
                mask = (labels == j)
                count = mask.sum()
                if count == 0:
                    dist[:, j] = float('inf')
                    continue
                K_j = K[mask][:, mask]
                sum_K_j = K[mask].sum(0)
                dist[:, j] = K.diag() - 2 * sum_K_j / count + K_j.sum() / (count ** 2)
            new_labels = dist.argmin(1)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
        self.labels_ = labels.cpu().numpy()

    def __fit_matrix_form(self):
        K = self.kernel(self.dataset)
        n_samples = self.dataset.shape[0]
        colidx = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)
        vals = torch.ones(n_samples, device=self.device)
        rowidx = torch.arange(self.n_clusters, device=self.device)
        V = torch.sparse_coo_tensor(rowidx.unsqueeze(0), colidx, (self.n_clusters, n_samples), device=self.device)
        for _ in range(self.max_iter):
            dist = torch.zeros((n_samples, self.n_clusters), device=self.device)
            for j in range(self.n_clusters):
                mask = (labels == j)
                count = mask.sum()
                if count == 0:
                    dist[:, j] = float('inf')
                    continue
                K_j = K[mask][:, mask]
                sum_K_j = K[mask].sum(0)
                dist[:, j] = K.diag() - 2 * sum_K_j / count + K_j.sum() / (count ** 2)
            new_labels = dist.argmin(1)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
        self.labels_ = labels.cpu().numpy()
        return self

    def predict(self):
        # For kernel k-means, prediction is not straightforward without refitting
        raise NotImplementedError("Prediction for new data is not supported in kernel k-means.")

# Example usage:
if __name__ == "__main__":
    # Dummy data
    X = torch.randn(100, 2)
    model = KernelKMeans(n_clusters=3, kernel='linear', gamma=0.5)
    model.read_csv('N1797_D64_digits-sklearn.csv', device='cpu')

    start_time = time.time()
    model.fit()
    end_time = time.time()
    print(f"Kernel K-means execution time: {end_time - start_time:.4f} seconds")