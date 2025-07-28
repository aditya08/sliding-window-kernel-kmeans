import torch
import pandas as pd
import numpy as np
import time
class KernelKMeans:

    def __init__(self, n_clusters=3, max_iter=100, kernel='rbf', gamma=None, device='cpu'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma
        self.device = device
        self.X = None
        self.ground_truth = None

    def read_csv(self, filepath, device='cpu'):
        df = pd.read_csv(filepath, header=None)
        self.X = torch.tensor(df.iloc[1:,:-1].values.astype(np.float32), dtype=torch.float32, device=device)
        self.ground_truth = torch.tensor(df.iloc[1:,-1].values.astype(np.int64), dtype=torch.int64, device=device)

    def _kernel_function(self, Y=None):
        if self.kernel == 'linear':
            if Y is None:
                Y = self.X
            return torch.mm(self.X, Y.t())
        elif self.kernel == 'rbf':
            if Y is None:
                Y = self.X
            X_norm = (self.X ** 2).sum(1).view(-1, 1)
            Y_norm = (Y ** 2).sum(1).view(1, -1)
            K = X_norm + Y_norm - 2.0 * torch.mm(self.X, Y.t())
            gamma = self.gamma if self.gamma is not None else 1.0 / self.X.shape[1]
            return torch.exp(-gamma * K)
        else:
            raise ValueError('Unsupported kernel')

    def fit(self):
        n_samples = self.X.shape[0]
        K = self._kernel_function(self.X)
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
        return self

    def fit_matrix_form(self):
        n_samples = self.X.shape[0]
        K = self._kernel_function(self.X)
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