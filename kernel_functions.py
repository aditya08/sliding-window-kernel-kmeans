import time
import torch
from typing import Optional

from dataset import Dataset

class KernelFunction:
    def __init__(self, kernel_type: str = 'rbf'):
        """
        Initializes the KernelFunction class with parameters for kernel computation.
        Args:
            kernel_type (str): Type of kernel ('linear', 'rbf', 'tanh').
            gamma (float, optional): Parameter for RBF kernel. If None, it is set to 1/n_features.
        """
        self.kernel_type = kernel_type

    def __call__(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: empty torch tensor.
        """
        return torch.empty(0)  # Placeholder for kernel computation

class RBF(KernelFunction):
    def __init__(self, gamma):
        super().__init__('rbf')
        self.gamma = gamma

    def __call__(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the RBF kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: RBF kernel matrix.
        """
        X = dataset.data
        sq_dists = torch.cdist(X, X) ** 2
        return torch.exp(-self.gamma * sq_dists)

class Linear(KernelFunction):
    def __init__(self):
        super().__init__('linear')

    def __call__(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the linear kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: Linear kernel matrix.
        """
        X = dataset.data
        return X.dot(X.t())

class Tanh(KernelFunction):
    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__('tanh')
        self.alpha = alpha
        self.beta = beta

    def __call__(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the tanh kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: Tanh kernel matrix.
        """
        X = dataset.data
        return torch.tanh(self.alpha * X.dot(X.t()) + self.beta)

class Polynomial(KernelFunction):
    def __init__(self, degree=2, coef0=1.0):
        super().__init__('polynomial')
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the polynomial kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: Polynomial kernel matrix.
        """
        X = dataset.data
        return (X.dot(X.t()) + self.coef0) ** self.degree