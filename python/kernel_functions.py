import torch
from typing import Optional
from abc import ABC, abstractmethod

from dataset import Dataset

class KernelFunction(ABC):
    def __init__(self, kernel_type: str = 'rbf'):
        """
        Initializes the KernelFunction class with parameters for kernel computation.
        Args:
            kernel_type (str): Type of kernel ('linear', 'rbf', 'tanh').
        """
        self.kernel_type = kernel_type

    @abstractmethod
    def __call__(self, X: torch.Tensor, Y: torch.Tensor, diag: Optional[bool] = False) -> torch.Tensor:
        """
        Computes the kernel matrix for the given dataset.
        Args:
            dataset (Dataset): Dataset object containing data.
        Returns:
            torch.Tensor: empty torch tensor.
        """
        pass  # Implementation of kernel computation is deferred.

class RBF(KernelFunction):
    def __init__(self, gamma):
        super().__init__('rbf')
        self.gamma = gamma

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, diag: Optional[bool] = False) -> torch.Tensor:
        """
        Computes the RBF kernel matrix between two tensors.
        Args:
            X (torch.Tensor): First tensor.
            Y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: RBF kernel matrix between X and Y.
        """
        if diag:
            return torch.exp(-self.gamma * torch.norm(X - Y, dim=1) ** 2)
        sq_dists = torch.cdist(X, Y) ** 2
        return torch.exp(-self.gamma * sq_dists)

    def __repr__(self):
        return f"RBF(gamma={self.gamma})"

class Linear(KernelFunction):
    def __init__(self):
        super().__init__('linear')

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, diag: Optional[bool] = False) -> torch.Tensor:
        """
        Computes the linear kernel matrix for the given dataset.
        Args:
            X (torch.Tensor): First tensor.
            Y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Linear kernel matrix.
        """
        if diag:
            return (X * Y).sum(dim=1)
        return X @ (Y.t())

class Tanh(KernelFunction):
    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__('tanh')
        self.alpha = alpha
        self.beta = beta

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, diag: Optional[bool] = False) -> torch.Tensor:
        """
        Computes the tanh kernel matrix between two tensors.
        Args:
            X (torch.Tensor): First tensor.
            Y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Tanh kernel matrix.
        """
        if diag:
            return torch.tanh(self.alpha * (X * Y).sum(dim=1) + self.beta)
        return torch.tanh(self.alpha * X.dot(Y.t()) + self.beta)

class Polynomial(KernelFunction):
    def __init__(self, degree=2, coef0=1.0):
        super().__init__('polynomial')
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, diag: Optional[bool] = False) -> torch.Tensor:
        """
        Computes the polynomial kernel matrix for the given dataset.
        Args:
            X (torch.Tensor): First tensor.
            Y (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Polynomial kernel matrix.
        """
        if diag:
            return (X * Y).sum(dim=1) + self.coef0 ** self.degree
        return (X.dot(Y.t()) + self.coef0) ** self.degree