import torch
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

class Dataset:
    def __init__(self, filepath: str, device: str ="cpu") -> None:
        self.filepath = filepath
        self.device = device
        if filepath.endswith('.csv'):
            (self.data, self.ground_truth) = self.read_csv(filepath, device=device)
        else:
            (self.data, self.ground_truth) = self.read_libsvm(filepath, device=device)
        self.shape = self.data.shape
        self.ndims = len(self.shape)

    def read_libsvm(self, filepath: str, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads a LIBSVM file and loads the data as Torch tensors.
        Args:
            filepath (str): Path to the LIBSVM file.
            device (str): Device to load the tensors onto (default is 'cpu').
        Returns:
            tuple: A tuple containing the data tensor and the ground truth labels tensor.
        """

        # TODO: Need to handle sparse data properly
        # Currently, we load data as a dense torch tensor
        X, y = load_svmlight_file(filepath)
        data = torch.tensor(X.todense(), dtype=torch.float32, device=device)
        ground_truth = torch.tensor(y, dtype=torch.int32, device=device)
        return (data, ground_truth)

    def read_csv(self, filepath: str, device: str ='cpu') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads a CSV file and loads the data as Torch tensors.
        Args:
            filepath (str): Path to the CSV file.
            device (str): Device to load the tensors onto (default is 'cpu').
        Returns:
            tuple: A tuple containing the data tensor and the ground truth labels tensor.
        """
        df = pd.read_csv(filepath, header=None)
        data = torch.tensor(df.iloc[1:,:-1].values.astype(np.float32), dtype=torch.float32, device=device)
        ground_truth = torch.tensor(df.iloc[1:,-1].values.astype(np.int32), dtype=torch.int32, device=device)
        return (data, ground_truth)

if __name__ == "__main__":
    # Example usage
    dataset = Dataset("./data/letter", device="cpu")
    print("Number of dimensions:", dataset.ndims)
    print("Data shape:", dataset.shape)
    print("Ground truth labels:", dataset.ground_truth)