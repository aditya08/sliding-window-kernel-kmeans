import torch
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath: str, device: str ='cpu') -> None:
        self.filepath = filepath
        self.device = device
        (self.X, self.ground_truth) = self.read_csv(filepath, device=device)

    def read_csv(self, filepath: str, device: str ='cpu') -> None:
        """
        Reads a CSV file and loads the data as Torch tensors.
        Args:
            filepath (str): Path to the CSV file.
            device (str): Device to load the tensors onto (default is 'cpu').
        Returns:
            tuple: A tuple containing the data tensor and the ground truth labels tensor.
        """
        df = pd.read_csv(filepath, header=None)
        self.X = torch.tensor(df.iloc[1:,:-1].values.astype(np.float32), dtype=torch.float32, device=device)
        self.ground_truth = torch.tensor(df.iloc[1:,-1].values.astype(np.int32), dtype=torch.int32, device=device)
        return (self.X, self.ground_truth)
