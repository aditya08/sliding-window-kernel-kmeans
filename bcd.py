import torch
import time
from dataloader import DataLoader

class KmeansBCD:
    def __init__(self, n_clusters: int =3, max_iter: int =100, filepath: str = None, device: str ='cpu') -> None:
        """
        Initializes the KmeansBCD class with parameters for clustering.
        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for convergence.
            filepath (str): Path to the CSV file containing the dataset.
            device (str): Device to load the tensors onto (default is 'cpu').
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.device = device
        if filepath is not None:
            dataloader = DataLoader(filepath=filepath, device=device)
            (self.X, self.ground_truth) = dataloader.read_csv(filepath, device=device)
        else:
            self.X = None
            self.ground_truth = None