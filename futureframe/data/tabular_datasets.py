import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from futureframe.utils import cast_to_ndarray


class SupervisedDataset(Dataset):
    """
    A PyTorch Dataset class for supervised learning with tabular data.

    Attributes:
        X (pd.DataFrame): The input features of the dataset.
        y (np.ndarray): The target labels of the dataset, converted to a numpy array.

    Methods:
        __init__(X: pd.DataFrame, y: pd.DataFrame | pd.Series):
            Initializes the dataset with input features and target labels.
        __getitem__(idx):
            Retrieves the feature and target label at the specified index.
        __len__():
            Returns the total number of samples in the dataset.
        collate_fn(batch):
            A custom collate function for combining a batch of data samples into a single batch.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        """
        Initializes the dataset with input features and target labels.

        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.DataFrame or pd.Series): The target labels, which will be cast to a numpy array.
        """
        self.X = X
        self.y = cast_to_ndarray(y)

        assert len(self.X) == len(self.y), "The number of samples in X and y must be equal."

    def __getitem__(self, idx):
        """
        Retrieves the feature and target label at the specified index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature (as a DataFrame) and the target label (as a numpy array) at the specified index.
        """
        x = self.X.iloc[[idx], :]
        y = self.y[idx]
        return x, y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.X)

    @classmethod
    def collate_fn(cls, batch):
        """
        A custom collate function for combining a batch of data samples into a single batch.

        Parameters:
            batch (list): A list of tuples, where each tuple contains a feature (DataFrame) and a target label (numpy array).

        Returns:
            tuple: A tuple containing a concatenated DataFrame of features and a tensor of target labels.
        """
        X = pd.concat([x[0] for x in batch])
        y = torch.from_numpy(np.array([x[1] for x in batch])).view(-1, 1).float()
        return X, y


class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract features and target from the DataFrame
        row = self.df.iloc[[idx], :]
        if self.tokenizer:
            row = self.tokenizer(row)
        return row

    @classmethod
    def collate_fn(cls, batch):
        X = pd.concat(batch)
        return X


class KaggleDataset(SupervisedDataset):
    def __init__(self, dataset_link="") -> None:
        X = ...
        y = ...
        super().__init__(X=X, y=y)