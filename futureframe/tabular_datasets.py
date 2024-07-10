import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from futureframe.utils import cast_to_ndarray


class SupervisedDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        self.X = X
        self.y = cast_to_ndarray(y)

        assert len(self.X) == len(self.y)

    def __getitem__(self, idx):
        x = self.X.iloc[[idx], :]
        y = self.y[idx]
        return x, y

    def __len__(self):
        return len(self.X)

    @classmethod
    def collate_fn(cls, batch):
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
