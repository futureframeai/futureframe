import logging

import pandas as pd
import torch.utils.data
from torch import Tensor, nn
from torch.utils.data import DataLoader

from futureframe.data.tabular_datasets import FeatureDataset

log = logging.getLogger(__name__)


@torch.no_grad()
def predict(model: nn.Module, X_test: pd.DataFrame, batch_size: int = 64, num_workers=0) -> Tensor:
    """
    Generates predictions for the given test data using the specified model.

    Parameters:
        model (torch.nn.Module): The trained model to be used for predictions.
        X_test (list): The input test data.
        batch_size (int, optional): The batch size for the DataLoader. Default is 64.
        num_workers (int, optional): The number of worker threads for data loading. Default is 0.

    Returns:
        numpy.ndarray: The predicted values.
    """
    val_dataset = FeatureDataset(X_test)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=FeatureDataset.collate_fn,
    )

    y_pred = []
    model.eval()
    for _, x in enumerate(val_dataloader):
        log.debug(f"{x=}")
        logits = model(x)
        log.debug(f"{logits=}")

        y_pred.append(logits.cpu())

    y_pred = torch.cat(y_pred, dim=0).squeeze()

    return y_pred