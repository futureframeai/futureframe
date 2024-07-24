import logging

import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

from futureframe.data.tabular_datasets import FeatureDataset
from futureframe.utils import send_to_device_recursively

log = logging.getLogger(__name__)


@torch.no_grad()
def predict(model: nn.Module, X_test: pd.DataFrame, batch_size: int = 64, num_workers=0):
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
    device = next(model.parameters()).device
    # assert model tokenizer is fit
    assert model.tokenizer.is_fit

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
        x = model.tokenizer(x)
        x = send_to_device_recursively(x.to_dict(), device)
        log.debug(f"{x=}")
        logits = model(x)
        log.debug(f"{logits=}")

        y_pred.append(logits.cpu())

    y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()

    return y_pred