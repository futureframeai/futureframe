from abc import ABC, abstractmethod

import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn


class Predictor(ABC):
    @abstractmethod
    def finetune(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test) -> list[int] | list[str] | list[float] | np.ndarray:
        pass


mse_loss = nn.MSELoss(reduction="none")
binary_ce_loss = nn.BCEWithLogitsLoss(reduction="none")
ce_loss = nn.CrossEntropyLoss(reduction="none")


def cos_sim_loss(y_pred, y_true):
    # _loss = nn.CosineEmbeddingLoss(reduction="none")
    # return _loss(y_pred, y_true, torch.ones(y_true.shape[0], device=y_true.device))
    return 1 - F.cosine_similarity(y_pred, y_true, dim=1)


def finetuning_loss(y_pred: Tensor, y_true: Tensor, num_classes: int):
    if num_classes == 1:
        return mse_loss(y_pred, y_true)
    elif num_classes == 2:
        return binary_ce_loss(y_pred, y_true)
    elif num_classes > 2:
        return ce_loss(y_pred, y_true)
    else:
        raise ValueError("num_classes must be >= 1")


def finetuning_metrics(y_pred: Tensor, y_true: Tensor, num_classes: int):
    if num_classes == 1:
        return {"mse": mse_loss(y_pred, y_true)}
    elif num_classes == 2:
        return {"bce": binary_ce_loss(y_pred, y_true)}
    elif num_classes > 2:
        acc = (y_pred.argmax(dim=1) == y_true).float().mean()
        return {"ce": ce_loss(y_pred, y_true), "acc": acc}
    else:
        raise ValueError("num_classes must be >= 1")


def get_finetuning_metrics_fn(): ...
