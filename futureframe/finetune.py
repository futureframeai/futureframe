import logging
from abc import ABC

import numpy as np
from torch import Tensor, nn, optim

from futureframe.data_types import TargetType
from futureframe.evaluate import eval_binary_clf, eval_multiclass_clf, eval_regression
from futureframe.utils import cast_to_ndarray, cast_to_tensor

log = logging.getLogger(__name__)

mse_loss = nn.MSELoss(reduction="none")
binary_ce_loss = nn.BCEWithLogitsLoss(reduction="none")
ce_loss = nn.CrossEntropyLoss(reduction="none")


def get_linear_warmup_cos_lr_scheduler(
    optimizer, max_steps, lr, start_factor=0.3, end_factor=0.1, warmup_fraction=0.02
):
    total_warmup_iters = int(warmup_fraction * max_steps)
    total_cosine_iters = int(max_steps * (1 - warmup_fraction))

    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=total_warmup_iters,
    )

    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_cosine_iters,
        eta_min=lr * end_factor,
    )

    lr_scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    return lr_scheduler


class Task(ABC):
    def __init__(self, loss_fn, eval_fn, num_classes):
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.num_classes = num_classes

    def compute_loss(self, y_true: TargetType, y_pred: TargetType):
        if not isinstance(y_pred, Tensor):
            y_pred = cast_to_tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = cast_to_tensor(y_true)

        return self.loss_fn(y_true, y_pred)

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        if not isinstance(y_pred, np.ndarray):
            y_pred = cast_to_ndarray(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = cast_to_ndarray(y_true)

        return self.eval_fn(y_true, y_pred)

    def plots(self, y_true: TargetType, y_pred: TargetType):
        raise NotImplementedError


class BinaryClassification(Task):
    def __init__(self):
        super().__init__(loss_fn=binary_ce_loss, eval_fn=eval_binary_clf, num_classes=2)


class MulticlassClassification(Task):
    def __init__(self, num_classes):
        super().__init__(loss_fn=ce_loss, eval_fn=eval_multiclass_clf, num_classes=num_classes)


class Regression(Task):
    def __init__(self):
        super().__init__(loss_fn=mse_loss, eval_fn=eval_regression, num_classes=1)


# le = LabelEncoder()
# y = le.fit_transform(y)


def get_task(num_classes: int):
    if num_classes == 1:
        return Regression()
    elif num_classes == 2:
        return BinaryClassification()
    elif num_classes > 2:
        return MulticlassClassification(num_classes)
    else:
        raise ValueError("num_classes must be >= 1")
