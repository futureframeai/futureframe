import logging
from abc import ABC
from typing import Callable

import torch.utils.data
from torch import Tensor, binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy, mse_loss

from futureframe.evaluation import eval_binary_classification, eval_multiclass_clf, eval_regression
from futureframe.types import TargetType
from futureframe.utils import cast_to_ndarray, cast_to_tensor

log = logging.getLogger(__name__)


class Task(ABC):
    """
    Base class for defining tasks in FutureFrame.

    Args:
        name (str): The name of the task.
        loss_fn (callable): The loss function used for training the task.
        eval_fn (callable): The evaluation function used for evaluating the task.
        num_classes (int): The number of classes in the task.
        best_metric (str): The metric used to determine the best model during training.
        less_is_better (bool, optional): Whether a lower value of the best metric is better.
    """

    def __init__(
        self, name: str, loss_fn: Callable, eval_fn: Callable, num_classes: int, best_metric: str, less_is_better=False
    ):
        """
        Initialize a Task object.

        Args:
            name (str): The name of the task.
            loss_fn (callable): The loss function for the task.
            eval_fn (callable): The evaluation function for the task.
            num_classes (int): The number of classes for classification tasks.
            best_metric (str): The metric used to determine the best model.
            less_is_better (bool, optional): Whether a lower value of the best metric is better.
        """
        self.name = name
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.num_classes = num_classes
        self.best_metric = best_metric
        self.less_is_better = less_is_better

    def compute_loss(self, y_true: TargetType, y_pred: TargetType) -> Tensor:
        """
        Compute the loss for the task.

        Args:
            y_true (TargetType): The true labels.
            y_pred (TargetType): The predicted labels.

        Returns:
            The computed loss value.
        """
        y_pred = cast_to_tensor(y_pred)
        y_true = cast_to_tensor(y_true)

        return self.loss_fn(y_pred, y_true)

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        """
        Evaluate the task.

        Args:
            y_true (TargetType): The true labels.
            y_pred (TargetType): The predicted labels.

        Returns:
            The evaluation result.
        """
        y_pred = cast_to_ndarray(y_pred)
        y_true = cast_to_ndarray(y_true)

        return self.eval_fn(y_true, y_pred)

    def plots(self, y_true: TargetType, y_pred: TargetType) -> None:
        """
        Generate plots for the task.

        Args:
            y_true (TargetType): The true labels.
            y_pred (TargetType): The predicted labels.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.
        """
        raise NotImplementedError


class BinaryClassification(Task):
    """
    A class representing a binary classification task.

    Inherits from the Task class and provides specific configuration for binary classification tasks.
    """

    def __init__(self):
        super().__init__(
            name="classification",
            loss_fn=binary_cross_entropy_with_logits,
            eval_fn=eval_binary_classification,
            num_classes=2,
            best_metric="auc",
            less_is_better=False,
        )


class Regression(Task):
    """
    A class representing a regression task.

    Inherits from the Task class and provides specific configuration for regression tasks.
    """

    def __init__(self):
        super().__init__(
            name="regression",
            loss_fn=mse_loss,
            eval_fn=eval_regression,
            num_classes=1,
            best_metric="mse",
            less_is_better=True,
        )


class MulticlassClassification(Task):
    """
    A task for multiclass classification.

    Args:
        num_classes (int): The number of classes in the classification problem.
    """

    def __init__(self, num_classes):
        super().__init__(
            name="multiclass_classification",
            loss_fn=cross_entropy,
            eval_fn=eval_multiclass_clf,
            num_classes=num_classes,
            best_metric="accuracy",
            less_is_better=False,
        )

    def compute_loss(self, y_true: TargetType, y_pred: TargetType):
        """
        Computes the loss given the true labels and predicted labels.

        Args:
            y_true (TargetType): The true labels.
            y_pred (TargetType): The predicted labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        y_pred = cast_to_tensor(y_pred)
        y_true = cast_to_tensor(y_true)

        y_true = y_true.to(dtype=torch.int64).view(-1)
        return self.loss_fn(y_pred, y_true)

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        """
        Evaluates the predictions given the true labels.

        Args:
            y_true (TargetType): The true labels.
            y_pred (TargetType): The predicted labels.

        Returns:
            float: The evaluation result.
        """
        y_pred = cast_to_ndarray(y_pred)
        y_true = cast_to_ndarray(y_true)

        y_true = y_true.astype(int)
        return self.eval_fn(y_true, y_pred)


def create_task(num_classes: int) -> Task:
    """
    Create a task based on the number of classes.

    Args:
        num_classes (int): The number of classes for the task.

    Returns:
        Task: An instance of the appropriate task class based on the number of classes.

    Raises:
        ValueError: If num_classes is less than 1.
    """
    if num_classes == 1:
        return Regression()
    elif num_classes == 2:
        return BinaryClassification()
    elif num_classes > 2:
        return MulticlassClassification(num_classes)
    else:
        raise ValueError("num_classes must be >= 1")


task_type_by_num_classes = {
    2: "classification",
    1: "regression",
}


def get_task_type(num_classes: int) -> str:
    return task_type_by_num_classes.get(num_classes, "classification")