"""
Whether you're working with regression, binary classification, or multiclass classification, \
this module is designed to offer flexibility in evaluating your model's predictions against the true outcomes.
"""

import logging

import numpy as np
import torchmetrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from tqdm import tqdm

from futureframe.data.features import prepare_target_for_eval

log = logging.getLogger(__name__)


METRICS = ["accuracy", "auc", "f1", "precision", "recall", "mse", "mae", "r2", "ap"]


def _accuracy_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the accuracy score.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The accuracy score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.accuracy(y_pred, y_true)
    return accuracy_score(y_true, y_pred)


def _auc_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the area under the ROC curve.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The AUC score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.auroc(y_pred, y_true)
    return roc_auc_score(y_true, y_pred)


def _f1_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the F1 score.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The F1 score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.f1(y_pred, y_true)
    return f1_score(y_true, y_pred)


def _mse_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the mean squared error.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The mean squared error.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.mean_squared_error(y_pred, y_true)
    return mean_squared_error(y_true, y_pred)


def _mae_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the mean absolute error.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The mean absolute error.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.mean_absolute_error(y_pred, y_true)
    return mean_absolute_error(y_true, y_pred)


def _r2_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the coefficient of determination (R^2).

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The R^2 score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.r2score(y_pred, y_true)
    return r2_score(y_true, y_pred)


def _precision_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the precision score.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The precision score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.precision(y_pred, y_true)
    return precision_score(y_true, y_pred)


def _recall_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the recall score.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The recall score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.recall(y_pred, y_true)
    return recall_score(y_true, y_pred)


def _ap_score(y_true: np.ndarray | Tensor, y_pred: np.ndarray | Tensor) -> float | Tensor:
    """
    Compute the average precision score.

    Args:
        y_true (np.ndarray | Tensor): The true labels.
        y_pred (np.ndarray | Tensor): The predicted labels.

    Returns:
        float: The average precision score.
    """
    if isinstance(y_true, Tensor):
        return torchmetrics.functional.average_precision(y_pred, y_true)
    return average_precision_score(y_true, y_pred)


def eval_binary_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_is_probability: bool = False, threshold: float | None = None
) -> dict[str, float]:
    """
    Evaluate the performance of a binary classification model.

    Parameters:
        y_true (np.ndarray): The true labels of the binary classification problem.
        y_pred (np.ndarray): The predicted labels or logits of the binary classification problem.
        y_pred_is_probability (bool, optional): Whether the predicted values are probabilities or logits.
        threshold (float | None, optional): The threshold value for converting probabilities to binary labels.
            If None, the default threshold is used.

    Returns:
        dict: A dictionary containing the evaluation metrics:

            - accuracy: The accuracy of the model.
            - auc: The area under the ROC curve.
            - f1: The F1 score.
            - precision: The precision score.
            - recall: The recall score.
            - ap: The average precision score.
    """
    if not y_pred_is_probability:  # then y_pred is logits
        threshold = 0.0 if threshold is None else threshold
    else:
        threshold = 0.5 if threshold is None else threshold

    y_pred_hard = (y_pred >= 0).astype(int)

    acc = accuracy_score(y_true, y_pred_hard)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_hard)
    precision = precision_score(y_true, y_pred_hard)
    recall = recall_score(y_true, y_pred_hard)
    ap = average_precision_score(y_true, y_pred)

    return dict(accuracy=acc, auc=auc, f1=f1, precision=precision, recall=recall, ap=ap)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Evaluate the performance of a regression model by calculating various metrics.

    Parameters:
        y_true (np.ndarray): The true values of the target variable.
        y_pred (np.ndarray): The predicted values of the target variable.

    Returns:
        dict: A dictionary containing the calculated metrics:

            - mse (float): The mean squared error.
            - mae (float): The mean absolute error.
            - r2 (float): The coefficient of determination ($R^2$).
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(mse=mse, mae=mae, r2=r2)


def eval_multiclass_clf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Evaluate the performance of a multiclass classification model.

    Parameters:
    - y_true (np.ndarray): True labels of the samples.
    - y_pred (np.ndarray): Predicted labels of the samples.

    Returns:
        dict: A dictionary containing the evaluation metrics:

            - accuracy: Accuracy score.
            - f1: F1 score.
            - precision: Precision score.
            - recall: Recall score.
    """
    y_pred_hard = np.argmax(y_pred, axis=1).reshape(-1)
    acc = accuracy_score(y_true, y_pred_hard)
    f1 = f1_score(y_true, y_pred_hard, average="macro")
    precision = precision_score(y_true, y_pred_hard, average="macro")
    recall = recall_score(y_true, y_pred_hard, average="macro")
    return dict(accuracy=acc, f1=f1, precision=precision, recall=recall)


def eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_non_none_metrics_only: bool = False,
    y_pred_is_probability: bool = False,
    num_classes: int | None = None,
) -> dict[str, float]:
    """
    Evaluate the performance of a model's predictions.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        return_non_none_metrics_only (bool, optional): Whether to return only the metrics with non-None values.
        y_pred_is_probability (bool, optional): Whether the predicted labels are probabilities.
        num_classes (int | None, optional): The number of classes.

    Returns:
        dict: A dictionary containing the evaluation metrics.

    Raises:
        ValueError: If num_classes is less than 1.
    """

    def determine_num_classes(y_pred: np.ndarray, num_classes: int | None) -> int:
        if num_classes is not None:
            return num_classes
        if y_pred.ndim > 1:
            return y_pred.shape[1]
        return 1

    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
        if num_classes == 1:
            y_pred = y_pred.reshape(-1)
            return eval_regression(y_true, y_pred)
        elif num_classes == 2:
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1].reshape(-1)
            return eval_binary_classification(y_true, y_pred, y_pred_is_probability=y_pred_is_probability)
        elif num_classes > 2:
            return eval_multiclass_clf(y_true, y_pred)
        else:
            raise ValueError("num_classes must be >= 1")

    def filter_none_metrics(results: dict) -> dict:
        return {k: v for k, v in results.items() if v is not None}

    num_classes = determine_num_classes(y_pred, num_classes)
    y_true = prepare_target_for_eval(y_true, num_classes)
    results = {k: None for k in METRICS}

    res = evaluate(y_true, y_pred, num_classes)
    results.update(res)

    if return_non_none_metrics_only:
        results = filter_none_metrics(results)

    return results


def bootstrap_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_iterations: int = 10,
    ci=95.0,
    verbose=False,
    non_none_only=False,
):
    bootstrap_results = {metric: [] for metric in METRICS}

    pbar = tqdm(range(n_iterations), disable=not verbose)
    for _ in pbar:
        # Sample with replacement
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        # Evaluate on the sample
        results = eval(y_true_sample, y_pred_sample, return_non_none_metrics_only=True)

        # Collect the results
        for metric in results:
            bootstrap_results[metric].append(results[metric])

    # Compute the 95% confidence intervals
    ci_results = {}
    lower_p = (100 - ci) / 2
    upper_p = ci + lower_p
    for metric, values in bootstrap_results.items():
        if len(values) == 0:
            if non_none_only:
                continue
            ci_results[metric] = None
            continue
        lower = np.percentile(values, lower_p)
        upper = np.percentile(values, upper_p)
        ci_results[metric] = {
            "mean": np.mean(values),
            "lower": lower,
            "upper": upper,
            "median": np.median(values),
            "std": np.std(values),
            "values": values,
            "ci": f"{ci}%",
            "n_iterations": n_iterations,
        }

    return ci_results
