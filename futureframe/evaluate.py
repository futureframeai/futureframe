"""Evaluation module."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

log = logging.getLogger(__name__)


METRICS = ["accuracy", "auc", "f1", "precision", "recall", "mse", "mae", "r2"]


def eval_binary_clf(y_true: np.ndarray, y_pred: np.ndarray):
    log.debug(f"{y_true=}")
    log.debug(f"{y_pred=}")
    y_pred_hard = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_hard)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_hard)
    precision = precision_score(y_true, y_pred_hard)
    recall = recall_score(y_true, y_pred_hard)
    return dict(accuracy=acc, auc=auc, f1=f1, precision=precision, recall=recall)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(mse=mse, mae=mae, r2=r2)


def eval_multiclass_clf(y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    return dict(accuracy=acc, f1=f1, precision=precision, recall=recall)


def eval(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, non_none_only: bool = False):
    results = {k: None for k in METRICS}
    res = {}
    if num_classes == 2:
        res = eval_binary_clf(y_true, y_pred)
    elif num_classes > 2:
        res = eval_multiclass_clf(y_true, y_pred)
    elif num_classes == 1:
        res = eval_regression(y_true, y_pred)
    else:
        raise ValueError("num_classes must be >= 1")

    for k, v in res.items():
        results[k] = v

    if non_none_only:
        results = {k: v for k, v in results.items() if v is not None}

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
        results = eval(y_true_sample, y_pred_sample, num_classes, non_none_only=True)

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
