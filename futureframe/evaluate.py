"""Evaluation module."""

from typing import Literal
import sklearn.metrics

from futureframe.types import Tasks


def eval_binary_clf(y_true, y_pred):
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    return dict(accuracy=acc, auc=auc, f1=f1, precision=precision, recall=recall)


def eval_regression(y_true, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    return dict(mse=mse, mae=mae, r2=r2)


def eval_multiclass_clf(y_true, y_pred):
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    return dict(accuracy=acc, f1=f1, precision=precision, recall=recall)


def eval(y_true, y_pred, task_type: Literal["binary_classification", "multiclass_classification", "regression"]):
    if task_type == Tasks.BINARY_CLASSIFICATION.value:
        results = eval_binary_clf(y_true, y_pred)
    elif task_type == Tasks.MULTICLASS_CLASSIFICATION.value:
        results = eval_multiclass_clf(y_true, y_pred)
    elif task_type == Tasks.REGRESSION.value:
        results = eval_regression(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    all_keys = ["accuracy", "auc", "f1", "precision", "recall", "mse", "mae", "r2"]
    for key in all_keys:
        if key not in results:
            results[key] = None
    return results