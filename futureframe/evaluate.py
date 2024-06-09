"""Evaluation module."""

from collections import defaultdict
from typing import Literal

import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

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


def eval(
    y_true, y_pred, task_type: Literal["binary_classification", "multiclass_classification", "regression"]
) -> dict:
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


# TODO: remove all of this that comes from cm2
def acc_fn(y, p, num_class=2):
    if num_class == 2:
        y_p = (p >= 0.5).astype(int)
    else:
        y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)


def auc_fn(y, p, num_class=2):
    if num_class > 2:
        return roc_auc_score(y, p, multi_class="ovo")
    else:
        return roc_auc_score(y, p)


def mse_fn(y, p, num_class=None):
    return mean_squared_error(y, p)


def r2_fn(y, p, num_class=None):
    y = y.values
    return r2_score(y, p)


def rae_fn(y_true: np.ndarray, y_pred: np.ndarray, num_class=None):
    y_true = y_true.values
    up = np.abs(y_pred - y_true).sum()
    down = np.abs(y_true.mean() - y_true).sum()
    score = 1 - up / down
    return score


def rmse_fn(y, p, num_class=None):
    return np.sqrt(mean_squared_error(y, p))


def get_eval_metric_fn(eval_metric):
    fn_dict = {
        "acc": acc_fn,
        "auc": auc_fn,
        "mse": mse_fn,
        "r2": r2_fn,
        "rae": rae_fn,
        "rmse": rmse_fn,
        "val_loss": None,
    }
    return fn_dict[eval_metric]


def evaluate(ypred, y_test, metric="auc", num_class=2, seed=123, bootstrap=False):
    np.random.seed(seed)
    eval_fn = get_eval_metric_fn(metric)
    res_list = []
    stats_dict = defaultdict(list)
    if bootstrap:
        for i in range(10):
            sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
            sub_ypred = ypred[sub_idx]
            sub_ytest = y_test.iloc[sub_idx]
            try:
                sub_res = eval_fn(sub_ytest, sub_ypred)
            except ValueError:
                print("evaluation went wrong!")
            stats_dict[metric].append(sub_res)
        for key in stats_dict.keys():
            stats = stats_dict[key]
            alpha = 0.95
            p = ((1 - alpha) / 2) * 100
            lower = max(0, np.percentile(stats, p))
            p = (alpha + ((1.0 - alpha) / 2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print(f"{key} {alpha:.2f} mean/interval {(upper + lower) / 2:.4f}({(upper - lower) / 2:.2f})")
            if key == metric:
                res_list.append((upper + lower) / 2)
    else:
        res = eval_fn(y_test, ypred, num_class)
        res_list.append(res)
    return res_list
