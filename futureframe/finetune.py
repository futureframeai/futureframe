import logging
import os
import time
from abc import ABC
from collections import defaultdict

import numpy as np
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import futureframe as ff
from futureframe.types import TargetType
from futureframe.evaluate import eval_binary_clf, eval_multiclass_clf, eval_regression
from futureframe.features import prepare_target_for_eval
from futureframe.tabular_datasets import FeatureDataset, SupervisedDataset
from futureframe.utils import cast_to_ndarray, cast_to_tensor, seed_all

log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    def __init__(self, loss_fn, eval_fn, num_classes, best_metric, less_is_better=False):
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.num_classes = num_classes
        self.best_metric = best_metric
        self.less_is_better = less_is_better

    def compute_loss(self, y_true: TargetType, y_pred: TargetType):
        if not isinstance(y_pred, Tensor):
            y_pred = cast_to_tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = cast_to_tensor(y_true)

        return self.loss_fn(y_pred, y_true)

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
        super().__init__(
            loss_fn=binary_ce_loss,
            eval_fn=eval_binary_clf,
            num_classes=2,
            best_metric="auc",
            less_is_better=False,
        )


class MulticlassClassification(Task):
    def __init__(self, num_classes):
        super().__init__(
            loss_fn=ce_loss,
            eval_fn=eval_multiclass_clf,
            num_classes=num_classes,
            best_metric="accuracy",
            less_is_better=False,
        )

    def compute_loss(self, y_true: TargetType, y_pred: TargetType):
        if not isinstance(y_pred, Tensor):
            y_pred = cast_to_tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = cast_to_tensor(y_true)

        y_true = y_true.to(dtype=torch.int64).view(-1)
        return self.loss_fn(y_pred, y_true)

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        if not isinstance(y_pred, np.ndarray):
            y_pred = cast_to_ndarray(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = cast_to_ndarray(y_true)

        y_true = y_true.astype(int)
        return self.eval_fn(y_true, y_pred)


class Regression(Task):
    def __init__(self):
        super().__init__(
            loss_fn=mse_loss,
            eval_fn=eval_regression,
            num_classes=1,
            best_metric="mse",
            less_is_better=True,
        )


def get_task(num_classes: int):
    if num_classes == 1:
        return Regression()
    elif num_classes == 2:
        return BinaryClassification()
    elif num_classes > 2:
        return MulticlassClassification(num_classes)
    else:
        raise ValueError("num_classes must be >= 1")


def finetune(
    model,
    X_train,
    y_train,
    num_classes,
    max_steps,
    checkpoints_dir=None,
    num_eval=10,
    patience=None,
    lr=1e-3,
    batch_size=64,
    num_workers=0,
    seed=42,
):
    seed_all(seed)
    device = next(model.parameters()).device
    log.info(f"Using device: {device}")
    task = get_task(num_classes)
    # fit tokenizer
    model.tokenizer(X_train, fit=True)
    y_train = prepare_target_for_eval(y_train, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    train_dataset = SupervisedDataset(X_train, y_train)
    val_dataset = SupervisedDataset(X_val, y_val)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=SupervisedDataset.collate_fn,
    )
    train_terator = iter(train_dataloader)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=SupervisedDataset.collate_fn,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_warmup_cos_lr_scheduler(optimizer, max_steps, lr=lr)
    # criterion = task.loss_fn
    trainable, non_trainable = ff.utils.get_num_parameters(model)
    log.debug(f"{trainable=}, {non_trainable=}")
    history = defaultdict(list)
    pbar = tqdm(range(max_steps))
    eval_freq = max_steps // num_eval
    best_eval_metric = 1e18 if task.less_is_better else -1e18
    if patience is None:
        patience = max_steps
    else:
        patience = max_steps // patience
    patience_counter = 0
    for i in pbar:
        model.train()
        try:
            x, y = next(train_terator)
        except StopIteration:
            train_terator = iter(train_dataloader)
            x, y = next(train_terator)

        assert len(y) > 0, "y is empty."

        t0_global = time.perf_counter()
        log.debug(f"{x=}, {y=}")
        t0 = time.perf_counter()
        x = model.tokenizer(x)
        t1 = time.perf_counter()
        t_tok = t1 - t0
        x = ff.utils.send_to_device_recursively(x.to_dict(), device)
        y = y.to(device)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(x)
        log.debug(f"{logits=}")
        # loss = criterion(logits.squeeze(), y.squeeze()).mean()
        loss = task.compute_loss(y, logits).mean()
        log.debug(f"{loss=}")
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t1 = time.perf_counter()
        t_train = t1 - t0

        history["t/loss"].append(loss.item())

        # validation step
        if i % eval_freq == 0:
            y_pred, y_true = [], []
            model.eval()
            t0 = time.perf_counter()
            y_pred, y_true = [], []
            for j, (x, y) in enumerate(val_dataloader):
                assert len(y) > 0, "y is empty."
                x = model.tokenizer(x)
                x = ff.utils.send_to_device_recursively(x.to_dict(), device)
                y = y.to(device)
                with torch.no_grad():
                    logits = model(x)
                loss = task.compute_loss(y, logits).mean()
                # loss = criterion(logits.squeeze(), y.squeeze()).mean()
                log.debug(f"{loss=}")

                history["v/loss"].append(loss.item())

                y_pred.append(logits)
                y_true.append(y)
            t1 = time.perf_counter()
            t_eval = t1 - t0
            y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
            y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()

            metrics = task.evaluate(y_true, y_pred)
            # TODO: put it to the task class
            best_metric_value = metrics[task.best_metric]
            if task.less_is_better:
                is_best = best_metric_value < best_eval_metric
            else:
                is_best = best_metric_value > best_eval_metric
            if is_best:
                patience_counter = 0
                best_eval_metric = best_metric_value
                if checkpoints_dir is not None:
                    path = os.path.join(checkpoints_dir, "best_model.pth")
                    torch.save(model.state_dict(), path)
                    log.info(f"Saved best model to {path}.")

            for k in metrics:
                history[f"v/{k}"].append(metrics[k])

            pretty_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
            pretty_metrics = ", ".join([f"{k}={v}" for k, v in pretty_metrics.items()])
            print(
                f"Val. took {t_eval:.2f}s: {loss.item()=}, {pretty_metrics}, Best {task.best_metric}: {best_eval_metric:4f}."
            )

        t1_global = time.perf_counter()
        t_global = t1_global - t0_global
        latest_history = {k: v[-1] for k, v in history.items()}
        pbar.set_postfix(
            **latest_history,
        )  # t_tok=t_tok, t_train=t_train, t_global=t_global)
        # pbar.update(1)

        patience_counter += 1
        if patience_counter >= patience:
            log.info(f"Early stopping at step {i}.")
            break

        if i >= max_steps:
            break

    return model, history


@torch.no_grad()
def predict(model, X_test, batch_size=64, num_workers=0):
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
        x = ff.utils.send_to_device_recursively(x.to_dict(), device)
        log.debug(f"{x=}")
        logits = model(x)
        log.debug(f"{logits=}")

        y_pred.append(logits.cpu())
    y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()

    return y_pred