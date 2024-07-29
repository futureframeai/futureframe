import logging
import os
import time
from collections import defaultdict
from typing import Optional

import torch.utils.data
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

import pandas as pd

from futureframe.data.encoding import BaseFeaturesToModelInput
from futureframe.data.features import prepare_target_for_eval
from futureframe.data.tabular_datasets import SupervisedDataset
from futureframe.optim import get_linear_warmup_cos_lr_scheduler
from futureframe.tasks import create_task
from futureframe.utils import get_num_parameters, seed_all, send_to_device_recursively

log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def finetune(
    model: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_classes: int,
    max_steps: int,
    checkpoints_dir: str,
    input_encoder: Optional[BaseFeaturesToModelInput] = None,
    num_eval: int = 10,
    patience: Optional[int] = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 8,
    val_size: float = 0.05,
    seed: int = 42,
):
    """
    Fine-tunes a model on the given training data.

    Parameters:
        model (torch.nn.Module): The model to be fine-tuned.
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target labels for the training data.
        num_classes (int): The number of output classes.
        max_steps (int): The maximum number of training steps.
        checkpoints_dir (str, optional): Directory to save the best model checkpoints.
        input_encoder (BaseFeaturesToModelInput, optional): The model input encoder.
        num_eval (int, optional): Number of evaluations during training.
        patience (int, optional): Number of evaluations to wait for improvement before early stopping.
        lr (float, optional): Learning rate for the optimizer.
        batch_size (int, optional): Batch size for data loading.
        num_workers (int, optional): Number of worker threads for data loading.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: The fine-tuned model and the training history.
    """
    seed_all(seed)
    device = next(model.parameters()).device
    log.info(f"Using device: {device}")
    task = create_task(num_classes)

    # fit tokenizer
    if input_encoder is not None:
        input_encoder.fit(X_train)

    y_train = prepare_target_for_eval(y_train, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
    train_dataset = SupervisedDataset(X_train, y_train)
    val_dataset = SupervisedDataset(X_val, y_val)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=SupervisedDataset.collate_fn,
        pin_memory=True,
        prefetch_factor=num_workers // 2,
    )
    train_terator = iter(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=SupervisedDataset.collate_fn,
        pin_memory=True,
        prefetch_factor=num_workers // 2,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_warmup_cos_lr_scheduler(optimizer, max_steps, lr=lr)

    trainable, non_trainable = get_num_parameters(model)
    log.debug(f"{trainable=}, {non_trainable=}")

    history = defaultdict(list)
    pbar = tqdm(range(max_steps))
    eval_freq = max_steps // num_eval
    best_eval_metric = 1e18 if task.less_is_better else -1e18
    if patience is None:
        patience = max_steps
    patience_steps = patience * eval_freq
    patience_counter = 0
    best_model_state = model.state_dict()

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
        if input_encoder is not None:
            x = input_encoder.encode(x)
        x = send_to_device_recursively(x, device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(**x)
        log.debug(f"{logits=}")
        loss = task.compute_loss(y, logits).mean()
        log.debug(f"{loss=}")
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t1 = time.perf_counter()
        t_train = t1 - t0

        history["t/loss"].append(loss.item())

        # validation step TODO: replace with prdict function
        if i % eval_freq == 0:
            y_pred, y_true = [], []
            model.eval()
            t0 = time.perf_counter()
            y_pred, y_true = [], []
            for j, (x, y) in enumerate(val_dataloader):
                assert len(y) > 0, "y is empty."
                if input_encoder is not None:
                    x = input_encoder.encode(x)
                x = send_to_device_recursively(x, device)
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
                best_model_state = model.state_dict()
                if checkpoints_dir is not None:
                    path = os.path.join(checkpoints_dir, "best_model.pth")
                    torch.save(best_model_state, path)
                    log.info(f"Saved best model to {path}.")

            for k in metrics:
                history[f"v/{k}"].append(metrics[k])
            history[f"best/{task.best_metric}"].append(best_eval_metric)
            history["pc"].append(patience_counter)

        t1_global = time.perf_counter()
        t_global = t1_global - t0_global
        latest_history = {k: v[-1] for k, v in history.items()}
        pbar.set_postfix(**latest_history)

        patience_counter += 1
        if patience_counter >= patience_steps:
            log.info(f"Early stopping at step {i}.")
            break

        if i >= max_steps:
            break

    model.load_state_dict(best_model_state)

    return model, history
