from typing import Optional
import pandas as pd
from torch import nn

from futureframe.finetuning import finetune

from futureframe.inference import predict


class BaseModelForFinetuning(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_encoder = None

    def finetune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        num_classes: int,
        max_steps: int,
        checkpoints_dir: str,
        num_eval: int = 10,
        patience: Optional[int] = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 8,
        val_size: float = 0.05,
        seed: int = 42,
    ):
        return finetune(
            model=self,
            input_encoder=self.input_encoder,
            X_train=X_train,
            y_train=y_train,
            num_classes=num_classes,
            max_steps=max_steps,
            checkpoints_dir=checkpoints_dir,
            num_eval=num_eval,
            patience=patience,
            lr=lr,
            batch_size=batch_size,
            num_workers=num_workers,
            val_size=val_size,
            seed=seed,
        )

    def predict(
        self,
        X_test: pd.DataFrame,
        batch_size: int = 64,
        num_workers=0,
    ):
        return predict(
            model=self,
            X_test=X_test,
            batch_size=batch_size,
            num_workers=num_workers,
        )
