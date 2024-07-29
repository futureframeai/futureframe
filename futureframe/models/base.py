from typing import Optional

import pandas as pd
from torch import nn

from futureframe.finetuning import finetune
from futureframe.inference import predict


class BaseModelForFinetuning(nn.Module):
    """
    Base class for finetuning models.

    Args:
        input_encoder: The input encoder for the model.
    """

    def __init__(self, input_encoder):
        super().__init__()
        self.input_encoder = input_encoder

    def finetune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        num_classes: Optional[int] = None,
        max_steps: int = 1000,
        checkpoints_dir: Optional[str] = None,
        num_eval: int = 10,
        patience: Optional[int] = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 8,
        val_size: float = 0.05,
        seed: int = 42,
    ):
        """
        Finetunes the model on the given training data.

        Args:
            X_train: The input training data.
            y_train: The target training data.
            num_classes: The number of classes in the target data.
            max_steps: The maximum number of training steps.
            checkpoints_dir: The directory to save model checkpoints.
            num_eval: The number of evaluation steps.
            patience: The number of epochs to wait for improvement in validation loss.
            lr: The learning rate for the optimizer.
            batch_size: The batch size for training.
            num_workers: The number of workers for data loading.
            val_size: The proportion of validation data.
            seed: The random seed for reproducibility.

        Returns:
            The trained model.
        """
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
        """
        Generates predictions for the given test data.

        Args:
            X_test: The input test data.
            batch_size: The batch size for prediction.
            num_workers: The number of workers for data loading.

        Returns:
            The predicted values.
        """
        return predict(
            model=self,
            X_test=X_test,
            batch_size=batch_size,
            num_workers=num_workers,
        )
