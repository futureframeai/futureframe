"""Predictor module."""

# from futureframe.types import Features
import numpy as np
import pandas as pd

from futureframe.models.base import Predictor
from futureframe.registry import create_predictor


def predict(
    X_test: pd.DataFrame | pd.Series | list[str] | list[int] | list[float] | np.ndarray,
    X_train: pd.DataFrame | pd.Series | list[str] | list[int] | list[float] | np.ndarray,
    y_train: pd.DataFrame | pd.Series | list[str] | list[int] | list[float] | np.ndarray,
    predictor: Predictor | str = "tabtext_xgb",
    **kwargs,
):
    """Predict on a DataFrame of features.

    Args:
        x: DataFrame of testing features.
        X: DataFrame of training features and targets.

    Returns:
        Predictions.

    Example:
        ```python
        >>> import pandas as pd
        >>> import futureframe as ff
        >>> X = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 0, 1]})
        >>> x = pd.DataFrame({"feature": [4, 5, 6]})
        >>> y_pred = ff.predict(x, X)
        ```
    """
    if isinstance(predictor, str):
        predictor = create_predictor(predictor, **kwargs)

    predictor.finetune(X_train, y_train)

    return predictor.predict(X_test)
