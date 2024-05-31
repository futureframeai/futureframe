from enum import Enum, auto

import numpy as np
import pandas as pd

Features = pd.DataFrame | pd.Series | list[str] | list[int] | list[float] | np.ndarray


class FeaturesType(Enum):
    CATEGORY = auto()
    NUMBER = auto()
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    OTHER = auto()


class Tasks(Enum):
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
