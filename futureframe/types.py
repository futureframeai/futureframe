from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
import pandas as pd
from torch import Tensor

TargetType = pd.DataFrame | pd.Series | list | np.ndarray | Tensor

# Define the type alias
TaskType = Literal["classification", "regression", "multiclass_classification"]


def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(dataclass_to_dict(item) for item in obj)
    elif obj is None:
        return None
    else:
        return obj


class BaseInput:
    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        super().__setattr__(key, value)


class BaseOutput(BaseInput):
    pass


class ColumnDtype(Enum):
    CATEGORICAL_STR = auto()  # 1
    CATEGORICAL_INT = auto()  # 2
    NUMERICAL_FLOAT = auto()
    NUMERICAL_INT = auto()
    BOOLEAN = auto()
    MISSING = auto()
    DATETIME = auto()
    OTHER = auto()

    @classmethod
    def get(cls, name: str):
        name = name.upper()
        if name in cls.__members__:
            return cls[name]
        return cls.OTHER

    @classmethod
    def name_to_value(cls, name: str):
        name = name.upper()
        if name in cls.__members__:
            return cls[name].value
        return cls.OTHER.value


class ValueDtype(Enum):
    """Python primitive data types, Pandas dtypes."""

    STR = auto()
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    DATETIME = auto()
    TIMEDELTA = auto()
    TIMESTAMP = auto()
    NAN = auto()
    COMPLEX = auto()
    BYTES = auto()
    NONE = auto()
    OBJECT = auto()
    OTHER = auto()

    @classmethod
    def get(cls, name: str):
        name = name.upper()
        if name in cls.__members__:
            return cls[name]
        return cls.OTHER

    @classmethod
    def name_to_value(cls, name: str):
        name = name.upper()
        if name in cls.__members__:
            return cls[name].value
        return cls.OTHER.value

    @classmethod
    def convert_dtype(cls, value):
        # TODO: finish this
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return cls.NAN
        elif isinstance(value, bool):
            return cls.BOOL
        elif isinstance(value, int) and not isinstance(value, bool):
            return cls.INT
        elif isinstance(value, float):
            return cls.FLOAT
        elif isinstance(value, str):
            return cls.STR
        elif isinstance(value, (datetime, np.datetime64, pd.Timestamp)):
            return cls.DATETIME
        else:
            return cls.OTHER


class FinetuningTasks(Enum):
    MULTICLASS_CLASSIFICATION = auto()
    BINARY_CLASSIFICATION = auto()
    REGRESSION = auto()


def get_finetuning_task_type(y_true):
    # TODO: num of classes
    if "float" in str(y_true.dtype):
        return FinetuningTasks.REGRESSION
    if len(np.unique(y_true)) == 2:
        return FinetuningTasks.BINARY_CLASSIFICATION
    elif len(np.unique(y_true)) > 2:
        return FinetuningTasks.MULTICLASS_CLASSIFICATION


def valuedtype_to_columndtype(valuedtype: ValueDtype):
    mapping = {
        ValueDtype.STR: ColumnDtype.CATEGORICAL_STR,
        ValueDtype.INT: ColumnDtype.CATEGORICAL_INT,
        ValueDtype.FLOAT: ColumnDtype.NUMERICAL_FLOAT,
        ValueDtype.BOOL: ColumnDtype.BOOLEAN,
        ValueDtype.DATETIME: ColumnDtype.DATETIME,
        ValueDtype.TIMESTAMP: ColumnDtype.DATETIME,
        ValueDtype.NAN: ColumnDtype.MISSING,
        ValueDtype.OTHER: ColumnDtype.OTHER,
    }
    return mapping.get(valuedtype, ColumnDtype.OTHER)


class Metrics(Enum):
    ACCURACY = "accuracy"
    AUC = "auc"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"