from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def finetune(self, X_train, y_train, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test, *args, **kwargs) -> list[int] | list[str] | list[float] | np.ndarray:
        pass