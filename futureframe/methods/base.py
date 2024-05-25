from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test) -> list[int] | list[str] | list[float] | np.ndarray:
        pass