import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from futureframe.config import CACHE_ROOT

log = logging.getLogger(__name__)


class TabText:
    def __init__(
        self,
        device="cpu",
        model_path="Alibaba-NLP/gte-large-en-v1.5",
        download=True,
        columns=None,
        target_variable=None,
        batch_size=32,
        seed=42,
        **estimators_params,
    ) -> None:
        self.device = device
        self.columns = columns
        self.batch_size = batch_size
        self.seed = seed
        self.target_variable = target_variable

        self.model_path = model_path
        if download:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model.save_pretrained(Path(CACHE_ROOT) / f"{model_path}-tokenizer")
            self.tokenizer.save_pretrained(Path(CACHE_ROOT) / f"{model_path}-model")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(Path(CACHE_ROOT) / f"{model_path}-tokenizer")
            self.model = AutoModel.from_pretrained(Path(CACHE_ROOT) / f"{model_path}-model")

        self.model.eval()
        self.model.to(device)

        # if task is classification
        self.estimator = self.estimator_cls(random_state=seed, **estimators_params)

    @property
    def estimator_cls(self):
        return LogisticRegression

    def finetune(self, X_train, y_train):
        if self.columns is None:
            if isinstance(X_train, np.ndarray):
                self.columns = range(X_train.shape[1])
            elif isinstance(X_train, list):
                self.columns = range(len(X_train[0]))
            elif isinstance(X_train, pd.DataFrame):
                self.columns = X_train.columns

        if self.target_variable is None:
            if isinstance(y_train, pd.Series):
                self.target_variable = y_train.name
            elif isinstance(y_train, pd.DataFrame):
                self.target_variable = y_train.columns[0]
            else:
                self.target_variable = "target"

        if isinstance(X_train, list):
            X_train = np.array(X_train)
        elif isinstance(X_train, pd.DataFrame):
            X_train = X_train.values

        all_embeddings = self._serialize_embed(X_train, "Fit")
        log.debug(f"Training embeddings shape: {all_embeddings.shape}")
        self.estimator.fit(all_embeddings, y_train)

    def predict(self, X_test):
        if isinstance(X_test, list):
            X_test = np.array(X_test)
        elif isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        all_embeddings = self._serialize_embed(X_test, "Predict")
        return self.estimator.predict(all_embeddings)

    def predict_proba(self, X_test):
        if isinstance(X_test, list):
            X_test = np.array(X_test)
        elif isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        all_embeddings = self._serialize_embed(X_test, "Predict proba")
        return self.estimator.predict_proba(all_embeddings)

    def _serialize_embed(self, X: np.ndarray, tqdm_suffix=""):
        # serialize the data
        log.debug(f"{X[:1]}")
        serialized_data = []
        for row in X:
            out = f"We want to predict {self.target_variable}. "
            for col, val in zip(self.columns, row):
                out += f"{col} is {val}, "
            out = out[:-2] + "."
            serialized_data.append(out)
        log.debug(f"Serialized data (top 10): {serialized_data[:10]}")

        # embed the data
        indexes = range(0, len(serialized_data), self.batch_size)
        all_embeddings = []
        for idx in tqdm(indexes, tqdm_suffix):
            batch = serialized_data[idx : idx + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    @torch.no_grad()
    def _embed_batch(self, input_text) -> np.ndarray:
        inp = self.tokenizer(
            input_text,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        embeddings = out.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()


class TabTextRegressor(TabText):
    @property
    def estimator_cls(self):
        return LinearRegression


class TabTextXGBoostClassifier(TabText):
    @property
    def estimator_cls(self):
        return GradientBoostingClassifier


class TabTextXGBoostRegressor(TabText):
    @property
    def estimator_cls(self):
        return GradientBoostingRegressor
