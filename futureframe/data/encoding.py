import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from torch import Tensor

from futureframe import config
from futureframe.data.features import clean_entity_names

log = logging.getLogger(__name__)

text_encoding_models = [
    "fasttext",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-base-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5 ",
    "intfloat/multilingual-e5-large-instruct",
]


class BaseFeaturesToModelInput(ABC):
    """
    Base class for encoding features into model inputs.

    This class provides a base implementation for encoding features into model inputs.
    Subclasses should implement the `encode` method to perform the actual encoding.

    Attributes:
        None

    Methods:
        fit(data: pd.DataFrame, *args, **kwargs) -> self:
            Fit the encoder to the data.

        encode(data: pd.DataFrame, *args, **kwargs) -> dict[str, Tensor]:
            Encode the data into model inputs.

        __call__(data, *args: Any, **kwds: Any):
            Call the `encode` method.

        download(path: str):
            Download the encoder to the specified path.

        load(path: str):
            Load the encoder from the specified path.

        save(path: str):
            Save the encoder to the specified path.
    """

    def __init__(self):
        raise NotImplementedError

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        """
        Fit the encoder to the data.

        Args:
            data (pd.DataFrame): The data to fit the encoder on.

        Returns:
            self: The fitted encoder object.
        """
        return self

    @abstractmethod
    def encode(self, data: pd.DataFrame, *args, **kwargs) -> dict[str, Tensor]:
        """
        Encode the data into model inputs.

        This method should be implemented by subclasses to perform the actual encoding.

        Args:
            data (pd.DataFrame): The data to encode.

        Returns:
            dict[str, Tensor]: A dictionary of encoded model inputs.
        """
        pass

    def __call__(self, data, *args: Any, **kwds: Any):
        """
        Call the `encode` method.

        Args:
            data: The data to encode.

        Returns:
            dict[str, Tensor]: A dictionary of encoded model inputs.
        """
        return self.encode(data, *args, **kwds)

    def download(self, path: str):
        """
        Download the encoder to the specified path.

        Args:
            path (str): The path to download the encoder to.
        """
        pass

    def load(self, path: str):
        """
        Load the encoder from the specified path.

        Args:
            path (str): The path to load the encoder from.
        """
        pass

    def save(self, path: str):
        """
        Save the encoder to the specified path.

        Args:
            path (str): The path to save the encoder to.
        """
        pass


def extract_fasttext_features(data: pd.DataFrame, extract_col_name: str):
    import fasttext

    # Preliminary Settings
    lm_model = fasttext.load_model()

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()

    # Entity Names
    ent_names = clean_entity_names(data[extract_col_name])
    ent_names = list(ent_names)

    # Data Fasttext for entity names
    data_fasttext = [lm_model.get_sentence_vector(str(x)) for x in ent_names]
    data_fasttext = np.array(data_fasttext)
    data_fasttext = pd.DataFrame(data_fasttext)
    col_names = [f"X{i}" for i in range(data_fasttext.shape[1])]
    data_fasttext = data_fasttext.set_axis(col_names, axis="columns")
    data_fasttext = pd.concat([data_fasttext, data[extract_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    data_fasttext = data_fasttext.reset_index(drop=True)

    return data_fasttext


def get_text_encoding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=config.CACHE_ROOT)
    return model
