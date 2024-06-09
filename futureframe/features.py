import logging
from typing import Optional

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def categorize_columns(df: pd.DataFrame):
    """
    Categorizes and extracts columns from a pandas DataFrame based on their data types, including detecting mixed types.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary containing lists of column names categorized by their data types.

    Examples:
    ```python
        df = pd.DataFrame(
            {
                "A": [1, 2, "three"],
                "B": ["a", "b", "c"],
                "C": [True, False, True],
                "D": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
                "E": pd.Categorical(["test", "train", "test"]),
            }
        )
        categorized_columns = categorize_columns(df)
        print(categorized_columns)
    ```
    """
    # Initialize dictionaries to hold the column names based on their types
    column_categories = {
        "categorical": [],
        "numerical": [],
        "datetime": [],
        "mixed": [],
        "other": [],
    }

    # Loop through the columns and categorize them
    for col in df.columns:
        unique_types = set(df[col].apply(type))
        unique_values = len(df[col].unique())
        log.debug(f"Column '{col}' has unique types: {unique_types} and {unique_values} unique values")
        if len(unique_types) > 1:
            column_categories["mixed"].append(col)
        else:
            unique_type = list(unique_types)[0]
            if unique_type == str:
                column_categories["categorical"].append(col)
            elif unique_type == int and unique_values == 2:
                column_categories["categorical"].append(col)
            elif unique_type == int or unique_type == float:
                column_categories["numerical"].append(col)
            elif unique_type == pd.Timestamp:
                column_categories["datetime"].append(col)
            else:
                column_categories["other"].append(col)

    return column_categories


def extract_target_variable(df: pd.DataFrame, target: Optional[str] = None):
    """
    Splits a DataFrame into features (X) and target variable (y).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The name of the target variable column.

    Returns:
    tuple: A tuple containing the features DataFrame (X) and the target Series (y).

    Examples:
    ```python
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
        X, y = split_target_variable(df, 'target')
        print(X)
        print(y)
    ```
    """
    if target is None:
        target = df.columns.tolist()[-1]

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame columns.")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def get_num_classes(y):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    num_classes = len(y.value_counts())
    return num_classes


def encode_target_label(y: pd.Series):
    # encode target label
    name = y.name
    index = y.index
    y = LabelEncoder().fit_transform(y.values)
    y = pd.Series(y, index=index, name=name)
    return y


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }
