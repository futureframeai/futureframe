import functools
import os
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import regex
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def freeze(layer, verbose=False):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


def print_non_frozen_layers(model: nn.Module):
    """
    Print the layers of a PyTorch model that are not frozen (i.e., require gradients).

    Parameters:
    - model: nn.Module
        The PyTorch model to inspect.
    """
    print("Non-frozen layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_auto_device():
    # "mps"
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def send_to_device_recursively(data, device):
    if isinstance(data, dict):
        return {k: send_to_device_recursively(v, device) for k, v in data.items()}
    if isinstance(data, Tensor):
        return data.to(device)
    return data


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_num_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params


def get_activation_fn(activation):
    mapping = {
        "relu": F.relu,
        "gelu": F.gelu,
        "selu": F.selu,
        "leakyrelu": F.leaky_relu,
    }
    if activation in mapping:
        return mapping[activation]
    raise RuntimeError(f"activation should be one of {list(mapping.keys())}, not {activation}")


def unzip(iterable):
    raise NotImplementedError


def curl_download(url, output_path):
    raise NotImplementedError


def read_parquet(filename: str):
    table = pq.read_table(filename)
    df = table.to_pandas()
    return df


def read_parquet_columns(filename: str, columns: list[str]) -> None:
    table = pq.read_pandas(filename, columns=columns)
    df = table.to_pandas()
    return df


def read_parquet_metadata(filename: str):
    parquet_file = pq.ParquetFile(filename)
    metadata = parquet_file.metadata
    schema = parquet_file.schema
    return {"metadata": metadata, "schema": schema}


def print_ndarray(a):
    return f"ndarray(shape={a.shape}, dtype={a.dtype})"


def print_tensor(t: torch.Tensor):
    return f"{repr(t)[:-1]}, \n\nshape={t.shape}, dtype={t.dtype})"


def preprocess_text(text: str) -> list[str]:
    # gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    gpt4_pattern = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )
    compiled_pattern = regex.compile(gpt4_pattern)
    # split the text up into text ch
    text_chunks = regex.findall(compiled_pattern, text)
    if len(text_chunks) == 0:
        text_chunks = [" "]
    return text_chunks


def time_benchmark(func):
    """Decorator to measure the inference time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in: {elapsed_time:.6f} seconds")
        return result

    return wrapper


def human_readable_bytes(size_in_bytes):
    # Define the size units in a tuple
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    # Start with the base size
    size = float(size_in_bytes)

    # Iterate over the units, dividing the size by 1024 for each unit
    for unit in units:
        if size < 1024:
            # Return the size formatted to 2 decimal places
            return f"{size:.2f} {unit}"
        size /= 1024

    # In case the size is extremely large, it will be in Yottabytes
    return f"{size:.2f} YB"


def cast_to_tensor(x: Tensor | list | np.ndarray | pd.Series | pd.DataFrame) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        return torch.Tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"unknown dtype: {type(x)}")


def cast_to_ndarray(x: Tensor | list | np.ndarray | pd.Series | pd.DataFrame | Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, pd.Series):
        return x.values
    elif isinstance(x, pd.DataFrame):
        return x.values
    else:
        raise ValueError(f"unknown dtype: {type(x)}")


def cast_to_series(x: Tensor | list | np.ndarray | pd.Series | pd.DataFrame | Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    elif isinstance(x, list):
        return pd.Series(x)
    elif isinstance(x, np.ndarray):
        return pd.Series(x)
    elif isinstance(x, torch.Tensor):
        return pd.Series(x.cpu().numpy())
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        else:
            raise ValueError("DataFrame has more than one column and cannot be converted to pd.Series")

    raise ValueError(f"unknown dtype: {type(x)}")


def get_last_two_folders(path):
    # Normalize the path to handle different OS path separators
    path = os.path.normpath(path)
    # Split the path into parts
    parts = path.split(os.sep)

    # Get the last two folders
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    elif len(parts) == 1:
        return parts[-1]
    else:
        return ""


def save_or_append_to_csv(df: pd.DataFrame, path: str):
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
