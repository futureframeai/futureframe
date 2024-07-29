import functools
import logging
import os
import random
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import regex
import requests
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

log = logging.getLogger(__name__)


def freeze(layer, verbose=False):
    """
    Freezes the parameters of a given layer.

    Args:
        layer (nn.Module): The layer whose parameters need to be frozen.
        verbose (bool, optional): If True, prints a message for each parameter that is frozen.
    """
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
            if verbose:
                print(f"Parameter '{param}' frozen.")


def unfreeze(layer, verbose=False):
    """
    Unfreezes the given layer by setting the `requires_grad` attribute of all its parameters to True.

    Args:
        layer (nn.Module): The layer to unfreeze.
    """
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True
            if verbose:
                print(f"Parameter '{param}' unfrozen.")


def print_non_frozen_layers(model: nn.Module):
    """
    Prints the names of non-frozen layers in the given model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        None
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


def get_auto_device(index: int | None = None):
    """
    Returns the appropriate torch device based on the availability of CUDA and MPS.

    Parameters:
        index (int | None): The index of the CUDA device to use. If None, the default CUDA device is used.

    Returns:
        torch.device: The selected torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        if index is not None:
            return torch.device(f"cuda:{index}")
        return torch.device("cuda")
    return torch.device("cpu")


def send_to_device_recursively(data: dict | Tensor | Any, device, **kwargs):
    if isinstance(data, dict):
        return {k: send_to_device_recursively(v, device, **kwargs) for k, v in data.items()}
    if isinstance(data, Tensor):
        return data.to(device, **kwargs)
    return data


def seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If using torch's multi-GPU functionality
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


def get_num_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Calculates the number of trainable and non-trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        tuple[int, int]: A tuple containing the number of trainable parameters and the number of non-trainable parameters.
    """
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
    raise ValueError(f"Invalid activation function '{activation}'. Available options are: {', '.join(mapping.keys())}.")


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
    pattern = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )
    compiled_pattern = regex.compile(pattern)
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


def human_readable_bytes(size_in_bytes: int) -> str:
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
    """
    Casts the input `x` to a PyTorch tensor.

    Args:
        x (Tensor | list | np.ndarray | pd.Series | pd.DataFrame): The input data to be casted.

    Returns:
        Tensor: The input data casted to a PyTorch tensor.

    Raises:
        ValueError: If the input data type is not supported.
    """
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, list):
        return Tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"unknown dtype: {type(x)}")


def cast_to_ndarray(x: Tensor | list | np.ndarray | pd.Series | pd.DataFrame | Any) -> np.ndarray:
    """
    Casts the input `x` to a NumPy ndarray.

    Args:
        x (Tensor | list | np.ndarray | pd.Series | pd.DataFrame | Any): The input object to be casted.

    Returns:
        np.ndarray: The input `x` casted to a NumPy ndarray.

    Raises:
        ValueError: If the input `x` has an unknown dtype.

    """
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
    """
    Casts the input `x` to a pandas Series.

    Parameters:
        x (Tensor | list | np.ndarray | pd.Series | pd.DataFrame | Any): The input object to be casted.

    Returns:
        pd.Series: The input object `x` casted to a pandas Series.

    Raises:
        ValueError: If the input object `x` has an unknown data type.
        ValueError: If the input object `x` is a DataFrame with more than one column.
    """
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
    path = os.path.normpath(path)
    parts = path.split(os.sep)

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


def download_file(url, local_path):
    log.info(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        log.info("Failed to download file")
        response.raise_for_status()
    else:
        log.info(f"File downloaded successfully and saved as {local_path}")


# Function to extract the zip file
def extract_zip(zip_path, extract_to="."):
    log.info(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    log.info("Extraction complete.")


def download_and_extract(url: str, root: str, filename=None, clean=False):
    if filename is None:
        filename = os.path.basename(url)
    dest = os.path.join(root, filename)
    download_file(url, dest)
    extract_zip(dest, root)
    if clean:
        os.remove(dest)
