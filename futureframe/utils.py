import random

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


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


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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
