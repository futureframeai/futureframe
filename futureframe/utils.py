import random

import numpy as np
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
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "selu":
        return F.selu
    elif activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError(f"activation should be relu/gelu/selu/leakyrelu, not {activation}")
