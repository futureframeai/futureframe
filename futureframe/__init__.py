from . import benchmarks, models
from . import baselines, config, data_types, evaluate, features, finetune, registry, tabular_datasets, utils


import importlib.metadata

__version__ = importlib.metadata.version("futureframe")