import importlib.metadata

from . import (
    baselines,
    benchmarks,
    config,
    types,
    evaluate,
    features,
    finetune,
    models,
    registry,
    tabular_datasets,
    utils,
)

__version__ = importlib.metadata.version("futureframe")
