import os

PRETRAINED_MODELS_ROOT = os.environ.get("PRETRAINED_MODELS_ROOT", "pretrained-models")
FINETUNED_MODELS_ROOT = os.environ.get("FINETUNED_MODELS_ROOT", "finetuned-models")
CHECKPOINTS_ROOT = os.environ.get("CHECKPOINTS_ROOT", "checkpoints")
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "datasets")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")
