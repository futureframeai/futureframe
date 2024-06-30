import json
import os

PRETRAINED_MODELS_ROOT = os.environ.get("PRETRAINED_MODELS_ROOT", "pretrained-models")
FINETUNED_MODELS_ROOT = os.environ.get("FINETUNED_MODELS_ROOT", "finetuned-models")
CHECKPOINTS_ROOT = os.environ.get("CHECKPOINTS_ROOT", "checkpoints")
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "data")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")
CACHE_ROOT = os.environ.get("CACHE_ROOT", "cache")


class BaseConfig:
    def save(self, path):
        self.to_json(path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}

    @classmethod
    def from_dict(cls, d):
        config = cls()
        for k, v in d.items():
            setattr(config, k, v)
        return config

    def to_json(self, path, indent=4):
        with open(path, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=4))

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
