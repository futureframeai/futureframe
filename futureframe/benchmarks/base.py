import logging
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from futureframe import config
from futureframe.benchmarks.download import download_dataset, get_dataset_dest_from_link
from futureframe.evaluate import eval
from futureframe.features import get_num_classes, prepare_target_for_eval
from futureframe.utils import cast_to_ndarray, get_last_two_folders

log = logging.getLogger(__name__)


class Benchmark:
    datasets_links = []

    def __init__(
        self,
        csv_results_name: str = "benchmark.csv",
        datasets_root: str = config.DATASETS_ROOT,
        csv_results_root: str = config.RESULTS_ROOT,
        download=True,
        force_download=False,
        resume=False,
        verbose=False,
    ) -> None:
        super().__init__()

        self.datasets_root = datasets_root
        self.csv_results_root = csv_results_root
        self.csv_results_name = csv_results_name
        self.verbose = verbose
        self.resume = resume

        if download:
            os.makedirs(self.datasets_root, exist_ok=True)
            for link in tqdm(self.datasets_links, desc="Downloading datasets"):
                dest_dir = get_dataset_dest_from_link(link, self.datasets_root)
                if os.path.exists(dest_dir) and not force_download:
                    log.info(f"Dataset {dest_dir} already exists, skipping download")
                    continue
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                    download_dataset(link, self.datasets_root)
                except Exception as e:
                    log.error(f"Failed to download dataset from {link}: {e}")
                    # remove directory if download failed
                    shutil.rmtree(dest_dir)

        self.subdirs = []
        for link in self.datasets_links:
            dest_dir = get_dataset_dest_from_link(link, self.datasets_root)
            self.subdirs.append(dest_dir)
        log.debug(f"{self.subdirs=}")

    def run(self, model, batch_size: int = 8, seed: int = 42, *args, **kwargs):
        results = []
        for subdir in tqdm(self.subdirs, desc="Running benchmark"):
            res = self.run_subdir(subdir, model, batch_size=batch_size, seed=seed, *args, **kwargs)
            log.info(f"{res=}")

            results.append(res)

        self.save_results(results)
        return results

    def save_results(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.csv_results_root, self.csv_results_name)
        os.makedirs(self.csv_results_root, exist_ok=True)
        # if file exists, append
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
            log.info(f"Results appended to {csv_path}")
        else:
            df.to_csv(csv_path, index=False)
            log.info(f"Results saved to {csv_path}")

    def benchmark_iter(self, seed=42):
        for subdir in tqdm(self.subdirs, desc="Running benchmark"):
            X_train, X_val, y_train, y_val = self._get_split(subdir, test_size=0.3, random_state=seed)
            yield X_train, y_train, X_val, y_val

    def run_subdir(self, subdir, model, batch_size: int = 8, seed: int = 42, *args, **kwargs):
        X_train, X_test, y_train, y_test = self._get_split(subdir, test_size=0.3, random_state=seed)
        num_classes = get_num_classes(y_train)
        log.debug(f"{num_classes=}")
        y_pred = self._run_model(model, num_classes, X_train, y_train, X_test, batch_size=batch_size, *args, **kwargs)
        y_pred = cast_to_ndarray(y_pred)

        y_test = prepare_target_for_eval(y_test, num_classes=num_classes)
        metrics = eval(y_test, y_pred, num_classes=num_classes)
        log.debug(f"{metrics=}")
        results = {
            "dataset": get_last_two_folders(subdir),
            "model": model.__class__.__name__,
            "seed": seed,
            **metrics,
        }

        return results

    def run_idx(self, idx, model, batch_size: int = 8, seed: int = 42, *args, **kwargs):
        subdir = self.subdirs[idx]
        return self.run_subdir(subdir, model, batch_size=batch_size, seed=seed, *args, **kwargs)

    @staticmethod
    def _run_model(
        model, num_class, X_train, y_train, X_test, batch_size=8, patience=3, num_epochs=10, *args, **kwargs
    ) -> Tensor:
        # TODO: best is to define task object instead of num_class
        model.finetune(
            X_train,
            y_train,
            num_class=num_class,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            *args,
            **kwargs,
        )
        y_pred = model.predict(X_test)
        return y_pred

    @staticmethod
    def _get_split(subdir, test_size=0.3, random_state=42):
        X_path = os.path.join(subdir, "X.csv")
        y_path = os.path.join(subdir, "y.csv")
        X = pd.read_csv(X_path, low_memory=False)
        y = pd.read_csv(y_path, low_memory=False)

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0
        assert y.shape[1] == 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


class BinaryClassificationBenchmark(Benchmark):
    def __init__(self) -> None:
        pass


class MulticlassClassificationBenchmark(Benchmark):
    def __init__(self) -> None:
        pass


class RegressionBenchmark(Benchmark):
    def __init__(self) -> None:
        pass


class ClusteringBenchmark(Benchmark):
    def __init__(self) -> None:
        pass


class FewShotBenchmark(Benchmark):
    def __init__(self) -> None:
        pass


class AnomalyDetectionBenchmark(Benchmark):
    def __init__(self) -> None:
        pass
