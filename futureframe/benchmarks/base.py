import json
import logging
import os
import shutil
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from futureframe import config
from futureframe.baselines import create_baseline_pipeline, get_task_type
from futureframe.benchmarks.download import download_dataset, get_dataset_dest_from_link
from futureframe.evaluate import eval
from futureframe.features import encode_target_label, get_num_classes, prepare_target_for_eval
from futureframe.finetune import finetune, predict
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

    def run(self, model_class, params, batch_size: int = 8, task_type=None, seed: int = 42, *args, **kwargs):
        results = []
        for subdir in tqdm(self.subdirs, desc="Running benchmark"):
            # for self.benchmark_iter():
            log.debug(f"Running benchmark for {subdir}")
            t0 = time.perf_counter()
            model = model_class(**params)
            try:
                res = self.run_subdir(
                    subdir, model, batch_size=batch_size, task_type=task_type, seed=seed, *args, **kwargs
                )
            except Exception as e:
                log.error(f"Failed to run benchmark for {subdir}: {e}")
                res = {"error": str(e)}
                raise e
            t1 = time.perf_counter()
            res["time_in_s"] = t1 - t0

            log.info(f"{res=}")
            results.append(res)
            # TODO: save intermediate results
            # resume, skip if already in csv

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

    def run_subdir(self, subdir, model, batch_size: int = 8, task_type=None, seed: int = 42, *args, **kwargs):
        X_train, X_test, y_train, y_test = self._get_split(subdir, test_size=0.3, random_state=seed)
        num_classes = get_num_classes(y_train, task_type)
        log.debug(f"{num_classes=}")
        y_pred = self._run_model(model, num_classes, X_train, y_train, X_test, batch_size=batch_size, *args, **kwargs)
        y_pred = cast_to_ndarray(y_pred)

        y_test = prepare_target_for_eval(y_test, num_classes=num_classes)
        return self.eval(model.__class__.__name__, get_last_two_folders(subdir), y_test, y_pred, num_classes, seed)

    @staticmethod
    def eval(model_name, dataset_name, y_test, y_pred, num_classes, seed):
        metrics = eval(y_test, y_pred, is_prob=True)
        log.debug(f"{metrics=}")
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "num_classes": num_classes,
            **metrics,
            "seed": seed,
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
        # TODO: reinit model
        print(f"{num_epochs=}, {patience=}")
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


class BaselineBenchmark(Benchmark):
    @staticmethod
    def _run_model(
        model, num_class, X_train, y_train, X_test, batch_size=None, patience=None, num_epochs=None, *args, **kwargs
    ) -> Tensor:
        print(f"{num_epochs=}, {patience=}")
        numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
        task_type = get_task_type(num_class)
        model_pipeline = create_baseline_pipeline(
            model,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            task_type=task_type,
        )
        model_pipeline.fit(X_train, y_train)
        if num_class >= 2:
            y_pred = model_pipeline.predict_proba(X_test)
        else:
            y_pred = model_pipeline.predict(X_test)
        return y_pred

    def run_subdir(self, subdir, model, batch_size: int = 8, task_type=None, seed: int = 42, *args, **kwargs):
        X_train, X_test, y_train, y_test = self._get_split(subdir, test_size=0.3, random_state=seed)
        num_classes = get_num_classes(y_train, task_type)
        y_train = prepare_target_for_eval(y_train, num_classes)
        log.debug(f"{num_classes=}")
        y_pred = self._run_model(model, num_classes, X_train, y_train, X_test, batch_size=batch_size, *args, **kwargs)
        y_pred = cast_to_ndarray(y_pred)

        y_test = prepare_target_for_eval(y_test, num_classes=num_classes)
        return self.eval(model.__class__.__name__, get_last_two_folders(subdir), y_test, y_pred, num_classes, seed)


class ModifiedBenchmark(Benchmark):
    def run_subdir(self, subdir, model, batch_size: int = 8, seed: int = 42, *args, **kwargs):
        X_train, X_test, y_train, y_test = self._get_split(subdir, test_size=0.1, random_state=seed)
        num_classes = get_num_classes(y_train)
        log.debug(f"{num_classes=}")
        y_train = prepare_target_for_eval(y_train, num_classes=num_classes)
        checkpoints_dir = os.path.join(
            config.CHECKPOINTS_ROOT, "finetune", model.__class__.__name__, get_last_two_folders(subdir)
        )
        os.makedirs(checkpoints_dir, exist_ok=True)
        y_pred = self._run_model(
            model,
            num_classes,
            X_train,
            y_train,
            X_test,
            checkpoints_dir=checkpoints_dir,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        y_pred = cast_to_ndarray(y_pred)

        y_test = prepare_target_for_eval(y_test, num_classes=num_classes)
        metrics = eval(y_test, y_pred, is_prob=False)
        log.debug(f"{metrics=}")
        results = {
            "dataset": get_last_two_folders(subdir),
            "model": model.__class__.__name__,
            "seed": seed,
            "num_classes": num_classes,
            **metrics,
        }

        return results

    @staticmethod
    def _run_model(
        model,
        num_class,
        X_train,
        y_train,
        X_test,
        checkpoints_dir=None,
        batch_size=8,
        patience=3,
        max_steps=None,
        num_epochs=20,
        *args,
        **kwargs,
    ) -> Tensor:
        if max_steps is None:
            max_steps = num_epochs * len(X_train) // batch_size

        model.build_head(num_class)

        # TODO: reinit model
        # model.init_weights()
        model, history = finetune(
            model,
            X_train,
            y_train,
            checkpoints_dir=checkpoints_dir,
            num_classes=num_class,
            batch_size=batch_size,
            max_steps=max_steps,
            patience=patience,
            *args,
            **kwargs,
        )

        if checkpoints_dir is not None:
            # save history as json
            with open(os.path.join(checkpoints_dir, "history.json"), "w") as f:
                json.dump(history, f)
            log.info(f"Saved history to {checkpoints_dir}.")
            # load best model
            w = torch.load(os.path.join(checkpoints_dir, "best_model.pth"))
            model.load_state_dict(w)
            log.info(f"Loaded best model from {checkpoints_dir}.")

        y_pred = predict(
            model,
            X_test,
            batch_size=batch_size,
        )
        return y_pred
