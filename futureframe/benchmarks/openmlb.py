import logging
import os
from pathlib import Path

import openml
import pandas as pd
from openml import config
from openml.tasks import TaskType
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from futureframe import config
from futureframe.benchmarks.base import Benchmark, ModifiedBenchmark
from futureframe.evaluate import eval
from futureframe.registry import create_predictor, get_predictor_class_by_idx

log = logging.getLogger(os.path.basename(__file__))


def list_dataset_ids_from_openml_benchmark_suite(benchmark_id: str = "OpenML-CC18"):
    benchmark_suite = openml.study.get_suite(benchmark_id)
    data_ids = []
    for task_id in benchmark_suite.tasks:
        task = openml.tasks.get_task(
            task_id,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
            download_splits=False,
        )
        data_ids.append(task.dataset_id)

    return data_ids


def get_links_from_dataset_ids(data_ids):
    return [f"https://openml.org/d/{data_id}" for data_id in data_ids]


def list_links_from_openml_benchmark_suite(benchmark_id: str = "OpenML-CC18"):
    data_ids = list_dataset_ids_from_openml_benchmark_suite(benchmark_id)
    return get_links_from_dataset_ids(data_ids)


class OpenMLBenchmark(Benchmark):
    def __init__(
        self,
        openmlb_id: str = "OpenML-CC18",
        csv_results_name: str = "benchmark.csv",
        datasets_root: str = config.DATASETS_ROOT,
        csv_results_root: str = config.RESULTS_ROOT,
        download=True,
        force_download=False,
        resume=False,
        verbose=False,
    ) -> None:
        self.datasets_links = list_links_from_openml_benchmark_suite(openmlb_id)
        log.debug(f"{len(self.datasets_links)=}")
        super().__init__(csv_results_name, datasets_root, csv_results_root, download, force_download, resume, verbose)


class OpenMLCC18Benchmark(Benchmark):
    datasets_links = [  # 72 datasets
        "https://openml.org/d/3",
        "https://openml.org/d/6",
        "https://openml.org/d/11",
        "https://openml.org/d/12",
        "https://openml.org/d/14",
        "https://openml.org/d/15",
        "https://openml.org/d/16",
        "https://openml.org/d/18",
        "https://openml.org/d/22",
        "https://openml.org/d/23",
        "https://openml.org/d/28",
        "https://openml.org/d/29",
        "https://openml.org/d/31",
        "https://openml.org/d/32",
        "https://openml.org/d/37",
        "https://openml.org/d/44",
        "https://openml.org/d/46",
        "https://openml.org/d/50",
        "https://openml.org/d/54",
        "https://openml.org/d/151",
        "https://openml.org/d/182",
        "https://openml.org/d/188",
        "https://openml.org/d/38",
        "https://openml.org/d/307",
        "https://openml.org/d/300",
        "https://openml.org/d/458",
        "https://openml.org/d/469",
        "https://openml.org/d/554",
        "https://openml.org/d/1049",
        "https://openml.org/d/1050",
        "https://openml.org/d/1053",
        "https://openml.org/d/1063",
        "https://openml.org/d/1067",
        "https://openml.org/d/1068",
        "https://openml.org/d/1590",
        "https://openml.org/d/4134",
        "https://openml.org/d/1510",
        "https://openml.org/d/1489",
        "https://openml.org/d/1494",
        "https://openml.org/d/1497",
        "https://openml.org/d/1501",
        "https://openml.org/d/1480",
        "https://openml.org/d/1485",
        "https://openml.org/d/1486",
        "https://openml.org/d/1487",
        "https://openml.org/d/1468",
        "https://openml.org/d/1475",
        "https://openml.org/d/1462",
        "https://openml.org/d/1464",
        "https://openml.org/d/4534",
        "https://openml.org/d/6332",
        "https://openml.org/d/1461",
        "https://openml.org/d/4538",
        "https://openml.org/d/1478",
        "https://openml.org/d/23381",
        "https://openml.org/d/40499",
        "https://openml.org/d/40668",
        "https://openml.org/d/40966",
        "https://openml.org/d/40982",
        "https://openml.org/d/40994",
        "https://openml.org/d/40983",
        "https://openml.org/d/40975",
        "https://openml.org/d/40984",
        "https://openml.org/d/40979",
        "https://openml.org/d/40996",
        "https://openml.org/d/41027",
        "https://openml.org/d/23517",
        "https://openml.org/d/40923",
        "https://openml.org/d/40927",
        "https://openml.org/d/40978",
        "https://openml.org/d/40670",
        "https://openml.org/d/40701",
    ]


class ModifOpenMLCC18Benchmark(OpenMLCC18Benchmark, ModifiedBenchmark):
    pass


def main(
    benchmark_id: str = "OpenML-CC18",
    method_name: str = "RandomForest",
    method_id: int = None,
    data_dir: str = "datasets/",
    results_path: str = "results/benchmark.csv",
    seed: int = 42,
    resume: bool = False,
    logging_level: str = "INFO",
    N=10240,
    n=1024,
):
    logging.basicConfig(level=logging.getLevelName(logging_level))
    path = Path(__file__).parent
    config.cache_directory = path / data_dir
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    benchmark_suite = openml.study.get_suite(benchmark_id)
    log.debug(f"{benchmark_suite=}")

    if method_id is not None:
        method_name = get_predictor_class_by_idx(idx).__name__

    cfg = dict(method=method_name, benchmark_suite=benchmark_id, seed=seed)

    for task_id in tqdm(benchmark_suite.tasks):  # iterate over all tasks
        log.debug(f"Processing task {task_id}")
        try:
            # check if task is completed
            df = pd.read_csv(results_path) if os.path.exists(results_path) else None
            if df is not None and resume:
                if df.query(f"task_id == {task_id} and method == '{method_name}'").shape[0] > 0:
                    log.info(f"Task {task_id} already completed for {method_name}. Skipping.")
                    continue

            task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False)

            dataset = fetch_openml(
                data_id=task.dataset_id,
                data_home=data_dir,
                as_frame=True,
                return_X_y=False,
                parser="auto",
            )
            dataset.frame.info()
            X, y = dataset.data, dataset.target
            name = dataset.details["name"]
            log.debug(f"{X=}")
            log.debug(f"{y=}")

            column_names = X.columns
            numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32", "number"]).columns
            categorical_features = X.select_dtypes(include=["object", "category"]).columns
            log.debug(f"{column_names=}")
            log.debug(f"{numeric_features=}")
            log.debug(f"{categorical_features=}")

            task = openml.tasks.get_task(task_id)  # download the OpenML task
            log.debug(f"{task=}")

            task_type = ""
            if task.task_type_id == TaskType.SUPERVISED_REGRESSION:
                task_type = "regression"
            elif task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION:
                class_labels = task.class_labels
                num_classes = len(class_labels)
                log.debug(f"{class_labels=}")

                if num_classes == 2:
                    task_type = "binary_classification"
                else:
                    task_type = "multiclass_classification"

                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                log.error(f"Unknown task type: {task.task_type_id}")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            X_train = X_train[:N]
            y_train = y_train[:N]
            X_test = X_test[:n]
            y_test = y_test[:n]

            predictor = create_predictor(
                method_name,
                column_names,
                task_type,
                numeric_features,
                categorical_features,
            )
            log.info("Fitting...")
            predictor.fit(X_train, y_train)
            log.info("Predicting...")
            y_pred = predictor.predict(X_test)

            res = eval(y_test, y_pred, task_type)
            results = {"task_id": task_id, "name": name, **cfg, **res}
            df = pd.DataFrame([results])

            # append the results to a CSV file
            if not os.path.exists(results_path):
                df.to_csv(results_path, index=False)
            else:
                df.to_csv(results_path, mode="a", header=False, index=False)

            print(df.to_markdown(index=False))

        except Exception as e:
            log.error(f"Error processing task {task_id}: {e}")
            continue


if __name__ == "__main__":
    from fire import Fire

    Fire({"main": main, "list": list_links_from_openml_benchmark_suite})
