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

from futureframe.evaluate import eval
from futureframe.registry import create_predictor, get_predictor_class_by_idx

log = logging.getLogger(os.path.basename(__file__))


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
            results = {
                "task_id": task_id,
                "name": name,
                **cfg,
                **res,
            }
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

    Fire(main)
