import logging
import os

import openml

from futureframe.benchmarks.base import BaselineBenchmark, Benchmark, ModifiedBenchmark

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
    links_list = get_links_from_dataset_ids(data_ids)
    # print in a way that I can copy and paste

    for link in links_list:
        print(f'"{link}",')


class OpenMLCC18Benchmark(Benchmark):
    """Benchmark id: "OpenML-CC18"""

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


class OpenMLCC18BaselineBenchmark(OpenMLCC18Benchmark, BaselineBenchmark):
    pass


class OpenMLBinaryClassificationBenchmark(Benchmark):
    """Benchmark id:"""

    datasets_links = [  # 72 datasets
    ]


class OpenMLRegressionBenchmark(Benchmark):
    """Benchmark id: OpenML-CTR23"""

    datasets_links = [
        "https://openml.org/d/44956",
        "https://openml.org/d/44957",
        "https://openml.org/d/44958",
        "https://openml.org/d/44959",
        "https://openml.org/d/44963",
        "https://openml.org/d/44964",
        "https://openml.org/d/44965",
        "https://openml.org/d/44966",
        "https://openml.org/d/44969",
        "https://openml.org/d/44971",
        "https://openml.org/d/44972",
        "https://openml.org/d/44973",
        "https://openml.org/d/44974",
        "https://openml.org/d/44975",
        "https://openml.org/d/44976",
        "https://openml.org/d/44977",
        "https://openml.org/d/44978",
        "https://openml.org/d/44979",
        "https://openml.org/d/44980",
        "https://openml.org/d/44981",
        "https://openml.org/d/44983",
        "https://openml.org/d/44984",
        "https://openml.org/d/44987",
        "https://openml.org/d/44989",
        "https://openml.org/d/44990",
        "https://openml.org/d/44992",
        "https://openml.org/d/44993",
        "https://openml.org/d/45012",
        "https://openml.org/d/41021",
        "https://openml.org/d/44960",
        "https://openml.org/d/44962",
        "https://openml.org/d/44967",
        "https://openml.org/d/44970",
        "https://openml.org/d/44994",
        "https://openml.org/d/45402",
    ]

    pass


class OpenMLRegressionBaselineBenchmark(OpenMLRegressionBenchmark, BaselineBenchmark):
    pass


if __name__ == "__main__":
    from fire import Fire

    Fire({"list": list_links_from_openml_benchmark_suite})