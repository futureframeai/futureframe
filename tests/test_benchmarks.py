import logging

import futureframe as ff

logging.basicConfig(level=logging.DEBUG)


def test_openmlb_benchmark_download():
    benchmark = ff.benchmarks.openmlb.OpenMLCC18Benchmark(download=True)


def test_benchmark_iter():
    logging.basicConfig(level=logging.DEBUG)
    benchmark = ff.benchmarks.openmlb.OpenMLCC18Benchmark(download=True)
    for idx, (X_train, y_train, X_val, y_val) in enumerate(benchmark.benchmark_iter()):
        pass


if __name__ == "__main__":
    test_openmlb_benchmark_download()
    test_benchmark_iter()
