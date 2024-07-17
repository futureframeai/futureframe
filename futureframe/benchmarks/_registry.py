from futureframe.benchmarks.cm2b import CM2Benchmark
from futureframe.benchmarks.openmlb import (
    OpenMLCC18BaselineBenchmark,
    OpenMLCC18Benchmark,
    OpenMLRegressionBaselineBenchmark,
    OpenMLRegressionBenchmark,
)

benchmarks_registry = dict(
    CM2Benchmark=CM2Benchmark,
    OpenMLCC18Benchmark=OpenMLCC18Benchmark,
    OpenMLCC18BaselineBenchmark=OpenMLCC18BaselineBenchmark,
    OpenMLRegressionBenchmark=OpenMLRegressionBenchmark,
    OpenMLRegressionBaselineBenchmark=OpenMLRegressionBaselineBenchmark,
)


def create_benchmark(benchmark_name: str, **kwargs):
    try:
        benchmark_cls = benchmarks_registry[benchmark_name]
    except KeyError:
        raise ValueError(f"Benchmark {benchmark_name} not found in the registry.")

    return benchmark_cls(**kwargs)