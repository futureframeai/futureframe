import logging
from pprint import pprint

import futureframe as ff


def main(baseline_name: str, benchmark_name: str, task_type: str | None = None, log_level="INFO"):
    logging.basicConfig(level=log_level)
    params = {}
    baseline_cls = ff.baselines.create_baseline_cls(baseline_name, task_type)
    benchmark = ff.benchmarks.create_benchmark(benchmark_name, download=True)
    results = benchmark.run(baseline_cls, params, task_type=task_type, seed=42)
    pprint(results)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

# python scripts/benchmark_baselines_cli.py \
#   --baseline_name XGB \
#   --benchmark_name OpenMLRegressionBaselineBenchmark \
#   --task_type regression \
#   --log_level DEBUG

# python scripts/benchmark_baselines_cli.py \
#   --baseline_name XGB \
#   --benchmark_name OpenMLRegressionBaselineBenchmark \
#   --task_type regression \
#   --log_level DEBUG
