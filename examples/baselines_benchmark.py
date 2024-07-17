import logging
from pprint import pprint

import futureframe as ff

logging.basicConfig(level=logging.DEBUG)

classifier = ff.baselines.create_baseline_cls("XGB", "classification")
benchmark = ff.benchmarks.openmlb.OpenMLCC18BaselineBenchmark(download=True)
params = {}
results = benchmark.run(classifier, params, seed=42)
pprint(results)