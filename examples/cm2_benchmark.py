from pprint import pprint

import futureframe as ff

model = ff.models.CM2Classifier()
benchmark = ff.benchmarks.CM2Benchmark()
results = benchmark.run(model, batch_size=64, seed=42, num_epochs=30, patience=10, lr=1e-3)
pprint(results)
