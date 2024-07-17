import logging

from futureframe.benchmarks.base import Benchmark, ModifiedBenchmark

log = logging.getLogger(__name__)


class CM2Benchmark(Benchmark):
    """CM2 Benchmark

    | Dataset Name     | R/C | Samples | Numerical | Categorical | Label Classes | Source                                                                  |
    |------------------|-----|---------|-----------|-------------|---------------|-------------------------------------------------------------------------|
    | Breast           | C   | 699     | 9         | 0           | 2             | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original |
    | Cmc              | C   | 1473    | 2         | 7           | 3             | https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice      |
    | Vehicle          | C   | 846     | 18        | 0           | 4             | https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes     |
    | Satimage         | C   | 6430    | 36        | 0           | 6             | https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite       |
    | Sick             | C   | 3772    | 7         | 22          | 2             | http://archive.ics.uci.edu/dataset/102/thyroid+disease                  |
    | Analcatdata      | C   | 797     | 0         | 4           | 6             | https://pages.stern.nyu.edu/~jsimonof/AnalCatData/Data/                 |
    | Adult            | C   | 48842   | 6         | 8           | 2             | https://archive.ics.uci.edu/dataset/2/adult                             |
    | PhishingWebsites | C   | 11055   | 0         | 30          | 2             | https://archive.ics.uci.edu/dataset/327/phishing+websites               |
    | Cylinder-bands   | C   | 540     | 18        | 21          | 2             | https://archive.ics.uci.edu/dataset/32/cylinder+bands                   |
    | MiceProtein      | C   | 1080    | 77        | 4           | 8             | https://archive.ics.uci.edu/dataset/342/mice+protein+expression         |
    | Car              | C   | 1728    | 0         | 6           | 4             | https://archive.ics.uci.edu/dataset/19/car+evaluation                   |
    | SAT11            | R   | 4440    | 115       | 1           | -             | https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/                |
    | Segment          | C   | 2310    | 19        | 0           | 7             | http://archive.ics.uci.edu/dataset/50/image+segmentation                |
    | Porto-seguro     | C   | 2000    | 26        | 31          | 2             | https://openml.org/d/44787                                              |
    | Pc1              | C   | 1109    | 21        | 0           | 2             | https://openml.org/d/1068                                               |
    | Elevators        | R   | 16599   | 18        | 19          | -             | https://openml.org/d/216                                                |
    | Diabetes         | C   | 768     | 8         | 0           | 2             | https://openml.org/d/37                                                 |
    | Yprop            | R   | 8885    | 251       | 0           | -             | https://openml.org/d/416                                                |
    | Topo             | R   | 8885    | 266       | 267         | -             | https://openml.org/d/422                                      |
    | Diamonds         | R   | 53940   | 6         | 3           | -             | https://openml.org/d/42225                                              |
    | House_sales      | R   | 21613   | 20        | 1           | -             | https://openml.org/d/42731                                              |
    | Amazon           | C   | 2000    | 0         | 9           | 2             | https://openml.org/d/44712                                              |
    """

    datasets_links = [
        # "http://archive.ics.uci.edu/dataset/102/thyroid+disease",
        # "http://archive.ics.uci.edu/dataset/50/image+segmentation",
        # "https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite",
        # "https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes",
        # "https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original",
        # "https://archive.ics.uci.edu/dataset/19/car+evaluation",
        # "https://archive.ics.uci.edu/dataset/2/adult",
        # "https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice",
        # "https://archive.ics.uci.edu/dataset/32/cylinder+bands",
        # "https://archive.ics.uci.edu/dataset/327/phishing+websites",
        # "https://archive.ics.uci.edu/dataset/342/mice+protein+expression",
        # "https://openml.org/d/42742", # too big
        "https://openml.org/d/44787",  # Porto-seguro - 0.5931 - binary classification
        "https://openml.org/d/1068",  # PC1 - 0.7764
        # "https://openml.org/d/216",
        "https://openml.org/d/37",  # Diabetes - 0.8369 - binary classification
        # "https://openml.org/d/416",
        # "https://openml.org/d/422",
        # "https://openml.org/d/42225",
        # "https://openml.org/d/42731",
        "https://openml.org/d/44712",  # Amazon - 0.6464 - binary classification
        # "https://pages.stern.nyu.edu/~jsimonof/AnalCatData/Data/",
        # "https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/",
    ]


class ModifiedCM2Benchmark(ModifiedBenchmark, CM2Benchmark):
    pass
