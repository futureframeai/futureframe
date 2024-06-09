# 💠 Future Frame

<p align="center">
  <i>Accelerate your data science workflow from months to days with foundation models for tabular data</i>
</p>

## Installation

```bash
pip install futureframe
```

## Quick-start guide

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import futureframe as ff

dataset_name = "tests/data/churn.csv"
target_variable = "Churn"
df = pd.read_csv(dataset_name)

X, y = df.drop(columns=[target_variable]), df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##############
# Future Frame
##############
model = ff.models.CM2Classifier()
model.finetune(X_train, y_train)

y_pred = model.predict(X_test)
##############

auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:0.2f}")
```

## Models

| Model Name | Paper Title                                                | Paper                                               | GitHub                                 |
| ---------- | ---------------------------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| CM2        | Towards Cross-Table Masked Pretraining for Web Data Mining | [Ye et al., 2024](https://arxiv.org/abs/2307.04308) | [Link](https://github.com/Chao-Ye/CM2) |

More to come!

## Important links

- [Future Frame Website](https://futureframe.ai/)
- [`futureframe` Pypi package index](https://pypi.python.org/pypi/futureframe)
- [`futureframe` Github repository](https://github.com/futureframeai/futureframe)
<!-- - [Documentation](https://futureframe.ai/docs/) -->

## Contributing

We are currently under heavy development. If you want to contribute, please send us an email at <i>eduardo(at)futureframe.ai</i>.

If you find any bugs, please write us an [issue](https://github.com/futureframeai/futureframe/issues/new) :).
