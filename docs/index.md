# 💠 Future Frame

Empowering Data Scientists with Foundation Models for Tabular data.

- This Python package allows you to interact with pre-trained foundation models for tabular data.
- Easily fine-tune them on your classification and regression use cases in a single line of code.
- Interested in what we're building? Join our [waitlist](https://futureframe.ai/)!

## Installation

1. Install Future Frame with `pip` – more details on our [PyPI page](https://pypi.org/project/futureframe/).

```bash
pip install futureframe
```

## Quick Start

Use Future Frame to fine-tune a pre-trained foundation model on a classification task.

```python linenums="1"
# Import standard libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Import Future Frame
import futureframe as ff

# Import data
dataset_name = "tests/data/churn.csv"
target_variable = "Churn"
df = pd.read_csv(dataset_name)

# Split data
X, y = df.drop(columns=[target_variable]), df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fine-tune a pre-trained classifier with Future Frame
model = ff.models.CM2Classifier()
model.finetune(X_train, y_train)

# Make predictions with Future Frame
y_pred = model.predict(X_test)

# Evaluate your model
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:0.2f}")
```

## Models

| Model Name | Paper Title                                                | Paper                                               | GitHub                                 |
| ---------- | ---------------------------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| CM2        | Towards Cross-Table Masked Pretraining for Web Data Mining | [Ye et al., 2024](https://arxiv.org/abs/2307.04308) | [Link](https://github.com/Chao-Ye/CM2) |

More foundation models will be integrated into the library soon. Stay stuned by joining our [waitlist](https://futureframe.ai/)!

## Links

- [Future Frame Official Website](https://futureframe.ai/)
- [Future Frame API Reference](https://futureframe.ai/api-reference/)
- [`futureframe` PyPI Page](https://pypi.python.org/pypi/futureframe)
- [`futureframe` GitHub Repository](https://github.com/futureframeai/futureframe)
- [`futureframe` Documentation](https://futureframe.ai/docs/)

## Contributing

- We are currently under heavy development.
- If you'd like to contribute, please send us an email at <i>eduardo(at)futureframe.ai</i>.
- To report a bug, please write an [issue](https://github.com/futureframeai/futureframe/issues/new).

---

Foundation models for tabular data can AI performance for Healthcare, Education, Telecommunications, Energy, and Finance industries.
