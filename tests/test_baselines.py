import logging

import pandas as pd
from sklearn.model_selection import train_test_split

import futureframe as ff

logging.basicConfig(level=logging.DEBUG)


def test_xgboost_classifier():
    data = pd.read_csv("tests/data/titanic.csv")
    X = data.drop(columns=["Survived"])
    y = data["Survived"]
    num_classes = ff.features.get_num_classes(y)
    task_type = ff.baselines.get_task_type(num_classes)
    print(f"{num_classes=}")
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    model = ff.baselines.create_baseline("XGB", task_type=task_type)
    model_pipeline = ff.baselines.create_baseline_pipeline(
        model, numerical_features=numerical_features, categorical_features=categorical_features, task_type=task_type
    )
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    y_pred_prob = model_pipeline.predict_proba(X_test)

    print(y_pred, y_pred_prob)

    results = ff.evaluate.eval(y_test, y_pred_prob, is_prob=True)
    print(results)

def test_xgboost_regressor(y_column_name="Fare"):
    data = pd.read_csv("tests/data/titanic.csv")
    X = data.drop(columns=[y_column_name])
    y = data[y_column_name]
    num_classes = ff.features.get_num_classes(y)
    task_type = ff.baselines.get_task_type(num_classes)
    print(f"{num_classes=}")
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    model = ff.baselines.create_baseline("XGB", task_type=task_type)
    model_pipeline = ff.baselines.create_baseline_pipeline(
        model, numerical_features=numerical_features, categorical_features=categorical_features, task_type=task_type
    )
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    if task_type == "classification":
        y_pred = model_pipeline.predict_proba(X_test)
    else:
        y_pred = model_pipeline.predict(X_test)

    results = ff.evaluate.eval(y_test, y_pred, is_prob=True)
    print(results)


if __name__ == "__main__":
    # test_xgboost_classifier()
    test_xgboost_regressor()
