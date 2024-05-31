import logging

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)


def create_baseline(predictor_name: str, task_type: str, numeric_features, categorical_features, **kwargs):
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    if predictor_name == "RandomForest":
        if task_type == "regression":
            p = RandomForestRegressor(**kwargs)
        else:
            p = RandomForestClassifier(**kwargs)

    elif predictor_name == "GradientBoosting":
        if task_type == "regression":
            p = GradientBoostingRegressor(**kwargs)
        else:
            p = GradientBoostingClassifier(**kwargs)

    else:
        raise ValueError(f"Predictor {predictor_name} not supported")

    predictor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("predictor", p),
        ]
    )

    return predictor
