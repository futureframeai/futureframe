import logging

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

log = logging.getLogger(__name__)

scaler_registry = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}

baselines_registry = {
    "RandomForest": (RandomForestClassifier, RandomForestRegressor),
    "GradientBoosting": (GradientBoostingClassifier, GradientBoostingRegressor),
    "XGB": (XGBClassifier, XGBRegressor),
    "CatBoost": (CatBoostClassifier, CatBoostRegressor),
}

task_type_by_num_classes = {
    2: "classification",
    1: "regression",
}

tast_type_index = {
    "classification": 0,
    "regression": 1,
}


regressor_registry = {}


def get_task_type(num_classes: int):
    return task_type_by_num_classes.get(num_classes, "classification")


def create_baseline(predictor_name: str, task_type: str, **kwargs):
    predictor_cls = create_baseline_cls(predictor_name, task_type)
    return predictor_cls(**kwargs)


def create_baseline_cls(predictor_name: str, task_type: str):
    index = tast_type_index[task_type]
    try:
        baseline = baselines_registry[predictor_name]
    except KeyError as e:
        log.error(f"Predictor {predictor_name} not found in the registry.")
        raise e
    return baseline[index]


def create_baseline_pipeline(
    predictor,
    numerical_features,
    categorical_features,
    numerical_scaler="StandardScaler",
    **kwargs,
):
    categorical_transformer = Pipeline(
        steps=[
            ("inputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("predictor", predictor),
        ]
    )

    return pipeline