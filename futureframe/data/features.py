import logging
import re
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from futureframe.types import ColumnDtype, TargetType, TaskType, ValueDtype, valuedtype_to_columndtype
from futureframe.utils import cast_to_ndarray, cast_to_series

warnings.filterwarnings("ignore", message="n_quantiles.*n_samples")

log = logging.getLogger(__name__)


def clean_entity_names(data):
    if not isinstance(data, str):
        return data

    data = data.replace("<", "").replace(">", "").replace("\n", "").replace("_", " ").lower()

    data = re.sub(r"[^a-z0-9\s\-\'\"]+", "", data)

    data = re.sub(r"\s+", " ", data).strip()

    return data


def remove_unique_vals_columns(df: pd.DataFrame):
    nunique = df.nunique()
    drop_columns = nunique[nunique == 1].index.tolist()
    df = df.drop(columns=drop_columns)
    return df


def clean_df(df: pd.DataFrame):
    df = remove_unique_vals_columns(df)
    df.columns = df.columns.map(clean_entity_names)
    df = df.drop_duplicates()
    return df


def get_type(val):
    # log.debug(f"{val=}")
    t = ValueDtype.OTHER.name
    if pd.isna(val):
        t = ValueDtype.NAN.name
    elif isinstance(val, bool):
        t = ValueDtype.BOOL.name
    elif pd.api.types.is_numeric_dtype(type(val)):
        fractional_parts, integral_parts = np.modf(val)
        if np.all(fractional_parts == 0):
            t = ValueDtype.INT.name
        else:
            t = ValueDtype.FLOAT.name
    else:
        conversions = [
            (int, ValueDtype.INT.name),
            (float, ValueDtype.FLOAT.name),
            (pd.to_numeric, ValueDtype.FLOAT.name),
            (pd.to_datetime, ValueDtype.DATETIME.name),
            (pd.to_timedelta, ValueDtype.TIMEDELTA.name),
        ]
        for func, t in conversions:
            try:
                val = func(val)
                break
            except Exception:
                pass
        else:
            t = ValueDtype.get(type(val).__name__).name

    # log.debug(f"{t=}")
    return t


def get_column_dtype_counts(df: pd.DataFrame):
    # Function to map values to type names, including NaNs
    dtypes = df.map(get_type)
    log.debug(f"{dtypes=}")

    # Count the occurrences of each data type in each column
    column_dtypes_count = dtypes.apply(pd.Series.value_counts).fillna(0).astype(int).to_dict()

    # Remove zero counts
    for col, dtype_counts in column_dtypes_count.items():
        column_dtypes_count[col] = {dtype: count for dtype, count in dtype_counts.items() if count > 0}

    return column_dtypes_count, dtypes


def infer_majority_dtype(df: pd.DataFrame):
    col_dtype_counts, dtypes_mask = get_column_dtype_counts(df)
    log.debug(f"{col_dtype_counts=}")

    categorized_columns = {}
    for col in df.columns:
        col_dtypes = col_dtype_counts[col]
        # most common dtype - majority vote
        majority_dtype = max(col_dtypes, key=col_dtypes.get)
        categorized_columns[col] = valuedtype_to_columndtype(ValueDtype.get(majority_dtype.upper())).name

    return categorized_columns, dtypes_mask


def numerical_quantile_transform(
    df: pd.DataFrame | pd.Series, n_quantiles: int = 100, output_distribution: Literal["uniform", "normal"] = "uniform"
):
    transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, copy=True)
    transformed_data = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed_data, index=df.index, columns=df.columns)

    return {"df": transformed_df, "numerical_transformer": transformer}


def get_numerical_transformation_fn_by_name(name: str):
    transformations = {
        "quantile": numerical_quantile_transform,
    }
    return transformations[name]


def fit_numerical_transformation(
    df: pd.DataFrame | pd.Series, numerical_transformation_name="quantile", **numerical_transform_kwargs
):
    numerical_transformation_fn = get_numerical_transformation_fn_by_name(numerical_transformation_name)
    df = df.apply(pd.to_numeric, errors="coerce")
    res = numerical_transformation_fn(df, **numerical_transform_kwargs)
    df, numerical_transformation = res["df"], res["numerical_transformer"]
    return df, numerical_transformation


def prepare_df(df: pd.DataFrame, numerical_transformation_name="quantile", **numerical_transform_kwargs):
    columns = df.columns
    categorized_columns, dtypes_mask = infer_majority_dtype(df)
    numerical_columns = [col for col, dtype in categorized_columns.items() if dtype == ColumnDtype.NUMERICAL_FLOAT.name]
    non_numerical_columns = [
        col for col, dtype in categorized_columns.items() if dtype != ColumnDtype.NUMERICAL_FLOAT.name
    ]

    # numerical transformation
    numerical_transformation = None
    if numerical_columns:
        tmp_df, numerical_transformation = fit_numerical_transformation(
            df[numerical_columns], numerical_transformation_name, **numerical_transform_kwargs
        )
        tmp_df = pd.concat([tmp_df, df[non_numerical_columns]], axis=1)
        # reorder by column
        tmp_df = tmp_df[columns]
        df = tmp_df.combine_first(df)

    df = df.fillna("missing")

    df[non_numerical_columns] = df[non_numerical_columns].astype(str)  # OK if none non-numerical columns

    return dict(
        df=df,
        columns=df.columns.tolist(),
        columns_dtypes=list(categorized_columns.values()),
        values=df.values.flatten(),
        values_dtypes=dtypes_mask.values.flatten(),
        shape=df.shape,
        numerical_columns=numerical_columns,
        numerical_transformation=numerical_transformation,
    )


def get_dtype_mask(df: pd.DataFrame, dtype):
    dtype_mask = df.map(lambda x: isinstance(x, dtype))
    return dtype_mask


def get_text_mask(df: pd.DataFrame):
    return get_dtype_mask(df, str)


def get_float_mask(df: pd.DataFrame):
    return get_dtype_mask(df, float)


def get_non_nan_values_indices(df: pd.DataFrame):
    mask_values = df.values
    mask_shape = mask_values.shape
    mask_flattened = mask_values.flatten()

    non_nan_indices = np.where(~pd.isna(mask_flattened))[0]
    non_nan_values = mask_flattened[non_nan_indices]
    return {
        "values": non_nan_values,
        "indices": non_nan_indices,
        "shape": mask_shape,
        "index": df.index,
        "columns": df.columns,
    }


def reconstruct_df(non_nan_values, non_nan_indices, shape, index, columns):
    mask_transformed = np.full(shape, np.nan, dtype=object)
    mask_transformed.flat[non_nan_indices] = non_nan_values
    mask_df_transformed = pd.DataFrame(mask_transformed, index=index, columns=columns)
    return mask_df_transformed


def from_flat_indices_to_column_names(indices, columns):
    column_names = []
    num_columns = len(columns)

    for idx in indices:
        col_idx = idx % num_columns
        if 0 <= col_idx < num_columns:
            column_names.append(columns[col_idx])
        else:
            column_names.append(None)  # Placeholder for out-of-range indices
    return column_names


def from_flat_indices_to_column_idx(indices, columns):
    column_idxs = []
    num_columns = len(columns)

    for idx in indices:
        col_idx = idx % num_columns
        if 0 <= col_idx < num_columns:
            column_idxs.append(col_idx)
        else:
            column_idxs.append(None)  # Placeholder for out-of-range indices
    return np.array(column_idxs)


def extract_target_variable(df: pd.DataFrame, target: Optional[str] = None):
    """
    Splits a DataFrame into features (X) and target variable (y).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The name of the target variable column.

    Returns:
    tuple: A tuple containing the features DataFrame (X) and the target Series (y).

    Examples:
    ```python
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
        X, y = split_target_variable(df, 'target')
        print(X)
        print(y)
    ```
    """
    if target is None:
        target = df.columns.tolist()[-1]

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame columns.")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def get_num_classes_classification(y):
    if not isinstance(y, pd.Series):
        y = cast_to_series(y)
    num_classes = len(y.value_counts())
    return num_classes


def get_num_classes(y: TargetType, task_type: Optional[TaskType] = None):
    if task_type is not None:
        if task_type == "classification" or task_type == "multiclass_classification":
            return get_num_classes_classification(y)
        if task_type == "regression":
            return 1

    y = cast_to_series(y)
    log.debug(f"{y=}")
    # Check it dtype is numeric and float
    log.debug(f"{pd.api.types.is_float_dtype(y)=}")
    if pd.api.types.is_numeric_dtype(y):
        fractional_parts, integral_parts = np.modf(y)
        # if all fractional parts are zero, then it is an integer
        if np.all(fractional_parts == 0):
            n_unique = y.nunique()
            if n_unique == 2:
                return 2
            if n_unique > 0.1 * len(y):
                # regression heuristics: if more than 10% of the values are int and unique, it is regression
                return 1
            # if not y.min() == 0 or y.min() == 1:
            #     return 1
            return get_num_classes_classification(y.astype(int))
        return 1
    return get_num_classes_classification(y)


def encode_target_label(y: pd.Series, return_le=False):
    # encode target label
    y = cast_to_series(y)
    name = y.name
    index = y.index
    le = LabelEncoder().fit(y)
    y = le.transform(y.values)
    y = pd.Series(y, index=index, name=name)
    if return_le:
        return y, le
    return y


def prepare_target_for_eval(y: TargetType, num_classes: Optional[int] = None) -> np.ndarray:
    y = cast_to_ndarray(y).ravel()
    if num_classes is None:
        num_classes = get_num_classes(y)

    if num_classes >= 2:
        le = LabelEncoder()
        y = le.fit_transform(y)

    return y.reshape(-1)


def prepare_pred_for_eval(y: TargetType, num_classes: Optional[int] = None) -> np.ndarray:
    y = cast_to_ndarray(y)
    return y


def get_batch_size(max_num_cells, num_cols):
    return max_num_cells // num_cols
