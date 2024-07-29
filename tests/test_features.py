import logging

import numpy as np
import pandas as pd

from futureframe.data.features import clean_df, infer_majority_dtype

data = {
    "float_only": [1.1, 2.2, 3.3, 4.4],
    "string_only": ["a", "b", "c", "d"],
    "int_only": [1, 2, 3, 4],
    "float_with_nan": [1.1, 2.2, np.nan, 4.4],
    "int_with_nan": [1, np.nan, 3, 4],
    "string_with_nan": ["a", np.nan, "c", "d"],
    "mix_float_int": [1.1, 2, 3.3, 4],
    "mix_float_int_string": [1.1, "b", 3, 4.4],
    "boolean": [True, False, True, False],
    "datetime": pd.to_datetime(["2021-01-01", "2022-02-02", "2023-03-03", "2024-04-04"]),
}
df = pd.DataFrame(data)
df.to_csv("tests/data/mixed_types.csv", index=False)


def test_column_values_dtypes():
    print(df)
    categorized_columns, dtypes_mask = infer_majority_dtype(df)
    print(categorized_columns)
    print(dtypes_mask)
    print("===" * 79)

    df2 = pd.read_csv("tests/data/mixed_types.csv")
    print(df2)
    categorized_columns2, dtypes_mask2 = infer_majority_dtype(df2)
    print(categorized_columns2)
    print(dtypes_mask2)


def test_clean_df():
    global df
    print(df)
    df = clean_df(df)
    print(df)

    df2 = pd.read_csv("tests/data/mixed_types.csv")
    print(df2)
    df2 = clean_df(df2)
    print(df2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_column_values_dtypes()
    test_clean_df()