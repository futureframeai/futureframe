import datetime
import torch

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from torch.utils.data import Dataset
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)


def detect_timestamp(df, col):
    try:
        ts_min = int(float(df.loc[~(df[col] == "") & (df[col].notnull()), col].min()))
        ts_max = int(float(df.loc[~(df[col] == "") & (df[col].notnull()), col].max()))
        datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime("%Y-%m-%d %H:%M:%S")
        datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime("%Y-%m-%d %H:%M:%S")
        if (
            datetime_min > "2000-01-01 00:00:01"
            and datetime_max < "2030-01-01 00:00:01"
            and datetime_max > datetime_min
        ):
            return True
    except:
        return False


def detect_datetime(df, col):
    is_DATETIME = False
    if df[col].dtypes == object or str(df[col].dtypes) == "category":
        is_DATETIME = True
        try:
            pd.to_datetime(df[col])
        except:
            is_DATETIME = False
    return is_DATETIME


class FeatureTypeRecognition:
    def __init__(self):
        self.df = None

    def detect_TIMESTAMP(self, col):
        try:
            ts_min = int(float(self.df.loc[~(self.df[col] == "") & (self.df[col].notnull()), col].min()))
            ts_max = int(float(self.df.loc[~(self.df[col] == "") & (self.df[col].notnull()), col].max()))
            datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime("%Y-%m-%d %H:%M:%S")
            datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime("%Y-%m-%d %H:%M:%S")
            if (
                datetime_min > "2000-01-01 00:00:01"
                and datetime_max < "2030-01-01 00:00:01"
                and datetime_max > datetime_min
            ):
                return True
        except:
            return False

    def detect_DATETIME(self, col):
        is_DATETIME = False
        if self.df[col].dtypes == object or str(self.df[col].dtypes) == "category":
            is_DATETIME = True
            try:
                pd.to_datetime(self.df[col])
            except:
                is_DATETIME = False
        return is_DATETIME

    def get_data_type(self, col):
        if self.detect_DATETIME(col):
            return "cat"
        if self.detect_TIMESTAMP(col):
            return "cat"
        if self.df[col].dtypes == object or self.df[col].dtypes == bool or str(self.df[col].dtypes) == "category":
            return "cat"
        if "int" in str(self.df[col].dtype) or "float" in str(self.df[col].dtype):
            if self.df[col].nunique() < 15:
                return "cat"
            return "num"

    def fit(self, df):
        self.df = df
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if cur_type == "num":
                self.num.append(col)
            elif cur_type == "cat":
                self.cat.append(col)
            elif cur_type == "bin":
                self.bin.append(col)
            else:
                raise RuntimeError("error feature type!")
        return self.cat, self.bin, self.num


def preprocess(df, target=None, auto_feature_type=None, encode_cat=False):
    if not target:
        target = df.columns.tolist()[-1]
    if not auto_feature_type:
        auto_feature_type = FeatureTypeRecognition()

    # Delete the sample whose label count is 1 or label is nan
    count_num = list(df[target].value_counts())
    count_value = list(df[target].value_counts().index)
    delete_index = []
    for i, cnt in enumerate(count_num):
        if cnt <= 1:
            index = df.loc[df[target] == count_value[i]].index.to_list()
            delete_index.extend(index)
    df.drop(delete_index, axis=0, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, subset=[target], inplace=True)

    y = df[target]
    X = df.drop([target], axis=1)
    all_cols = [col.lower() for col in X.columns.tolist()]
    X.columns = all_cols
    attribute_names = all_cols

    # divide cat/bin/num feature
    cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)

    # encode target label
    y = LabelEncoder().fit_transform(y.values)
    y = pd.Series(y, index=X.index, name=target)

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        # BUG: this has to change
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())

    if len(bin_cols) > 0:
        for col in bin_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        X[bin_cols] = (
            X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ["yes", "true", "1", "t"] else 0).values
        )
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError("bin feature process error!")

    X = X[bin_cols + num_cols + cat_cols]

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print(
        f"# data: {len(X)}, # feat: {len(attribute_names)}, # cate: {len(cat_cols)},  # bin: {len(bin_cols)}, # numerical: {len(num_cols)}, pos rate: {(y == 1).sum() / len(y):.2f}"
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_class = len(y.value_counts())
    return X, y, cat_cols, num_cols, bin_cols, num_class


class TrainDataset(Dataset):
    def __init__(self, trainset):
        (self.x, self.y), self.table_flag = trainset

    def __len__(self):
        # return len(self.x)
        if self.x["x_num"] is not None:
            return self.x["x_num"].shape[0]
        else:
            return self.x["x_cat_input_ids"].shape[0]

    def __getitem__(self, index):
        if self.x["x_cat_input_ids"] is not None:
            x_cat_input_ids = self.x["x_cat_input_ids"][index : index + 1]
            x_cat_att_mask = self.x["x_cat_att_mask"][index : index + 1]
            col_cat_input_ids = self.x["col_cat_input_ids"]
            col_cat_att_mask = self.x["col_cat_att_mask"]
        else:
            x_cat_input_ids = None
            x_cat_att_mask = None
            col_cat_input_ids = None
            col_cat_att_mask = None

        if self.x["x_num"] is not None:
            x_num = self.x["x_num"][index : index + 1]
            num_col_input_ids = self.x["num_col_input_ids"]
            num_att_mask = self.x["num_att_mask"]
        else:
            x_num = None
            num_col_input_ids = None
            num_att_mask = None

        if self.y is not None:
            y = self.y.iloc[index : index + 1]
        else:
            y = None
        return (
            x_cat_input_ids,
            x_cat_att_mask,
            x_num,
            col_cat_input_ids,
            col_cat_att_mask,
            num_col_input_ids,
            num_att_mask,
            y,
            self.table_flag,
        )


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }
