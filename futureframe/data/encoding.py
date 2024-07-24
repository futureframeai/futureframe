import logging

import numpy as np
import pandas as pd

from futureframe import config

text_encoding_models = [
    "fasttext",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-base-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5 ",
    "intfloat/multilingual-e5-large-instruct",
]


def extract_fasttext_features(data: pd.DataFrame, extract_col_name: str):
    import fasttext

    # Preliminary Settings
    lm_model = fasttext.load_model(config_directory["fasttext"])

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()

    # Entity Names
    ent_names = _clean_entity_names(data[extract_col_name])
    ent_names = list(ent_names)

    # Data Fasttext for entity names
    data_fasttext = [lm_model.get_sentence_vector(str(x)) for x in ent_names]
    data_fasttext = np.array(data_fasttext)
    data_fasttext = pd.DataFrame(data_fasttext)
    col_names = [f"X{i}" for i in range(data_fasttext.shape[1])]
    data_fasttext = data_fasttext.set_axis(col_names, axis="columns")
    data_fasttext = pd.concat([data_fasttext, data[extract_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    data_fasttext = data_fasttext.reset_index(drop=True)

    return data_fasttext


def extract_llm_features(
    data: pd.DataFrame,
    extract_col_name: str,
    device: str = "cuda:0",
):
    # Load LLM Model
    from sentence_transformers import SentenceTransformer

    lm_model = SentenceTransformer("intfloat/e5-large-v2", device=device)

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)

    # Entity Names
    ent_names = _clean_entity_names(data_[extract_col_name].copy())
    ent_names = ent_names.astype(str)
    ent_names = "query: " + ent_names  # following the outlined procedure using "query: "
    ent_names = list(ent_names)

    # Data for entity names
    embedding = lm_model.encode(ent_names, convert_to_numpy=True)
    embedding = pd.DataFrame(embedding)
    col_names = [f"X{i}" for i in range(embedding.shape[1])]
    embedding = embedding.set_axis(col_names, axis="columns")
    embedding = pd.concat([embedding, data[extract_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    embedding = embedding.reset_index(drop=True)

    return embedding


def get_text_encoding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-zh-v1.5", cache_folder=config.CACHE_ROOT)
    return model


class Cache:
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model = get_text_encoding_model(text_encoding_models[1])
    sentences = ["Hi"]
    print(model(sentences))