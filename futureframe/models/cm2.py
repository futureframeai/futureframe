"""CM2
Authors: Chao Ye, Guoshan Lu, Haobo Wang, Liyao Li, Sai Wu, Gang Chen, Junbo Zhao

Modified by: Eduardo Dadalto

References:
- Paper: https://arxiv.org/abs/2307.04308
- Code: https://github.com/Chao-Ye/CM2

License: Apache-2.0
"""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor, nn
from transformers import BertTokenizerFast

from futureframe import config
from futureframe.data.encoding import BaseFeaturesToModelInput
from futureframe.data.features import infer_majority_dtype
from futureframe.models.base import BaseModelForFinetuning
from futureframe.types import BaseInput, ColumnDtype
from futureframe.utils import (
    download_and_extract,
    get_activation_fn,
    send_to_device_recursively,
)

log = logging.getLogger(__name__)


@dataclass
class CM2EncodedInputs(BaseInput):
    """
    Represents the encoded inputs for the CM2 model.

    Attributes:
        x_num (Tensor): The numerical input tensor.
        num_col_input_ids (Tensor): The input IDs for numerical column tokens.
        num_col_attn_mask (Tensor): The attention mask for numerical column tokens.
        x_cat_input_ids (Tensor): The input IDs for categorical column tokens.
        x_cat_attn_mask (Tensor): The attention mask for categorical column tokens.
        cat_col_input_ids (Tensor): The input IDs for categorical column tokens.
        cat_col_attn_mask (Tensor): The attention mask for categorical column tokens.
    """

    x_num: Tensor
    num_col_input_ids: Tensor
    num_col_attn_mask: Tensor
    x_cat_input_ids: Tensor
    x_cat_attn_mask: Tensor
    cat_col_input_ids: Tensor
    cat_col_attn_mask: Tensor


class CM2FeaturesToModelInput(BaseFeaturesToModelInput):
    """Class for encoding features to CM2 model input"""

    def __init__(
        self,
        weights_dir=os.path.join(config.CACHE_ROOT, "cm2"),
        download=True,
        disable_tokenizer_parallel=False,
        categorical_columns=None,
        numerical_columns=None,
    ):
        """
        Initialize the CM2FeaturesToModelInput class.

        Args:
            weights_dir (str, optional): Directory to store the weights. Defaults to os.path.join(config.CACHE_ROOT, "cm2").
            download (bool, optional): Whether to download the tokenizer weights. Defaults to True.
            disable_tokenizer_parallel (bool, optional): Whether to disable tokenizer parallelism. Defaults to False.
            categorical_columns (list, optional): List of categorical column names. Defaults to None.
            numerical_columns (list, optional): List of numerical column names. Defaults to None.
        """
        tokenizer_dir = os.path.join(weights_dir, "tokenizer")
        if download:
            self.download(tokenizer_dir)
            self.save(tokenizer_dir)
        else:
            self.load(tokenizer_dir)

        self.tokenizer.__dict__["model_max_length"] = 512

        if disable_tokenizer_parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def download(self, path):
        """
        Download the tokenizer weights.

        Args:
            path (str): Directory to save the tokenizer weights.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def load(self, path):
        """
        Load the tokenizer weights from a directory.

        Args:
            path (str): Directory containing the tokenizer weights.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(path)

    def save(self, path):
        """
        Save the tokenizer weights to a directory.

        Args:
            path (str): Directory to save the tokenizer weights.
        """
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def _get_columns_types(categorized_columns: dict):
        """
        Get the types of columns (numerical or categorical) based on a dictionary of categorized columns.

        Args:
            categorized_columns (dict): Dictionary mapping column names to their categorized types.

        Returns:
            tuple: A tuple containing two lists - numerical_columns and categorical_columns.
        """
        numerical_columns = []
        categorical_columns = []
        for k, v in categorized_columns.items():
            if ColumnDtype.get(v) in [ColumnDtype.NUMERICAL_FLOAT, ColumnDtype.NUMERICAL_INT]:
                numerical_columns.append(k)
            else:
                categorical_columns.append(k)

        return numerical_columns, categorical_columns

    def tokenize(self, data):
        """
        Tokenize the input data using the tokenizer.

        Args:
            data (list): List of input data to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized input data.
        """
        log.debug(f"Tokenizing data: {data=}")
        if len(data) == 0:
            input_ids = torch.tensor([], dtype=torch.long).reshape(0, 0)
            attention_mask = torch.tensor([], dtype=torch.long).reshape(0, 0)
            out = dict(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.tokenizer(
                data, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt", max_length=512
            )
        return out

    @torch.no_grad()
    def encode(self, data: pd.DataFrame) -> dict[str, Tensor]:
        """
        Encode the training data into CM2 model inputs.

        Args:
            data (pd.DataFrame): Training data to be encoded.

        Returns:
            CM2EncodedInputs: Encoded inputs for the CM2 model.
        """
        if self.categorical_columns is None or self.numerical_columns is None:
            categorized_columns, _ = infer_majority_dtype(data)
            log.debug(f"{categorized_columns=}")

            self.numerical_columns, self.categorical_columns = self._get_columns_types(categorized_columns)
            log.debug(f"{self.numerical_columns=}")
            log.debug(f"{self.categorical_columns=}")

        # tokenize columns
        num_col_t = self.tokenize(self.numerical_columns)
        num_col_input_ids = num_col_t["input_ids"]
        num_col_att_mask = num_col_t["attention_mask"]

        cat_col_t = self.tokenize(self.categorical_columns)
        cat_col_input_ids = cat_col_t["input_ids"]
        cat_col_att_mask = cat_col_t["attention_mask"]

        # tokenize categorical values
        x_cat = data[self.categorical_columns].astype(str)
        x_cat = x_cat.fillna("missing")
        x_cat_shape = x_cat.shape
        log.debug(f"{x_cat_shape=}")
        x_cat = x_cat.values.reshape(-1)
        x_cat_t = self.tokenize(x_cat.tolist())
        x_cat_input_ids = x_cat_t["input_ids"].reshape(x_cat_shape[0], x_cat_shape[1], -1)
        x_cat_attn_mask = x_cat_t["attention_mask"].reshape(x_cat_shape[0], x_cat_shape[1], -1)
        log.debug(f"{x_cat_input_ids.shape=}")

        # preprocess numerical values
        x_num = data[self.numerical_columns].apply(pd.to_numeric, errors="coerce")
        x_num = x_num.apply(lambda x: x.fillna(x.median()), axis=0)
        x_num = torch.from_numpy(x_num.values).float()

        return CM2EncodedInputs(
            x_num=x_num,
            num_col_input_ids=num_col_input_ids,
            num_col_attn_mask=num_col_att_mask,
            cat_col_input_ids=cat_col_input_ids,
            cat_col_attn_mask=cat_col_att_mask,
            x_cat_input_ids=x_cat_input_ids,
            x_cat_attn_mask=x_cat_attn_mask,
        ).to_dict()


class CM2WordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        vocab_dim=768,
        padding_idx=0,
        hidden_dropout_prob: float = 0,
        layer_norm_eps=1e-5,
        vocab_freeze=True,
        download=False,
        weights_dir=os.path.join(config.CACHE_ROOT, "cm2"),
    ) -> None:
        super().__init__()
        if download:
            self.download(weights_dir)

        weights_path = Path(weights_dir)
        word2vec_weight = torch.load(weights_path / "bert_emb.pt")
        self.word_embeddings_header = nn.Embedding.from_pretrained(
            word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx
        )
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)

        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        weight_emb = torch.load(weights_path / "bert_layernorm_weight.pt")
        bias_emb = torch.load(weights_path / "bert_layernorm_bias.pt")
        self.norm_header.weight.data.copy_(weight_emb)
        self.norm_header.bias.data.copy_(bias_emb)

        self.norm_value = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, emb_type) -> Tensor:
        if emb_type == "header":
            embeddings = self.word_embeddings_header(input_ids)
            embeddings = self.norm_header(embeddings)
        elif emb_type == "value":
            embeddings = self.word_embeddings_value(input_ids)
            embeddings = self.norm_value(embeddings)
        else:
            raise RuntimeError(f"no {emb_type} word_embedding method!")

        embeddings = self.dropout(embeddings)
        return embeddings

    def download(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        if not os.path.isfile(os.path.join(ckpt_dir, "bert_emb.pt")):
            download_and_extract(
                "https://github.com/futureframeai/futureframe/releases/download/cm2_weights/cm2.zip",
                ckpt_dir,
            )


class CM2NumEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0], -1, -1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb


class CM2FeatureProcessor(nn.Module):
    def __init__(
        self,
        vocab_dim=768,
        hidden_dim=128,
        hidden_dropout_prob: float = 0,
        pad_token_id=0,
        vocab_freeze=False,
        pool_policy="avg",
        weights_dir=os.path.join(config.CACHE_ROOT, "cm2"),
    ) -> None:
        super().__init__()
        self.word_embedding = CM2WordEmbedding(
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
            vocab_freeze=vocab_freeze,
            weights_dir=weights_dir,
        )
        self.num_embedding = CM2NumEmbedding(vocab_dim)
        self.align_layer = nn.Linear(vocab_dim, hidden_dim, bias=False)
        self.pool_policy = pool_policy

    def _avg_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is None:
            return embs.mean(-2)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(-2) / (att_mask.sum(-1, keepdim=True).to(embs.device) + eps)
            return embs

    def _max_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is not None:
            embs[att_mask == 0] = -1e12
        embs = torch.max(embs, dim=-2)[0]
        return embs

    def _sa_block(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x[:, 0, :]

    def _check_nan(self, value):
        return torch.isnan(value).any().item()

    def forward(
        self,
        x_num: Optional[Tensor] = None,
        num_col_input_ids: Optional[Tensor] = None,
        num_col_attn_mask: Optional[Tensor] = None,
        x_cat_input_ids: Optional[Tensor] = None,
        x_cat_attn_mask: Optional[Tensor] = None,
        cat_col_input_ids: Optional[Tensor] = None,
        cat_col_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.device = x_num.device

        num_feat_embedding = None
        cat_feat_embedding = None
        other_info = {
            "col_emb": None,  # [num_fs+cat_fs]
            "num_cnt": 0,  # num_fs
            "x_num": x_num,  # [bs, num_fs]
            "cat_bert_emb": None,  # [bs, cat_fs, dim]
        }

        if other_info["x_num"] is not None:
            other_info["x_num"] = other_info["x_num"].to(self.device)

        if self.pool_policy == "avg":
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type="header")
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_col_attn_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, x_cat_attn_mask)
                col_cat_feat_embedding = self.word_embedding(cat_col_input_ids.to(self.device), emb_type="header")
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, cat_col_attn_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))

                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)

                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_attn_mask)

                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)

                other_info["cat_bert_emb"] = x_cat_bert_embedding.detach()
        elif self.pool_policy == "no":
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(
                    num_col_input_ids.to(self.device), emb_type="header"
                )  # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_col_attn_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                col_cat_feat_embedding = self.word_embedding(cat_col_input_ids.to(self.device), emb_type="header")
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand(
                    (x_cat_feat_embedding.shape[0], -1, -1, -1)
                )
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                bs, emb_dim = cat_feat_embedding.shape[0], cat_feat_embedding.shape[-1]
                cat_feat_embedding = cat_feat_embedding.reshape(bs, -1, emb_dim)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                # mask
                cat_col_attn_mask = cat_col_attn_mask.unsqueeze(0).expand((x_cat_attn_mask.shape[0], -1, -1))
                cat_att_mask = torch.cat((cat_col_attn_mask, x_cat_attn_mask), dim=-1)
                cat_att_mask = cat_att_mask.reshape(bs, -1)

                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_attn_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info["cat_bert_emb"] = x_cat_bert_embedding.detach()
        elif self.pool_policy == "max":
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(
                    num_col_input_ids.to(self.device), emb_type="header"
                )  # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._max_embedding_by_mask(num_col_emb, num_col_attn_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                x_cat_feat_embedding = self._max_embedding_by_mask(x_cat_feat_embedding, x_cat_attn_mask)
                col_cat_feat_embedding = self.word_embedding(cat_col_input_ids.to(self.device), emb_type="header")
                cat_col_emb = self._max_embedding_by_mask(col_cat_feat_embedding, cat_col_attn_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))

                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._max_embedding_by_mask(cat_feat_embedding)

                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._max_embedding_by_mask(x_cat_bert_embedding, x_cat_attn_mask)

                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)

                other_info["cat_bert_emb"] = x_cat_bert_embedding.detach()
        elif self.pool_policy == "self-attention":
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(
                    num_col_input_ids.to(self.device), emb_type="header"
                )  # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_emb_mask = self.add_cls(num_col_emb, num_col_attn_mask)
                num_col_emb = num_emb_mask["embedding"]
                num_col_attn_mask = num_emb_mask["attention_mask"].to(num_col_emb.device)
                num_col_emb = self._sa_block(num_col_emb, num_col_attn_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                col_cat_feat_embedding = self.word_embedding(cat_col_input_ids.to(self.device), emb_type="header")
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand(
                    (x_cat_feat_embedding.shape[0], -1, -1, -1)
                )
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                # mask
                cat_col_attn_mask = cat_col_attn_mask.unsqueeze(0).expand((x_cat_attn_mask.shape[0], -1, -1))
                cat_att_mask = torch.cat((cat_col_attn_mask, x_cat_attn_mask), dim=-1)

                bs, fs, ls = cat_feat_embedding.shape[0], cat_feat_embedding.shape[1], cat_feat_embedding.shape[2]
                cat_feat_embedding = cat_feat_embedding.reshape(bs * fs, ls, -1)
                cat_att_mask = cat_att_mask.reshape(bs * fs, ls)
                cat_embedding_mask = self.add_cls(cat_feat_embedding, cat_att_mask)
                cat_feat_embedding = cat_embedding_mask["embedding"]
                cat_att_mask = cat_embedding_mask["attention_mask"].to(cat_feat_embedding.device)
                cat_feat_embedding = self._sa_block(cat_feat_embedding, cat_att_mask).reshape(bs, fs, -1)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_attn_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info["cat_bert_emb"] = x_cat_bert_embedding.detach()
        else:
            raise RuntimeError(f"no such {self.pool_policy} pooling policy!!!")

        emb_list = []
        att_mask_list = []
        col_emb = []
        if num_feat_embedding is not None:
            col_emb += [num_col_emb]
            other_info["num_cnt"] = num_col_emb.shape[0]
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device)]

        if cat_feat_embedding is not None:
            col_emb += [cat_col_emb]
            emb_list += [cat_feat_embedding]
            if self.pool_policy == "no":
                att_mask_list += [cat_att_mask.to(self.device)]
            else:
                att_mask_list += [torch.ones(cat_feat_embedding.shape[0], cat_feat_embedding.shape[1]).to(self.device)]

        if len(emb_list) == 0:
            raise Exception("no feature found belonging into numerical, categorical, or binary, check your data!")
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        other_info["col_emb"] = torch.cat(col_emb, 0).float()
        return {"embedding": all_feat_embedding, "attention_mask": attention_mask}, other_info


class CM2TransformerLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        device=None,
        dtype=None,
        use_layer_norm=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, *args, **kwargs) -> Tensor:
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:  # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        return x


class CM2Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob: float = 0,
        ffn_dim=256,
        activation="relu",
    ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
                CM2TransformerLayer(
                    d_model=hidden_dim,
                    nhead=num_attention_head,
                    dropout=hidden_dropout_prob,
                    dim_feedforward=ffn_dim,
                    batch_first=True,
                    layer_norm_eps=1e-5,
                    norm_first=False,
                    use_layer_norm=True,
                    activation=activation,
                )
            ]
        )
        if num_layer > 1:
            encoder_layer = CM2TransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
            )
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer - 1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        outputs = embedding
        for _, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs


class CM2InputEncoder(nn.Module):
    def __init__(self, feature_extractor, feature_processor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized.to_dict())
        return embeds


class CM2CLSToken(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {"embedding": embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1).to(attention_mask.device), attention_mask], 1
            )
        outputs["attention_mask"] = attention_mask
        return outputs


class CM2Model(nn.Module):
    def __init__(
        self,
        download=True,
        load_pre_trained=True,
        checkpoints_dir=config.CACHE_ROOT,
        categorical_columns=None,
        numerical_columns=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        vocab_freeze=False,
        pool_policy="avg",
    ) -> None:
        super().__init__()

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.hidden_dim = hidden_dim

        if download:
            self.download(checkpoints_dir)

        self.input_encoder = CM2FeaturesToModelInput(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            weights_dir=checkpoints_dir,
        )

        self.feature_processor = CM2FeatureProcessor(
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_freeze=vocab_freeze,
            pool_policy=pool_policy,
            weights_dir=checkpoints_dir,
        )

        self.encoder = CM2Encoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        self.cls_token = CM2CLSToken(hidden_dim=hidden_dim)

        if load_pre_trained:
            self.load(checkpoints_dir)

    def forward(self, x):
        if isinstance(x, pd.DataFrame):
            device = next(self.parameters()).device
            # input is the pre-tokenized encoded inputs
            x = self.input_encoder(x)
            x = send_to_device_recursively(x, device)

        embeded, _ = self.feature_processor(**x)
        embeded = self.cls_token(**embeded)
        encoder_output = self.encoder(**embeded)

        return encoder_output

    def download(self, ckpt_dir):
        if not os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin")):
            os.makedirs(ckpt_dir, exist_ok=True)
            download_and_extract(
                "https://github.com/futureframeai/futureframe/releases/download/cm2_weights/cm2.zip",
                ckpt_dir,
            )

    def load(self, ckpt_dir):
        model_name = os.path.join(ckpt_dir, "pytorch_model.bin")
        state_dict = torch.load(model_name, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        log.info(f"loaded pre-trained model weights from {ckpt_dir}")
        log.info(f"with missing keys: {missing_keys} and unexpected keys: {unexpected_keys}")

        # load feature extractor
        self.input_encoder.load(os.path.join(ckpt_dir, "tokenizer"))

    def save(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))


class CM2LinearClassifierHead(nn.Module):
    def __init__(self, num_class, hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:, 0, :]  # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits


class CM2Classifier(CM2Model, BaseModelForFinetuning):
    """
    CM2Classifier is a classifier model based on the CM2Model architecture.

    Args:
        download (bool, optional): Whether to download pre-trained weights if not already available. Defaults to True.
        load_pre_trained (bool, optional): Whether to load pre-trained weights if available. Defaults to True.
        checkpoint_dir (str, optional): Directory to save/load checkpoints. Defaults to os.path.join(config.CACHE_ROOT, "cm2").
        categorical_columns (list, optional): List of categorical column names. Defaults to None.
        numerical_columns (list, optional): List of numerical column names. Defaults to None.
        num_class (int, optional): Number of classes. Defaults to 2.
        hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to 128.
        num_layer (int, optional): Number of layers. Defaults to 3.
        num_attention_head (int, optional): Number of attention heads. Defaults to 8.
        hidden_dropout_prob (float, optional): Dropout probability for the hidden layers. Defaults to 0.1.
        ffn_dim (int, optional): Dimensionality of the feed-forward network. Defaults to 256.
        activation (str, optional): Activation function to use. Defaults to "relu".
        vocab_freeze (bool, optional): Whether to freeze the vocabulary during training. Defaults to True.
        pool_policy (str, optional): Pooling policy for the final output. Defaults to "avg".
    """

    def __init__(
        self,
        download=True,
        load_pre_trained=True,
        checkpoint_dir=os.path.join(config.CACHE_ROOT, "cm2"),
        categorical_columns=None,
        numerical_columns=None,
        num_class=2,
        hidden_dim=128,
        num_layer=3,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        vocab_freeze=True,
        pool_policy="avg",
    ) -> None:
        super().__init__(
            download=download,
            load_pre_trained=load_pre_trained,
            checkpoints_dir=checkpoint_dir,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            pool_policy=pool_policy,
        )

        self.head = CM2LinearClassifierHead(num_class=num_class, hidden_dim=self.hidden_dim)

    def forward(self, x):
        x = super().forward(x)
        logits = self.head(x)
        return logits


class CM2LinearRegressionHead(CM2LinearClassifierHead):
    def __init__(self, hidden_dim=128) -> None:
        super().__init__(1, hidden_dim)


class CM2Regression(CM2Model, BaseModelForFinetuning):
    """
    CM2Regression is a class that represents a regression model based on the CM2 architecture.

    Args:
        download (bool, optional): Whether to download pre-trained weights. Defaults to True.
        load_pre_trained (bool, optional): Whether to load pre-trained weights. Defaults to True.
        checkpoint_dir (str, optional): Directory to save/load checkpoints. Defaults to os.path.join(config.CACHE_ROOT, "cm2").
        categorical_columns (list, optional): List of categorical column names. Defaults to None.
        numerical_columns (list, optional): List of numerical column names. Defaults to None.
        hidden_dim (int, optional): Dimension of the hidden layers. Defaults to 128.
        num_layer (int, optional): Number of layers. Defaults to 2.
        num_attention_head (int, optional): Number of attention heads. Defaults to 8.
        hidden_dropout_prob (float, optional): Dropout probability for hidden layers. Defaults to 0.
        ffn_dim (int, optional): Dimension of the feed-forward network. Defaults to 256.
        activation (str, optional): Activation function for the hidden layers. Defaults to "relu".
        vocab_freeze (bool, optional): Whether to freeze the vocabulary. Defaults to False.
    """

    def __init__(
        self,
        download=True,
        load_pre_trained=True,
        checkpoint_dir=os.path.join(config.CACHE_ROOT, "cm2"),
        categorical_columns=None,
        numerical_columns=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation="relu",
        vocab_freeze=False,
    ) -> None:
        super().__init__(
            download=download,
            load_pre_trained=load_pre_trained,
            checkpoints_dir=checkpoint_dir,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
        )
        self.head = CM2LinearRegressionHead(hidden_dim=hidden_dim)

    def forward(self, x):
        x = super().forward(x)
        logits = self.head(x)
        return logits
