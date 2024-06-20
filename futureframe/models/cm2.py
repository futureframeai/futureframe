"""CM2
Authors: Chao Ye, Guoshan Lu, Haobo Wang, Liyao Li, Sai Wu, Gang Chen, Junbo Zhao

References:
- Paper: https://arxiv.org/abs/2307.04308
- Code: https://github.com/Chao-Ye/CM2

License: Apache-2.0
"""

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.init as nn_init
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from futureframe.evaluate import evaluate, get_eval_metric_fn
from futureframe.finetune import LinearWarmupScheduler, SupervisedTrainCollator
from futureframe.utils import freeze, get_activation_fn, get_parameter_names

logger = logging.getLogger(__name__)


TRAINING_ARGS_NAME = "training_args.json"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
TOKENIZER_DIR = "tokenizer"
EXTRACTOR_STATE_DIR = "extractor"
INPUT_ENCODER_NAME = "input_encoder.bin"


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, output_dir="ckpt", trace_func=print, less_is_better=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0:  # no early stop
            self.early_stop = False
            return

        if self.less_is_better:
            score = val_loss
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), os.path.join(self.path, WEIGHTS_NAME))
        self.val_loss_min = val_loss


class CM2WordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        vocab_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        vocab_freeze=False,
        use_bert=True,
        weights_dir="./weights/CM2/CM2-v1",
    ) -> None:
        super().__init__()
        weights_path = Path(weights_dir)
        word2vec_weight = torch.load(weights_path / "bert_emb.pt")
        self.word_embeddings_header = nn.Embedding.from_pretrained(
            word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx
        )
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings_value.weight)

        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        weight_emb = torch.load(weights_path / "bert_layernorm_weight.pt")
        bias_emb = torch.load(weights_path / "bert_layernorm_bias.pt")
        self.norm_header.weight.data.copy_(weight_emb)
        self.norm_header.bias.data.copy_(bias_emb)
        if vocab_freeze:
            freeze(self.norm_header)
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


class CM2NumEmbedding(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, *args, **kwargs) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0], -1, -1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb


class CM2FeatureExtractor:
    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        weights_dir="./weights/CM2/CM2-v1",
        *args,
        **kwargs,
    ) -> None:
        weights_path = Path(weights_dir)
        if os.path.exists(weights_path / "tokenizer"):
            self.tokenizer = BertTokenizerFast.from_pretrained(weights_path / "tokenizer")
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.tokenizer.save_pretrained(weights_path / "tokenizer")
        self.tokenizer.__dict__["model_max_length"] = 512
        if disable_tokenizer_parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

    def __call__(self, x, x_cat=None, table_flag=0) -> Dict:
        encoded_inputs = {
            "x_num": None,
            "num_col_input_ids": None,
            "x_cat_input_ids": None,
        }
        col_names = x.columns.tolist()
        cat_cols = (
            [c for c in col_names if c in self.categorical_columns[table_flag]]
            if self.categorical_columns[table_flag] is not None
            else []
        )
        num_cols = (
            [c for c in col_names if c in self.numerical_columns[table_flag]]
            if self.numerical_columns[table_flag] is not None
            else []
        )

        if len(cat_cols + num_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0)
            x_num_ts = torch.tensor(x_num.values, dtype=torch.float32)
            num_col_ts = self.tokenizer(
                num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt"
            )
            encoded_inputs["x_num"] = x_num_ts
            encoded_inputs["num_col_input_ids"] = num_col_ts["input_ids"]
            encoded_inputs["num_att_mask"] = num_col_ts["attention_mask"]

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_cat = x_cat.fillna("")

            x_cat_str = x_cat.values.tolist()
            encoded_inputs["x_cat_input_ids"] = []
            encoded_inputs["x_cat_att_mask"] = []
            max_y = 0
            cat_cnt = len(cat_cols)
            # max_token_len = max(int(1), int(4096/cat_cnt))
            max_token_len = max(1, int(2048 / cat_cnt))
            for sample in x_cat_str:
                x_cat_ts = self.tokenizer(
                    sample, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt"
                )
                x_cat_ts["input_ids"] = x_cat_ts["input_ids"][:, :max_token_len]
                x_cat_ts["attention_mask"] = x_cat_ts["attention_mask"][:, :max_token_len]
                encoded_inputs["x_cat_input_ids"].append(x_cat_ts["input_ids"])
                encoded_inputs["x_cat_att_mask"].append(x_cat_ts["attention_mask"])
                max_y = max(max_y, x_cat_ts["input_ids"].shape[1])
            for i in range(len(encoded_inputs["x_cat_input_ids"])):
                # tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp = torch.full((cat_cnt, max_y), self.pad_token_id, dtype=int)
                tmp[:, : encoded_inputs["x_cat_input_ids"][i].shape[1]] = encoded_inputs["x_cat_input_ids"][i]
                encoded_inputs["x_cat_input_ids"][i] = tmp
                tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp[:, : encoded_inputs["x_cat_att_mask"][i].shape[1]] = encoded_inputs["x_cat_att_mask"][i]
                encoded_inputs["x_cat_att_mask"][i] = tmp
            encoded_inputs["x_cat_input_ids"] = torch.stack(encoded_inputs["x_cat_input_ids"], dim=0)
            encoded_inputs["x_cat_att_mask"] = torch.stack(encoded_inputs["x_cat_att_mask"], dim=0)

            col_cat_ts = self.tokenizer(
                cat_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt"
            )
            encoded_inputs["col_cat_input_ids"] = col_cat_ts["input_ids"]
            encoded_inputs["col_cat_att_mask"] = col_cat_ts["attention_mask"]

        return encoded_inputs

    def save(self, path):
        save_path = os.path.join(path, EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

    def load(self, path):
        tokenizer_path = os.path.join(path, TOKENIZER_DIR)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def update(self, cat=None, num=None, bin=None):
        if cat is not None:
            self.categorical_columns = cat

        if num is not None:
            self.numerical_columns = num

        if bin is not None:
            self.binary_columns = bin


class CM2FeatureProcessor(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        vocab_dim=768,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        vocab_freeze=False,
        use_bert=True,
        pool_policy="avg",
        device="cpu",
        weights_dir="./weights/CM2/CM2-v1",
    ) -> None:
        super().__init__()
        self.word_embedding = CM2WordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            weights_dir=weights_dir,
        )
        self.num_embedding = CM2NumEmbedding(vocab_dim)

        self.align_layer = nn.Linear(vocab_dim, hidden_dim, bias=False)

        self.pool_policy = pool_policy
        self.device = device

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
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        x_cat_att_mask=None,
        col_cat_input_ids=None,
        col_cat_att_mask=None,
        **kwargs,
    ) -> Tensor:
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
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type="header")
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))

                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)

                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)

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
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type="header")
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand(
                    (x_cat_feat_embedding.shape[0], -1, -1, -1)
                )
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                bs, emb_dim = cat_feat_embedding.shape[0], cat_feat_embedding.shape[-1]
                cat_feat_embedding = cat_feat_embedding.reshape(bs, -1, emb_dim)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0], -1, -1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)
                cat_att_mask = cat_att_mask.reshape(bs, -1)

                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info["cat_bert_emb"] = x_cat_bert_embedding.detach()
        elif self.pool_policy == "max":
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(
                    num_col_input_ids.to(self.device), emb_type="header"
                )  # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._max_embedding_by_mask(num_col_emb, num_att_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                x_cat_feat_embedding = self._max_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type="header")
                cat_col_emb = self._max_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))

                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._max_embedding_by_mask(cat_feat_embedding)

                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="header")
                x_cat_bert_embedding = self._max_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)

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
                num_emb_mask = self.add_cls(num_col_emb, num_att_mask)
                num_col_emb = num_emb_mask["embedding"]
                num_att_mask = num_emb_mask["attention_mask"].to(num_col_emb.device)
                num_col_emb = self._sa_block(num_col_emb, num_att_mask)

                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)

            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type="value")
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type="header")
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand(
                    (x_cat_feat_embedding.shape[0], -1, -1, -1)
                )
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0], -1, -1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)

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
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
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
        hidden_dropout_prob=0,
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
    def __init__(
        self,
        feature_extractor,
        feature_processor,
        device="cpu",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds

    def load(self, ckpt_dir):
        self.feature_extractor.load(os.path.join(ckpt_dir, EXTRACTOR_STATE_DIR))


class CM2Model(nn.Module):
    def __init__(
        self,
        checkpoints_dir,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        device="cpu",
        vocab_freeze=False,
        use_bert=True,
        pool_policy="avg",
        **kwargs,
    ) -> None:
        super().__init__()

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.hidden_dim = hidden_dim

        if feature_extractor is None:
            feature_extractor = CM2FeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                weights_dir=checkpoints_dir,
                **kwargs,
            )

        feature_processor = CM2FeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            weights_dir=checkpoints_dir,
        )

        self.input_encoder = CM2InputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
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
        self.device = device
        self.to(device)

        # tie weights
        self.input_encoder.feature_processor.word_embedding.word_embeddings_value.weight = (
            self.input_encoder.feature_processor.word_embedding.word_embeddings_header.weight
        )

    def forward(self, x, y=None):
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        encoder_output = self.encoder(**embeded)

        return encoder_output

    def load(self, ckpt_dir):
        model_name = os.path.join(ckpt_dir, WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f"load model from {ckpt_dir}")

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, INPUT_ENCODER_NAME))

    def update(self, config):
        col_map = {}
        for k, v in config.items():
            if k in ["cat", "num", "bin"]:
                col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if "num_class" in config:
            num_class = config["num_class"]
            self.clf = CM2LinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            logger.info(f"Build a new classifier with num {num_class} classes outputs, need further finetune to work.")


class CM2LinearClassifier(nn.Module):
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


class CM2MaskToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mask_emb = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.mask_emb, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, embedding, masked_indices, header_emb):
        embedding[masked_indices.bool()] = 0
        bs, fs = embedding.shape[0], embedding.shape[1]
        all_mask_token = self.mask_emb.unsqueeze(0).unsqueeze(0).expand(bs, fs, -1) + header_emb.unsqueeze(0).expand(
            bs, -1, -1
        )
        embedding = embedding + all_mask_token * masked_indices.unsqueeze(-1)

        return embedding


class CM2Classifier(CM2Model):
    def __init__(
        self,
        checkpoint_dir="./weights/CM2/CM2-v1",
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=3,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        vocab_freeze=True,
        use_bert=True,
        pool_policy="avg",
        device=None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(
            checkpoint_dir,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
        )
        self.to(device)
        self.load(checkpoint_dir)
        self.update({"cat": categorical_columns, "num": numerical_columns, "bin": binary_columns})

    def forward(self, x, y=None, table_flag=0):
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f"CM2Classifier takes inputs with dict or pd.DataFrame, find {type(x)}.")

        outputs, _ = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)
        encoder_output = self.encoder(**outputs)  # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        if y is not None:
            # compute classification loss
            if self.num_class == 2:
                if isinstance(y, pd.Series):
                    y_ts = torch.tensor(y.values).to(self.device).float()
                else:
                    y_ts = y.float().to(self.device)
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                if isinstance(y, pd.Series):
                    y_ts = torch.tensor(y.values).to(self.device).long()
                else:
                    y_ts = y.long().to(self.device)
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss

    def finetune(
        self,
        X_train,
        y_train,
        device="cpu",
        num_epochs=50,
        batch_size=64,
        lr=1e-4,
        eval_metric="auc",
        eval_less_is_better=False,
        output_dir="./models/checkpoint-finetune",
        patience=5,
        num_workers=0,
        flag=1,
        eval_batch_size=256,
        weight_decay=0,
        warmup_ratio=0.1,
        warmup_steps=0,
        collate_fn=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=True,
        data_weight=None,
    ):
        X, y, cat_cols, num_cols, bin_cols, num_classes, num_cols_processing = preprocess(X_train, y_train)
        self.num_cols_processing = num_cols_processing

        self.num_class = num_classes
        self.clf = CM2LinearClassifier(num_class=self.num_class, hidden_dim=self.hidden_dim)
        self.clf.to(device)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        trainer = Trainer(
            self,
            (X_train, y_train),
            (X_val, y_val),
            collate_fn=collate_fn,
            output_dir=output_dir,
            num_epoch=num_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            eval_batch_size=eval_batch_size,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            balance_sample=balance_sample,
            load_best_at_last=load_best_at_last,
            ignore_duplicate_cols=ignore_duplicate_cols,
            eval_metric=eval_metric,
            eval_less_is_better=eval_less_is_better,
            num_workers=num_workers,
            regression_task=False,
            flag=flag,
            data_weight=data_weight,
            device=device,
        )

        trainer.train((X_val, y_val))

    def predict(
        self,
        x_test,
        y_test=None,
        return_loss=False,
        eval_batch_size=256,
        table_flag=0,
        regression_task=False,
    ):
        self.eval()

        if self.numerical_columns is not None and self.num_cols_processing is not None:
            x_test[self.numerical_columns] = self.num_cols_processing.transform(x_test[self.numerical_columns])

        pred_list, loss_list = [], []
        x_test = self.input_encoder.feature_extractor(x_test, table_flag=table_flag)
        if x_test["x_cat_input_ids"] is not None:
            x_len = x_test["x_cat_input_ids"].shape[0]
        else:
            x_len = x_test["x_num"].shape[0]
        for i in range(0, x_len, eval_batch_size):
            if x_test["x_cat_input_ids"] is not None:
                x_cat_input_ids = x_test["x_cat_input_ids"][i : i + eval_batch_size]
                x_cat_att_mask = x_test["x_cat_att_mask"][i : i + eval_batch_size]
                col_cat_input_ids = x_test["col_cat_input_ids"]
                col_cat_att_mask = x_test["col_cat_att_mask"]
            else:
                x_cat_input_ids = None
                x_cat_att_mask = None
                col_cat_input_ids = None
                col_cat_att_mask = None

            if x_test["x_num"] is not None:
                x_num = x_test["x_num"][i : i + eval_batch_size]
                num_col_input_ids = x_test["num_col_input_ids"]
                num_att_mask = x_test["num_att_mask"]
            else:
                x_num = None
                num_col_input_ids = None
                num_att_mask = None

            bs_x_test = {
                "x_cat_input_ids": x_cat_input_ids,
                "x_cat_att_mask": x_cat_att_mask,
                "x_num": x_num,
                "col_cat_input_ids": col_cat_input_ids,
                "col_cat_att_mask": col_cat_att_mask,
                "num_col_input_ids": num_col_input_ids,
                "num_att_mask": num_att_mask,
            }
            # bs_x_test = x_test.iloc[i:i+eval_batch_size]
            with torch.no_grad():
                logits, loss = self(bs_x_test, y_test, table_flag=table_flag)

            if loss is not None:
                loss_list.append(loss.item())

            if regression_task:
                pred_list.append(logits.detach().cpu().numpy())
            elif logits.shape[-1] == 1:  # binary classification
                pred_list.append(logits.sigmoid().detach().cpu().numpy())
            else:  # multi-class classification
                pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())
        pred_all = np.concatenate(pred_list, 0)
        if logits.shape[-1] == 1:
            pred_all = pred_all.flatten()

        if return_loss:
            avg_loss = np.mean(loss_list)
            return avg_loss
        else:
            return pred_all


class CM2LinearRegression(nn.Module):
    def __init__(self, hidden_dim=128) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activate_fn = nn.ReLU()

    def forward(self, x) -> Tensor:
        x = x[:, 0, :]  # take the cls token embedding
        x = self.activate_fn(x)
        logits = self.fc(x)
        return logits


class CM2Regression(CM2Model):
    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation="relu",
        vocab_freeze=False,
        device="cpu",
        **kwargs,
    ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            device=device,
            **kwargs,
        )
        self.res = CM2LinearRegression(hidden_dim=hidden_dim)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        if isinstance(x, dict):
            inputs = x
        elif isinstance(x, pd.DataFrame):
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f"CM2Classifier takes inputs with dict or pd.DataFrame, find {type(x)}.")

        outputs, _ = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)
        encoder_output = self.encoder(**outputs)  # bs, seqlen+1, hidden_dim

        logits = self.res(encoder_output)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1, 1)
                y = torch.tensor(y, dtype=torch.float32)
            y = y.float().to(self.device)
            y = y.reshape(-1, 1)
            loss = self.loss_fn(logits, y)
        else:
            loss = None

        return logits, loss


class Trainer:
    def __init__(
        self,
        model,
        train_set_list,
        test_set_list=None,
        collate_fn=None,
        output_dir="./ckpt",
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=True,
        eval_metric="auc",
        eval_less_is_better=False,
        num_workers=0,
        regression_task=False,
        flag=0,
        data_weight=None,
        device=None,
        **kwargs,
    ):
        self.flag = flag
        self.model = model
        self.device = device
        self.data_weight = data_weight
        if isinstance(train_set_list, tuple):
            train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple):
            test_set_list = [test_set_list]

        new_train_list = []
        new_test_list = []
        self.collate_fn = collate_fn
        self.regression_task = regression_task
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )

        self.feature_extractor = CM2FeatureExtractor(
            categorical_columns=model.categorical_columns,
            numerical_columns=model.numerical_columns,
            binary_columns=model.binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
        # prepare collate_fn for all train datasets once
        for dataindex, trainset in enumerate(train_set_list):
            x = trainset[0]
            if trainset[1] is not None:
                y = trainset[1]
            else:
                y = None
            inputs = self.feature_extractor(x, table_flag=dataindex)
            new_train_list.append((inputs, y))
        self.trainloader_list = [
            self._build_dataloader(
                (trainset, dataindex), batch_size=batch_size, collator=self.collate_fn, num_workers=num_workers
            )
            for dataindex, trainset in enumerate(new_train_list)
        ]
        # prepare collate_fn for test/val datasets once
        if test_set_list is not None:
            for dataindex, testset in enumerate(test_set_list):
                x = testset[0]
                if testset[1] is not None:
                    y = testset[1]
                else:
                    y = None
                inputs = self.feature_extractor(x, table_flag=dataindex)
                new_test_list.append((inputs, y))
            self.testloader_list = [
                self._build_dataloader(
                    (testset, dataindex),
                    batch_size=eval_batch_size,
                    collator=self.collate_fn,
                    num_workers=num_workers,
                    shuffle=False,
                )
                for dataindex, testset in enumerate(new_test_list)
            ]
        else:
            self.testloader_list = None

        self.train_set_list = new_train_list
        self.test_set_list = new_test_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(
            output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better
        )
        self.args = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epoch": num_epoch,
            "eval_batch_size": eval_batch_size,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "num_training_steps": self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            "eval_metric": get_eval_metric_fn(eval_metric),
            "eval_metric_name": eval_metric,
        }
        self.args["steps_per_epoch"] = int(self.args["num_training_steps"] / (num_epoch * len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self, eval_data=None):
        args = self.args
        self.create_optimizer()
        # self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.96)
        # self.lr_scheduler = StepLR(self.optimizer, step_size=5, gamma=0.95)
        # self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30], gamma=0.8)
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0)
        if args["warmup_ratio"] is not None or args["warmup_steps"] is not None:
            self.lr_scheduler = LinearWarmupScheduler(
                optimizer=self.optimizer,
                base_lr=self.args["lr"],
                warmup_epochs=args["warmup_steps"],
            )
            self.lr_scheduler.init_optimizer()

        start_time = time.time()
        real_res_list = []
        for epoch in tqdm(range(args["num_epoch"]), desc="Epoch"):
            ite = 0
            train_loss_all = 0
            # for all datasets
            self.model.train()
            for dataindex in range(len(self.trainloader_list)):
                # for each batch of one dataset
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()
                    for key in data[0]:
                        if isinstance(data[0][key], list):
                            for i in range(len(data[0][key])):
                                data[0][key][i] = self.change_device(data[0][key][i], self.device)
                        else:
                            data = list(data)
                            data[0] = self.change_device(data[0], self.device)
                        break
                    if data[1] is not None:
                        data[1] = torch.tensor(data[1].values).to(self.device)
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                    # print(f'{dataindex} :::: {loss.item()}')
                    loss.backward()
                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(cur_epoch=epoch)

            if self.test_set_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                # print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                if self.early_stopping(-eval_res, self.model) and eval_data:
                    if self.regression_task:
                        ypred = predict(self.model, eval_data[0], regression_task=True)
                        ans = evaluate(ypred, eval_data[1], metric="rmse")
                    else:
                        ypred = predict(self.model, eval_data[0])
                        ans = evaluate(ypred, eval_data[1], metric="auc", num_class=self.model.num_class)
                    real_res_list.append(ans[0])
                    # logging.info(f'eval_res_list: {real_res_list}')
                if self.early_stopping.early_stop:
                    logging.info("early stopped")
                    break
                logging.info(
                    "epoch: {}, train loss: {:.4f}, test {}: {:.6f}, lr: {:.6f}, spent: {:.1f} secs".format(
                        epoch,
                        train_loss_all,
                        self.args["eval_metric_name"],
                        eval_res,
                        self.optimizer.param_groups[0]["lr"],
                        time.time() - start_time,
                    )
                )
            else:
                logging.info(
                    "epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs".format(
                        epoch, train_loss_all, self.optimizer.param_groups[0]["lr"], time.time() - start_time
                    )
                )

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints
                state_dict = torch.load(os.path.join(self.output_dir, WEIGHTS_NAME), map_location="cpu")
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        logger.info(f"training complete, cost {time.time() - start_time:.1f} secs.")
        return real_res_list

    def change_device(self, data, dev):
        for key in data:
            if data[key] is not None:
                data[key] = data[key].to(dev)
        return data

    def save_epoch(self):
        save_path = "./openml_pretrain_model"
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        self.save_model(save_path)

    def evaluate(self):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in self.testloader_list[dataindex]:
                y_test.append(data[1])
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if self.regression_task:
                        pred_list.append(logits.detach().cpu().numpy())
                    elif logits.shape[-1] == 1:  # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else:  # multi-class classification
                        pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())

            if len(pred_list) > 0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args["eval_metric_name"] == "val_loss":
                eval_res = np.mean(loss_list)
            else:
                y_test = pd.concat(y_test, axis=0)
                if self.regression_task:
                    eval_res = self.args["eval_metric"](y_test, pred_all)
                else:
                    eval_res = self.args["eval_metric"](y_test, pred_all, self.model.num_class)

            eval_res_list.append(eval_res)

        return eval_res_list

    def train_no_dataloader(
        self,
        resume_from_checkpoint=None,
    ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        args = self.args
        self.create_optimizer()
        if args["warmup_ratio"] is not None or args["warmup_steps"] is not None:
            print("set warmup training.")
            self.create_scheduler(args["num_training_steps"], self.optimizer)

        for epoch in range(args["num_epoch"]):
            ite = 0
            # go through all train sets
            for train_set in self.train_set_list:
                x_train, y_train = train_set
                train_loss_all = 0
                for i in range(0, len(x_train), args["batch_size"]):
                    self.model.train()
                    if self.balance_sample:
                        bs_x_train_pos = x_train.loc[y_train == 1].sample(int(args["batch_size"] / 2))
                        bs_y_train_pos = y_train.loc[bs_x_train_pos.index]
                        bs_x_train_neg = x_train.loc[y_train == 0].sample(int(args["batch_size"] / 2))
                        bs_y_train_neg = y_train.loc[bs_x_train_neg.index]
                        bs_x_train = pd.concat([bs_x_train_pos, bs_x_train_neg], axis=0)
                        bs_y_train = pd.concat([bs_y_train_pos, bs_y_train_neg], axis=0)
                    else:
                        bs_x_train = x_train.iloc[i : i + args["batch_size"]]
                        bs_y_train = y_train.loc[bs_x_train.index]

                    self.optimizer.zero_grad()
                    logits, loss = self.model(bs_x_train, bs_y_train)
                    loss.backward()

                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set is not None:
                # evaluate in each epoch
                self.model.eval()
                x_test, y_test = self.test_set
                pred_all = predict(self.model, x_test, self.args["eval_batch_size"])
                eval_res = self.args["eval_metric"](y_test, pred_all)
                print("epoch: {}, test {}: {}".format(epoch, self.args["eval_metric_name"], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print("early stopped")
                    break

            print(
                "epoch: {}, train loss: {}, lr: {:.6f}".format(
                    epoch, train_loss_all, self.optimizer.param_groups[0]["lr"]
                )
            )

        if os.path.exists(self.output_dir):
            if self.test_set is not None:
                # load checkpoints
                print("load best at last from", self.output_dir)
                state_dict = torch.load(os.path.join(self.output_dir, WEIGHTS_NAME), map_location="cpu")
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    def save_model(self, output_dir=None):
        if output_dir is None:
            print("no path assigned for save mode, default saved to ./ckpt/model.pt !")
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logger.info(f"saving model checkpoint to {output_dir}")
        self.model.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args["weight_decay"],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args["lr"])

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args["warmup_steps"]
            if self.args["warmup_steps"] is not None
            else math.ceil(num_training_steps * self.args["warmup_ratio"])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return trainloader


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


def preprocess(X, y, auto_feature_type=None, encode_cat=False):
    target = y.name

    df = pd.concat([X, y], axis=1)

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
    all_cols = [col for col in X.columns.tolist()]
    X.columns = all_cols
    attribute_names = all_cols

    # divide cat/bin/num feature
    cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)

    # encode target label
    y = LabelEncoder().fit_transform(y.values)
    y = pd.Series(y, index=X.index, name=target)

    # start processing features
    # process num
    num_cols_processing = None
    if len(num_cols) > 0:
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        num_cols_processing = MinMaxScaler()
        X[num_cols] = num_cols_processing.fit_transform(X[num_cols])

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
    return X, y, cat_cols, num_cols, bin_cols, num_class, num_cols_processing


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


def train(
    model,
    trainset,
    valset=None,
    cmd_args=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric="auc",
    output_dir="./ckpt",
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=True,
    eval_less_is_better=False,
    flag=0,
    regression_task=False,
    train_method="normal",
    device=None,
    data_weight=None,
    **kwargs,
):
    if isinstance(trainset, tuple):
        trainset = [trainset]

    train_args = {
        "num_epoch": num_epoch,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "eval_metric": eval_metric,
        "output_dir": output_dir,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "balance_sample": balance_sample,
        "load_best_at_last": load_best_at_last,
        "ignore_duplicate_cols": ignore_duplicate_cols,
        "eval_less_is_better": eval_less_is_better,
        "flag": flag,
        "regression_task": regression_task,
        "device": device,
        "data_weight": data_weight,
    }
    trainer = Trainer(
        model,
        trainset,
        valset,
        **train_args,
    )
    return trainer


def predict(
    clf,
    x_test,
    y_test=None,
    return_loss=False,
    eval_batch_size=256,
    table_flag=0,
    regression_task=False,
):
    clf.eval()
    pred_list, loss_list = [], []
    x_test = clf.input_encoder.feature_extractor(x_test, table_flag=table_flag)
    if x_test["x_cat_input_ids"] is not None:
        x_len = x_test["x_cat_input_ids"].shape[0]
    else:
        x_len = x_test["x_num"].shape[0]
    for i in range(0, x_len, eval_batch_size):
        if x_test["x_cat_input_ids"] is not None:
            x_cat_input_ids = x_test["x_cat_input_ids"][i : i + eval_batch_size]
            x_cat_att_mask = x_test["x_cat_att_mask"][i : i + eval_batch_size]
            col_cat_input_ids = x_test["col_cat_input_ids"]
            col_cat_att_mask = x_test["col_cat_att_mask"]
        else:
            x_cat_input_ids = None
            x_cat_att_mask = None
            col_cat_input_ids = None
            col_cat_att_mask = None

        if x_test["x_num"] is not None:
            x_num = x_test["x_num"][i : i + eval_batch_size]
            num_col_input_ids = x_test["num_col_input_ids"]
            num_att_mask = x_test["num_att_mask"]
        else:
            x_num = None
            num_col_input_ids = None
            num_att_mask = None

        bs_x_test = {
            "x_cat_input_ids": x_cat_input_ids,
            "x_cat_att_mask": x_cat_att_mask,
            "x_num": x_num,
            "col_cat_input_ids": col_cat_input_ids,
            "col_cat_att_mask": col_cat_att_mask,
            "num_col_input_ids": num_col_input_ids,
            "num_att_mask": num_att_mask,
        }
        # bs_x_test = x_test.iloc[i:i+eval_batch_size]
        with torch.no_grad():
            logits, loss = clf(bs_x_test, y_test, table_flag=table_flag)

        if loss is not None:
            loss_list.append(loss.item())

        if regression_task:
            pred_list.append(logits.detach().cpu().numpy())
        elif logits.shape[-1] == 1:  # binary classification
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else:  # multi-class classification
            pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())
    pred_all = np.concatenate(pred_list, 0)
    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    if return_loss:
        avg_loss = np.mean(loss_list)
        return avg_loss
    else:
        return pred_all
