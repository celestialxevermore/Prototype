"""Fast, metadata-free replacement for table_to_embedding_carte.py.

Drop-in interface (fit / transform / fit_transform). Produces pkl with the
same dict schema as the original so downstream loaders / TabularFLM_S.py
work unchanged.

Key differences vs table_to_embedding_carte.py:
  1. skip_desc (default True):
     description / label_description JSON 을 요구하지 않음.
     TabularFLM_S.py forward 에서 cat_desc_embeddings / num_desc_embeddings /
     label_description_embeddings 는 gather 만 하고 실제로는 쓰지 않는 dead-data
     라는 사실에 근거. 0 tensor 로 채워 downstream key-presence check 통과.
  2. dedup:
     row-wise 로 같은 (feature_name, unique_value) 텍스트를 반복 encoding 하던
     걸 dataset 전체에서 유일한 텍스트 집합에 대해 한 번만 LLM forward.
     ICU eicu 기준 9M+ forward → 수백 회 수준으로 줄어듦.
  3. GPU batched encoding:
     args.encode_batch_size (default 128) 만큼 한 forward 에 태움.
     mean pooling 은 attention_mask masked mean 으로 (no-pad 상황은 기존과 동일,
     pad 상황은 올바른 값).
  4. 옵션 fp16 (args.fp16=True, cuda 일 때).

출력 key:
    cat_name_embeddings, cat_value_embeddings, cat_desc_embeddings (zeros),
    cat_desc_texts, num_name_embeddings, num_prompt_embeddings,
    num_desc_embeddings (zeros), num_desc_texts, label_description_embeddings
    (zeros if skip_desc), y, s_idx.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from transformers import (BertModel, BertTokenizer, GPT2Config, GPT2Model,
                          GPT2Tokenizer, LlamaConfig, LlamaModel,
                          LlamaTokenizer)


class Table2EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, args, source_dataset_name):
        self.args = args
        self.input_dim = args.input_dim
        self.scaler_type = args.scaler_type
        self.llm_model_name = args.llm_model
        self.source_dataset_name = source_dataset_name
        self.skip_desc = bool(getattr(args, "skip_desc", True))
        self.encode_batch_size = int(getattr(args, "encode_batch_size", 128))
        # node-drop mode: per-sample drop never-measured nodes instead of
        # median/mode-imputing them. NaN (num/cat) or count-stat==0 -> drop.
        self.drop_missing = bool(getattr(args, "drop_missing", False))
        self.count_cols_ = set()
        device_str = getattr(args, "device",
                             "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.fp16 = bool(getattr(args, "fp16", True)) and self.device.type == "cuda"

        self._load_lm_model()
        self.label_embedding = self._transform_label()

        self.cat_col_names: List[str] | None = None
        self.num_col_names: List[str] | None = None
        self.num_transformer = None
        self.is_fitted_ = False
        self.y_ = None

    # ---------- LLM loading ----------
    def _load_lm_model(self):
        name = self.llm_model_name
        if name in ("gpt2_mean", "gpt2_auto"):
            cfg = GPT2Config.from_pretrained("openai-community/gpt2")
            cfg.num_hidden_layers = 12
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2", config=cfg)
        elif name in ("LLAMA_mean", "LLAMA_auto"):
            cfg = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.llm_model = LlamaModel.from_pretrained(
                "huggyllama/llama-7b", config=cfg)
        elif name == "bio-bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1")
            self.llm_model = BertModel.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1")
        elif name == "bio-clinical-bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT")
            self.llm_model = BertModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT")
        elif name == "sentence-bert":
            from sentence_transformers import SentenceTransformer
            self.llm_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=str(self.device))
            self.tokenizer = None
        elif name == "gemma-medical":
            from sentence_transformers import SentenceTransformer
            self.llm_model = SentenceTransformer(
                "sentence-transformers/embeddinggemma-300m-medical",
                device=str(self.device),
                truncate_dim=self.input_dim)
            if self.fp16:
                # Gemma-3 is bf16-native; fp16 yields all-NaN output.
                self.llm_model = self.llm_model.to(dtype=torch.bfloat16)
            self.tokenizer = None
        else:
            raise ValueError(f"unsupported llm_model: {name}")

        if name not in ("sentence-bert", "gemma-medical"):
            for p in self.llm_model.parameters():
                p.requires_grad = False
            self.llm_model = self.llm_model.to(self.device).eval()
            if self.fp16:
                self.llm_model = self.llm_model.half()

    # ---------- sklearn interface ----------
    def fit(self, X, y=None):
        self.y_ = y

        cat_cols = list(X.select_dtypes(include="object").columns
                        .str.replace("\n", " ", regex=True))
        num_cols = list(X.select_dtypes(exclude="object").columns
                        .str.replace("\n", " ", regex=True))
        self.cat_col_names = cat_cols
        self.num_col_names = num_cols

        # count-stat columns (ICU 24h aggregation): a value of 0 literally means
        # "measured 0 times in the window" == never measured -> node dropped.
        self.count_cols_ = set(
            c for c in num_cols
            if c.endswith("_count") or c.endswith("_count_24h"))

        # NaN imputation stats: median for num, mode for cat.
        # Without this the per-row tensors had variable length and DataLoader's
        # default collate (torch.stack) blew up on mixed-NaN batches.
        self.num_medians_ = None
        if len(num_cols) > 0:
            X_num_fit = X.select_dtypes(exclude="object").copy()
            X_num_fit.columns = num_cols
            X_num_fit = X_num_fit.replace([np.inf, -np.inf], np.nan)
            self.num_medians_ = X_num_fit.median()

        self.cat_modes_ = None
        if len(cat_cols) > 0:
            X_cat_fit = X.select_dtypes(include="object").copy()
            X_cat_fit.columns = cat_cols
            modes = {}
            for c in cat_cols:
                m = X_cat_fit[c].mode(dropna=True)
                modes[c] = m.iloc[0] if len(m) > 0 else "missing"
            self.cat_modes_ = modes

        # Degenerate (constant after imputation / all-NaN) columns break
        # PowerTransformer's MLE optimization (BracketError).
        # Zero them out and exclude from the transformer.
        self.degenerate_num_cols_ = []
        if len(num_cols) > 0:
            X_num_fit = X.select_dtypes(exclude="object").copy()
            X_num_fit.columns = num_cols
            X_num_fit = X_num_fit.replace([np.inf, -np.inf], np.nan)
            X_num_fit = X_num_fit.fillna(self.num_medians_)
            nu = X_num_fit.nunique(dropna=False)
            self.degenerate_num_cols_ = nu[nu <= 1].index.tolist()

        if self.scaler_type == "pow" and len(num_cols) > 0:
            X_num_fit = X.select_dtypes(exclude="object").copy()
            X_num_fit.columns = num_cols
            X_num_fit = X_num_fit.replace([np.inf, -np.inf], np.nan)
            X_num_fit = X_num_fit.fillna(self.num_medians_)
            non_deg = [c for c in num_cols if c not in self.degenerate_num_cols_]
            if non_deg:
                self.num_transformer = PowerTransformer().set_output(transform="pandas")
                self.num_transformer.fit(X_num_fit[non_deg])
                self.num_transformer_cols_ = non_deg
            else:
                self.num_transformer = None
                self.num_transformer_cols_ = []

        return self

    def transform(self, X, y=None):
        assert self.cat_col_names is not None, "call fit() first"
        X_ = X.copy().replace("\n", " ", regex=True)
        n = X_.shape[0]

        X_cat = X_.select_dtypes(include="object").copy()
        X_cat.columns = self.cat_col_names
        X_num = X_.select_dtypes(exclude="object").copy()
        X_num.columns = self.num_col_names

        # ---- node-drop masks (from RAW values, BEFORE imputation) ----
        # present[i, j] == True  -> keep node j for row i.
        # cat: drop if NaN.  num: drop if NaN/inf, or (count-stat and value==0).
        cat_present = num_present = None
        if self.drop_missing:
            if len(self.cat_col_names) > 0:
                cat_present = (~X_cat.isna()).to_numpy()
            if len(self.num_col_names) > 0:
                _Xr = X_num.replace([np.inf, -np.inf], np.nan)
                _pres = ~_Xr.isna()
                for c in self.num_col_names:
                    if c in self.count_cols_:
                        _pres[c] = _pres[c] & (_Xr[c] != 0)
                num_present = _pres.to_numpy()

        # Impute BEFORE per-row assembly so every row has identical shape.
        if len(self.num_col_names) > 0 and self.num_medians_ is not None:
            X_num = X_num.replace([np.inf, -np.inf], np.nan)
            X_num = X_num.fillna(self.num_medians_)
            if self.num_transformer is not None:
                non_deg = self.num_transformer_cols_
                X_num_t = self.num_transformer.transform(X_num[non_deg])
                for c in self.degenerate_num_cols_:
                    X_num_t[c] = 0.0
                X_num = X_num_t[self.num_col_names]
            else:
                for c in self.degenerate_num_cols_:
                    X_num[c] = 0.0
        if len(self.cat_col_names) > 0 and self.cat_modes_ is not None:
            X_cat = X_cat.fillna(value=self.cat_modes_)

        y_ = None
        if self.y_ is not None:
            y_ = torch.tensor(np.array(self.y_)).reshape(n, 1)

        # ---------- build text list ----------
        texts: List[str] = []
        tags:  List[Tuple[str, str, str]] = []  # (kind, col, text)

        def add(kind: str, col: str, text: str):
            tags.append((kind, col, text))
            texts.append(text)

        for c in self.cat_col_names:
            add("cat_name", c, c)
            for v in X_cat[c].dropna().astype(str).unique():
                add("cat_value", c, v)

        for c in self.num_col_names:
            add("num_name", c, c)

        # ---------- dedup + batched encode ----------
        emb_all = self._batched_encode(texts)  # [len(texts), D], float32 cpu

        cat_name_emb: Dict[str, torch.Tensor] = {}
        cat_value_emb: Dict[str, Dict[str, torch.Tensor]] = {
            c: {} for c in self.cat_col_names}
        num_name_emb: Dict[str, torch.Tensor] = {}
        for (kind, col, text), emb in zip(tags, emb_all):
            if kind == "cat_name":
                cat_name_emb[col] = emb
            elif kind == "cat_value":
                cat_value_emb[col][text] = emb
            elif kind == "num_name":
                num_name_emb[col] = emb

        # ---------- per-row assembly (no LLM calls) ----------
        label_emb = self.label_embedding
        out = []
        for idx in range(n):
            data = {
                "label_description_embeddings": label_emb,
                "y": y_[idx].clone() if y_ is not None else torch.tensor([]),
                "s_idx": idx,
            }
            # per-row kept columns (node-drop: subset; else: all)
            cat_cols_here = list(self.cat_col_names)
            num_cols_here = list(self.num_col_names)
            if cat_present is not None:
                cat_cols_here = [c for k, c in enumerate(self.cat_col_names)
                                 if cat_present[idx, k]]
            if num_present is not None:
                num_cols_here = [c for k, c in enumerate(self.num_col_names)
                                 if num_present[idx, k]]
            # safety: never emit a node-less sample (would break downstream
            # collate). Fall back to full node set for a fully-missing row.
            if self.drop_missing and not cat_cols_here and not num_cols_here:
                cat_cols_here = list(self.cat_col_names)
                num_cols_here = list(self.num_col_names)

            if len(self.cat_col_names) > 0 and len(cat_cols_here) > 0:
                row = X_cat.iloc[idx]
                row = row.astype(str).str.replace("\n", " ", regex=True)
                c_name = torch.stack([cat_name_emb[c] for c in cat_cols_here], 0)
                c_val = torch.stack(
                    [cat_value_emb[c][row[c]] for c in cat_cols_here], 0)
                c_desc = torch.zeros_like(c_name)
                data.update({
                    "cat_name_embeddings":  c_name,
                    "cat_value_embeddings": c_val,
                    "cat_desc_embeddings":  c_desc,
                    "cat_desc_texts":       cat_cols_here,
                })
            if len(self.num_col_names) > 0 and len(num_cols_here) > 0:
                row = X_num.iloc[idx]
                n_name = torch.stack([num_name_emb[c] for c in num_cols_here], 0)
                x_num = torch.tensor(row[num_cols_here].values.astype("float32"))
                n_prompt = x_num.view(-1, 1) * n_name
                n_desc = torch.zeros_like(n_name)
                data.update({
                    "num_prompt_embeddings": n_prompt,
                    "num_name_embeddings":   n_name,
                    "num_desc_embeddings":   n_desc,
                    "num_desc_texts":        num_cols_here,
                })
            out.append(data)

        self.is_fitted_ = True
        return out

    # ---------- label handling ----------
    def _transform_label(self) -> torch.Tensor:
        if self.skip_desc:
            return torch.zeros(1, self.input_dim)

        metadata_path = (
            f"/storage/personal/eungyeop/dataset/feature_description/"
            f"{self.source_dataset_name}/"
            f"{self.source_dataset_name}-metadata.json")
        with open(metadata_path) as f:
            md = json.load(f)
        text = md.get("target_binary", md.get("target_multiclass", ""))
        emb = self._batched_encode([text])   # (1, D)
        return emb

    # ---------- encoder ----------
    def _batched_encode(self, texts: List[str]) -> torch.Tensor:
        """Dedup + batched LLM forward. Returns [len(texts), D] on CPU float32."""
        if len(texts) == 0:
            return torch.zeros(0, self.input_dim)

        unique: Dict[str, int] = {}
        for t in texts:
            if t not in unique:
                unique[t] = len(unique)
        unique_texts = list(unique.keys())

        if self.llm_model_name in ("sentence-bert", "gemma-medical"):
            # gemma-medical was trained with query/document prompt asymmetry;
            # feature names/values are closer to "documents" than queries.
            # sentence-transformers<3.3 has no encode_document; use prompt_name.
            kw = dict(convert_to_tensor=True,
                      batch_size=self.encode_batch_size,
                      show_progress_bar=False)
            if self.llm_model_name == "gemma-medical":
                kw["prompt_name"] = "document"
            with torch.no_grad():
                emb_unique = self.llm_model.encode(unique_texts, **kw)
            emb_unique = emb_unique.detach().float().cpu()
        else:
            all_emb = []
            bs = self.encode_batch_size
            with torch.inference_mode():
                for i in range(0, len(unique_texts), bs):
                    chunk = unique_texts[i:i + bs]
                    tok = self.tokenizer(chunk, padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)
                    out = self.llm_model(**tok).last_hidden_state  # [b, L, D]
                    mask = tok.attention_mask.unsqueeze(-1).to(out.dtype)
                    if self.llm_model_name in ("bio-bert", "bio-clinical-bert"):
                        pooled = out[:, 0, :]
                    elif self.llm_model_name in ("gpt2_auto", "LLAMA_auto"):
                        last = tok.attention_mask.sum(-1) - 1
                        pooled = out[torch.arange(out.size(0), device=out.device),
                                     last, :]
                    else:  # gpt2_mean / LLAMA_mean / default
                        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                    all_emb.append(pooled.float().cpu())
            emb_unique = torch.cat(all_emb, 0)

        D = emb_unique.size(-1)
        out_all = torch.empty(len(texts), D)
        for i, t in enumerate(texts):
            out_all[i] = emb_unique[unique[t]]
        return out_all
