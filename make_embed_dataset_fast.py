"""Fast version of make_embed_dataset.py.

기존 make_embed_dataset.py + table_to_embedding_carte.py 는 row 마다 LLM
forward 를 다시 태워서 큰 dataset (eicu 114k+) 기준 시간이 매우 오래 걸림.
이 버전은 table_to_embedding_carte_fast.Table2EmbeddingTransformer 를 써서
(col, unique_value) 단위 dedup + GPU batched encoding 으로 한 dataset 을
수분 내에 끝낸다.

기본 출력 경로 (기존과 충돌 회피):
  /storage/personal/eungyeop/dataset/embedding/
      tabular_embeddings_carte_fast/{llm_model}/embedding_{dataset}.pkl

기존 make_embed_dataset.py 는 그대로 두고 side-by-side 로 비교 가능.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from table_to_embedding_carte_fast import Table2EmbeddingTransformer  # noqa: E402


DATASET_AND_CLASS = {
    "heart":         ["target_binary", ["no", "yes"]],
    "heart_target_1": ["target_binary", ["no", "yes"]],
    "heart_target_2": ["target_binary", ["no", "yes"]],
    "heart_target_3": ["target_binary", ["no", "yes"]],
    "heart_target_4": ["target_binary", ["no", "yes"]],
    "cleveland":     ["target_binary", ["no", "yes"]],
    "credit-g":      ["target_binary", ["no", "yes"]],
    "hungarian":     ["target_binary", ["no", "yes"]],
    "blood":         ["target_binary", ["no", "yes"]],
    "switzerland":   ["target_binary", ["no", "yes"]],
    "heart_statlog": ["target_binary", ["no", "yes"]],
    "adult":         ["target_binary", ["no", "yes"]],
    "diabetes":      ["target_binary", ["no", "yes"]],
    "breast":        ["target_binary", [0, 1]],
    "higgs":         ["target_binary", [0, 1]],
    "bank":          ["target_binary", ["no", "yes"]],
    "higgs_sampled": ["target_binary", [0.0, 1.0]],
    "magic_telescope": ["target_binary", ["g", "h"]],
    "car":           ["target_multiclass",
                      ["unacceptable", "acceptable", "very good", "good"]],
    "forest_covertype_sampled": ["target_multiclass", [1, 2, 3, 4, 5, 6, 7]],
    "communities":   ["target_multiclass", ["medium", "high", "low"]],
    "Cardiovascular_Disease_Dataset": ["target_binary", ["no", "yes"]],
    "Medicaldataset": ["target_binary", ["no", "yes"]],
    "heart1":        ["target_binary", ["no", "yes"]],
    "Heart_disease_statlog": ["target_binary", ["no", "yes"]],
    "heart_disease": ["target_binary", ["no", "yes"]],
    "heart_disease_clean": ["target_binary", ["no", "yes"]],
    "mimic_mortality":   ["label", [0, 1]],
    "eicu_mortality":    ["label", [0, 1]],
    "hirid_mortality":   ["label", [0, 1]],
    "support_mortality": ["label", [0, 1]],
    "zigong_mortality":  ["label", [0, 1]],
    "sic_mortality":     ["label", [0, 1]],
}

# mirror table target-col rename logic from the original script
_TARGET_COL_RENAME = {
    "adult": ("income", "target_binary"),
    "diabetes": ("Outcome", "target_binary"),
    "breast": ("target", "target_binary"),
    "bank": ("Class", "target_binary"),
    "higgs_sampled": ("target", "target_binary"),
    "magic_telescope": ("Class", "target_binary"),
    "credit-g": ("class", "target_binary"),
    "blood": ("Class", "target_binary"),
    "Cardiovascular_Disease_Dataset": ("target", "target_binary"),
    "Medicaldataset": ("Result", "target_binary"),
    "heart1": ("output", "target_binary"),
    "car": ("class", "target_multiclass"),
    "forest_covertype_sampled": ("Cover_Type", "target_multiclass"),
    "communities": ("ViolentCrimesPerPop", "target_multiclass"),
}


def preprocessing(df: pd.DataFrame, data_name: str):
    assert data_name in DATASET_AND_CLASS, f"{data_name} not registered"
    class_name, class_values = DATASET_AND_CLASS[data_name]

    if data_name in _TARGET_COL_RENAME:
        src, dst = _TARGET_COL_RENAME[data_name]
        if src in df.columns:
            df[dst] = df[src]
            df = df.drop(src, axis=1)

    X = df.drop(class_name, axis=1).reset_index(drop=True)
    y = df[class_name]
    if class_name in ("target_binary", "target_multiclass"):
        mapping = {lbl: i for i, lbl in enumerate(class_values)}
        y = y.map(mapping)
    y = y.astype(int).reset_index(drop=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    help="e.g. eicu_mortality")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--input_dim", type=int, default=768)
    ap.add_argument("--scaler_type", type=str, default="pow",
                    choices=["std", "pow"])
    ap.add_argument("--llm_model", type=str, default="gpt2_mean",
                    choices=["gpt2_mean", "gpt2_auto", "sentence-bert",
                             "bio-bert", "bio-clinical-bert",
                             "LLAMA_mean", "LLAMA_auto",
                             "gemma-medical"])
    ap.add_argument("--label", action="store_true",
                    help="use label_table subdir instead of origin_table")
    ap.add_argument("--base_path", type=str,
                    default="/storage/personal/eungyeop/dataset/table/")
    ap.add_argument("--output_root", type=str,
                    default="/storage/personal/eungyeop/dataset/embedding")
    ap.add_argument("--output_subdir", type=str,
                    default="tabular_embeddings_carte_fast",
                    help="top-level subdir under output_root")
    ap.add_argument("--skip_desc", action="store_true", default=True,
                    help="(default on) skip description / label_description LLM forward")
    ap.add_argument("--use_desc", dest="skip_desc", action="store_false",
                    help="re-enable description encoding (needs metadata JSON)")
    ap.add_argument("--encode_batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cuda:0 / cpu (default: cuda if available)")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--cpu_affinity", type=int, default=None,
                    help="pin to cpu range [N, N+3). None = no pinning")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    if args.cpu_affinity is not None:
        import psutil
        psutil.Process().cpu_affinity(
            range(args.cpu_affinity, args.cpu_affinity + 3))

    data_source = "label_table" if args.label else "origin_table"
    file_path = os.path.join(args.base_path, data_source, f"{args.dataset}.csv")
    print(f"[fast-embed] loading {file_path}", flush=True)
    df = pd.read_csv(file_path)
    X, y = preprocessing(df, args.dataset)
    print(f"[fast-embed] X.shape={X.shape}  y.unique={sorted(set(y.tolist()))}",
          flush=True)

    import time
    t0 = time.time()
    maker = Table2EmbeddingTransformer(args=args, source_dataset_name=args.dataset)
    data = maker.fit_transform(X, y)
    dt = time.time() - t0
    print(f"[fast-embed] fit_transform done in {dt:.1f}s  "
          f"(n_rows={len(data)})", flush=True)

    save_dir = os.path.join(args.output_root, args.output_subdir, args.llm_model)
    os.makedirs(save_dir, exist_ok=True)
    fname = (f"embedding_label_{args.dataset}.pkl" if args.label
             else f"embedding_{args.dataset}.pkl")
    save_path = os.path.join(save_dir, fname)
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": data,
            "feature_names": list(X.columns),
            "random_seed": args.random_seed,
            "num_classes": len(set(y)),
        }, f)
    print(f"[fast-embed] saved -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
