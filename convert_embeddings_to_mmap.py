"""
Convert per-sample-dict .pkl embedding files into per-field memmap binaries.

Input layout (existing):
  <base>/tabular_embeddings_<embed_type>/<llm_model>/embedding_<dataset>.pkl
    dict with keys: embeddings (list of per-sample dicts), feature_names, num_classes, random_seed

Output layout (new, memmap-friendly):
  <base>/tabular_embeddings_<embed_type>_mmap/<llm_model>/<dataset>/
    - meta.json                               (shapes, N, num_classes, feature_names, cat/num desc_texts)
    - label_description_embeddings.memmap     (N, 1, 768)  float32
    - cat_name_embeddings.memmap              (N, n_cat, 768)  float32
    - cat_value_embeddings.memmap             (N, n_cat, 768)  float32
    - cat_desc_embeddings.memmap              (N, n_cat, 768)  float32
    - num_prompt_embeddings.memmap            (N, n_num, 768)  float32
    - num_name_embeddings.memmap              (N, n_num, 768)  float32
    - num_desc_embeddings.memmap              (N, n_num, 768)  float32
    - y.memmap                                (N,)  int64
    - s_idx.memmap                            (N,)  int64

Usage:
    python convert_embeddings_to_mmap.py --dataset zigong_mortality
    python convert_embeddings_to_mmap.py --dataset mimic_mortality --llm_model gemma-medical --embed_type carte
    python convert_embeddings_to_mmap.py --all  # convert all 6 ICU mortality datasets (sequential, size-ascending)
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

TENSOR_KEYS = (
    'label_description_embeddings',
    'cat_name_embeddings', 'cat_value_embeddings', 'cat_desc_embeddings',
    'num_prompt_embeddings', 'num_name_embeddings', 'num_desc_embeddings',
)


def convert_one(pkl_path: str, out_dir: str, strict_shape: bool = True) -> int:
    """Convert one .pkl into a mmap directory. Returns N samples kept.

    When strict_shape=False (permissive), samples whose tensor shapes don't match
    the canonical (majority) shape are skipped instead of raising. The canonical
    shape is the most common shape across all samples for each tensor key.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"[load] {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    embs = data['embeddings']
    N = len(embs)
    feature_names = list(data.get('feature_names', []))
    num_classes = int(data.get('num_classes', 2))
    print(f"[load] done in {time.time()-t0:.1f}s, N={N}")

    # Some datasets have 0 categorical features (or 0 numerical); only convert the
    # tensor keys that actually exist in this dataset.
    present_keys = [k for k in TENSOR_KEYS if k in embs[0]]
    missing_keys = [k for k in TENSOR_KEYS if k not in embs[0]]
    if missing_keys:
        print(f"[info] missing keys (dataset has no features of that kind): {missing_keys}")
    cat_desc_texts = list(embs[0].get('cat_desc_texts', []))
    num_desc_texts = list(embs[0].get('num_desc_texts', []))

    # Determine canonical shapes. In strict mode just trust sample 0; in
    # permissive mode, scan all samples and take the most common shape per key.
    if strict_shape:
        shapes = {k: tuple(embs[0][k].shape) for k in present_keys}
        valid_idx = list(range(N))
    else:
        from collections import Counter
        print("[scan] permissive mode: scanning shape distribution")
        counters = {k: Counter() for k in present_keys}
        for i in range(N):
            e = embs[i]
            for k in present_keys:
                counters[k][tuple(e[k].shape)] += 1
        shapes = {}
        for k in present_keys:
            top = counters[k].most_common(3)
            shapes[k] = top[0][0]
            print(f"  {k}: {top}")
        valid_idx = []
        for i in range(N):
            ok = all(tuple(embs[i][k].shape) == shapes[k] for k in present_keys)
            if ok:
                valid_idx.append(i)
        N_valid = len(valid_idx)
        print(f"[scan] valid {N_valid}/{N} ({100*N_valid/max(N,1):.2f}%)")
        if N_valid == 0:
            raise RuntimeError("No samples match canonical shape; pkl is inconsistent")

    N_out = len(valid_idx)

    # Write meta
    meta = {
        'N': N_out,
        'N_raw': N,
        'num_classes': num_classes,
        'feature_names': feature_names,
        'cat_desc_texts': cat_desc_texts,
        'num_desc_texts': num_desc_texts,
        'present_keys': present_keys,
        'shapes': {k: list(v) for k, v in shapes.items()},
        'dtype': 'float32',
        'label_dtype': 'int64',
        'strict_shape': bool(strict_shape),
        'samples_dropped': N - N_out,
    }
    with open(os.path.join(out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # Allocate memmaps sized to N_out (kept samples only)
    mm = {
        k: np.memmap(
            os.path.join(out_dir, f"{k}.memmap"),
            dtype=np.float32, mode='w+', shape=(N_out,) + shapes[k],
        )
        for k in present_keys
    }
    y_mm = np.memmap(os.path.join(out_dir, "y.memmap"), dtype=np.int64, mode='w+', shape=(N_out,))
    sidx_mm = np.memmap(os.path.join(out_dir, "s_idx.memmap"), dtype=np.int64, mode='w+', shape=(N_out,))

    t1 = time.time()
    for out_i, i in enumerate(valid_idx):
        e = embs[i]
        for k in present_keys:
            arr = e[k]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if strict_shape and tuple(arr.shape) != shapes[k]:
                raise ValueError(f"Sample {i} {k}: shape {arr.shape} != expected {shapes[k]}")
            mm[k][out_i] = arr
        y_mm[out_i] = int(e['y'].item() if hasattr(e['y'], 'item') else e['y'])
        sidx_mm[out_i] = int(e.get('s_idx', i))
        embs[i] = None
        if (out_i + 1) % 5000 == 0 or (out_i + 1) == N_out:
            elapsed = time.time() - t1
            rate = (out_i + 1) / max(elapsed, 1e-6)
            eta = (N_out - (out_i + 1)) / max(rate, 1e-6)
            print(f"  [{out_i+1}/{N_out}]  {rate:.0f} samp/s  eta {eta:.0f}s")

    for arr in mm.values():
        arr.flush()
    y_mm.flush()
    sidx_mm.flush()

    del mm, y_mm, sidx_mm, embs, data
    print(f"[done] {out_dir}  total {time.time()-t0:.1f}s  kept {N_out}/{N}")
    return N_out


def default_paths(args):
    base = "/storage/personal/eungyeop/dataset/embedding"
    if args.embed_type == "_":
        src_sub = f"tabular_embeddings_/{args.llm_model}"
        dst_sub = f"tabular_embeddings__mmap/{args.llm_model}"
    else:
        src_sub = f"tabular_embeddings_{args.embed_type}/{args.llm_model}"
        dst_sub = f"tabular_embeddings_{args.embed_type}_mmap/{args.llm_model}"
    return base, src_sub, dst_sub


DEFAULT_ALL = [
    # smallest → largest, so peak RAM during conversion ramps gradually
    'support_mortality',
    'zigong_mortality',
    'sic_mortality',
    'hirid_mortality',
    'mimic_mortality',
    'eicu_mortality',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default=None, help="dataset name (e.g. mimic_mortality)")
    p.add_argument('--all', action='store_true', help="convert the ICU mortality set sequentially")
    p.add_argument('--llm_model', type=str, default='gpt2_mean')
    p.add_argument('--embed_type', type=str, default='carte')
    p.add_argument('--force', action='store_true', help="overwrite existing output dir")
    p.add_argument('--permissive', action='store_true',
                   help="allow per-sample shape mismatches by dropping minority-shape samples")
    args = p.parse_args()

    if not args.dataset and not args.all:
        p.error("pass --dataset <name> or --all")

    base, src_sub, dst_sub = default_paths(args)
    targets = DEFAULT_ALL if args.all else [args.dataset]

    for name in targets:
        pkl_path = os.path.join(base, src_sub, f"embedding_{name}.pkl")
        out_dir = os.path.join(base, dst_sub, name)
        if os.path.isdir(out_dir) and not args.force:
            meta_path = os.path.join(out_dir, 'meta.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        existing_meta = json.load(f)
                    present = existing_meta.get('present_keys', list(TENSOR_KEYS))
                except Exception:
                    present = list(TENSOR_KEYS)
            else:
                present = list(TENSOR_KEYS)
            needed = ['meta.json'] + [f"{k}.memmap" for k in present] + ['y.memmap', 's_idx.memmap']
            if all(os.path.exists(os.path.join(out_dir, fn)) for fn in needed):
                print(f"[skip] {out_dir} already converted (use --force to redo)")
                continue
        print(f"\n=== {name} ===")
        convert_one(pkl_path, out_dir, strict_shape=not args.permissive)


if __name__ == '__main__':
    main()
