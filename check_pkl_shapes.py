"""Load a gpt2_mean mortality pkl and report the shape distribution of each
tensor key across all samples. Exits 0 if every sample has the same shape for
every key, else 1.
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--llm_model', default='gpt2_mean')
    ap.add_argument('--embed_type', default='carte')
    args = ap.parse_args()

    base = "/storage/personal/eungyeop/dataset/embedding"
    pkl = os.path.join(base, f"tabular_embeddings_{args.embed_type}",
                       args.llm_model, f"embedding_{args.dataset}.pkl")
    print(f"[load] {pkl}", flush=True)
    t0 = time.time()
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    embs = data['embeddings']
    N = len(embs)
    print(f"[load] done in {time.time()-t0:.1f}s  N={N}", flush=True)

    TENSOR_KEYS = (
        'label_description_embeddings',
        'cat_name_embeddings', 'cat_value_embeddings', 'cat_desc_embeddings',
        'num_prompt_embeddings', 'num_name_embeddings', 'num_desc_embeddings',
    )
    present_keys = [k for k in TENSOR_KEYS if k in embs[0]]
    print(f"present_keys: {present_keys}")

    counters = {k: Counter() for k in present_keys}
    for i in range(N):
        e = embs[i]
        for k in present_keys:
            arr = e[k]
            shape = tuple(arr.shape) if hasattr(arr, 'shape') else None
            counters[k][shape] += 1

    ok = True
    for k in present_keys:
        c = counters[k]
        unique = len(c)
        top = c.most_common(3)
        flag = "OK" if unique == 1 else "VARIABLE"
        print(f"  {k:30s} unique_shapes={unique}  top3={top}  [{flag}]")
        if unique != 1:
            ok = False

    print(f"[result] {'ALL FIXED' if ok else 'HAS VARIABLE-SHAPE SAMPLES'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
