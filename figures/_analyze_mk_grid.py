"""
Per-seed correlation analysis between (K, T) and {AUC, H_s, H_p}.
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

CSV = Path(__file__).parent / "mk_grid_summary.csv"
K_VALS = [4, 8, 12, 16, 32]
T_VALS = [4, 8, 12, 16, 32]


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["K"] = int(r["K"])
            r["T"] = int(r["T"])
            r["AUC"] = float(r["AUC"]) if r["AUC"] else None
            r["H_s"] = float(r["H_s"]) if r["H_s"] else None
            r["H_p"] = float(r["H_p"]) if r["H_p"] else None
            rows.append(r)
    return rows


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def to_grid(rows, key):
    """Return 5x5 grid indexed [T_idx, K_idx] (row = T ascending bottom-to-top)."""
    G = np.full((len(T_VALS), len(K_VALS)), np.nan)
    for r in rows:
        ti = T_VALS.index(r["T"])
        ki = K_VALS.index(r["K"])
        G[ti, ki] = r[key] if r[key] is not None else np.nan
    return G


def print_grid(name, G, fmt="{:6.3f}"):
    print(f"\n[{name}] rows: T (top=32) x cols: K (left=4)")
    header = "     " + " ".join(f"K={k:>3}" for k in K_VALS)
    print(header)
    for ti in range(len(T_VALS) - 1, -1, -1):
        t = T_VALS[ti]
        cells = []
        for ki in range(len(K_VALS)):
            v = G[ti, ki]
            cells.append("  nan " if np.isnan(v) else fmt.format(v))
        print(f"T={t:>3} " + " ".join(cells))


def main():
    rows = load()
    seeds = sorted(set(r["seed"] for r in rows))

    print("=" * 70)
    print("GLOBAL (pooling all seeds)")
    print("=" * 70)
    auc = [r["AUC"] for r in rows]
    Ks = [r["K"] for r in rows]
    Ts = [r["T"] for r in rows]
    Hs = [r["H_s"] for r in rows]
    Hp = [r["H_p"] for r in rows]
    KT = [r["K"] * r["T"] for r in rows]  # total "capacity"

    print(f"Pearson corr(AUC, K)   = {pearson(Ks, auc):+.3f}")
    print(f"Pearson corr(AUC, T)   = {pearson(Ts, auc):+.3f}")
    print(f"Pearson corr(AUC, K*T) = {pearson(KT, auc):+.3f}")
    print(f"Pearson corr(AUC, H_s) = {pearson(Hs, auc):+.3f}")
    print(f"Pearson corr(AUC, H_p) = {pearson(Hp, auc):+.3f}")
    print(f"Pearson corr(H_s, K)   = {pearson(Ks, Hs):+.3f}")
    print(f"Pearson corr(H_p, K)   = {pearson(Ks, Hp):+.3f}")
    print(f"Pearson corr(H_s, T)   = {pearson(Ts, Hs):+.3f}")
    print(f"Pearson corr(H_p, T)   = {pearson(Ts, Hp):+.3f}")

    for s in seeds:
        sr = [r for r in rows if r["seed"] == s]
        print("\n" + "=" * 70)
        print(f"SEED {s}  (n={len(sr)})")
        print("=" * 70)
        A = to_grid(sr, "AUC")
        Hs_g = to_grid(sr, "H_s")
        Hp_g = to_grid(sr, "H_p")
        print_grid("AUC", A)
        print_grid("H_s", Hs_g)
        print_grid("H_p", Hp_g)

        # best/worst cells
        flat = [(r["K"], r["T"], r["AUC"]) for r in sr if r["AUC"] is not None]
        flat.sort(key=lambda x: x[2], reverse=True)
        print("\n  top-5 (K, T, AUC):")
        for k, t, a in flat[:5]:
            print(f"    K={k:>3} T={t:>3}  AUC={a:.4f}")
        print("  bot-5 (K, T, AUC):")
        for k, t, a in flat[-5:]:
            print(f"    K={k:>3} T={t:>3}  AUC={a:.4f}")

        # correlations within this seed
        aucs = [r["AUC"] for r in sr]
        Ks_s = [r["K"] for r in sr]
        Ts_s = [r["T"] for r in sr]
        Hs_s = [r["H_s"] for r in sr]
        Hp_s = [r["H_p"] for r in sr]
        print(f"\n  corr(AUC, K)={pearson(Ks_s, aucs):+.3f}"
              f"  corr(AUC, T)={pearson(Ts_s, aucs):+.3f}"
              f"  corr(AUC, H_s)={pearson(Hs_s, aucs):+.3f}"
              f"  corr(AUC, H_p)={pearson(Hp_s, aucs):+.3f}")


if __name__ == "__main__":
    main()
