"""
Plot per-seed 3-panel heatmaps of (AUC, H_s, H_p) over the (K, T) grid,
matching the style of the user's reference figure.

Saves PNGs to /storage/personal/eungyeop/Graph2/
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CSV = Path(__file__).parent / "mk_grid_summary.csv"
OUT = Path("/storage/personal/eungyeop/Graph2")
OUT.mkdir(parents=True, exist_ok=True)

K_VALS = [4, 8, 12, 16, 32]
T_VALS = [4, 8, 12, 16, 32]


def load_rows():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["K"] = int(r["K"])
            r["T"] = int(r["T"])
            r["AUC"] = float(r["AUC"]) if r["AUC"] else np.nan
            r["H_s"] = float(r["H_s"]) if r["H_s"] else np.nan
            r["H_p"] = float(r["H_p"]) if r["H_p"] else np.nan
            rows.append(r)
    return rows


def to_grid(rows, key):
    """Return 5x5 grid indexed [T_idx, K_idx], with T increasing upward (row 0 = T=32, last row = T=4)."""
    G = np.full((len(T_VALS), len(K_VALS)), np.nan)
    for r in rows:
        ti = T_VALS.index(r["T"])
        ki = K_VALS.index(r["K"])
        G[ti, ki] = r[key]
    # flip so T=32 is at the top row (imshow origin='upper')
    return np.flipud(G)


def annotate(ax, G, fmt="{:.3f}", **_unused):
    """Annotate cells. All text rendered in black for visual consistency."""
    nrows, ncols = G.shape
    for i in range(nrows):
        for j in range(ncols):
            v = G[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=9, color="black", fontweight="semibold")


def plot_one_seed(seed, rows_for_seed, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    A = to_grid(rows_for_seed, "AUC")
    Hs = to_grid(rows_for_seed, "H_s")
    Hp = to_grid(rows_for_seed, "H_p")

    # --- AUC panel ---
    ax = axes[0]
    # pick vmin/vmax tight around the data for nicer contrast (match reference style ~0.625..0.83)
    vmin = float(np.nanmin(A)); vmax = float(np.nanmax(A))
    pad = 0.005
    vmin = max(0.45, vmin - pad); vmax = min(1.0, vmax + pad)
    im = ax.imshow(A, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    # RdYlBu_r: both ends are dark (red at top, blue at bottom)
    annotate(ax, A, "{:.3f}",
             dark_above=vmax - 0.07, dark_below=vmin + 0.07)
    ax.set_title("Zero-shot AUC", fontsize=13)
    ax.set_xlabel(r"$K$ (n_graphs)")
    ax.set_ylabel(r"$T$ (n_nodes)")
    ax.set_xticks(range(len(K_VALS)), [str(k) for k in K_VALS])
    ax.set_yticks(range(len(T_VALS)), [str(t) for t in reversed(T_VALS)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC")

    # --- H_s panel (purple) ---
    ax = axes[1]
    # Data-driven vmax so contrast is usable even though H_s is small (~0.05-0.5)
    hs_vmax = float(np.nanmax(Hs)) * 1.05
    hs_vmax = max(hs_vmax, 0.3)  # keep a sane floor
    im = ax.imshow(Hs, cmap="Purples", vmin=0.0, vmax=hs_vmax, aspect="equal")
    annotate(ax, Hs, "{:.2f}", dark_above=hs_vmax * 0.65)
    ax.set_title(r"$H_s$ (per-sample entropy)", fontsize=13)
    ax.set_xlabel(r"$K$ (n_graphs)")
    ax.set_ylabel(r"$T$ (n_nodes)")
    ax.set_xticks(range(len(K_VALS)), [str(k) for k in K_VALS])
    ax.set_yticks(range(len(T_VALS)), [str(t) for t in reversed(T_VALS)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$H_s/\log K$")

    # --- H_p panel (green) ---
    ax = axes[2]
    # Data-driven vmin/vmax for better contrast (observed range ~0.55-0.90)
    hp_lo = float(np.nanmin(Hp)) - 0.02
    hp_hi = float(np.nanmax(Hp)) + 0.02
    hp_lo = max(0.0, hp_lo); hp_hi = min(1.0, hp_hi)
    im = ax.imshow(Hp, cmap="YlGn", vmin=hp_lo, vmax=hp_hi, aspect="equal")
    annotate(ax, Hp, "{:.2f}", dark_above=hp_lo + (hp_hi - hp_lo) * 0.65)
    ax.set_title(r"$H_p$ (batch-level entropy)", fontsize=13)
    ax.set_xlabel(r"$K$ (n_graphs)")
    ax.set_ylabel(r"$T$ (n_nodes)")
    ax.set_xticks(range(len(K_VALS)), [str(k) for k in K_VALS])
    ax.set_yticks(range(len(T_VALS)), [str(t) for t in reversed(T_VALS)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$H_p/\log K$")

    fig.tight_layout()
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


def main():
    rows = load_rows()
    seeds = sorted(set(r["seed"] for r in rows))
    for s in seeds:
        srows = [r for r in rows if r["seed"] == s]
        out_png = OUT / f"mk_grid_seed{s}.png"
        plot_one_seed(s, srows, out_png)

    # also: mean-over-seeds panel — helps the central-sweet-spot story
    print("\nBuilding mean-over-seeds figure...")
    mean_rows = []
    for K in K_VALS:
        for T in T_VALS:
            sub = [r for r in rows if r["K"] == K and r["T"] == T]
            mean_rows.append({
                "seed": "mean",
                "K": K, "T": T,
                "AUC": float(np.nanmean([r["AUC"] for r in sub])),
                "H_s": float(np.nanmean([r["H_s"] for r in sub])),
                "H_p": float(np.nanmean([r["H_p"] for r in sub])),
            })
    plot_one_seed("mean", mean_rows, OUT / "mk_grid_mean.png")

    print(f"\nAll figures saved under: {OUT}")


if __name__ == "__main__":
    main()
