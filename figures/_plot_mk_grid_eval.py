"""
Plot per-seed 2x2 figures using EVAL-TIME entropies.
  (0,0) Zero-shot AUC heatmap
  (0,1) Utilization heatmap   eff_K / K = K^(H_p - 1)
  (1,0) Ideal K vs effective K (bar, averaged over T)
  (1,1) Utilization & p_max vs K (line, averaged over T)

H_s / H_p panels intentionally dropped in favor of utilization / eff_K,
which carry the same information in an interpretable scale.

eff_K = exp(raw H_p) = K^(normalized H_p)  -- classic codebook perplexity.

Saves to /storage/personal/eungyeop/Graph2/mk_grid_eval_*.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CSV = Path(__file__).parent / "mk_grid_summary_eval.csv"
OUT = Path("/storage/personal/eungyeop/Graph2")
OUT.mkdir(parents=True, exist_ok=True)

K_VALS = [4, 8, 12, 16, 32]
T_VALS = [4, 8, 12, 16, 32]

# Soft accent palette (pastel orange + pastel blue, similar to the reference figure)
COLOR_ORANGE = "#f4a582"
COLOR_ORANGE_EDGE = "#c87a53"
COLOR_BLUE = "#7aa8d8"
COLOR_BLUE_EDGE = "#456e9e"
ALPHA_FILL = 0.78
ALPHA_LINE = 0.95

GRID_KW = dict(linestyle=":", linewidth=0.8, color="#888888", alpha=0.55)

# Soft diverging colormap (pastel blue -> cream -> pastel orange)
CMAP_AUC = LinearSegmentedColormap.from_list(
    "soft_blue_orange",
    ["#7aa8d8", "#cfdbe8", "#f3ede2", "#f6c9a8", "#f4a582"],
    N=256,
)
# Softer utilization colormap (pastel blue tones only)
CMAP_UTIL = LinearSegmentedColormap.from_list(
    "soft_blue",
    ["#f3ede2", "#d8e3ef", "#a8c2de", "#7aa8d8", "#4f84bd"],
    N=256,
)


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
            r["p_max"] = float(r["p_max"]) if r.get("p_max") else np.nan
            rows.append(r)
    return rows


def to_grid(rows, key):
    G = np.full((len(T_VALS), len(K_VALS)), np.nan)
    for r in rows:
        ti = T_VALS.index(r["T"])
        ki = K_VALS.index(r["K"])
        G[ti, ki] = r[key]
    return np.flipud(G)


def util_grid(rows):
    """eff_K / K = K^(H_p - 1)."""
    G = np.full((len(T_VALS), len(K_VALS)), np.nan)
    for r in rows:
        ti = T_VALS.index(r["T"])
        ki = K_VALS.index(r["K"])
        Hp = r["H_p"]
        if not np.isnan(Hp):
            G[ti, ki] = float(r["K"]) ** (Hp - 1.0)
    return np.flipud(G)


def annotate(ax, G, fmt="{:.3f}"):
    nrows, ncols = G.shape
    for i in range(nrows):
        for j in range(ncols):
            v = G[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=9, color="black", fontweight="semibold")


def style_cartesian_axes(ax):
    """Light spines + dotted grid, background behind data."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", **GRID_KW)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.9)


def plot_one(seed, rows_for_seed, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))

    # ---------- (0,0) AUC heatmap ----------
    A = to_grid(rows_for_seed, "AUC")
    ax = axes[0, 0]
    vmin = max(0.45, float(np.nanmin(A)) - 0.005)
    vmax = min(1.0, float(np.nanmax(A)) + 0.005)
    im = ax.imshow(A, cmap=CMAP_AUC, vmin=vmin, vmax=vmax, aspect="equal")
    annotate(ax, A, "{:.3f}")
    ax.set_xlabel(r"$K$ (n_graphs)"); ax.set_ylabel(r"$T$ (n_nodes)")
    ax.set_xticks(range(len(K_VALS)), [str(k) for k in K_VALS])
    ax.set_yticks(range(len(T_VALS)), [str(t) for t in reversed(T_VALS)])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("AUC")

    # ---------- (0,1) Utilization heatmap ----------
    U = util_grid(rows_for_seed)
    u_lo = max(0.3, float(np.nanmin(U)) - 0.02)
    u_hi = min(1.0, float(np.nanmax(U)) + 0.02)
    ax = axes[0, 1]
    im = ax.imshow(U, cmap=CMAP_UTIL, vmin=u_lo, vmax=u_hi, aspect="equal")
    annotate(ax, U, "{:.2f}")
    ax.set_xlabel(r"$K$ (n_graphs)"); ax.set_ylabel(r"$T$ (n_nodes)")
    ax.set_xticks(range(len(K_VALS)), [str(k) for k in K_VALS])
    ax.set_yticks(range(len(T_VALS)), [str(t) for t in reversed(T_VALS)])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"$\mathrm{eff}_K/K$")

    # ---------- K-wise aggregates (avg over T) ----------
    by_K_eff = {k: [] for k in K_VALS}
    by_K_pmx = {k: [] for k in K_VALS}
    for r in rows_for_seed:
        k = r["K"]
        if not np.isnan(r["H_p"]):
            by_K_eff[k].append(float(k) ** r["H_p"])
        if not np.isnan(r.get("p_max", np.nan)):
            by_K_pmx[k].append(r["p_max"])
    eff_means = np.array([np.mean(by_K_eff[k]) if by_K_eff[k] else np.nan for k in K_VALS])
    pmx_means = np.array([np.mean(by_K_pmx[k]) if by_K_pmx[k] else np.nan for k in K_VALS])
    util_means = eff_means / np.array(K_VALS, dtype=float)

    # ---------- (1,0) Ideal vs effective K (bar) ----------
    ax = axes[1, 0]
    style_cartesian_axes(ax)
    x = np.arange(len(K_VALS))
    w = 0.38
    ax.bar(x - w/2, K_VALS, w,
           color=COLOR_ORANGE, alpha=ALPHA_FILL,
           edgecolor=COLOR_ORANGE_EDGE, linewidth=0.8,
           label=r"$K$  (ideal)")
    ax.bar(x + w/2, eff_means, w,
           color=COLOR_BLUE, alpha=ALPHA_FILL,
           edgecolor=COLOR_BLUE_EDGE, linewidth=0.8,
           label=r"$\mathrm{eff}_K$  (actual)")
    for xi, k, e in zip(x, K_VALS, eff_means):
        ax.text(xi - w/2, k + 0.6, f"{k}", ha="center",
                fontsize=8, color=COLOR_ORANGE_EDGE, fontweight="semibold")
        if not np.isnan(e):
            ax.text(xi + w/2, e + 0.6, f"{e:.1f}", ha="center",
                    fontsize=8, color=COLOR_BLUE_EDGE, fontweight="semibold")
    ax.set_xticks(x, [str(k) for k in K_VALS])
    ax.set_xlabel(r"$K$"); ax.set_ylabel("# graphs")
    ax.set_ylim(0, max(K_VALS) * 1.22)
    leg = ax.legend(loc="upper left", fontsize=10,
                    frameon=True, framealpha=0.92, edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.6)

    # ---------- (1,1) Utilization & p_max lines ----------
    ax = axes[1, 1]
    style_cartesian_axes(ax)
    ax.plot(x, util_means, marker="o", linewidth=2.2, markersize=8,
            color=COLOR_BLUE, alpha=ALPHA_LINE,
            markerfacecolor=COLOR_BLUE, markeredgecolor=COLOR_BLUE_EDGE,
            label=r"utilization $\mathrm{eff}_K / K$")
    ax.plot(x, pmx_means, marker="s", linewidth=2.2, markersize=8,
            color=COLOR_ORANGE, alpha=ALPHA_LINE,
            markerfacecolor=COLOR_ORANGE, markeredgecolor=COLOR_ORANGE_EDGE,
            label=r"$p_{\max}$ (top-1 share)")
    for xi, u in zip(x, util_means):
        if not np.isnan(u):
            ax.text(xi, u + 0.035, f"{u:.2f}", ha="center",
                    fontsize=8, color=COLOR_BLUE_EDGE, fontweight="semibold")
    for xi, p in zip(x, pmx_means):
        if not np.isnan(p):
            ax.text(xi, p - 0.055, f"{p:.2f}", ha="center",
                    fontsize=8, color=COLOR_ORANGE_EDGE, fontweight="semibold")
    ax.set_xticks(x, [str(k) for k in K_VALS])
    ax.set_xlabel(r"$K$"); ax.set_ylabel("ratio / share")
    ax.set_ylim(0.0, 1.18)
    leg = ax.legend(loc="upper right", fontsize=10,
                    frameon=True, framealpha=0.92, edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


def main():
    rows = load_rows()
    seeds = sorted(set(r["seed"] for r in rows))
    for s in seeds:
        srows = [r for r in rows if r["seed"] == s]
        plot_one(s, srows, OUT / f"mk_grid_eval_v2_seed{s}.pdf")

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
                "p_max": float(np.nanmean([r["p_max"] for r in sub])),
            })
    plot_one("mean", mean_rows, OUT / "mk_grid_eval_v2_mean.pdf")

    print(f"\nAll figures saved under: {OUT}")


if __name__ == "__main__":
    main()
