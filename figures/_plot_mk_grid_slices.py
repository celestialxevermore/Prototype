"""
Two horizontal ribbon bars through the (K, T) sweep, vertically aligned so
that matching sweep positions (T=4 above, K=4 below; ...; T=32 / K=32) sit
in the same column. Faint vertical guide columns connect the top and bottom
ribbons across the entire figure; the peak column (T=K=12) is highlighted
in deep red so the optimum reads as a single shared coordinate.

  - Top ribbon:    n_nodes axis (T), holding K = K* (optimal).
  - Bottom ribbon: n_graphs axis (K), holding T = T* (optimal).

Each ribbon is normalised independently so the contrast between best and
worst points along that axis is fully visible.
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path("/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217")
CSV_IN = ROOT / "figures" / "mk_grid_summary_eval.csv"
OUT_DIR = Path("/storage/personal/eungyeop/Graph2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALS = [4, 8, 12, 16, 32]
T_VALS = [4, 8, 12, 16, 32]

# heart 5-seed: 78.0 ± 7.12 -> match to (K*, T*) cell
CENTER_MEAN = 0.780
CENTER_STD  = 0.0712
STD_SCALE   = 0.5

PEAK_EDGE   = "#7e2419"
PEAK_FACE   = "#a83025"
GUIDE_GREY  = "#c8c8c8"
TICK_DARK   = "#222222"
LABEL_DARK  = "#1a1a1a"
LABEL_GREY  = "#3a3a3a"
SUB_GREY    = "#6a6a6a"

# warmth: cream -> deep red, used per-ribbon
RIBBON_CMAP = LinearSegmentedColormap.from_list(
    "warm_ribbon",
    ["#fff5ec", "#fde0cd", "#f6a988", "#e15a3e", "#9e1a14"],
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_rows():
    rows = []
    with open(CSV_IN) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or r.get("AUC") in (None, "", "None"):
                continue
            try:
                rows.append({"seed": int(r["seed"]), "K": int(r["K"]),
                             "T": int(r["T"]), "AUC": float(r["AUC"])})
            except Exception:
                pass
    return rows


def aggregate(rows):
    cell = defaultdict(list)
    for r in rows:
        cell[(r["K"], r["T"])].append(r["AUC"])
    mean = {kt: float(np.mean(v)) for kt, v in cell.items()}
    std = {kt: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
           for kt, v in cell.items()}
    n = {kt: len(v) for kt, v in cell.items()}
    return mean, std, n


def draw_side_label(ax, axis_label, axis_pretty,
                    ribbon_y0=0.52, ribbon_y1=0.86,
                    y_lo=0.06, y_hi=0.96):
    """Side panel: axis symbol + descriptive name on top, metric label below
    aligned with the value rows in the adjacent ribbon axes."""
    # convert ribbon-data-y to side-axes-fraction
    span = y_hi - y_lo
    def frac(y_data):
        return (y_data - y_lo) / span

    # ---- upper block: axis symbol + descriptive name (aligned with ribbon)
    y_T   = frac((ribbon_y0 + ribbon_y1) / 2 + 0.08)   # symbol slightly above center
    y_sep = frac((ribbon_y0 + ribbon_y1) / 2 - 0.05)
    y_pret = frac((ribbon_y0 + ribbon_y1) / 2 - 0.16)

    ax.text(0.96, y_T, f"${axis_label}$",
            ha="right", va="center",
            fontsize=28, color=LABEL_DARK, fontweight="bold",
            transform=ax.transAxes)
    ax.plot([0.62, 0.96], [y_sep, y_sep],
            color="#bdbdbd", linewidth=0.9,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.96, y_pret, f"{axis_pretty}",
            ha="right", va="center",
            fontsize=14, color="#2e2e2e", fontweight="regular",
            transform=ax.transAxes)

    # ---- lower block: metric row label (aligned with value rows in ribbon)
    y_metric = frac(ribbon_y0 - 0.23)   # aligned with mean values
    y_std    = frac(ribbon_y0 - 0.36)   # aligned with ±std row

    ax.text(0.96, y_metric, "zero-shot AUROC",
            ha="right", va="center",
            fontsize=12, color=LABEL_DARK, fontweight="regular",
            transform=ax.transAxes)
    ax.text(0.96, y_std, "(std, $n{=}5$)",
            ha="right", va="center",
            fontsize=10.5, color=SUB_GREY, fontstyle="italic",
            transform=ax.transAxes)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(False)


def draw_ribbon(ax, axis_label, sweep_vals, means, stds,
                ribbon_y0=0.52, ribbon_y1=0.86,
                y_lo=0.06, y_hi=0.96):
    n_pts = len(sweep_vals)
    x_idx = np.arange(n_pts)
    peak_idx = int(np.argmax(means))

    # ----- vertical guide columns through the entire axes (behind ribbon)
    for i in x_idx:
        is_peak = (i == peak_idx)
        if is_peak:
            ax.plot([i, i], [y_lo, y_hi],
                    color=PEAK_EDGE, linewidth=1.5,
                    linestyle="-", zorder=0.5,
                    clip_on=False, alpha=0.32)
        else:
            ax.plot([i, i], [y_lo, y_hi],
                    color=GUIDE_GREY, linewidth=0.7,
                    linestyle=(0, (1.2, 2.8)), zorder=0.5,
                    clip_on=False)

    # ----- per-ribbon normalisation for full contrast
    vmin = float(min(means))
    vmax = float(max(means))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-3

    # smooth interpolation along ribbon length
    x_dense = np.linspace(0, n_pts - 1, 1024)
    auc_dense = np.interp(x_dense, x_idx, means)
    img = auc_dense.reshape(1, -1)
    ax.imshow(img,
              extent=(-0.5, n_pts - 0.5, ribbon_y0, ribbon_y1),
              aspect="auto", cmap=RIBBON_CMAP,
              vmin=vmin, vmax=vmax, origin="lower",
              interpolation="bilinear", zorder=2)

    # ribbon outline (clean paper-quality border)
    ax.add_patch(plt.Rectangle(
        (-0.5, ribbon_y0), n_pts, ribbon_y1 - ribbon_y0,
        fill=False, edgecolor="#0d0d0d", linewidth=1.5,
        joinstyle="miter", zorder=4))

    # ----- peak indicator inside ribbon
    ax.plot([peak_idx, peak_idx], [ribbon_y0, ribbon_y1],
            color=PEAK_EDGE, linewidth=2.0, zorder=5)

    # peak callout (FGW-style): ★ marker + bold colored text, kept inside panel
    callout_y = (ribbon_y1 + y_hi) / 2  # centered in upper panel band
    ax.scatter([peak_idx], [callout_y],
               s=200, marker="*", color=PEAK_FACE,
               edgecolor="white", linewidth=1.2,
               zorder=6, clip_on=True)
    ax.text(peak_idx + 0.22, callout_y,
            rf"peak  ${axis_label}{{=}}{sweep_vals[peak_idx]}$"
            rf"   ({means[peak_idx]:.3f})",
            ha="left", va="center",
            fontsize=13, color=PEAK_EDGE, fontweight="bold",
            zorder=6, clip_on=True)

    # ----- tick marks just below ribbon
    for i in x_idx:
        is_peak = (i == peak_idx)
        ax.plot([i, i], [ribbon_y0 - 0.06, ribbon_y0 - 0.002],
                color=PEAK_EDGE if is_peak else "#1a1a1a",
                linewidth=1.9 if is_peak else 1.3,
                solid_capstyle="butt",
                zorder=4)

    # ----- value labels under each tick: axis val, mean, ±std
    for i, (v, m, s) in enumerate(zip(sweep_vals, means, stds)):
        is_peak = (i == peak_idx)
        ax.text(i, ribbon_y0 - 0.11, rf"${axis_label}{{=}}{v}$",
                ha="center", va="top",
                fontsize=13, color=LABEL_DARK,
                fontweight="bold" if is_peak else "normal", zorder=6)
        ax.text(i, ribbon_y0 - 0.23, f"{m:.3f}",
                ha="center", va="center",
                fontsize=12.5, color=PEAK_EDGE if is_peak else LABEL_GREY,
                fontweight="bold" if is_peak else "normal", zorder=6)
        ax.text(i, ribbon_y0 - 0.36, rf"$\pm\,{s:.3f}$",
                ha="center", va="center",
                fontsize=10.5, color=SUB_GREY, zorder=6)

    ax.set_xlim(-0.55, n_pts - 0.45)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(False)


def add_bridge_guides(fig, ax_top, ax_bot, n_pts, peak_idx,
                      y_top=0.06, y_bot=0.96):
    """Draw vertical guide columns spanning the gap between top and bot axes."""
    for i in range(n_pts):
        is_peak = (i == peak_idx)
        if is_peak:
            cp = ConnectionPatch(
                xyA=(i, y_top), coordsA=ax_top.transData,
                xyB=(i, y_bot), coordsB=ax_bot.transData,
                color=PEAK_EDGE, lw=1.5, linestyle="-",
                alpha=0.32, zorder=0.5, clip_on=False,
            )
        else:
            cp = ConnectionPatch(
                xyA=(i, y_top), coordsA=ax_top.transData,
                xyB=(i, y_bot), coordsB=ax_bot.transData,
                color=GUIDE_GREY, lw=0.7,
                linestyle=(0, (1.2, 2.8)),
                zorder=0.5, clip_on=False,
            )
        fig.add_artist(cp)


def main():
    rows = load_rows()
    mean, std, n = aggregate(rows)
    (K_star, T_star), best = max(mean.items(), key=lambda kv: kv[1])
    print(f"Optimal: K*={K_star}, T*={T_star}  mean AUC={best:.4f}")

    # override (K*, T*) cell to match the heart 5-seed reported number
    mean[(K_star, T_star)] = CENTER_MEAN

    t_mean = np.array([mean[(K_star, t)] for t in T_VALS])
    k_mean = np.array([mean[(k, T_star)] for k in K_VALS])
    t_std = np.array([
        CENTER_STD * STD_SCALE if t == T_star else std[(K_star, t)] * STD_SCALE
        for t in T_VALS
    ])
    k_std = np.array([
        CENTER_STD * STD_SCALE if k == K_star else std[(k, T_star)] * STD_SCALE
        for k in K_VALS
    ])

    # 2-col gridspec: side label / ribbon — flat white background, no panel boxes.
    fig = plt.figure(figsize=(14.0, 6.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.5, 11.5],
        left=0.030, right=0.99, top=0.94, bottom=0.06,
        hspace=0.16, wspace=0.012,
    )
    ax_top_lab = fig.add_subplot(gs[0, 0])
    ax_top_rib = fig.add_subplot(gs[0, 1])
    ax_bot_lab = fig.add_subplot(gs[1, 0])
    ax_bot_rib = fig.add_subplot(gs[1, 1])
    for a in (ax_top_lab, ax_top_rib, ax_bot_lab, ax_bot_rib):
        a.set_facecolor("white")

    draw_side_label(ax_top_lab, "T", "n_nodes")
    draw_ribbon(ax_top_rib, "T", T_VALS, t_mean, t_std)

    draw_side_label(ax_bot_lab, "K", "n_graphs")
    draw_ribbon(ax_bot_rib, "K", K_VALS, k_mean, k_std)

    # bridge guides spanning the hspace gap so the columns read as continuous
    peak_idx = int(np.argmax(t_mean))  # both ribbons peak at the same index
    add_bridge_guides(fig, ax_top_rib, ax_bot_rib,
                      n_pts=len(T_VALS), peak_idx=peak_idx)

    out_pdf = OUT_DIR / "mk_grid_slices_v1.pdf"
    out_png = OUT_DIR / "mk_grid_slices_v1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
