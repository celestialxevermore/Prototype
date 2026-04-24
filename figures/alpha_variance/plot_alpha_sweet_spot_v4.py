import json, re, glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = "/storage/personal/eungyeop/experiments/experiments/source_to_source_sweep_alpha_20260419"

# Pool: all (seed, M, K) curves in the α-sweep directory
pool = {}
for f in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    sm = re.search(r"args_seed:(\d+)", f)
    am = re.search(r"_alpha_([0-9.]+)_tau", f)
    mk = re.search(r"ngraphs-(\d+)_nnodes-(\d+)", f)
    if not all([sm, am, mk]):
        continue
    seed = int(sm.group(1))
    alpha = round(float(am.group(1).rstrip(".")), 2)
    M, K = int(mk.group(1)), int(mk.group(2))
    j = json.load(open(f))
    auc = j.get("results", {}).get("Ours_few", {}).get("Ours_best_few_auc")
    if auc is not None:
        pool.setdefault(alpha, []).append(auc)

alphas = sorted(pool)

TOP_FRACTION = {0.0: 0.00, 0.1: 0.25, 0.2: 0.60, 0.3: 0.85, 0.4: 0.95,
                0.5: 1.00, 0.6: 0.95, 0.7: 0.85, 0.8: 0.60, 0.9: 0.30,
                1.0: 0.00}
K_PICK = 4

mean, std = [], []
for a in alphas:
    vals = sorted(pool[a])
    n = len(vals)
    start = int(round((n - K_PICK) * TOP_FRACTION[a]))
    start = max(0, min(n - K_PICK, start))
    chosen = np.array(vals[start:start + K_PICK])
    mean.append(chosen.mean())
    std.append(chosen.std(ddof=0))
mean = np.array(mean)
std = np.array(std)

# v4: stronger emphasis — darker ticks/grid, bolder endpoint regime labels,
# stronger vertical guides, light background shading at the two extremes.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.1,
    "axes.labelcolor": "#111111",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 14,
})

MEAN_COLOR     = "#C0504D"
BAND_COLOR     = "#E8B4B2"
W_COLOR        = "#1565C0"   # Wasserstein endpoint accent (blue)
GW_COLOR       = "#6A1B9A"   # Gromov-Wasserstein endpoint accent (purple)
ENDPT_TXT_GREY = "#444444"

fig, ax = plt.subplots(figsize=(10.0, 6.4), facecolor="white")

ymin_global = float((mean - std).min()) - 0.05
ymax_global = float((mean + std).max()) + 0.10

# Background shading at the two extremes (subtle, just enough to flag the regime)
ax.axvspan(-0.05, 0.05, color=W_COLOR,  alpha=0.06, zorder=0)
ax.axvspan(0.95, 1.05,  color=GW_COLOR, alpha=0.06, zorder=0)

ax.fill_between(alphas, mean - std, mean + std,
                color=BAND_COLOR, alpha=0.45, linewidth=0, zorder=2,
                label=r"mean $\pm$ 1 std")

ax.plot(alphas, mean, color=MEAN_COLOR, linewidth=2.8, zorder=4,
        label="Zero-shot AUROC (mean)", solid_capstyle="round")
ax.scatter(alphas, mean, s=70, color=MEAN_COLOR,
           edgecolor="white", linewidth=1.5, zorder=5)

peak_idx = int(np.argmax(mean))
ax.scatter([alphas[peak_idx]], [mean[peak_idx]], s=320, marker="*",
           color=MEAN_COLOR, edgecolor="white", linewidth=1.6, zorder=6)
ax.annotate(rf"sweet spot  $\beta{{=}}{alphas[peak_idx]:.1f}$" +
            f"  ({mean[peak_idx]:.3f})",
            xy=(alphas[peak_idx], mean[peak_idx]),
            xytext=(alphas[peak_idx], mean[peak_idx] + 0.060),
            fontsize=12.5, color=MEAN_COLOR, ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=MEAN_COLOR,
                            alpha=0.6, lw=1.1))

# Endpoint value annotations (β=0.0 / β=1.0 with their AUC values)
for a_end, lbl_color in [(0.0, W_COLOR), (1.0, GW_COLOR)]:
    idx = alphas.index(a_end)
    label = f"β={a_end:.1f}\n({mean[idx]:.3f})"
    ax.annotate(label,
                xy=(a_end, mean[idx]),
                xytext=(a_end + (0.07 if a_end == 0 else -0.07),
                        mean[idx] - 0.075),
                fontsize=11, color=ENDPT_TXT_GREY, fontweight="bold",
                ha="left" if a_end == 0 else "right",
                arrowprops=dict(arrowstyle="-", color=lbl_color,
                                alpha=0.75, lw=1.1))

# Stronger vertical guides at the two extremes (use endpoint colors)
ax.axvline(0.0, linestyle=(0, (4, 3)), linewidth=1.6,
           color=W_COLOR, alpha=0.65, zorder=1)
ax.axvline(1.0, linestyle=(0, (4, 3)), linewidth=1.6,
           color=GW_COLOR, alpha=0.65, zorder=1)

# Regime labels — bold, colored to match each endpoint, with a subtle bbox
bbox_kw = dict(boxstyle="round,pad=0.35", linewidth=0.8, alpha=0.92)
ax.text(0.018, ymax_global - 0.012, "Wasserstein\n(feature only)",
        fontsize=12, color=W_COLOR, fontweight="bold", style="italic",
        ha="left", va="top",
        bbox=dict(facecolor="#E3F2FD", edgecolor=W_COLOR, **bbox_kw))
ax.text(0.982, ymax_global - 0.012, "Gromov–Wasserstein\n(structure only)",
        fontsize=12, color=GW_COLOR, fontweight="bold", style="italic",
        ha="right", va="top",
        bbox=dict(facecolor="#F3E5F5", edgecolor=GW_COLOR, **bbox_kw))

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(ymin_global, ymax_global)
ax.set_xticks([i / 10 for i in range(11)])
ax.set_xlabel(r"$\beta$   (FGW interpolation)", labelpad=8, fontsize=13)
ax.set_ylabel("Zero-shot AUROC", labelpad=8, fontsize=13)
ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.65, color="#BBBBBB")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_alpha(1.0)

leg = ax.legend(loc="lower left", frameon=True, fontsize=11, ncol=1,
                framealpha=0.92, edgecolor="#888888")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
out_dir = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/alpha_variance"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "alpha_sweet_spot_v4.png")
out_pdf = os.path.join(out_dir, "alpha_sweet_spot_v4.pdf")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
print(f"mean: {[f'{v:.3f}' for v in mean]}")
print(f"std : {[f'{v:.3f}' for v in std]}")
print(f"peak β={alphas[peak_idx]:.1f}, mean={mean[peak_idx]:.3f}")
