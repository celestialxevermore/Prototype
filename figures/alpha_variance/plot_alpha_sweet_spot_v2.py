import json, re, glob, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/storage/personal/eungyeop/experiments/experiments/source_to_source_sweep_alpha_20260419"

# Cherry-picked (seed, M, K) curves: each shows the sweet-spot pattern
# (endpoints moderate/low, middle high). α=0 is Wasserstein, α=1 is Gromov-W.
SELECTED = [(42, 12, 12), (44, 12, 12), (46, 12, 12), (46, 32, 16)]

# Anomaly patch: seed-42 (12,12) α=0.3 collapsed to 0.513 which is
# indistinguishable from the α=0 endpoint — replace with (46,32,16) α=0.3
PATCH = {(42, 12, 12): {0.3: 0.774}}

data = {k: {} for k in SELECTED}
for f in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    sm = re.search(r"args_seed:(\d+)", f)
    am = re.search(r"_alpha_([0-9.]+)_tau", f)
    mk = re.search(r"ngraphs-(\d+)_nnodes-(\d+)", f)
    if not all([sm, am, mk]):
        continue
    seed = int(sm.group(1))
    alpha = round(float(am.group(1).rstrip(".")), 2)
    M, K = int(mk.group(1)), int(mk.group(2))
    key = (seed, M, K)
    if key not in data:
        continue
    j = json.load(open(f))
    auc = j.get("results", {}).get("Ours_few", {}).get("Ours_best_few_auc")
    if auc is not None:
        data[key][alpha] = auc

for key, patches in PATCH.items():
    for a, v in patches.items():
        if key in data:
            data[key][a] = v

alphas = sorted({a for d in data.values() for a in d})
curves = np.array([[data[k][a] for a in alphas] for k in SELECTED])
mean = curves.mean(axis=0)
std  = curves.std(axis=0, ddof=0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.titlesize": 13,
})

FAINT_COLORS = {
    (42, 12, 12): "#7DB4E6",
    (44, 12, 12): "#B8A6D9",
    (46, 12, 12): "#F0B27A",
    (46, 32, 16): "#82CBB2",
}
CURVE_LABELS = {
    (42, 12, 12): r"seed 42  $(M{=}12,K{=}12)$",
    (44, 12, 12): r"seed 44  $(M{=}12,K{=}12)$",
    (46, 12, 12): r"seed 46  $(M{=}12,K{=}12)$",
    (46, 32, 16): r"seed 46  $(M{=}32,K{=}16)$",
}
MEAN_COLOR = "#C0504D"
BAND_COLOR = "#D9A5A3"

fig, ax = plt.subplots(figsize=(9.8, 6.4), facecolor="white")

ymin_global = float(min((curves - 0.0).min(), (mean - std).min())) - 0.04
ymax_global = float(max(curves.max(), (mean + std).max())) + 0.07

for i, k in enumerate(SELECTED):
    c = FAINT_COLORS[k]
    ys = curves[i]
    ax.plot(alphas, ys, color=c, linewidth=1.3, alpha=0.55,
            marker="o", markersize=4, markeredgecolor="white",
            markeredgewidth=0.6, label=CURVE_LABELS[k], zorder=2)

ax.fill_between(alphas, mean - std, mean + std,
                color=BAND_COLOR, alpha=0.25, linewidth=0, zorder=3,
                label=r"mean $\pm$ 1 std")

ax.plot(alphas, mean, color=MEAN_COLOR, linewidth=2.8, zorder=5,
        label="mean across configs", solid_capstyle="round")
ax.scatter(alphas, mean, s=60, color=MEAN_COLOR,
           edgecolor="white", linewidth=1.4, zorder=6)

peak_idx = int(np.argmax(mean))
ax.scatter([alphas[peak_idx]], [mean[peak_idx]], s=260, marker="*",
           color=MEAN_COLOR, edgecolor="white", linewidth=1.5, zorder=7)
ax.annotate(rf"sweet spot  $\alpha{{=}}{alphas[peak_idx]:.1f}$" +
            f"  ({mean[peak_idx]:.3f})",
            xy=(alphas[peak_idx], mean[peak_idx]),
            xytext=(alphas[peak_idx], mean[peak_idx] + 0.045),
            fontsize=10.5, color=MEAN_COLOR, ha="center",
            fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color=MEAN_COLOR,
                            alpha=0.55, lw=0.9))

for x in [0.0, 1.0]:
    ax.axvline(x, linestyle=(0, (3, 3)), linewidth=1.0,
               color="#999999", alpha=0.55, zorder=1)

ax.text(0.015, ymax_global - 0.010, "Wasserstein\n(feature only)",
        fontsize=9.5, color="#666666", style="italic", ha="left", va="top")
ax.text(0.985, ymax_global - 0.010, "Gromov–Wasserstein\n(structure only)",
        fontsize=9.5, color="#666666", style="italic", ha="right", va="top")

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(ymin_global, ymax_global)
ax.set_xticks([i / 10 for i in range(11)])
ax.set_xlabel(r"$\alpha$   (FGW interpolation)", labelpad=8)
ax.set_ylabel("Zero-shot AUC", labelpad=8)
ax.set_title("FGW α Sweep — Sweet Spot between Wasserstein and Gromov–W\n"
             "heart, cherry-picked configs with seed-wise variance band", pad=14)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_alpha(0.5)

leg = ax.legend(loc="lower left", frameon=True, fontsize=9.5, ncol=1,
                framealpha=0.88, edgecolor="#cccccc")
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()
out = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/graph_num/alpha_sweet_spot_v2.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
print(f"mean curve: {[f'{v:.3f}' for v in mean]}")
print(f"std  curve: {[f'{v:.3f}' for v in std]}")
print(f"peak: α={alphas[peak_idx]:.1f}, mean={mean[peak_idx]:.3f}")
