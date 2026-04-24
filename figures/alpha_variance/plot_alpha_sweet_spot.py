import json, re, glob, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/storage/personal/eungyeop/experiments/experiments/source_to_source_sweep_alpha_20260419"
TARGET = {42: (12, 12), 48: (8, 16)}

data = {s: {} for s in TARGET}
for f in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    sm = re.search(r"args_seed:(\d+)", f)
    am = re.search(r"_alpha_([0-9.]+)_tau", f)
    mk = re.search(r"ngraphs-(\d+)_nnodes-(\d+)", f)
    if not all([sm, am, mk]):
        continue
    seed = int(sm.group(1))
    alpha = float(am.group(1).rstrip("."))
    M, K = int(mk.group(1)), int(mk.group(2))
    if seed not in TARGET or (M, K) != TARGET[seed]:
        continue
    j = json.load(open(f))
    auc = j.get("results", {}).get("Ours_few", {}).get("Ours_best_few_auc")
    if auc is not None:
        data[seed][alpha] = auc

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

COLORS = {42: "#4A90D9", 48: "#E88B8B"}
LABELS = {42: r"seed 42   $(M{=}12,\ K{=}12)$", 48: r"seed 48   $(M{=}8,\ K{=}16)$"}

fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor="white")

all_y = [v for d in data.values() for v in d.values()]
ymin_global = min(all_y) - 0.05
ymax_global = max(all_y) + 0.09

for seed in sorted(data):
    d = data[seed]
    xs = sorted(d.keys())
    ys = np.array([d[a] for a in xs])
    c = COLORS[seed]

    ax.fill_between(xs, ys, ymin_global, color=c, alpha=0.13, zorder=1, linewidth=0)
    ax.plot(xs, ys, color=c, linewidth=2.4, zorder=3,
            label=LABELS[seed], solid_capstyle="round")
    ax.scatter(xs, ys, s=55, color=c, edgecolor="white", linewidth=1.3, zorder=4)

    peak_idx = int(np.argmax(ys))
    ax.scatter([xs[peak_idx]], [ys[peak_idx]], s=210, marker="*",
               color=c, edgecolor="white", linewidth=1.4, zorder=5)
    ax.annotate(rf"peak  $\alpha{{=}}{xs[peak_idx]:.1f}$ " + f"({ys[peak_idx]:.3f})",
                xy=(xs[peak_idx], ys[peak_idx]),
                xytext=(xs[peak_idx], ys[peak_idx] + 0.028),
                fontsize=9.5, color=c, ha="center", fontweight="semibold",
                arrowprops=dict(arrowstyle="-", color=c, alpha=0.45, lw=0.8))

for x in [0.0, 1.0]:
    ax.axvline(x, linestyle=(0, (3, 3)), linewidth=1.0, color="#999999", alpha=0.55, zorder=2)

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
             "heart, per-seed best $(M,K)$", pad=14)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_alpha(0.5)

leg = ax.legend(loc="lower left", frameon=True, fontsize=10.5, ncol=1,
                framealpha=0.85, edgecolor="#cccccc")
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()
out = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/graph_num/alpha_sweet_spot.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
