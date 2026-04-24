"""
Ideal basis activation radar for cardiovascular / heart-failure datasets.

Concept: each axis = a clinical concept basis the ProtoLLM backbone might
learn. The radar shows which bases we *expect* each dataset to activate
based on its numerical features. Standard CVD-screening datasets should
cluster together; the heart-failure dataset should deviate.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- bases (radar axes) ----------------------------------------------------
BASES = [
    "Demographic",
    "Vitals & Labs\n(CVD risk)",
    "Body\ncomposition",
    "Lifestyle",
    "Cardiac\nfunction",
    "Renal /\nelectrolyte",
]

# ---- ideal activation per dataset (numerical features only) ----------------
# Columns follow BASES order above. Values in [0, 1].
DATASETS = {
    # Should cluster together (CVD screening with vitals/labs dominant)
    "Medical\n(Age, HR, SBP, DBP, BS, CK-MB, Troponin)":
        [0.60, 1.00, 0.00, 0.00, 0.00, 0.00],
    "Cardiovascular\n(age, BP, chol, HR, oldpeak, #vessels)":
        [0.60, 0.90, 0.00, 0.00, 0.00, 0.00],
    "Heart (Statlog)\n(age, trestbps, chol, thalach, oldpeak, ca)":
        [0.60, 0.90, 0.00, 0.00, 0.00, 0.00],
    "heart\n(age, BP, chol, maxHR, oldpeak)":
        [0.60, 0.85, 0.00, 0.00, 0.00, 0.00],

    # Variants (body / lifestyle bumps)
    "Erbil\n(+ height, weight, yrs-smoke, IVSD)":
        [0.70, 0.80, 0.80, 0.50, 0.45, 0.00],
    "SA-heart\n(tobacco, adiposity, obesity, alcohol, type-A)":
        [0.60, 0.50, 0.80, 0.85, 0.00, 0.00],

    # Heart failure — structurally different profile
    "Heart Failure\n(CPK, EF, platelets, creatinine, sodium)":
        [0.50, 0.30, 0.00, 0.00, 0.95, 0.95],
}

# ---- styling ---------------------------------------------------------------
# Blue family = CVD-screening cluster (should look similar).
# Warm neutrals for body/lifestyle variants.
# Red = HF (the outlier).
COLORS = {
    "Medical\n(Age, HR, SBP, DBP, BS, CK-MB, Troponin)":      "#1f77b4",
    "Cardiovascular\n(age, BP, chol, HR, oldpeak, #vessels)": "#2a8fd4",
    "Heart (Statlog)\n(age, trestbps, chol, thalach, oldpeak, ca)": "#4ea8e0",
    "heart\n(age, BP, chol, maxHR, oldpeak)":                 "#7cc0ec",
    "Erbil\n(+ height, weight, yrs-smoke, IVSD)":             "#8c7853",
    "SA-heart\n(tobacco, adiposity, obesity, alcohol, type-A)": "#b59a6b",
    "Heart Failure\n(CPK, EF, platelets, creatinine, sodium)": "#d62728",
}

# ---- plot ------------------------------------------------------------------
N = len(BASES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for name, values in DATASETS.items():
    v = values + values[:1]
    color = COLORS[name]
    lw = 2.8 if "Heart Failure" in name else 1.8
    alpha_line = 1.0 if "Heart Failure" in name else 0.85
    alpha_fill = 0.18 if "Heart Failure" in name else 0.08
    ax.plot(angles, v, color=color, linewidth=lw, label=name, alpha=alpha_line)
    ax.fill(angles, v, color=color, alpha=alpha_fill)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(BASES, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
ax.set_ylim(0, 1.05)
ax.set_rlabel_position(90)
ax.spines["polar"].set_alpha(0.3)
ax.grid(alpha=0.3)

ax.set_title(
    "Ideal basis activation per dataset\n"
    "(CVD-screening datasets cluster; Heart Failure deviates)",
    pad=30, fontsize=13,
)
ax.legend(loc="upper left", bbox_to_anchor=(1.20, 1.05), fontsize=9, frameon=False)

plt.tight_layout()
out = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/ideal_basis_radar.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved: {out}")
