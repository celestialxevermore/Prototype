"""Ideal radar mock v2 — less overlap, jagged shapes, spikes pulled back from edge.
HF panel kept (user said it looks natural). cardio_SAheart de-rectangularized.
"""
import numpy as np
import matplotlib.pyplot as plt

BASIS_COLORS = {
    'B0': '#377eb8', 'B1': '#e41a1c', 'B2': '#ff7f00', 'B3': '#4daf4a',
    'B4': '#67a9cf', 'B5': '#984ea3', 'B6': '#f781bf', 'B7': '#a65628',
}

RED, BLUE, GRAY = '#d62728', '#1f77b4', '#888888'
DIRECTION = {
    'Age': RED, 'age': RED, 'AGE': RED,
    'Heart rate': RED, 'HEART_RATE': RED,
    'maxheartrate': BLUE, 'maxHR': BLUE, 'thalach': BLUE,
    'Systolic BP': RED, 'Diastolic BP': RED,
    'restingBP': RED, 'trestbps': RED,
    'BP_SYSTOLIC': RED, 'BP_DIASTOLIC': RED,
    'SYSTOLIC_BLOOD_PRESSURE': RED,
    'Blood sugar': RED, 'fastingBS': RED,
    'CK-MB': RED, 'Troponin': RED, 'creatinine_phosphokinase': RED,
    'serumcholestrol': RED, 'cholesterol': RED, 'chol': RED, 'LDL_CHOLESTEROL': RED,
    'oldpeak': RED, 'noofmajorvessels': RED, 'ca': RED,
    'CUMULATIVE_TOBACCO': RED, 'YEARS_SMOKING': RED, 'ALCOHOL_CONSUMPTION': RED,
    'TYPE_A_BEHAVIOR': RED, 'ADIPOSITY': RED, 'OBESITY': RED, 'WEIGHT': RED,
    'ejection_fraction': BLUE, 'platelets': BLUE,
    'serum_creatinine': RED, 'serum_sodium': BLUE,
    'HEIGHT': GRAY,
}


def plot_panel(ax, short_name, num_cols, polygons, wpr_thr=0.71):
    n = len(num_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + angles[:1]

    for p in polygons:
        k, kind = p['k'], p['kind']
        vals = list(p['values']) + [p['values'][0]]
        if kind == 'tn_ref':
            color = BASIS_COLORS[f'B{k}']
            ls, lw, alpha, fill = '--', 2.0, 1.0, 0.05
            risk, tag = 'L', 'TN ref'
        elif kind == 'dim':
            color = '#888888'
            ls, lw, alpha, fill = ':', 1.5, 0.4, 0.03
            risk, tag = 'H*', p['label_extra']
        else:
            color = BASIS_COLORS[f'B{k}']
            ls, lw, alpha, fill = '-', 2.5, 1.0, 0.10
            risk, tag = 'H', p['label_extra']
        ax.plot(angles_c, vals, color=color, linewidth=lw, linestyle=ls, alpha=alpha,
                label=f"B{k} [{risk}] ({tag} n_eff={p['n_eff']:.1f})")
        ax.fill(angles_c, vals, color=color, alpha=fill)

    ax.set_xticks(angles)
    ax.set_xticklabels(num_cols, fontsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_color(DIRECTION.get(lbl.get_text(), GRAY))

    all_vals = []
    for line in ax.get_lines():
        all_vals.extend(line.get_ydata())
    vmax = max(2, np.ceil(max(all_vals) + 0.5))
    vmin = min(-2, np.floor(min(all_vals) - 0.5))
    ax.set_ylim(vmin, vmax)
    ticks = [t for t in range(int(vmin), int(vmax) + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:+d}σ' if t != 0 else 'median' for t in ticks],
                       fontsize=6, color='gray')
    ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.32, 1.12))
    ax.set_title(f'[{short_name}] (z-median, clipped ±3σ | WPR≥{wpr_thr:.2f})',
                 fontsize=9, pad=15)


# ============================================================
# v2 polygons: spikes ~1.3-1.5 (not 1.7), more jagged variation,
# less uniform mid-range overlap. Some axes carry only 1-2 bases.
# ============================================================
sources = [
    # Medicaldataset (7 axes)
    ('Medicaldataset', 'Medicaldataset',
     ['Age', 'Heart rate', 'Systolic BP', 'Diastolic BP', 'Blood sugar', 'CK-MB', 'Troponin'],
     [
        # B1: BP focused, jagged elsewhere
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.78', 'n_eff': 412.0,
         'values': [0.4, 0.7, 1.4, 1.2, 0.3, 0.2, 0.3]},
        # B2: scattered mid (general risk)
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 198.0,
         'values': [0.5, 0.3, 0.6, 0.4, 0.7, 0.8, 0.5]},
        # B5: HR focused
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.72', 'n_eff': 156.0,
         'values': [0.3, 1.4, 0.6, 0.4, 0.2, 0.3, 0.2]},
        # B6: cardiac biomarker
        {'k': 6, 'kind': 'high', 'label_extra': 'WPR=0.81', 'n_eff': 287.0,
         'values': [0.4, 0.2, 0.3, 0.4, 0.5, 1.5, 1.4]},
        # B7: Age + BS metabolic
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 134.0,
         'values': [1.4, 0.3, 0.5, 0.4, 1.2, 0.4, 0.2]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 295.0,
         'values': [-0.3, -0.2, -0.5, -0.4, -0.3, -0.6, -0.7]},
     ]),
    # Cardiovascular (6 axes)
    ('Cardiovascular', 'Cardiovascular_Disease_Dataset',
     ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 'noofmajorvessels'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.81', 'n_eff': 389.0,
         'values': [0.5, 1.5, 0.5, -0.3, 1.0, 0.4]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 215.0,
         'values': [0.3, 0.5, 1.5, -0.1, 0.4, 0.6]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 142.0,
         'values': [0.3, 0.3, 0.4, -1.4, 0.4, 0.2]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.76', 'n_eff': 168.0,
         'values': [1.4, 0.5, 0.5, -0.2, 0.6, 1.3]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 180.0,
         'values': [-0.3, -0.4, -0.5, 0.4, -0.6, -0.5]},
     ]),
    # Heart_disease_statlog (6 axes)
    ('Heart_disease_statlog', 'Heart_disease_statlog',
     ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.79', 'n_eff': 105.0,
         'values': [0.4, 1.4, 0.5, -0.3, 1.2, 0.5]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 78.0,
         'values': [0.3, 0.4, 1.5, -0.2, 0.3, 0.6]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 52.0,
         'values': [0.3, 0.4, 0.3, -1.4, 0.3, 0.2]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.75', 'n_eff': 60.0,
         'values': [1.4, 0.5, 0.4, -0.2, 0.5, 1.4]},
        {'k': 6, 'kind': 'dim', 'label_extra': 'WPR=0.52<0.71', 'n_eff': 28.0,
         'values': [0.2, 0.2, 0.1, -0.2, 0.3, 0.7]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 92.0,
         'values': [-0.2, -0.5, -0.6, 0.5, -0.7, -0.4]},
     ]),
    # Erbil (8 axes)
    ('Erbil', 'Erbil_Cardiovascular_Health_Dataset',
     ['AGE', 'HEIGHT', 'WEIGHT', 'YEARS_SMOKING', 'LDL_CHOLESTEROL',
      'HEART_RATE', 'BP_SYSTOLIC', 'BP_DIASTOLIC'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.80', 'n_eff': 98.0,
         'values': [0.4, 0.0, 0.3, 0.5, 0.3, 0.3, 1.4, 1.3]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 55.0,
         'values': [0.4, 0.1, 0.5, 0.4, 1.5, 0.3, 0.4, 0.3]},
        {'k': 4, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 42.0,
         'values': [0.3, 0.2, 1.3, 1.4, 0.5, 0.3, 0.4, 0.4]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 38.0,
         'values': [0.3, 0.0, 0.2, 0.3, 0.4, 1.4, 0.5, 0.4]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 45.0,
         'values': [1.4, 0.0, 0.4, 0.6, 0.4, 0.2, 0.5, 0.4]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 175.0,
         'values': [-0.3, -0.1, -0.4, -0.3, -0.5, -0.2, -0.6, -0.5]},
     ]),
    # cardio_SAheart (8 axes) — jagged, no rectangle
    ('cardio_SAheart', 'cardio_SAheart',
     ['SYSTOLIC_BLOOD_PRESSURE', 'CUMULATIVE_TOBACCO', 'LDL_CHOLESTEROL', 'ADIPOSITY',
      'TYPE_A_BEHAVIOR', 'OBESITY', 'ALCOHOL_CONSUMPTION', 'AGE'],
     [
        # B1: BP spike, varied valleys elsewhere
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 88.0,
         'values': [1.4, 0.3, 0.4, 0.2, 0.5, 0.3, 0.2, 0.7]},
        # B2: LDL spike, scattered
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 67.0,
         'values': [0.3, 0.5, 1.5, 0.7, 0.2, 0.4, 0.3, 0.4]},
        # B4: lifestyle-cluster (tobacco+adiposity+obesity+alcohol multi)
        {'k': 4, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 51.0,
         'values': [0.4, 1.3, 0.5, 1.3, 0.3, 1.4, 1.2, 0.3]},
        # B5: TYPE_A behavior spike
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 44.0,
         'values': [0.3, 0.4, 0.2, 0.5, 1.4, 0.3, 0.4, 0.4]},
        # B7: Age spike
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 58.0,
         'values': [0.5, 0.3, 0.4, 0.3, 0.2, 0.4, 0.3, 1.4]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 215.0,
         'values': [-0.4, -0.3, -0.5, -0.4, -0.2, -0.4, -0.3, -0.3]},
     ]),
    # heart_failure (kept similar — user said this looks natural)
    ('heart_failure', 'heart_failure_clinical_records',
     ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
      'serum_creatinine', 'serum_sodium'],
     [
        {'k': 3, 'kind': 'high', 'label_extra': 'WPR=0.83', 'n_eff': 178.0,
         'values': [0.6, 1.1, -1.4, -0.3, 1.3, -1.2]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 65.0,
         'values': [0.4, 0.5, -0.4, -1.4, 0.5, -0.3]},
        {'k': 6, 'kind': 'high', 'label_extra': 'WPR=0.75', 'n_eff': 95.0,
         'values': [0.4, 1.4, -0.5, -0.3, 0.6, -0.4]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 88.0,
         'values': [1.4, 0.5, -0.4, -0.3, 1.2, -0.4]},
        {'k': 1, 'kind': 'dim', 'label_extra': 'WPR=0.42<0.71', 'n_eff': 38.0,
         'values': [0.2, 0.0, -0.3, 0.0, 0.2, -0.2]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 140.0,
         'values': [-0.3, -0.5, 0.6, 0.3, -0.5, 0.5]},
     ]),
    # heart (with fastingBS)
    ('heart', 'heart',
     ['age', 'restingBP', 'cholesterol', 'fastingBS', 'maxHR', 'oldpeak'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.76', 'n_eff': 298.0,
         'values': [0.4, 1.4, 0.6, 0.3, -0.4, 1.2]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 164.0,
         'values': [0.3, 0.4, 1.5, 0.4, -0.2, 0.4]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 112.0,
         'values': [0.3, 0.4, 0.4, 0.3, -1.4, 0.3]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 138.0,
         'values': [1.4, 0.5, 0.4, 1.2, -0.3, 0.5]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 152.0,
         'values': [-0.2, -0.5, -0.6, -0.3, 0.5, -0.7]},
     ]),
]

M = len(sources)
fig, axes = plt.subplots(1, M, figsize=(6 * M, 6), subplot_kw=dict(polar=True))
for i, (short, _, num_cols, polys) in enumerate(sources):
    plot_panel(axes[i], short, num_cols, polys)

plt.tight_layout()
out_path = '/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/ideal_radar_mock_4panels_2.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
