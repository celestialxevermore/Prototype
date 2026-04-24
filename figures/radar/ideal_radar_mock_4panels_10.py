"""Ideal radar mock v10 — same shapes as v9 but legend shows only realistic sample count
(n=NNN) instead of WPR / n_eff jargon. Matches the look of actual model radar outputs.
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


def plot_panel(ax, short_name, num_cols, polygons, wpr_thr=0.71, show_n=True):
    n_axes = len(num_cols)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_c = angles + angles[:1]

    for p in polygons:
        k, kind = p['k'], p['kind']
        vals = list(p['values']) + [p['values'][0]]
        if kind == 'tn_ref':
            color = BASIS_COLORS[f'B{k}']
            ls, lw, alpha, fill = '--', 2.0, 1.0, 0.05
            risk, set_tag = 'L', 'TN'
        elif kind == 'dim':
            color = '#888888'
            ls, lw, alpha, fill = ':', 1.5, 0.4, 0.03
            risk, set_tag = 'H*', 'TP'
        else:
            color = BASIS_COLORS[f'B{k}']
            ls, lw, alpha, fill = '-', 2.5, 1.0, 0.10
            risk, set_tag = 'H', 'TP'

        if show_n:
            label = f"B{k} [{risk}] ({set_tag} n={p['n']})"
        else:
            label = f"B{k} [{risk}]"

        ax.plot(angles_c, vals, color=color, linewidth=lw, linestyle=ls, alpha=alpha,
                label=label)
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
# v10 — same polygon geometry as v9; only legend metadata changed.
# Sample counts (n) are realistic integers reflecting per-basis TP mass
# observed in actual model radar outputs.
# ============================================================
sources = [
    ('Medicaldataset', 'Medicaldataset',
     ['Age', 'Heart rate', 'Systolic BP', 'Diastolic BP', 'Blood sugar', 'CK-MB', 'Troponin'],
     [
        {'k': 1, 'kind': 'high', 'n': 264,
         'values': [0.3, 0.4, 1.1, 1.0, 0.2, 0.3, 0.3]},
        {'k': 2, 'kind': 'high', 'n': 87,
         'values': [0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2]},
        {'k': 5, 'kind': 'high', 'n': 150,
         'values': [0.2, 1.1, 0.4, 0.3, -0.2, 0.2, 0.2]},
        {'k': 6, 'kind': 'high', 'n': 148,
         'values': [0.2, 0.3, 0.2, 0.1, 0.3, 1.1, 1.0]},
        {'k': 7, 'kind': 'high', 'n': 94,
         'values': [1.1, 0.3, 0.0, 0.0, 0.9, 0.2, 0.3]},
        {'k': 0, 'kind': 'tn_ref', 'n': 206,
         'values': [-0.3, -0.2, -0.5, -0.4, -0.3, -0.6, -0.7]},
     ]),
    ('Cardiovascular', 'Cardiovascular_Disease_Dataset',
     ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 'noofmajorvessels'],
     [
        {'k': 1, 'kind': 'high', 'n': 145,
         'values': [0.3, 1.1, 0.4, -0.2, 0.8, 0.3]},
        {'k': 2, 'kind': 'high', 'n': 70,
         'values': [0.2, 0.3, 1.1, -0.1, 0.3, 0.4]},
        {'k': 5, 'kind': 'high', 'n': 95,
         'values': [0.3, 0.4, 0.3, -0.5, 1.0, 0.2]},
        {'k': 7, 'kind': 'high', 'n': 52,
         'values': [1.1, 0.4, 0.3, -0.2, 0.3, 1.0]},
        {'k': 0, 'kind': 'tn_ref', 'n': 139,
         'values': [-0.3, -0.4, -0.5, 0.4, -0.6, -0.5]},
     ]),
    ('Erbil', 'Erbil_Cardiovascular_Health_Dataset',
     ['AGE', 'HEIGHT', 'WEIGHT', 'YEARS_SMOKING', 'LDL_CHOLESTEROL',
      'HEART_RATE', 'BP_SYSTOLIC', 'BP_DIASTOLIC'],
     [
        {'k': 1, 'kind': 'high', 'n': 189,
         'values': [0.3, 0.0, 0.3, 0.4, 0.3, 0.3, 1.1, 1.0]},
        {'k': 2, 'kind': 'high', 'n': 58,
         'values': [0.3, 0.1, 0.3, 0.3, 1.1, 0.2, 0.3, 0.2]},
        {'k': 4, 'kind': 'high', 'n': 44,
         'values': [0.2, 0.1, 1.0, 1.1, 0.4, 0.2, 0.4, 0.3]},
        {'k': 5, 'kind': 'high', 'n': 31,
         'values': [0.1, 0.0, 0.1, 0.1, 0.2, 1.1, 0.2, 0.1]},
        {'k': 7, 'kind': 'high', 'n': 22,
         'values': [1.1, 0.0, 0.3, 0.4, -0.2, -0.1, 0.3, 0.3]},
        {'k': 0, 'kind': 'tn_ref', 'n': 139,
         'values': [-0.3, -0.1, -0.4, -0.3, -0.5, -0.2, -0.6, -0.5]},
     ]),
    ('cardio_SAheart', 'cardio_SAheart',
     ['SYSTOLIC_BLOOD_PRESSURE', 'CUMULATIVE_TOBACCO', 'LDL_CHOLESTEROL', 'ADIPOSITY',
      'TYPE_A_BEHAVIOR', 'OBESITY', 'ALCOHOL_CONSUMPTION', 'AGE'],
     [
        {'k': 1, 'kind': 'high', 'n': 28,
         'values': [1.1, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.4]},
        {'k': 2, 'kind': 'high', 'n': 22,
         'values': [0.2, 0.3, 1.1, 0.3, 0.2, 0.3, 0.2, 0.3]},
        {'k': 4, 'kind': 'high', 'n': 18,
         'values': [0.2, 1.1, 0.2, 0.5, 0.2, 1.0, 0.3, 0.2]},
        {'k': 5, 'kind': 'high', 'n': 15,
         'values': [0.1, 0.1, 0.1, 0.2, 1.1, 0.1, 0.1, 0.1]},
        {'k': 7, 'kind': 'high', 'n': 20,
         'values': [0.2, -0.1, 0.2, 0.1, 0.1, 0.2, -0.1, 1.1]},
        {'k': 0, 'kind': 'tn_ref', 'n': 165,
         'values': [-0.4, -0.3, -0.5, -0.4, -0.2, -0.4, -0.3, -0.3]},
     ]),
    ('heart_failure', 'heart_failure_clinical_records',
     ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
      'serum_creatinine', 'serum_sodium'],
     [
        {'k': 3, 'kind': 'high', 'n': 54,
         'values': [0.4, 0.8, -1.2, -0.3, 1.0, -1.0]},
        {'k': 5, 'kind': 'high', 'n': 14,
         'values': [0.2, 0.3, -0.4, -1.2, 0.3, -0.3]},
        {'k': 6, 'kind': 'high', 'n': 18,
         'values': [0.2, 1.1, -0.5, -0.3, 0.3, -0.4]},
        {'k': 7, 'kind': 'high', 'n': 16,
         'values': [1.1, 0.3, -0.4, -0.3, 0.9, -0.4]},
        {'k': 1, 'kind': 'dim', 'n': 8,
         'values': [0.1, 0.0, -0.3, 0.0, 0.1, -0.2]},
        {'k': 0, 'kind': 'tn_ref', 'n': 75,
         'values': [-0.3, -0.5, 0.6, 0.3, -0.5, 0.5]},
     ]),
    ('heart', 'heart',
     ['age', 'restingBP', 'cholesterol', 'fastingBS', 'maxHR', 'oldpeak'],
     [
        {'k': 1, 'kind': 'high', 'n': 165,
         'values': [0.3, 1.1, 0.4, 0.3, -0.3, 0.9]},
        {'k': 2, 'kind': 'high', 'n': 121,
         'values': [0.4, 0.4, 1.1, 0.3, 0.0, 0.4]},
        {'k': 5, 'kind': 'high', 'n': 54,
         'values': [0.2, 0.2, 0.8, 0.3, -1.2, 0.2]},
        {'k': 7, 'kind': 'high', 'n': 87,
         'values': [1.1, 0.0, 0.3, 0.9, -0.2, 0.3]},
        {'k': 0, 'kind': 'tn_ref', 'n': 125,
         'values': [-0.2, -0.5, -0.6, -0.3, 0.5, -0.7]},
     ]),
]


OUT_DIR = '/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/radar'


def render(out_path_base, show_n):
    M = len(sources)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 6), subplot_kw=dict(polar=True))
    for i, (short, _, num_cols, polys) in enumerate(sources):
        plot_panel(axes[i], short, num_cols, polys, show_n=show_n)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = f'{out_path_base}.{ext}'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    render(f'{OUT_DIR}/ideal_radar_mock_4panels_10_with', show_n=True)
    render(f'{OUT_DIR}/ideal_radar_mock_4panels_10_no', show_n=False)
