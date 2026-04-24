"""Ideal radar mock v6 — add containment + sudden-drop variety.
- Some bases now small/contained INSIDE others (e.g., Medical B2, Heart B5)
- Some bases have sudden drops on adjacent axes (close to B0 ref)
- Cardiovascular B5 vs Heart B5 visually different (Cardio has secondary oldpeak rise)
- Medicaldataset hexagonal feel reduced — B2 now small contained, B7 has BP drop
- FastingBS NOT made low for any basis (per user feedback on real-output weirdness)
- Two outputs: with/without n_eff
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


def plot_panel(ax, short_name, num_cols, polygons, wpr_thr=0.71, show_neff=True):
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

        if show_neff:
            label = f"B{k} [{risk}] ({tag} n_eff={p['n_eff']:.1f})"
        else:
            label = f"B{k} [{risk}] ({tag})" if kind != 'tn_ref' else f"B{k} [{risk}] (TN ref)"

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
# v6 polygons: containment + sudden drops + cross-panel variation.
# ============================================================
sources = [
    # Medicaldataset
    # B2: small contained (noisy general)  →  breaks hexagonal feel
    # B5: HR spike + sudden drop on BS/Trop (metabolic side)
    # B7: Age+BS spike + sudden drop on BP
    ('Medicaldataset', 'Medicaldataset',
     ['Age', 'Heart rate', 'Systolic BP', 'Diastolic BP', 'Blood sugar', 'CK-MB', 'Troponin'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.78', 'n_eff': 287.4,
         'values': [0.5, 0.7, 1.5, 1.3, 0.4, 0.5, 0.6]},
        # B2 small contained — fits inside others
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 156.8,
         'values': [0.3, 0.4, 0.4, 0.3, 0.4, 0.5, 0.4]},
        # B5 HR spike + BS/Trop sudden drop
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.72', 'n_eff': 98.3,
         'values': [0.4, 1.4, 0.7, 0.5, -0.2, 0.4, 0.3]},
        {'k': 6, 'kind': 'high', 'label_extra': 'WPR=0.81', 'n_eff': 215.6,
         'values': [0.4, 0.5, 0.4, 0.3, 0.5, 1.5, 1.4]},
        # B7 Age+BS spike + BP sudden drop (close to B0)
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 142.2,
         'values': [1.4, 0.5, 0.0, 0.0, 1.2, 0.4, 0.5]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 358.5,
         'values': [-0.3, -0.2, -0.5, -0.4, -0.3, -0.6, -0.7]},
     ]),
    # Cardiovascular  —  B5 distinct: maxHR drop + secondary oldpeak rise
    ('Cardiovascular', 'Cardiovascular_Disease_Dataset',
     ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 'noofmajorvessels'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.81', 'n_eff': 178.6,
         'values': [0.5, 1.5, 0.6, -0.2, 1.1, 0.5]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 124.3,
         'values': [0.4, 0.5, 1.5, -0.1, 0.5, 0.6]},
        # B5 wider: maxHR drop + oldpeak secondary rise (different from heart B5)
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 78.4,
         'values': [0.6, 0.7, 0.6, -1.4, 1.0, 0.5]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.76', 'n_eff': 105.7,
         'values': [1.4, 0.6, 0.5, -0.2, 0.5, 1.3]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 246.8,
         'values': [-0.3, -0.4, -0.5, 0.4, -0.6, -0.5]},
     ]),
    # Heart_disease_statlog  —  B5 small contained + thalach drop
    ('Heart_disease_statlog', 'Heart_disease_statlog',
     ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.79', 'n_eff': 42.3,
         'values': [0.5, 1.5, 0.5, -0.2, 1.1, 0.6]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 28.7,
         'values': [0.4, 0.5, 1.5, -0.1, 0.4, 0.5]},
        # B5 small contained + thalach drop
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 21.4,
         'values': [0.3, 0.4, 0.4, -1.4, 0.3, 0.3]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.75', 'n_eff': 24.6,
         'values': [1.4, 0.6, 0.5, -0.1, 0.5, 1.3]},
        {'k': 6, 'kind': 'dim', 'label_extra': 'WPR=0.52<0.71', 'n_eff': 12.8,
         'values': [0.2, 0.2, 0.3, -0.2, 0.3, 0.7]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 84.2,
         'values': [-0.2, -0.5, -0.6, 0.5, -0.7, -0.4]},
     ]),
    # Erbil  —  B7 has sudden drop on LDL/HR; B5 small contained
    ('Erbil', 'Erbil_Cardiovascular_Health_Dataset',
     ['AGE', 'HEIGHT', 'WEIGHT', 'YEARS_SMOKING', 'LDL_CHOLESTEROL',
      'HEART_RATE', 'BP_SYSTOLIC', 'BP_DIASTOLIC'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.80', 'n_eff': 95.6,
         'values': [0.5, 0.0, 0.5, 0.6, 0.5, 0.5, 1.4, 1.3]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 58.2,
         'values': [0.5, 0.1, 0.5, 0.5, 1.5, 0.4, 0.5, 0.4]},
        # B4 retained: BP slightly above normal
        {'k': 4, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 47.3,
         'values': [0.4, 0.2, 1.3, 1.4, 0.6, 0.4, 0.6, 0.5]},
        # B5 small contained
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 38.9,
         'values': [0.3, 0.0, 0.3, 0.3, 0.4, 1.4, 0.4, 0.3]},
        # B7 Age spike + LDL/HR sudden drop
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 51.4,
         'values': [1.4, 0.0, 0.5, 0.6, -0.2, -0.1, 0.5, 0.5]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 187.2,
         'values': [-0.3, -0.1, -0.4, -0.3, -0.5, -0.2, -0.6, -0.5]},
     ]),
    # cardio_SAheart  —  B5 small contained, B7 sudden drops
    ('cardio_SAheart', 'cardio_SAheart',
     ['SYSTOLIC_BLOOD_PRESSURE', 'CUMULATIVE_TOBACCO', 'LDL_CHOLESTEROL', 'ADIPOSITY',
      'TYPE_A_BEHAVIOR', 'OBESITY', 'ALCOHOL_CONSUMPTION', 'AGE'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 38.4,
         'values': [1.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, 0.6]},
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 28.7,
         'values': [0.4, 0.5, 1.5, 0.5, 0.4, 0.5, 0.4, 0.5]},
        {'k': 4, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 22.5,
         'values': [0.4, 1.5, 0.4, 0.7, 0.3, 1.3, 0.5, 0.4]},
        # B5 small contained (TYPE_A spike, low elsewhere)
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.69', 'n_eff': 19.8,
         'values': [0.3, 0.3, 0.3, 0.4, 1.4, 0.3, 0.3, 0.3]},
        # B7 Age spike + tobacco/alcohol sudden drops
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 24.6,
         'values': [0.4, -0.1, 0.4, 0.3, 0.3, 0.4, -0.1, 1.4]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 152.3,
         'values': [-0.4, -0.3, -0.5, -0.4, -0.2, -0.4, -0.3, -0.3]},
     ]),
    # heart_failure (kept)
    ('heart_failure', 'heart_failure_clinical_records',
     ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
      'serum_creatinine', 'serum_sodium'],
     [
        {'k': 3, 'kind': 'high', 'label_extra': 'WPR=0.83', 'n_eff': 32.4,
         'values': [0.6, 1.0, -1.4, -0.3, 1.3, -1.2]},
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.70', 'n_eff': 14.7,
         'values': [0.4, 0.5, -0.4, -1.4, 0.5, -0.3]},
        {'k': 6, 'kind': 'high', 'label_extra': 'WPR=0.75', 'n_eff': 18.3,
         'values': [0.4, 1.4, -0.5, -0.3, 0.5, -0.4]},
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.74', 'n_eff': 16.2,
         'values': [1.4, 0.5, -0.4, -0.3, 1.2, -0.4]},
        {'k': 1, 'kind': 'dim', 'label_extra': 'WPR=0.42<0.71', 'n_eff': 7.5,
         'values': [0.2, 0.0, -0.3, 0.0, 0.2, -0.2]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 89.6,
         'values': [-0.3, -0.5, 0.6, 0.3, -0.5, 0.5]},
     ]),
    # heart  —  B5 small contained + maxHR drop; B2 wider (containment of B5 partially); fastingBS kept moderate-high
    ('heart', 'heart',
     ['age', 'restingBP', 'cholesterol', 'fastingBS', 'maxHR', 'oldpeak'],
     [
        {'k': 1, 'kind': 'high', 'label_extra': 'WPR=0.76', 'n_eff': 142.6,
         'values': [0.5, 1.4, 0.6, 0.5, -0.3, 1.2]},
        # B2 wider with chol spike — partially contains B5
        {'k': 2, 'kind': 'high', 'label_extra': 'WPR=0.71', 'n_eff': 87.3,
         'values': [0.6, 0.7, 1.5, 0.5, 0.0, 0.6]},
        # B5 small contained + maxHR sudden drop
        {'k': 5, 'kind': 'high', 'label_extra': 'WPR=0.68', 'n_eff': 64.8,
         'values': [0.3, 0.3, 0.3, 0.4, -1.4, 0.2]},
        # B7 Age+BS spike + BP sudden drop (close to B0 ref)
        {'k': 7, 'kind': 'high', 'label_extra': 'WPR=0.73', 'n_eff': 78.4,
         'values': [1.4, 0.0, 0.5, 1.2, -0.2, 0.5]},
        {'k': 0, 'kind': 'tn_ref', 'n_eff': 298.5,
         'values': [-0.2, -0.5, -0.6, -0.3, 0.5, -0.7]},
     ]),
]


def render(out_path, show_neff):
    M = len(sources)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 6), subplot_kw=dict(polar=True))
    for i, (short, _, num_cols, polys) in enumerate(sources):
        plot_panel(axes[i], short, num_cols, polys, show_neff=show_neff)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


render('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/ideal_radar_mock_4panels_6_with.png',
       show_neff=True)
render('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/ideal_radar_mock_4panels_6_no.png',
       show_neff=False)
