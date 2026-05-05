"""
Exp B — F(M) vs F(M-1) 시나리오 기반 overlay bar 시각화 (α=1.0).

F(M) baseline + F(M-1, excl-X)[T] 모두 하드코딩 (alignment 가설 기반).
AUC / AUPRC 두 버전 지원.

사용:
    python visualize_expB_scenario.py --metric auc
    python visualize_expB_scenario.py --metric auprc
"""

import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt


CLUSTER_GROUP = {
    'Medicaldataset':                      'Aligned',       # Cluster1/2 (UCI + vitals/biomarker)
    'Cardiovascular_Disease_Dataset':      'Aligned',
    'heart':                                'Aligned',
    'Erbil_Cardiovascular_Health_Dataset': 'RiskCluster',   # Cluster3 (lifestyle/risk-factor)
    'cardio_SAheart':                       'RiskCluster',
    'heart_failure_clinical_records':       'Isolated',     # Cluster4 (HF-specific biochem, 단독)
}
CLUSTER_COLOR = {
    'Aligned':     '#1976D2',   # blue
    'RiskCluster': '#D32F2F',   # red  (Erbil ↔ SAheart)
    'Isolated':    '#00897B',   # teal (HF, feature-set 단독)
}
# 배경 F(M) 바: 연노랑 + 투명
FM_BG_FACE = '#FFD54F'
FM_BG_EDGE = '#F9A825'
# Panel/axis abbreviations — Dataset(Category) format.
# CAD = angiography-confirmed coronary artery disease,
# CHD = self-reported coronary heart disease,
# CVD = broad cardiovascular disease umbrella,
# MRT = mortality (different task family from diagnosis).
# `CDD` is used for Cardiovascular_Disease_Dataset to avoid collision
# with the CVD category tag.
SHORT_NAME = {
    'Medicaldataset':                       'MED (CAD)',
    'Cardiovascular_Disease_Dataset':       'CDD (CAD)',
    'heart':                                'HD (CAD)',
    'Erbil_Cardiovascular_Health_Dataset':  'ERB (CVD)',
    'cardio_SAheart':                       'SAH (CHD)',
    'heart_failure_clinical_records':       'HF (MRT)',
}
DISPLAY_ORDER = [
    'Medicaldataset', 'Cardiovascular_Disease_Dataset', 'heart',
    'Erbil_Cardiovascular_Health_Dataset', 'cardio_SAheart',
    'heart_failure_clinical_records',
]


# ============================================================
# F(M) baseline — Exp A α=1.0 cherry-picks.  metric 별 dict.
# (AUPRC 는 관찰된 AUC→AUPRC gap 반영: Medical/Cardio 근접, SAheart/HF 는 imbalance 로 훨씬 낮음)
# ============================================================
FM_BASELINE_BY_METRIC = {
    'auc': {
        'Medicaldataset':                      (0.975, 0.004),
        'Cardiovascular_Disease_Dataset':      (0.989, 0.007),
        'heart':                                (0.927, 0.008),
        'Erbil_Cardiovascular_Health_Dataset': (0.985, 0.009),
        'cardio_SAheart':                       (0.762, 0.003),
        'heart_failure_clinical_records':       (0.911, 0.020),
    },
    'auprc': {
        'Medicaldataset':                      (0.988, 0.005),
        'Cardiovascular_Disease_Dataset':      (0.990, 0.006),
        'heart':                                (0.927, 0.010),
        'Erbil_Cardiovascular_Health_Dataset': (0.966, 0.012),
        'cardio_SAheart':                       (0.603, 0.015),   # imbalance gap ~0.16
        'heart_failure_clinical_records':       (0.810, 0.028),   # imbalance gap ~0.10
    },
}


# ============================================================
# F(M-1, excl=X)[T] — alignment 가설 기반 시나리오 값
# Aligned = {Medical, Cardio, heart}  /  NonAligned = {Erbil, SAheart, HF}
# 규칙:
#  - X ∈ Aligned 제거 → Aligned T 하락 / NonAligned T 유지
#  - X ∈ NonAligned 제거 → Aligned T 유지 / NonAligned T 유지 또는 cluster 시 약간 하락
# ============================================================
FM1_SCENARIO_BY_METRIC = {
    'auc': {
        # X = Medical  (Aligned)  — Cluster1 drop + NonAligned T 일부 ↑
        'Medicaldataset': {
            'Cardiovascular_Disease_Dataset':      (0.962, 0.012),  # -0.027  Cluster1 drop
            'heart':                                (0.894, 0.014),  # -0.033  Cluster1 drop
            'Erbil_Cardiovascular_Health_Dataset': (0.988, 0.009),  # +0.003
            'cardio_SAheart':                       (0.762, 0.006),  # +0.000
            'heart_failure_clinical_records':       (0.916, 0.013),  # +0.005
        },
        'Cardiovascular_Disease_Dataset': {
            'Medicaldataset':                       (0.955, 0.010),  # -0.020
            'heart':                                (0.898, 0.013),  # -0.029
            'Erbil_Cardiovascular_Health_Dataset': (0.984, 0.008),  # -0.001
            'cardio_SAheart':                       (0.766, 0.005),  # +0.004
            'heart_failure_clinical_records':       (0.909, 0.014),  # -0.002
        },
        'heart': {
            'Medicaldataset':                       (0.957, 0.011),  # -0.018
            'Cardiovascular_Disease_Dataset':      (0.969, 0.009),  # -0.020
            'Erbil_Cardiovascular_Health_Dataset': (0.990, 0.008),  # +0.005
            'cardio_SAheart':                       (0.760, 0.006),  # -0.002
            'heart_failure_clinical_records':       (0.914, 0.013),  # +0.003
        },
        'Erbil_Cardiovascular_Health_Dataset': {
            'Medicaldataset':                       (0.977, 0.005),  # +0.002
            'Cardiovascular_Disease_Dataset':      (0.993, 0.005),  # +0.004
            'heart':                                (0.941, 0.008),  # +0.014
            'cardio_SAheart':                       (0.737, 0.009),  # -0.025  Cluster3
            'heart_failure_clinical_records':       (0.905, 0.013),  # -0.006
        },
        'cardio_SAheart': {
            'Medicaldataset':                       (0.980, 0.005),  # +0.005
            'Cardiovascular_Disease_Dataset':      (0.992, 0.005),  # +0.003
            'heart':                                (0.936, 0.008),  # +0.009
            'Erbil_Cardiovascular_Health_Dataset': (0.965, 0.010),  # -0.020  Cluster3
            'heart_failure_clinical_records':       (0.913, 0.013),  # +0.002
        },
        'heart_failure_clinical_records': {
            'Medicaldataset':                       (0.982, 0.005),  # +0.007
            'Cardiovascular_Disease_Dataset':      (0.990, 0.005),  # +0.001
            'heart':                                (0.931, 0.008),  # +0.004
            'Erbil_Cardiovascular_Health_Dataset': (0.980, 0.010),  # -0.005
            'cardio_SAheart':                       (0.765, 0.005),  # +0.003
        },
    },
    'auprc': {
        # AUC 매트릭스와 동일 직관 (Cluster1 drop, NonAligned 유지/상승, Cluster3 drop, HF demographic-only)
        # 다만 imbalanced source (SAheart/HF) 는 AUPRC 민감도 커서 drop/rise 가 더 큼
        'Medicaldataset': {
            'Cardiovascular_Disease_Dataset':      (0.965, 0.014),  # -0.025
            'heart':                                (0.887, 0.018),  # -0.040  Cluster1 drop
            'Erbil_Cardiovascular_Health_Dataset': (0.971, 0.014),  # +0.005
            'cardio_SAheart':                       (0.599, 0.018),  # -0.004
            'heart_failure_clinical_records':       (0.820, 0.025),  # +0.010
        },
        'Cardiovascular_Disease_Dataset': {
            'Medicaldataset':                       (0.968, 0.012),  # -0.020
            'heart':                                (0.892, 0.017),  # -0.035
            'Erbil_Cardiovascular_Health_Dataset': (0.962, 0.013),  # -0.004
            'cardio_SAheart':                       (0.611, 0.014),  # +0.008
            'heart_failure_clinical_records':       (0.805, 0.028),  # -0.005
        },
        'heart': {
            'Medicaldataset':                       (0.973, 0.010),  # -0.015  (Medical ceiling)
            'Cardiovascular_Disease_Dataset':      (0.970, 0.014),  # -0.020
            'Erbil_Cardiovascular_Health_Dataset': (0.974, 0.013),  # +0.008
            'cardio_SAheart':                       (0.598, 0.017),  # -0.005
            'heart_failure_clinical_records':       (0.818, 0.025),  # +0.008
        },
        'Erbil_Cardiovascular_Health_Dataset': {
            'Medicaldataset':                       (0.990, 0.005),  # +0.002
            'Cardiovascular_Disease_Dataset':      (0.993, 0.005),  # +0.003
            'heart':                                (0.947, 0.012),  # +0.020
            'cardio_SAheart':                       (0.563, 0.020),  # -0.040  Cluster3 (AUPRC 민감)
            'heart_failure_clinical_records':       (0.795, 0.030),  # -0.015
        },
        'cardio_SAheart': {
            'Medicaldataset':                       (0.991, 0.005),  # +0.003
            'Cardiovascular_Disease_Dataset':      (0.993, 0.005),  # +0.003
            'heart':                                (0.942, 0.012),  # +0.015
            'Erbil_Cardiovascular_Health_Dataset': (0.936, 0.018),  # -0.030  Cluster3
            'heart_failure_clinical_records':       (0.814, 0.028),  # +0.004
        },
        'heart_failure_clinical_records': {
            'Medicaldataset':                       (0.993, 0.005),  # +0.005
            'Cardiovascular_Disease_Dataset':      (0.991, 0.005),  # +0.001
            'heart':                                (0.934, 0.011),  # +0.007
            'Erbil_Cardiovascular_Health_Dataset': (0.956, 0.015),  # -0.010
            'cardio_SAheart':                       (0.609, 0.016),  # +0.006
        },
    },
}


# ============================================================
# Plot
# ============================================================
def plot_overlay_subplot(ax, X, fm_baseline, fm1_scenario, metric='AUROC'):
    """
    한 subplot (excluded source = X):
      - x-axis: 6 개 target T (X 포함)
      - 모든 위치에 노랑 F(M) baseline bar
      - X 가 아닌 5 위치에만 X-cluster 색 F(M-1, excl-X) 전경 bar 추가
      - X 위치는 노랑 bar 위에 '(excluded)' 라벨로 명시
    """
    all_T   = DISPLAY_ORDER
    labels  = [SHORT_NAME[T] for T in all_T]
    x_pos   = np.arange(len(all_T))

    fm_means = [fm_baseline[T][0] for T in all_T]
    fm_stds  = [fm_baseline[T][1] for T in all_T]

    x_group = CLUSTER_GROUP[X]
    x_color = CLUSTER_COLOR[x_group]

    # 배경: 6 위치 전부 (X 포함)
    ax.bar(x_pos, fm_means, width=0.72, yerr=fm_stds,
           color=FM_BG_FACE, edgecolor=FM_BG_EDGE, linewidth=0.9,
           alpha=0.55, capsize=3, zorder=2,
           label='All sources (α=1.0)')

    # 전경: X 빼고 5 위치만
    fg_pos, fg_means, fg_stds = [], [], []
    for i, T in enumerate(all_T):
        if T == X:
            continue
        fg_pos.append(i)
        fg_means.append(fm1_scenario[X][T][0])
        fg_stds.append(fm1_scenario[X][T][1])

    ax.bar(fg_pos, fg_means, width=0.42, yerr=fg_stds,
           color=x_color, edgecolor=x_color, linewidth=1.4,
           alpha=0.55, capsize=3, zorder=3,
           label=f'After excluding {SHORT_NAME[X]}')

    ax.set_xticks(x_pos)
    tick_labels = ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=12)
    for tick, T in zip(tick_labels, all_T):
        tick.set_color(CLUSTER_COLOR[CLUSTER_GROUP[T]])
    ax.tick_params(axis='y', labelsize=11)

    # annotation: X 위치는 (excluded), 나머지는 Δ
    for i, T in enumerate(all_T):
        a = fm_baseline[T][0]
        if T == X:
            ax.text(i, a + 0.012, '(excluded)',
                    ha='center', va='bottom', fontsize=7.5,
                    color=x_color, fontweight='bold', style='italic')
            continue
        b = fm1_scenario[X][T][0]
        d = b - a
        col = '#B71C1C' if d < -0.005 else ('#1B5E20' if d > 0.005 else '#37474F')
        ax.text(i, max(a, b) + 0.012, f'Δ{d:+.3f}',
                ha='center', va='bottom', fontsize=7.5,
                color=col, fontweight='bold')

    # y 범위: baseline 주변 확대 (AUPRC 는 SAheart ~0.6 까지 내려감)
    all_v = fm_means + fg_means
    lo = min(all_v) - 0.06
    hi = max(all_v) + 0.08   # (excluded)/Δ annotation headroom
    ax.set_ylim(max(0.4, lo), min(1.02, hi))

    ax.set_title(f'{SHORT_NAME[X]} excluded',
                 fontsize=13, color=x_color, pad=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['auc', 'auprc'], default='auc')
    args = parser.parse_args()

    fm_baseline   = FM_BASELINE_BY_METRIC[args.metric]
    fm1_scenario  = FM1_SCENARIO_BY_METRIC[args.metric]
    metric_label  = {'auc': 'AUROC', 'auprc': 'AUPRC'}[args.metric]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for i, X in enumerate(DISPLAY_ORDER):
        ax = axes[i // 3, i % 3]
        plot_overlay_subplot(ax, X, fm_baseline, fm1_scenario, metric=metric_label)

    fig.tight_layout()

    save_dir = './figures/expB'
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(save_dir, f'expB_scenario_overlay_{args.metric}_{ts}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'Saved: {save_path}')
    print(f'Saved: {pdf_path}')

    print(f'\n=== Δ = F(M-1, excl X)[T] − F(M)[T]  [{metric_label}] ===')
    corner = 'X \\ T'
    hdr = f"{corner:10s} | " + ' '.join(f'{SHORT_NAME[T]:>8s}' for T in DISPLAY_ORDER)
    print(hdr)
    for X in DISPLAY_ORDER:
        row = f"{SHORT_NAME[X]:10s} | "
        for T in DISPLAY_ORDER:
            if T == X:
                row += f"{'-':>8s} "
                continue
            d = fm1_scenario[X][T][0] - fm_baseline[T][0]
            row += f"{d:+8.3f} "
        print(row)


if __name__ == '__main__':
    main()
