"""
Exp B — F(M) vs F(M-1) Source Exclusion 시각화
2026.04.10

각 source를 제외했을 때 나머지 source들의 성능 변화를 F(M) baseline과 비교.
x축: remaining source, y축: AUC
각 excluded case별 subplot.

사용법:
    python visualize_expB_exclude.py --fm_dir /path/to/FM --fm1_dir /path/to/FM1
    python visualize_expB_exclude.py --fm_dir /path/to/FM --fm1_dir /path/to/FM1 --metric auprc
    python visualize_expB_exclude.py --fm_dir /path/to/FM --fm1_dir /path/to/FM1 --save_dir ./figures
"""

import os, json, glob, argparse, math
import matplotlib.pyplot as plt
from collections import defaultdict


DEFAULT_FM_DIR = "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/case2_freeze"
DEFAULT_FM1_DIR = "/storage/personal/eungyeop/experiments/experiments/source_to_source_exp_b_exclude_20260410/case2_exclude_freeze"

SOURCES = [
    'Medicaldataset',
    'Cardiovascular_Disease_Dataset',
    'Heart_disease_statlog',
    'Erbil_Cardiovascular_Health_Dataset',
    'cardio_SAheart',
    'heart_failure_clinical_records',
    'heart',
]

SHORT_NAMES = {
    'Medicaldataset': 'Medical',
    'Cardiovascular_Disease_Dataset': 'Cardio',
    'Heart_disease_statlog': 'Statlog',
    'Erbil_Cardiovascular_Health_Dataset': 'Erbil',
    'cardio_SAheart': 'SAheart',
    'heart_failure_clinical_records': 'HF_records',
    'heart': 'Heart',
}


def mean_std(vals):
    if not vals:
        return 0, 0, 0
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return m, s, len(vals)


def load_fm(fm_dir, metric='auc'):
    """F(M) baseline alpha=1.0 로드"""
    files = glob.glob(os.path.join(fm_dir, "case2_freeze_alpha1.00_*.json"))
    per_source = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        ps = d.get('per_source_test', {})
        srcs = ps.get('sources', [])
        vals = ps.get(metric, [])
        for src, v in zip(srcs, vals):
            per_source[src].append(v)
    return per_source


def load_fm1(fm1_dir, metric='auc'):
    """F(M-1) 결과 로드: {excluded: {src: [vals]}}"""
    files = glob.glob(os.path.join(fm1_dir, "*.json"))
    results = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = json.load(open(f))
        excl = d['excluded_sources'][0]
        ps = d.get('per_source_test', {})
        srcs = ps.get('sources', [])
        vals = ps.get(metric, [])
        for src, v in zip(srcs, vals):
            results[excl][src].append(v)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fm_dir', default=DEFAULT_FM_DIR, help='F(M) baseline 결과 디렉토리')
    parser.add_argument('--fm1_dir', default=DEFAULT_FM1_DIR, help='F(M-1) exclusion 결과 디렉토리')
    parser.add_argument('--metric', default='auc', choices=['auc', 'auprc', 'acc', 'f1'])
    parser.add_argument('--save_dir', default='./figures')
    args = parser.parse_args()

    fm = load_fm(args.fm_dir, args.metric)
    fm1 = load_fm1(args.fm1_dir, args.metric)

    n_excl = len(SOURCES)
    cols = 3
    rows = (n_excl + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, excl in enumerate(SOURCES):
        ax = axes[idx]
        remaining = [s for s in SOURCES if s != excl]
        short_remaining = [SHORT_NAMES.get(s, s) for s in remaining]

        # F(M) baseline
        fm_means, fm_stds = [], []
        for src in remaining:
            m, s, n = mean_std(fm.get(src, []))
            fm_means.append(m)
            fm_stds.append(s)

        # F(M-1)
        fm1_means, fm1_stds = [], []
        fm1_data = fm1.get(excl, {})
        for src in remaining:
            m, s, n = mean_std(fm1_data.get(src, []))
            fm1_means.append(m)
            fm1_stds.append(s)

        x = range(len(remaining))

        # 바 차트
        width = 0.35
        bars1 = ax.bar([i - width / 2 for i in x], fm_means, width,
                        yerr=fm_stds, label='F(M)', color='#2196F3', alpha=0.8,
                        capsize=3, error_kw={'linewidth': 1})
        bars2 = ax.bar([i + width / 2 for i in x], fm1_means, width,
                        yerr=fm1_stds, label=f'F(M-1)', color='#FF5722', alpha=0.8,
                        capsize=3, error_kw={'linewidth': 1})

        # diff 표시
        for i, (fm_m, fm1_m) in enumerate(zip(fm_means, fm1_means)):
            diff = fm1_m - fm_m
            color = '#4CAF50' if diff >= 0 else '#F44336'
            ax.annotate(f'{diff:+.3f}',
                        xy=(i + width / 2, fm1_m + (fm1_stds[i] if fm1_stds[i] else 0) + 0.005),
                        ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

        ax.set_title(f'Excluded: {SHORT_NAMES.get(excl, excl)}', fontsize=12, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_remaining, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(args.metric.upper())
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.2, axis='y')

        # n runs 표시
        n_fm = len(fm.get(remaining[0], []))
        n_fm1 = len(fm1_data.get(remaining[0], []))
        ax.text(0.98, 0.02, f'F(M) n={n_fm}, F(M-1) n={n_fm1}',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='gray')

    # 빈 subplot 숨기기
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Exp B: F(M) vs F(M-1) Source Exclusion [{args.metric.upper()}]\n'
                 f'F(M): {args.fm_dir}\nF(M-1): {args.fm1_dir}',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"expB_exclude_{args.metric}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
