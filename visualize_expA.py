"""
Exp A 시각화: Single-source vs Multi-source, alpha별 성능 추이
2026.04.03

사용법:
    python visualize_expA.py
    python visualize_expA.py --metric auprc
    python visualize_expA.py --sources Medicaldataset cardio_SAheart
    python visualize_expA.py --base_dir expA_20260402
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='expA_EE_20260404',
                        help='multi/case 결과가 있는 base_dir')
    parser.add_argument('--single_base_dir', type=str, default=None,
                        help='single 결과가 있는 base_dir (생략하면 --base_dir과 동일)')
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'auprc', 'acc', 'f1'])
    parser.add_argument('--sources', nargs='*', default=None,
                        help='특정 source만 시각화. 생략하면 전부.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='그래프 저장 경로. 생략하면 화면 출력.')
    parser.add_argument('--multi_mode', type=str, default='multi_source',
                        choices=['multi_source', 'case1', 'case2', 'case1_freeze', 'case2_freeze'],
                        help='multi 결과를 읽을 폴더')
    args = parser.parse_args()
    return args


def load_multi_results(result_dir):
    """
    multi_source / case1 / case2 JSON 로드 (flat 형식)
    returns: { (alpha, seed): { source_name: { metric_name: value }, ... }, ... }
    """
    results = {}
    if not os.path.exists(result_dir):
        return results
    # flat JSON: multi_alpha*.json / case1_alpha*.json / case2_alpha*.json
    for f in glob.glob(os.path.join(result_dir, "*_alpha*_seed*.json")):
        with open(f, 'r') as fp:
            data = json.load(fp)
        alpha = data['sampling_alpha']
        seed = data['seed']
        test_info = data.get('per_source_test', {})
        sources = test_info.get('sources', [])
        if not sources:
            continue
        if (alpha, seed) in results:
            continue
        metrics_all = {}
        for metric_name in ['auc', 'auprc', 'acc', 'f1']:
            vals = test_info.get(metric_name, [])
            for src, val in zip(sources, vals):
                if src not in metrics_all:
                    metrics_all[src] = {}
                metrics_all[src][metric_name] = val
        results[(alpha, seed)] = metrics_all
    return results


def load_single_results(result_dir):
    """
    single_source JSON 로드 (디렉토리 구조: {source}/{seed}/single_alpha*.json)
    returns: { (source, alpha, seed): { metric_name: value, ... }, ... }
    """
    results = {}
    if not os.path.exists(result_dir):
        return results
    for source_name in os.listdir(result_dir):
        source_dir = os.path.join(result_dir, source_name)
        if not os.path.isdir(source_dir):
            continue
        for seed_str in os.listdir(source_dir):
            seed_dir = os.path.join(source_dir, seed_str)
            if not os.path.isdir(seed_dir):
                continue
            for jf in glob.glob(os.path.join(seed_dir, "single_alpha*.json")):
                with open(jf, 'r') as fp:
                    data = json.load(fp)
                alpha = data['sampling_alpha']
                seed = data['seed']
                src = data['source']
                test_info = data.get('per_source_test', {})
                metrics = {}
                for metric_name in ['auc', 'auprc', 'acc', 'f1']:
                    vals = test_info.get(metric_name, [])
                    if len(vals) == 1:
                        metrics[metric_name] = vals[0]
                results[(src, alpha, seed)] = metrics
    return results


def aggregate_by_alpha(values_by_alpha_seed):
    """
    { alpha: [val1, val2, ...] } → { alpha: (mean, std) }
    alpha 기준 정렬
    """
    agg = {}
    for alpha in sorted(values_by_alpha_seed.keys()):
        vals = values_by_alpha_seed[alpha]
        agg[alpha] = (np.mean(vals), np.std(vals))
    return agg


DATASET_SIZES = {
    'Medicaldataset': 1319,
    'Cardiovascular_Disease_Dataset': 1000,
    'Heart_disease_statlog': 270,
    'Erbil_Cardiovascular_Health_Dataset': 287,
    'cardio_SAheart': 462,
    'heart_failure_clinical_records': 299,
    'heart': 918,
}


def plot_source(ax, source_name, metric, multi_results, single_results, multi_label='Multi-source'):

    # ── Multi-source 집계 ──
    multi_by_alpha = defaultdict(list)
    for (alpha, seed), src_metrics in multi_results.items():
        if source_name in src_metrics and metric in src_metrics[source_name]:
            multi_by_alpha[alpha].append(src_metrics[source_name][metric])
    multi_agg = aggregate_by_alpha(multi_by_alpha)

    # ── Single-source 집계 ──
    single_by_alpha = defaultdict(list)
    for (src, alpha, seed), metrics in single_results.items():
        if src == source_name and metric in metrics:
            single_by_alpha[alpha].append(metrics[metric])
    single_agg = aggregate_by_alpha(single_by_alpha)

    # ── 그리기: Multi ──
    if multi_agg:
        alphas = sorted(multi_agg.keys())
        means = [multi_agg[a][0] for a in alphas]
        stds = [multi_agg[a][1] for a in alphas]
        means = np.array(means)
        stds = np.array(stds)
        n_seeds = [len(multi_by_alpha[a]) for a in alphas]

        ax.plot(alphas, means, 'o-', color='#2196F3', linewidth=2, markersize=5,
                label=f'{multi_label} (n={n_seeds[0]})', zorder=3)
        ax.fill_between(alphas, means - stds, means + stds,
                        color='#2196F3', alpha=0.15, zorder=1)

    # ── 그리기: Single ──
    if single_agg:
        alphas = sorted(single_agg.keys())
        means = [single_agg[a][0] for a in alphas]
        stds = [single_agg[a][1] for a in alphas]
        means = np.array(means)
        stds = np.array(stds)
        n_seeds = [len(single_by_alpha[a]) for a in alphas]

        ax.plot(alphas, means, 's--', color='#FF5722', linewidth=2, markersize=5,
                label=f'Single-source (n={n_seeds[0]})', zorder=3)
        ax.fill_between(alphas, means - stds, means + stds,
                        color='#FF5722', alpha=0.15, zorder=1)

    n_samples = DATASET_SIZES.get(source_name, '?')
    ax.set_title(f"{source_name} (n={n_samples})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Sampling Alpha')
    ax.set_ylabel(metric.upper())
    ax.set_xlim(0.05, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()

    root = f"/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}"
    single_root = f"/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.single_base_dir or args.base_dir}"
    multi_dir = os.path.join(root, args.multi_mode)
    single_dir = os.path.join(single_root, "single_source")

    print(f"Loading {args.multi_mode} results from: {multi_dir}")
    print(f"Loading single-source results from: {single_dir}")

    multi_results = load_multi_results(multi_dir)
    single_results = load_single_results(single_dir)

    print(f"Multi-source: {len(multi_results)} entries loaded")
    print(f"Single-source: {len(single_results)} entries loaded")

    if not multi_results and not single_results:
        print("No results found. Check paths.")
        return

    # source 목록 결정
    all_sources = set()
    for (alpha, seed), src_metrics in multi_results.items():
        all_sources.update(src_metrics.keys())
    for (src, alpha, seed), _ in single_results.items():
        all_sources.add(src)
    all_sources = sorted(all_sources)

    if args.sources:
        all_sources = [s for s in args.sources if s in all_sources]

    if not all_sources:
        print("No matching sources found.")
        return

    print(f"Sources to plot: {all_sources}")
    print(f"Metric: {args.metric}")

    # ── Figure 구성 ──
    n = len(all_sources)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, source_name in enumerate(all_sources):
        plot_source(axes[i], source_name, args.metric, multi_results, single_results,
                    multi_label=args.multi_mode)

    # 빈 subplot 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Exp A: Single vs {args.multi_mode} ({args.metric.upper()})\n'
                 f'Multi: {args.base_dir}/{args.multi_mode}  |  Single: {args.single_base_dir or args.base_dir}/single_source',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        single_tag = args.single_base_dir or args.base_dir
        save_path = os.path.join(args.save_dir, f"expA_{args.multi_mode}_{args.base_dir}_single_{single_tag}_{args.metric}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
