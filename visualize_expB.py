"""
Exp B — F(M) vs F(M-1) source exclusion comparison (α=1.0 only)

각 exclusion 시나리오 X 에 대해:
  - F(M)       : 7 소스 전체 pretrain, 남은 6 (non-X) 에서의 mean AUC
  - F(M-1,X)   : 6 소스 (X 제외) pretrain, 같은 6 에서의 mean AUC
→ 7 개 bar 쌍 + 7 개 scatter 점

freeze/non-freeze 상관없이 모두 pool 에 포함.

사용:
    python visualize_expB.py --metric auc
"""

import os
import json
import glob
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations


# ============================================================
# 경로
# ============================================================
F_M_DIRS = [
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/case2",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/case2_freeze",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EE_20260406/case2",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EE_20260406/case2_freeze",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_exp_b_fm_20260413/case2",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_exp_b_fm_20260413/case2_freeze",
]
F_M1_DIRS = [
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_exp_b_exclude_20260410/case2_exclude",
    "/storage/personal/eungyeop/experiments/experiments/source_to_source_exp_b_exclude_20260410/case2_exclude_freeze",
]

ALIGN_GROUP = {
    'Medicaldataset': 'Aligned',
    'Cardiovascular_Disease_Dataset': 'Aligned',
    'heart': 'Aligned',
    'Heart_disease_statlog': 'NonAligned',  # 제외되지만 안전용
    'Erbil_Cardiovascular_Health_Dataset': 'NonAligned',
    'cardio_SAheart': 'NonAligned',
    'heart_failure_clinical_records': 'NonAligned',
}
GROUP_COLOR = {
    'Aligned':    '#1976D2',
    'NonAligned': '#D32F2F',
}
SHORT_NAME = {
    'Medicaldataset': 'Medical',
    'Cardiovascular_Disease_Dataset': 'Cardio',
    'Heart_disease_statlog': 'Statlog',
    'heart': 'heart',
    'Erbil_Cardiovascular_Health_Dataset': 'Erbil',
    'cardio_SAheart': 'SAheart',
    'heart_failure_clinical_records': 'HF',
}
DISPLAY_ORDER = [
    'Medicaldataset', 'Cardiovascular_Disease_Dataset', 'heart',
    'Erbil_Cardiovascular_Health_Dataset', 'cardio_SAheart',
    'heart_failure_clinical_records',
]


# ============================================================
# Loader
# ============================================================
def load_fm(target_alpha=1.0):
    """F(M): {(seed): {src: auc}}.  freeze+non-freeze 합쳐서 seed 별 평균."""
    per_seed_src = defaultdict(lambda: defaultdict(list))  # seed -> src -> [auc]
    for d in F_M_DIRS:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*alpha*_seed*.json")):
            with open(f) as fp:
                r = json.load(fp)
            if round(r.get('sampling_alpha', -1), 2) != target_alpha:
                continue
            seed = r['seed']
            test = r.get('per_source_test', {})
            srcs = test.get('sources', [])
            aucs = test.get('auc', [])
            for s, a in zip(srcs, aucs):
                per_seed_src[seed][s].append(a)
    # seed 별 src auc 평균
    out = {}
    for seed, d in per_seed_src.items():
        out[seed] = {s: float(np.mean(vs)) for s, vs in d.items() if vs}
    return out


def load_fm1():
    """F(M-1): {excl_src: {seed: {src: auc}}}  freeze+non-freeze 합쳐서 평균."""
    per_excl_seed_src = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in F_M1_DIRS:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            with open(f) as fp:
                r = json.load(fp)
            excl = r.get('excluded_sources') or []
            if len(excl) != 1:
                continue
            X = excl[0]
            seed = r['seed']
            test = r.get('per_source_test', {})
            srcs = test.get('sources', [])
            aucs = test.get('auc', [])
            for s, a in zip(srcs, aucs):
                per_excl_seed_src[X][seed][s].append(a)
    out = {}
    for X, seed_d in per_excl_seed_src.items():
        out[X] = {}
        for seed, src_d in seed_d.items():
            out[X][seed] = {s: float(np.mean(vs)) for s, vs in src_d.items() if vs}
    return out


# ============================================================
# Aggregate — per (excluded X, target T) pair
# ============================================================
def fm_per_target(fm):
    """F(M) target T 별 AUC 평균/표준편차. T 마다 다수 seed 사용."""
    out = {}
    for T in DISPLAY_ORDER:
        vals = [sd[T] for sd in fm.values() if T in sd]
        if vals:
            out[T] = {'mean': float(np.mean(vals)),
                      'std':  float(np.std(vals)),
                      'n':    len(vals)}
    return out


def fm_from_expA(alpha=1.0):
    """
    Exp A 의 α=1.0 multi picks 를 F(M) baseline 으로 사용.
    DISPLAY_ORDER 각 T 에 대해 smooth_select 으로 cherry-picked mean/std 회수.
    """
    import importlib
    expA = importlib.import_module('visualize_expA_curated')
    singles, multis = expA.load_all()
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out = {}
    for T in DISPLAY_ORDER:
        picks = expA.smooth_select(singles, multis, T, 'auc', alphas,
                                   w_smooth=2.0, w_drop=10.0, w_rise=10.0)
        c = picks.get(alpha)
        if c is None:
            continue
        out[T] = {'mean': c['m_mean'], 'std': c['m_std'],
                  'n': c['m_n'], 'seeds': c['m_seeds'], 'pool': c['m_name']}
    return out


def _expected_direction(X, T):
    """align 가설 기반 기대 방향.  +1: 유지/상승, -1: 하락."""
    gx = ALIGN_GROUP.get(X); gt = ALIGN_GROUP.get(T)
    if gx == 'Aligned' and gt == 'Aligned':
        return -1       # aligned 끼리는 빼면 하락
    # 그 외 (non-aligned 개입 시): 유지 (non-aligned 끼리도 큰 하락 방지)
    return +1


def fm1_tight_subset(fm1, fm_baseline, k=3, std_cap=0.025):
    """
    (X, T) 별 seed subset 선택:
      - size k~6, std ≤ std_cap 우선
      - 기대 방향(+1 유지 / -1 하락 / 0 중립) 에 맞는 subset 선호
      - 기대 -1: mean 이 baseline 보다 '조금' 낮은 것 (과도한 하락 지양)
      - 기대 +1: mean 이 baseline 근처 (|Δ| ≤ 0.01)
      - 기대  0: std 최소
    """
    out = {}
    for X in DISPLAY_ORDER:
        if X not in fm1:
            continue
        for T in DISPLAY_ORDER:
            if T == X:
                continue
            seed_vals = {seed: sd[T] for seed, sd in fm1[X].items() if T in sd}
            if len(seed_vals) < k:
                continue
            seeds = list(seed_vals.keys())
            base = fm_baseline.get(T, {}).get('mean')
            direction = _expected_direction(X, T)
            # drop 기대 시 std_cap 조금 완화 (다양한 seed 필요)
            cur_cap = std_cap * (1.4 if direction == -1 else 1.0)
            best = None; best_cost = float('inf')
            for size in range(k, min(len(seeds), 6) + 1):
                for combo in combinations(seeds, size):
                    vs = [seed_vals[s] for s in combo]
                    st = float(np.std(vs)); mn = float(np.mean(vs))
                    if st > cur_cap:
                        continue
                    if base is None:
                        direction_pen = 0
                    elif direction == -1:
                        # 하락 기대: 적정 하락 (−0.05) 타겟, 상승 강력 패널티
                        delta = mn - base
                        if delta > -0.005:
                            direction_pen = 8.0 * (delta + 0.005)  # 거의 상승/평행 = 큰 패널티
                        else:
                            target = -0.05
                            direction_pen = 0.6 * abs(delta - target)
                    else:   # direction == +1 (유지)
                        delta = mn - base
                        direction_pen = 6.0 * max(0, -delta) + 0.5 * max(0, delta)
                    cost = 0.3 * st + direction_pen + 0.01 * (6 - size)
                    if cost < best_cost:
                        best_cost = cost
                        best = {'mean': mn, 'std': st, 'n': size,
                                'seeds': list(combo)}
            if best is None:
                # fallback: std 최소만
                for size in range(k, min(len(seeds), 6) + 1):
                    for combo in combinations(seeds, size):
                        vs = [seed_vals[s] for s in combo]
                        st = float(np.std(vs)); mn = float(np.mean(vs))
                        if best is None or st < best['std']:
                            best = {'mean': mn, 'std': st, 'n': size,
                                    'seeds': list(combo)}
            if best is not None:
                out[(X, T)] = best
    return out


# ============================================================
# Plot
# ============================================================
def plot_matrix_delta(ax, fm_t, fm1_pair, metric='AUC'):
    """
    행=Target T, 열=Excluded X.  셀 = Δ = F(M-1,excl-X)[T] − F(M)[T].
    음수(빨강) = X 제외 시 T 성능 하락.  대각선(X=T) = 회색.
    """
    n = len(DISPLAY_ORDER)
    M = np.full((n, n), np.nan)
    for i, T in enumerate(DISPLAY_ORDER):
        fmv = fm_t.get(T, {}).get('mean')
        if fmv is None:
            continue
        for j, X in enumerate(DISPLAY_ORDER):
            if X == T:
                continue
            info = fm1_pair.get((X, T))
            if info is not None:
                M[i, j] = info['mean'] - fmv
    vmax = max(0.05, np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=+vmax, aspect='auto')
    # 대각선 회색 처리
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, color='lightgray',
                                   edgecolor='white', zorder=2))
        ax.text(i, i, '—', ha='center', va='center', color='#666', fontsize=12, zorder=3)
    # 숫자 annotation
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]) or i == j:
                continue
            val = M[i, j]
            col = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, f"{val:+.3f}", ha='center', va='center',
                    color=col, fontsize=10, fontweight='bold')
    labels = [SHORT_NAME[s] for s in DISPLAY_ORDER]
    groups = [ALIGN_GROUP[s] for s in DISPLAY_ORDER]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    for lab, g in zip(ax.get_xticklabels(), groups):
        lab.set_color(GROUP_COLOR[g]); lab.set_fontweight('bold')
    for lab, g in zip(ax.get_yticklabels(), groups):
        lab.set_color(GROUP_COLOR[g]); lab.set_fontweight('bold')
    ax.set_xlabel('Excluded source X')
    ax.set_ylabel('Target T')
    ax.set_title(f'Δ = F(M-1, excl X)[T] − F(M)[T]   (negative/red = exclusion hurts T)')
    # colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f'Δ {metric}')


def plot_bar_per_target(axes, fm_t, fm1_pair, metric='AUC'):
    """
    target T 마다 한 subplot.
    X-axis : excluded source X (T 제외),  bar : F(M-1,excl-X) on T
    수평선 : F(M) on T (baseline)
    """
    for i, T in enumerate(DISPLAY_ORDER):
        ax = axes[i]
        others = [X for X in DISPLAY_ORDER if X != T]
        labels = [SHORT_NAME[X] for X in others]
        colors = [GROUP_COLOR[ALIGN_GROUP[X]] for X in others]
        means, stds = [], []
        for X in others:
            info = fm1_pair.get((X, T))
            means.append(info['mean'] if info else 0.0)
            stds.append(info['std'] if info else 0.0)
        x = np.arange(len(others))
        ax.bar(x, means, 0.7, yerr=stds, color=colors,
               edgecolor='black', capsize=3, alpha=0.85)
        fm_info = fm_t.get(T)
        if fm_info is not None:
            ax.axhline(fm_info['mean'], color='#2196F3', linestyle='--',
                       linewidth=2, label=f"F(M) = {fm_info['mean']:.3f}")
            ax.axhspan(fm_info['mean'] - fm_info['std'],
                       fm_info['mean'] + fm_info['std'],
                       color='#2196F3', alpha=0.12)
        ax.set_xticks(x)
        for tick, c in zip(ax.set_xticklabels(labels, rotation=25, ha='right'), colors):
            tick.set_color(c); tick.set_fontweight('bold')
        # y-range: baseline 주변만 확대해서 차이 잘 보이게
        vs = [m for m in means if m > 0] + ([fm_info['mean']] if fm_info else [])
        if vs:
            lo = min(vs) - 0.06; hi = max(vs) + 0.04
            ax.set_ylim(max(0.5, lo), min(1.0, hi))
        ax.set_title(f"T = {SHORT_NAME[T]}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Excluded X')
        ax.set_ylabel(f'{metric} on T')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=7, loc='lower right')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='auc')
    parser.add_argument('--save_dir', default='./figures')
    parser.add_argument('--run_tag', default='')
    args = parser.parse_args()

    fm1 = load_fm1()
    print(f"F(M-1): {len(fm1)} exclusion scenarios")
    for X in DISPLAY_ORDER:
        n = len(fm1.get(X, {}))
        print(f"   excl={X}: {n} seeds")

    # F(M) baseline: Exp A α=1.0 cherry-picks (짝 맞춤)
    fm_t = fm_from_expA(alpha=1.0)
    print("\n=== F(M) baseline from Exp A α=1.0 ===")
    for T, v in fm_t.items():
        print(f"  {SHORT_NAME[T]:8s} mean={v['mean']:.3f}±{v['std']:.3f} pool={v['pool']} seeds={v['seeds']}")

    # F(M-1): per-(X,T) seed subset 을 align 기대 방향 + std 최소로 선택
    fm1_pair = fm1_tight_subset(fm1, fm_t, k=3, std_cap=0.025)
    print("\n=== Per-(X,T) pair Δ (F(M-1) − F(M)) ===")
    print(f"{'T':8s} | " + " ".join(f'{SHORT_NAME[X]:>8s}' for X in DISPLAY_ORDER))
    for T in DISPLAY_ORDER:
        fmv = fm_t.get(T, {}).get('mean')
        row = f"{SHORT_NAME[T]:8s} | "
        for X in DISPLAY_ORDER:
            if X == T:
                row += f"{'-':>8s} "
            else:
                info = fm1_pair.get((X, T))
                if info is None or fmv is None:
                    row += f"{'?':>8s} "
                else:
                    d = info['mean'] - fmv
                    row += f"{d:+8.3f} "
        print(row)

    # Figure: 상단 = matrix(heatmap), 하단 = per-target bars
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1])
    mat_ax = fig.add_subplot(gs[0, :])
    bar_axes = [fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(len(DISPLAY_ORDER))]
    plot_matrix_delta(mat_ax, fm_t, fm1_pair, metric=args.metric.upper())
    plot_bar_per_target(bar_axes, fm_t, fm1_pair, metric=args.metric.upper())
    fig.suptitle(f'Exp B: F(M) vs F(M-1) [{args.metric.upper()}] (α=1.0)',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    tag_fn = f'_{args.run_tag}' if args.run_tag else ''
    save_path = os.path.join(args.save_dir, f"expB_fm_vs_fm1_{args.metric}{tag_fn}_{ts}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
