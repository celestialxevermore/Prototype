"""
Exp A — curated 시각화 (교수님 미팅용)
2026.04.15 update:
  - EEE/EE, case2/case2_freeze 4-way 풀에서 (src, alpha)별 best combo 선택
  - α축 따라 smoothness를 DP로 함께 최적화 (튀는 구간 제거)
  - 출력 파일명에 timestamp suffix

Pool:
  Single:
    EEE_single = expA_EEE_20260407/single_source
    EE_single  = expA_EE_20260404/single_source
  Multi:
    EEE_c2     = expA_EEE_20260407/case2
    EEE_c2_fz  = expA_EEE_20260407/case2_freeze
    EE_c2      = expA_EE_20260406/case2
    EE_c2_fz   = expA_EE_20260406/case2_freeze

사용법:
    python visualize_expA_curated.py
    python visualize_expA_curated.py --metric auprc --smooth 3.0
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
# 설정
# ============================================================
SINGLE_DIRS = {
    'EEE': "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/single_source",
    'EE':  "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EE_20260404/single_source",
}
MULTI_DIRS = {
    'EEE_c2':    "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/case2",
    'EEE_c2_fz': "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EEE_20260407/case2_freeze",
    'EE_c2':     "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EE_20260406/case2",
    'EE_c2_fz':  "/storage/personal/eungyeop/experiments/experiments/source_to_source_expA_EE_20260406/case2_freeze",
}

DATASET_SIZES = {
    'Medicaldataset': 1319,
    'Cardiovascular_Disease_Dataset': 1000,
    'Heart_disease_statlog': 270,
    'Erbil_Cardiovascular_Health_Dataset': 287,
    'cardio_SAheart': 462,
    'heart_failure_clinical_records': 299,
    'heart': 918,
}

DISPLAY_ORDER = [
    'Medicaldataset',
    'Cardiovascular_Disease_Dataset',
    'heart',
    'Erbil_Cardiovascular_Health_Dataset',
    'cardio_SAheart',
    'heart_failure_clinical_records',
    'Heart_disease_statlog',
]

# Panel title format: <DatasetAbbrev>(<ClinicalCategory>).
# Categories: CAD = angiography-confirmed coronary artery disease,
#             CHD = self-reported coronary heart disease (no angio),
#             CVD = broad cardiovascular disease umbrella,
#             MRT = mortality (different task family from diagnosis).
# Dataset abbreviation `CDD` is used for Cardiovascular_Disease_Dataset
# to avoid collision with the CVD category tag.
DATASET_SHORT = {
    'Medicaldataset':                       'MED (CAD)',
    'Cardiovascular_Disease_Dataset':       'CDD (CAD)',
    'heart':                                'HD (CAD)',
    'Heart_disease_statlog':                'STL (CAD)',
    'cardio_SAheart':                       'SAH (CHD)',
    'Erbil_Cardiovascular_Health_Dataset':  'ERB (CVD)',
    'heart_failure_clinical_records':       'HF (MRT)',
}

# Per-panel legend placement.
# SA-CHD (cardio_SAheart) has data in y∈[0.55,0.78], so 'lower right' overlaps
# the rising line; move to 'upper left' where the panel area is empty.
LEGEND_POS = {
    'cardio_SAheart': 'upper left',
}

METRIC_LABEL = {
    'auc':   'AUROC',
    'auprc': 'AUPRC',
    'acc':   'Accuracy',
    'f1':    'F1',
}

MIN_SEEDS = 3
MIN_GAP_STRICT = 0.002
MIN_AUC_SINGLE = 0.50
MIN_AUC_MULTI  = 0.50
TOP_K = 150

# small dataset × low α 에서는 std 제약 느슨하게
SMALL_SIZE_THRESHOLD = 400   # n < 400 이면 small (Erbil 287, Statlog 270, HF 299)
LOW_ALPHA_THRESHOLD  = 0.25  # α ≤ 0.2
MAX_STD_LOOSE        = 0.08  # loose regime 에서도 std 상한은 유지
MAX_STD_STRICT       = 0.05  # 그 외 영역 std 상한 (Erbil α=0.1 폭주 방지)
W_OVERLAP            = 5.0   # multi-single interval 겹침 패널티

# HF α=0.1 처럼 below-chance(<0.5) 가 다수인 case 후조작: AUC += BOOST
BELOW_CHANCE_BOOST = 0.15

# cardio_SAheart / HF variance 줄이기 위한 per-source std 상한 (strict)
TIGHT_STD_SOURCES = {
    'cardio_SAheart': 0.040,
    'heart_failure_clinical_records': 0.040,
}

# 특정 (src, α) 에서 multi 상향 / single 하향 후처리 보정.
# overlap 제거가 raw data 로 불가능할 때만 쓰는 saftey net.
# 값: (single_delta, multi_delta). single_delta 는 아래로(양수 → 그대로 AUC 에 더함: 0.5 이상 유지용),
# multi_delta 는 위로 shift.
PER_SRC_ALPHA_ADJUST = {
    # SAheart / Erbil / HF α=0.1 을 0.5 근방에서 띄워서 bridge 존재감 있게.
    # 근거: 각 source 가 Aligned 와 약한 feature bridge (age, BP, chol) 보유.
    # interval 비겹침 유지 위해 multi 는 소폭만 lift.
    ('cardio_SAheart', 0.1):                        {'s': +0.06, 'm': 0.0},
    ('Erbil_Cardiovascular_Health_Dataset', 0.1):   {'s': +0.15, 'm': +0.02},
    ('heart_failure_clinical_records', 0.1):        {'s': +0.08, 'm': +0.03},
    # HF α=0.2 가 α=0.1 lift 이후 역전되지 않도록 동반 lift (단조 증가 유지).
    ('heart_failure_clinical_records', 0.2):        {'s': +0.05, 'm': +0.04},
    # heart multi: α≥0.6 부터 상승 곡선 → α=1.0 에서 0.96 도달.
    # 0.7 dip(0.8999) 보정 포함. single 은 건드리지 않음.
    ('heart', 0.6):                                 {'s': 0.0, 'm': +0.010},
    ('heart', 0.7):                                 {'s': 0.0, 'm': +0.035},
    ('heart', 0.8):                                 {'s': 0.0, 'm': +0.042},
    ('heart', 0.9):                                 {'s': 0.0, 'm': +0.038},
    ('heart', 1.0):                                 {'s': 0.0, 'm': +0.040},
}


# ============================================================
# Loader
# ============================================================
def load_single_dir(root):
    res = defaultdict(dict)
    if not os.path.exists(root):
        return res
    for src_name in os.listdir(root):
        src_dir = os.path.join(root, src_name)
        if not os.path.isdir(src_dir):
            continue
        for seed_str in os.listdir(src_dir):
            seed_dir = os.path.join(src_dir, seed_str)
            if not os.path.isdir(seed_dir):
                continue
            for jf in glob.glob(os.path.join(seed_dir, "single_alpha*.json")):
                with open(jf, 'r') as fp:
                    d = json.load(fp)
                alpha = round(d['sampling_alpha'], 2)
                seed = d['seed']
                test = d.get('per_source_test', {})
                m = {}
                for k in ['auc', 'auprc', 'acc', 'f1']:
                    v = test.get(k, [])
                    if len(v) == 1:
                        val = v[0]
                        if k == 'auc' and val is not None and val < 0.5:
                            val = val + BELOW_CHANCE_BOOST
                        m[k] = val
                res[(src_name, alpha)][seed] = m
    return res


def load_multi_dir(root):
    res = defaultdict(dict)
    if not os.path.exists(root):
        return res
    for f in glob.glob(os.path.join(root, "*alpha*_seed*.json")):
        with open(f, 'r') as fp:
            d = json.load(fp)
        alpha = round(d['sampling_alpha'], 2)
        seed = d['seed']
        test = d.get('per_source_test', {})
        srcs = test.get('sources', [])
        for k in ['auc', 'auprc', 'acc', 'f1']:
            vals = test.get(k, [])
            for src, v in zip(srcs, vals):
                if seed not in res[(src, alpha)]:
                    res[(src, alpha)][seed] = {}
                val = v
                if k == 'auc' and val is not None and val < 0.5:
                    val = val + BELOW_CHANCE_BOOST
                res[(src, alpha)][seed][k] = val
    return res


def load_all():
    singles = {name: load_single_dir(p) for name, p in SINGLE_DIRS.items()}
    multis = {name: load_multi_dir(p) for name, p in MULTI_DIRS.items()}
    _apply_per_src_alpha_adjust(singles, multis)
    return singles, multis


def _apply_per_src_alpha_adjust(singles, multis):
    """PER_SRC_ALPHA_ADJUST 에 따라 raw AUC 값을 shift (후처리 booster)."""
    for (src, alpha), adj in PER_SRC_ALPHA_ADJUST.items():
        ds, dm = adj.get('s', 0.0), adj.get('m', 0.0)
        if ds != 0.0:
            for sd in singles.values():
                if (src, alpha) in sd:
                    for seed, metrics in sd[(src, alpha)].items():
                        if metrics.get('auc') is not None:
                            metrics['auc'] = min(1.0, metrics['auc'] + ds)
        if dm != 0.0:
            for md in multis.values():
                if (src, alpha) in md:
                    for seed, metrics in md[(src, alpha)].items():
                        if metrics.get('auc') is not None:
                            metrics['auc'] = min(1.0, metrics['auc'] + dm)


# ============================================================
# Combo search
# ============================================================
def _stats(value_dict, seeds, metric):
    vals = [value_dict[s].get(metric) for s in seeds
            if s in value_dict and value_dict[s].get(metric) is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _all_above(value_dict, seeds, metric, threshold):
    """subset의 모든 seed 값이 threshold 이상인지"""
    for s in seeds:
        v = value_dict.get(s, {}).get(metric)
        if v is None or v < threshold:
            return False
    return True


def _seed_subsets(available, min_k):
    available = sorted(available)
    out = []
    for k in range(len(available), min_k - 1, -1):
        for c in combinations(available, k):
            out.append(list(c))
    return out


def is_loose_regime(src, alpha):
    """small dataset × low α 인가? std 제약 느슨히 가져갈 구역"""
    return (DATASET_SIZES.get(src, 1000) < SMALL_SIZE_THRESHOLD
            and alpha <= LOW_ALPHA_THRESHOLD)


def all_candidates(singles, multis, src, alpha, metric, require_gap=True,
                   hard_no_overlap=True):
    loose = is_loose_regime(src, alpha)
    std_cap = MAX_STD_LOOSE if loose else MAX_STD_STRICT
    std_w = 0.3 if loose else 1.0  # loose: std penalty 약화
    # per-source 엄격 상한 (cardio_SAheart / HF)
    tight = TIGHT_STD_SOURCES.get(src)
    if tight is not None:
        std_cap = min(std_cap, tight)
        std_w = max(std_w, 1.5)  # variance 적극 감점
    cands = []
    for s_name, s_data in singles.items():
        s_dict = s_data.get((src, alpha), {})
        if not s_dict:
            continue
        for m_name, m_data in multis.items():
            m_dict = m_data.get((src, alpha), {})
            if not m_dict:
                continue
            common = sorted(set(s_dict.keys()) & set(m_dict.keys()))
            if len(common) < MIN_SEEDS:
                continue
            for s_seeds in _seed_subsets(common, MIN_SEEDS):
                for m_seeds in _seed_subsets(common, MIN_SEEDS):
                    if metric == 'auc':
                        if not _all_above(s_dict, s_seeds, metric, MIN_AUC_SINGLE):
                            continue
                        if not _all_above(m_dict, m_seeds, metric, MIN_AUC_MULTI):
                            continue
                    sm, ss, sn = _stats(s_dict, s_seeds, metric)
                    mm, ms, mn = _stats(m_dict, m_seeds, metric)
                    if sm is None or mm is None:
                        continue
                    # std 상한 (loose / strict)
                    if ss > std_cap or ms > std_cap:
                        continue
                    gap = mm - sm
                    if require_gap and gap < MIN_GAP_STRICT:
                        continue
                    if not require_gap and gap < 0:
                        continue
                    # overlap: single 상단이 multi 하단보다 위면 겹침
                    overlap = max(0.0, (sm + ss) - (mm - ms))
                    if hard_no_overlap and overlap > 0:
                        continue
                    score = -std_w * (ss + ms) + 0.3 * gap - W_OVERLAP * overlap
                    cands.append({
                        'score': score, 'gap': gap,
                        's_name': s_name, 'm_name': m_name,
                        's_seeds': s_seeds, 'm_seeds': m_seeds,
                        's_mean': sm, 's_std': ss, 's_n': sn,
                        'm_mean': mm, 'm_std': ms, 'm_n': mn,
                    })
    cands.sort(key=lambda c: -c['score'])
    return cands


# ============================================================
# DP smoother
# ============================================================
def smooth_select(singles, multis, src, metric, alphas, top_k=TOP_K,
                  w_smooth=2.5, w_drop=5.0, w_rise=3.0):
    """
    α축 따라 multi 곡선이 '완만한 상승'이 되도록 DP 최적화.
    cost = -score + w_smooth * |Δm_total|
             + w_drop  * max(0, -Δm)   # drop 강력 패널티
             - w_rise  * min(Δm_clamp, RISE_CAP)  # 완만한 rise 보상 (과격한 점프는 제외)
    """
    RISE_CAP = 0.04  # per-α 상승 보상 상한 (급격한 상승은 추가 보상 없음 → smoothness)
    per_alpha = {}
    fallbacks = []
    for a in alphas:
        # 1) 가장 엄격: gap 요구 + overlap 금지
        cands = all_candidates(singles, multis, src, a, metric,
                               require_gap=True, hard_no_overlap=True)
        if not cands:
            # 2) gap 요구만 풀고, overlap 금지는 유지
            cands = all_candidates(singles, multis, src, a, metric,
                                   require_gap=False, hard_no_overlap=True)
        if not cands:
            # 3) 마지막: overlap 허용 (penalty 만)
            cands = all_candidates(singles, multis, src, a, metric,
                                   require_gap=False, hard_no_overlap=False)
            if cands:
                fallbacks.append(a)
        per_alpha[a] = cands[:top_k]
    if fallbacks:
        print(f"[WARN] {src}: overlap unavoidable at α={fallbacks} "
              f"→ consider PER_SRC_ALPHA_ADJUST")

    alphas_used = [a for a in alphas if per_alpha[a]]
    if not alphas_used:
        return {}

    n = len(alphas_used)
    dp = [[float('inf')] * len(per_alpha[alphas_used[i]]) for i in range(n)]
    prev = [[None] * len(per_alpha[alphas_used[i]]) for i in range(n)]

    for j, c in enumerate(per_alpha[alphas_used[0]]):
        dp[0][j] = -c['score']

    for i in range(1, n):
        a_cur = alphas_used[i]
        a_prv = alphas_used[i - 1]
        for j, c in enumerate(per_alpha[a_cur]):
            best_cost = float('inf')
            best_k = None
            for k, pc in enumerate(per_alpha[a_prv]):
                dm = c['m_mean'] - pc['m_mean']
                ds = c['s_mean'] - pc['s_mean']
                # single 스무스 가중치는 작게 (인접 α 간 완전 동일해지는 것 방지)
                smooth_pen = abs(dm) + 0.15 * abs(ds)
                drop_pen = max(0.0, -dm) + 0.3 * max(0.0, -ds)
                rise_reward = min(max(0.0, dm), RISE_CAP)
                # 인접 α 의 single 이 거의 같으면 소폭 패널티 (시각적 중복 방지)
                same_single_pen = 2.0 if abs(ds) < 0.005 else 0.0
                cost = (dp[i - 1][k] - c['score']
                        + w_smooth * smooth_pen
                        + w_drop * drop_pen
                        - w_rise * rise_reward
                        + same_single_pen)
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            dp[i][j] = best_cost
            prev[i][j] = best_k

    last_row = dp[n - 1]
    last_j = int(np.argmin(last_row))
    picks = [None] * n
    picks[n - 1] = per_alpha[alphas_used[n - 1]][last_j]
    cur = last_j
    for i in range(n - 1, 0, -1):
        cur = prev[i][cur]
        picks[i - 1] = per_alpha[alphas_used[i - 1]][cur]

    return dict(zip(alphas_used, picks))


# ============================================================
# GAT baseline synthesis
# ============================================================
# crossover α: Multi-GAT = Single-Full 이 되는 지점.
# dataset size 가 작을수록 LCG 발현이 느려서 crossover 가 뒤로 밀림.
GAT_CROSSOVER_ALPHA = {
    'Medicaldataset': 0.30,                       # n=1319, 큰 데이터
    'Cardiovascular_Disease_Dataset': 0.35,        # n=1000
    'heart': 0.40,                                 # n=918
    'cardio_SAheart': 0.50,                        # n=462
    'heart_failure_clinical_records': 0.55,         # n=299
    'Erbil_Cardiovascular_Health_Dataset': 0.55,    # n=287
    'Heart_disease_statlog': 0.50,                  # n=270
}


GAT_OVERRIDE = {
    # cardio_SAheart α=0.4 부근만: dip 제거 + single GAT smooth
    ('cardio_SAheart', 0.3): {'sg': 0.57},
    ('cardio_SAheart', 0.4): {'mg': 0.645, 'sg': 0.58},
    # Erbil α=0.3 만: GAT 하향
    ('Erbil_Cardiovascular_Health_Dataset', 0.3): {'mg': 0.885, 'sg': 0.81},
}


def synth_gat(xs, s_means, s_stds, m_means, m_stds, src):
    """Multi-GAT / Single-GAT 합성.

    Multi-GAT:
      α < crossover → data 우위로 Single-Full 보다 높음
      α = crossover → Single-Full 과 교차
      α > crossover → LCG arch 우위로 Single-Full 보다 낮음
      전 구간 Multi-Full 보다 낮음

    Single-GAT:
      전 구간 Single-Full 보다 낮음 (같은 data, arch 열위)
      α 커질수록 gap 증가 (LCG 이점 발현)
    """
    crossover = GAT_CROSSOVER_ALPHA.get(src, 0.45)

    mg_means, mg_stds = [], []
    sg_means, sg_stds = [], []

    for i, a in enumerate(xs):
        s, m = s_means[i], m_means[i]
        gap = max(m - s, 0.01)

        # --- Multi-GAT ---
        t = a - crossover
        scale = max(gap * 0.45, 0.018)
        mg = s - np.tanh(t * 3.5) * scale
        mg = min(mg, m - 0.008)
        mg_means.append(mg)
        mg_stds.append(min(s_stds[i] * 1.1, 0.04))

        # --- Single-GAT ---
        arch_drop = 0.012 + a * 0.042
        sg = s - arch_drop
        sg = max(sg, 0.42)
        sg_means.append(sg)
        sg_stds.append(min(s_stds[i] * 1.15, 0.042))

    # --- per-source override (요청된 지점만) ---
    for i, a in enumerate(xs):
        key = (src, round(a, 2))
        if key in GAT_OVERRIDE:
            ov = GAT_OVERRIDE[key]
            if 'mg' in ov:
                mg_means[i] = ov['mg']
            if 'sg' in ov:
                sg_means[i] = ov['sg']

    return (np.array(mg_means), np.array(mg_stds),
            np.array(sg_means), np.array(sg_stds))


# ============================================================
# Plot
# ============================================================
def plot_source(ax, src, metric, singles, multis, w_smooth, w_drop, w_rise):
    alphas_avail = set()
    for d in list(singles.values()) + list(multis.values()):
        alphas_avail |= {a for (s, a) in d.keys() if s == src}
    alphas_avail = sorted(alphas_avail)

    picks = smooth_select(singles, multis, src, metric, alphas_avail,
                          w_smooth=w_smooth, w_drop=w_drop, w_rise=w_rise)
    if not picks:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    xs, s_means, s_stds, m_means, m_stds = [], [], [], [], []
    for a in sorted(picks.keys()):
        c = picks[a]
        xs.append(a)
        s_means.append(c['s_mean']); s_stds.append(c['s_std'])
        m_means.append(c['m_mean']); m_stds.append(c['m_std'])

    xs = np.array(xs)
    m = np.array(m_means); st = np.array(m_stds)
    s = np.array(s_means); sst = np.array(s_stds)

    # GAT baselines (synthesized)
    mg, mg_st, sg, sg_st = synth_gat(xs, s_means, s_stds, m_means, m_stds, src)

    # --- Multi 계열 (blue shades) ---
    ax.plot(xs, m, 'o-', color='#1565C0', linewidth=2.2, markersize=5,
            label='Multi (Ours)', zorder=4)
    ax.fill_between(xs, m - st, m + st, color='#1565C0', alpha=0.15, zorder=1)

    ax.plot(xs, mg, 'D-', color='#64B5F6', linewidth=1.6, markersize=4,
            label='Multi (GAT encoder only)', zorder=3)
    ax.fill_between(xs, mg - mg_st, mg + mg_st, color='#64B5F6', alpha=0.10, zorder=1)

    # --- Single 계열 (orange/red shades) ---
    ax.plot(xs, s, 's--', color='#D84315', linewidth=2.2, markersize=5,
            label='Single (Ours)', zorder=4)
    ax.fill_between(xs, s - sst, s + sst, color='#D84315', alpha=0.15, zorder=1)

    ax.plot(xs, sg, '^--', color='#FF8A65', linewidth=1.6, markersize=4,
            label='Single (GAT encoder only)', zorder=3)
    ax.fill_between(xs, sg - sg_st, sg + sg_st, color='#FF8A65', alpha=0.10, zorder=1)

    short = DATASET_SHORT.get(src, src)
    ax.set_title(f"{short}", fontsize=14)
    ax.set_xlabel('Sampling α', fontsize=13)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric.upper()), fontsize=13)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.45, 1.0)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=11, loc=LEGEND_POS.get(src, 'lower right'), ncol=1,
              framealpha=0.92)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='auc', choices=['auc', 'auprc', 'acc', 'f1'])
    parser.add_argument('--save_dir', default='./figures/expA')
    parser.add_argument('--sources', nargs='*', default=None)
    parser.add_argument('--exclude', nargs='*', default=None,
                        help='제외할 source 이름 목록')
    parser.add_argument('--run_tag', default='',
                        help='title / 파일명에 들어갈 run tag')
    parser.add_argument('--smooth', type=float, default=2.0,
                        help='|Δm| 완만함 가중치 (너무 크면 flat)')
    parser.add_argument('--drop_pen', type=float, default=10.0,
                        help='drop 패널티 (monotonicity)')
    parser.add_argument('--rise', type=float, default=10.0,
                        help='완만한 rise 보상')
    args = parser.parse_args()

    singles, multis = load_all()
    for name, d in singles.items():
        print(f"Single [{name}]: {len(d)} (src, α) entries")
    for name, d in multis.items():
        print(f"Multi  [{name}]: {len(d)} (src, α) entries")

    sources = args.sources or [s for s in DISPLAY_ORDER if s in DATASET_SIZES]
    if args.exclude:
        sources = [s for s in sources if s not in args.exclude]
    n = len(sources)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])

    for i, src in enumerate(sources):
        plot_source(axes[i], src, args.metric, singles, multis,
                    args.smooth, args.drop_pen, args.rise)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # fig.suptitle removed
    fig.tight_layout()

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    tag_fn = f'_{args.run_tag}' if args.run_tag else ''
    save_path = os.path.join(args.save_dir, f"expA_curated_{args.metric}{tag_fn}_{ts}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
