"""
Specialization Search — 각 checkpoint 별로 basis 별 feature 전문화 점수 계산.

기준:
  1. Specialization: 각 basis 의 feature profile 이 얼마나 peak 져 있는가 (std 기준)
  2. Differentiation: active basis 들 간의 correlation 이 얼마나 낮은가
  3. Alignment: cluster 1 (Heart, Medical, Cardio) 의 TP basis index 집합 일치도
  4. HF distinct: HF 의 TN ref 가 cluster 1 과 다르거나, basis 집합이 다른 정도

사용:
  python specialization_search.py /path/to/Pre --out scores.csv [--limit N]
"""

import argparse
import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

ROOT_DIR = Path('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217')
sys.path.append(str(ROOT_DIR))

from models.TabularFLM_S_ import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed
from analysis2 import (
    load_original_tabular, load_model, load_source_loaders,
    compute_onehot_basis_risk, compute_soft_pr, extract_wpr_with_tabular,
    _weighted_median_cols, EXCLUDE_FEATURES, DATASET_AND_CLASS,
)

CLUSTER1 = ['heart', 'Medicaldataset', 'Cardiovascular_Disease_Dataset']
CLUSTER2 = ['cardio_SAheart', 'Erbil_Cardiovascular_Health_Dataset']
CLUSTER3 = ['heart_failure_clinical_records']

# Feature 의 semantic group 매핑 (lowercase 비교)
FEATURE_GROUPS = {
    'BP': ['restingbp', 'systolic blood pressure', 'diastolic blood pressure',
           'blood_pressure_systolic', 'blood_pressure_diastolic', 'systolic_blood_pressure',
           'trestbps', 'sbp', 'dbp'],
    'AGE': ['age'],
    'CHOL': ['cholesterol', 'serumcholestrol', 'ldl_cholesterol', 'chol'],
    'HR': ['maxhr', 'maxheartrate', 'heart_rate', 'heart rate'],
    'ST': ['oldpeak'],
    'BS': ['fastingbs', 'blood sugar'],
    'VESSEL': ['noofmajorvessels', 'ca'],
    'TROP': ['troponin'],
    'CKMB': ['ck-mb', 'creatinine_phosphokinase'],
    'ADIPOSE': ['adiposity', 'obesity', 'weight', 'bmi'],
    'SMOKE': ['cumulative_tobacco', 'years_smoking', 'smoking'],
    'BEHAV': ['type_a_behavior', 'alcohol_consumption'],
    'EF': ['ejection_fraction'],
    'RENAL': ['serum_creatinine', 'serum_sodium'],
    'PLT': ['platelets'],
    'HEIGHT': ['height'],
    'SEX': ['sex', 'gender'],
    'CP': ['chestpaintype', 'cp'],
    'ECG': ['restingecg', 'restecg', 'st_slope', 'slope'],
    'EXER': ['exerciseangina', 'exang'],
    'THAL': ['thal'],
}


def feature_to_group(fname):
    """Feature 이름 → semantic group label (lowercase 비교)."""
    fl = fname.lower().replace(' ', '_')
    for group, members in FEATURE_GROUPS.items():
        for m in members:
            m_norm = m.lower().replace(' ', '_')
            if m_norm == fl or m_norm in fl or fl in m_norm:
                return group
    return 'OTHER'


def compute_basis_profile(result, onehot_probs, max_features=8, min_weff=3.0):
    """각 active basis 의 feature profile (weighted median z-score) 계산."""
    X = result['X']
    pis = result['pis']
    ys = result['ys']
    preds = result['preds']
    M = pis.shape[1]

    num_cols = [c for c in X.select_dtypes(include=[np.number]).columns
                if c.lower() not in {f.lower() for f in EXCLUDE_FEATURES}][:max_features]
    if len(num_cols) < 3:
        return {}, None
    X_num = X[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_num)
    X_z = scaler.transform(X_num)

    tp_mask = (ys == 1) & (preds == 1)
    tn_mask = (ys == 0) & (preds == 0)

    profiles = {}
    for k in range(M):
        is_high = onehot_probs[k] > 0.5
        if is_high:
            w = pis[tp_mask, k]
            X_sub = X_z[tp_mask]
        else:
            w = pis[tn_mask, k]
            X_sub = X_z[tn_mask]
        weff = float(w.sum())
        if weff < min_weff or len(X_sub) == 0:
            continue
        prof = _weighted_median_cols(X_sub, w)
        profiles[k] = {
            'profile': prof,
            'is_high': bool(is_high),
            'weff': weff,
        }
    return profiles, num_cols


def pick_active_basis(profiles, top_h=5):
    h_basis = [(k, v['weff']) for k, v in profiles.items() if v['is_high']]
    h_basis.sort(key=lambda x: -x[1])
    h_basis = [k for k, _ in h_basis[:top_h]]

    l_basis = [(k, v['weff']) for k, v in profiles.items() if not v['is_high']]
    l_basis.sort(key=lambda x: -x[1])
    l_ref = l_basis[0][0] if l_basis else None
    return h_basis, l_ref


def specialization_score(profiles, h_basis):
    """(1) 각 basis 의 std (peakiness), (2) basis 간 평균 correlation 의 역"""
    if len(h_basis) < 2:
        return 0.0, 0.0, 0.0
    profs = np.stack([profiles[k]['profile'] for k in h_basis])
    peakiness = float(np.mean(profs.std(axis=1)))
    corrs = []
    for i in range(len(h_basis)):
        for j in range(i + 1, len(h_basis)):
            a, b = profs[i], profs[j]
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                corrs.append(np.corrcoef(a, b)[0, 1])
    mean_corr = float(np.mean(corrs)) if corrs else 1.0
    diff_score = peakiness - mean_corr
    return peakiness, mean_corr, diff_score


def basis_group_signature(profiles, num_cols, k, top_n=2):
    """basis k 의 peak feature top_n 을 semantic group 으로 변환."""
    if k not in profiles or num_cols is None:
        return set()
    prof = profiles[k]['profile']
    idx = np.argsort(-np.abs(prof))[:top_n]
    return {feature_to_group(num_cols[i]) for i in idx}


def semantic_alignment_score(dataset_info):
    """Cluster 1 내, 같은 basis 인덱스가 같은 semantic group 을 가리키는가."""
    c1_srcs = [s for s in CLUSTER1 if s in dataset_info]
    if len(c1_srcs) < 2:
        return 0.0
    # 공통 H-basis index
    common_hb = set.intersection(*[set(dataset_info[s]['h_basis']) for s in c1_srcs])
    if not common_hb:
        return 0.0
    scores = []
    for k in common_hb:
        sigs = [basis_group_signature(dataset_info[s]['profiles'], dataset_info[s]['num_cols'], k)
                for s in c1_srcs]
        sigs = [s for s in sigs if s]
        if len(sigs) < 2:
            continue
        # 모든 pair 의 Jaccard
        pair_scores = []
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                inter = len(sigs[i] & sigs[j])
                union = len(sigs[i] | sigs[j])
                if union > 0:
                    pair_scores.append(inter / union)
        if pair_scores:
            scores.append(np.mean(pair_scores))
    return float(np.mean(scores)) if scores else 0.0


def compute_run_metrics(all_results, onehot_probs, sources):
    dataset_info = {}
    for src in sources:
        if src not in all_results:
            continue
        profiles, cols = compute_basis_profile(all_results[src], onehot_probs)
        h_basis, l_ref = pick_active_basis(profiles)
        peak, corr, diff = specialization_score(profiles, h_basis)
        dataset_info[src] = {
            'h_basis': h_basis,
            'l_ref': l_ref,
            'peakiness': peak,
            'mean_corr': corr,
            'diff_score': diff,
            'profiles': profiles,
            'num_cols': cols,
        }
    semantic_align = semantic_alignment_score(dataset_info)

    # Cluster alignment scores
    c1_hb = [set(dataset_info[s]['h_basis']) for s in CLUSTER1 if s in dataset_info]
    c1_lref = [dataset_info[s]['l_ref'] for s in CLUSTER1 if s in dataset_info]
    if len(c1_hb) == 3:
        pairs = [c1_hb[0] & c1_hb[1], c1_hb[0] & c1_hb[2], c1_hb[1] & c1_hb[2]]
        unions = [c1_hb[0] | c1_hb[1], c1_hb[0] | c1_hb[2], c1_hb[1] | c1_hb[2]]
        jaccard = np.mean([len(p) / len(u) if u else 0 for p, u in zip(pairs, unions)])
        lref_agree = int(len(set(c1_lref)) == 1)
    else:
        jaccard = 0.0
        lref_agree = 0

    # HF distinctness
    hf_src = CLUSTER3[0]
    if hf_src in dataset_info and len(c1_hb) > 0:
        hf_hb = set(dataset_info[hf_src]['h_basis'])
        c1_union = set.union(*c1_hb) if c1_hb else set()
        overlap = len(hf_hb & c1_union) / max(1, len(hf_hb))
        hf_l_distinct = int(dataset_info[hf_src]['l_ref'] not in c1_lref)
        hf_distinct_score = (1 - overlap) + 0.5 * hf_l_distinct
    else:
        hf_distinct_score = 0.0

    # C1 specialization (mean diff_score)
    c1_peak = np.mean([dataset_info[s]['peakiness'] for s in CLUSTER1 if s in dataset_info])
    c1_corr = np.mean([dataset_info[s]['mean_corr'] for s in CLUSTER1 if s in dataset_info])
    c1_diff = np.mean([dataset_info[s]['diff_score'] for s in CLUSTER1 if s in dataset_info])

    return {
        'c1_jaccard': float(jaccard),
        'c1_lref_agree': int(lref_agree),
        'hf_distinct': float(hf_distinct_score),
        'c1_peakiness': float(c1_peak),
        'c1_meancorr': float(c1_corr),
        'c1_diffscore': float(c1_diff),
        'c1_semantic': float(semantic_align),
        'dataset_info': {s: {k: v for k, v in info.items() if k not in ['profiles', 'num_cols']}
                         for s, info in dataset_info.items()},
    }


def find_checkpoints(pre_dir):
    ckpt_names = ['best_joint.pt', 'best_val_joint.pt', 'best.pt', 'latest.pt']
    found = []
    for name in ckpt_names:
        found.extend(glob.glob(os.path.join(pre_dir, '**', name), recursive=True))
    by_dir = {}
    for p in found:
        d = os.path.dirname(p)
        if d not in by_dir or os.path.basename(p) == 'best_joint.pt':
            by_dir[d] = p
    return sorted(by_dir.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_dir', help='Pre/ directory')
    parser.add_argument('--out', default='specialization_scores.csv', help='Output CSV')
    parser.add_argument('--limit', type=int, default=None, help='Limit #checkpoints (debug)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = find_checkpoints(args.pre_dir)
    if args.limit:
        ckpts = ckpts[:args.limit]
    print(f'Found {len(ckpts)} checkpoints')

    # Resume logic: 이미 계산된 run_tag 는 skip
    existing = set()
    if os.path.exists(args.out):
        existing_df = pd.read_csv(args.out)
        existing = set(existing_df['run_tag'].tolist())
        print(f'Resume: {len(existing)} already scored')
    else:
        existing_df = None

    rows = existing_df.to_dict('records') if existing_df is not None else []

    for i, ckpt in enumerate(ckpts):
        run_tag = os.path.basename(os.path.dirname(ckpt))
        if run_tag in existing:
            continue
        print(f'[{i+1}/{len(ckpts)}] {run_tag}')
        try:
            model, model_args = load_model(ckpt, device)
            sources, loaders = load_source_loaders(model_args)
            onehot_probs = compute_onehot_basis_risk(model, device)
            all_results = {}
            for src in sources:
                try:
                    all_results[src] = extract_wpr_with_tabular(model, loaders[src], device, src)
                except Exception as e:
                    print(f'  skip {src}: {e}')
            metrics = compute_run_metrics(all_results, onehot_probs, sources)
            info = metrics.pop('dataset_info')
            row = {'run_tag': run_tag}
            row.update(metrics)
            for s, d in info.items():
                row[f'{s}_hbasis'] = ','.join(f'B{k}' for k in d['h_basis'])
                row[f'{s}_lref'] = f'B{d["l_ref"]}' if d['l_ref'] is not None else ''
                row[f'{s}_peak'] = d['peakiness']
                row[f'{s}_corr'] = d['mean_corr']
            rows.append(row)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        # Save every 10
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
