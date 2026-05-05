"""
Batch Qualitative Analysis v4
  - Signed Radar: -log10(p) with direction (outward=higher, inward=lower)
  - Numerical: t-test, Categorical: chi-square, unified p-value scale
  - Matrix: soft coordinate WPR + Pearson correlation
  - Clinical risk direction arrows from dictionary
Usage: python analysis4.py /path/to/base_dir
"""

import argparse
import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind, chi2_contingency, pearsonr

# ============================================================
# Setup
# ============================================================

ROOT_DIR = Path('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217')
sys.path.append(str(ROOT_DIR))

from models.TabularFLM_S_ import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed
from torch.utils.data import ConcatDataset, DataLoader

OUTPUT_DIR_NAME = 'analysis4_results'

BASE_TABLE_PATH = "/storage/personal/eungyeop/dataset/table/origin_table"

DATASET_SHORT_NAMES = {
    'Medicaldataset': 'MED (CAD)',
    'Cardiovascular_Disease_Dataset': 'CDD (CAD)',
    'heart': 'HD (CAD)',
    'Heart_disease_statlog': 'HD_S (CAD)',
    'Erbil_Cardiovascular_Health_Dataset': 'ERB (CVD)',
    'cardio_SAheart': 'SAH (CHD)',
    'heart_failure_clinical_records': 'HF (MRT)',
}

DATASET_AND_CLASS = {
    "heart": ['target_binary', ['no', 'yes']],
    "Cardiovascular_Disease_Dataset": ['target_binary', ['no', 'yes']],
    "Medicaldataset": ['target_binary', ['no', 'yes']],
    "Heart_disease_statlog": ['target_binary', ['no', 'yes']],
    "Erbil_Cardiovascular_Health_Dataset": ['target_binary', ['no', 'yes']],
    "cardio_SAheart": ['target_binary', ['no', 'yes']],
    "heart_failure_clinical_records": ['target_binary', ['no', 'yes']],
}

LABEL_COL_MAP = {
    "Cardiovascular_Disease_Dataset": "target",
    "Medicaldataset": "Result",
    "heart1": "output",
}

# Clinical risk direction: ↑ = higher is risky, ↓ = lower is risky
RISK_DIRECTION = {
    # === Blood pressure ===
    'Systolic blood pressure': '↑', 'SYSTOLIC_BLOOD_PRESSURE': '↑',
    'BLOOD_PRESSURE_SYSTOLIC': '↑', 'restingBP': '↑', 'trestbps': '↑',
    'Diastolic blood pressure': '↑', 'BLOOD_PRESSURE_DIASTOLIC': '↑',
    # === Cholesterol / lipids ===
    'serumcholestrol': '↑', 'cholesterol': '↑', 'Cholesterol': '↑',
    'chol': '↑', 'LDL_CHOLESTEROL': '↑',
    # === Blood sugar ===
    'Blood sugar': '↑',
    # === Cardiac biomarkers ===
    'CK-MB': '↑', 'Troponin': '↑',
    'creatinine_phosphokinase': '↑', 'serum_creatinine': '↑',
    # === Age ===
    'Age': '↑', 'age': '↑', 'AGE': '↑',
    # === Heart rate (resting: higher is risky) ===
    'Heart rate': '↑', 'HEART_RATE': '↑',
    # === Max heart rate (lower is risky) ===
    'maxheartrate': '↓', 'thalach': '↓', 'maxHR': '↓',
    # === Exercise / ST ===
    'oldpeak': '↑', 'Oldpeak': '↑',
    # === Major vessels ===
    'noofmajorvessels': '↑', 'ca': '↑',
    # === Ejection fraction (lower = worse) ===
    'ejection_fraction': '↓',
    # === Serum sodium (lower = worse in HF) ===
    'serum_sodium': '↓',
    # === Platelets (lower = risky in HF) ===
    'platelets': '↓',
    # === Body composition / metabolic ===
    'OBESITY': '↑', 'ADIPOSITY': '↑', 'WEIGHT': '↑', 'weight': '↑',
    'HEIGHT': '↑',
    # === Smoking / tobacco ===
    'CUMULATIVE_TOBACCO': '↑', 'YEARS_SMOKING': '↑',
    # === Behavioral / lifestyle ===
    'TYPE_A_BEHAVIOR': '↑', 'ALCOHOL_CONSUMPTION': '↑',
    # === Follow-up ===
    'time': '↓',
    # === Hypertension ===
    'HYPERTENSION': '↑',
}

EXCLUDE_FEATURES = {'patientid', 'patient_id', 'id', 'ID'}

BASIS_COLORS = {
    'B0': '#377eb8', 'B1': '#e41a1c', 'B2': '#ff7f00', 'B3': '#4daf4a',
    'B4': '#67a9cf', 'B5': '#984ea3', 'B6': '#f781bf', 'B7': '#a65628',
    'B8': '#999999', 'B9': '#17becf', 'B10': '#bcbd22', 'B11': '#8c564b',
}


# ============================================================
# Data Helpers
# ============================================================

def load_original_tabular(dataset_name):
    csv_path = f"{BASE_TABLE_PATH}/{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    class_info = DATASET_AND_CLASS.get(dataset_name)
    if class_info is None:
        for col in ['target_binary', 'target', 'Result', 'output', 'target_multiclass']:
            if col in df.columns:
                return df.drop(col, axis=1).reset_index(drop=True), df[col].reset_index(drop=True)
        raise ValueError(f"Cannot find label column for {dataset_name}")
    class_name = class_info[0]
    if dataset_name in LABEL_COL_MAP:
        orig_col = LABEL_COL_MAP[dataset_name]
        if orig_col in df.columns:
            df[class_name] = df[orig_col]
            df = df.drop(orig_col, axis=1)
    if class_name in df.columns:
        return df.drop(class_name, axis=1).reset_index(drop=True), df[class_name].reset_index(drop=True)
    raise ValueError(f"Label column '{class_name}' not found")


# ============================================================
# Model / Data Loading
# ============================================================

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']
    model = Model(
        args, args.input_dim, args.hidden_dim, args.output_dim,
        args.dropout_rate, args.llm_model,
        experiment_id="viz", mode="viz"
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model, args


def load_source_loaders(args):
    fix_seed(args.random_seed)
    sources = args.source_data if isinstance(args.source_data, (list, tuple)) else [args.source_data]
    loaders = {}
    for src in sources:
        res = prepare_embedding_dataloaders(args, src, is_source=True)
        tr, va, te = res['loaders']
        datasets = [l.dataset for l in [tr, va, te] if l is not None]
        loaders[src] = DataLoader(ConcatDataset(datasets), batch_size=32, shuffle=False)
    return sources, loaders


# ============================================================
# Compute Functions
# ============================================================

@torch.no_grad()
def compute_onehot_basis_risk(model, device):
    model.eval()
    lcg_feat, lcg_struct = model.latent_graph()
    M = lcg_feat.shape[0]
    expert_outputs = model.gnn_experts(lcg_feat.unsqueeze(0), lcg_struct.unsqueeze(0))
    onehot_probs = []
    for k in range(M):
        pi_onehot = torch.zeros(1, M, device=device)
        pi_onehot[0, k] = 1.0
        weighted = (pi_onehot.unsqueeze(-1) * expert_outputs).sum(dim=1)
        current_mode = getattr(model, 'mode', 'Full')
        logit = model.ghead2(weighted) if current_mode == 'Few' else model.ghead(weighted)
        onehot_probs.append(torch.sigmoid(logit).item())
    return np.array(onehot_probs)


@torch.no_grad()
def compute_soft_pr(model, loaders, sources, device):
    model.eval()
    all_results = {}
    for src_name in sources:
        all_pi, all_y = [], []
        for batch in loaders[src_name]:
            batch_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            model.predict(batch_t, return_all=True)
            pi = model.graph_quantizer.last_pi
            all_pi.append(pi.cpu().numpy())
            all_y.append(batch_t['y'].cpu().view(-1).numpy())
        pis = np.concatenate(all_pi, axis=0)
        ys = np.concatenate(all_y, axis=0).astype(float)
        M = pis.shape[1]
        basis_info = {}
        for k in range(M):
            w = pis[:, k]
            w_sum = w.sum()
            if w_sum > 0:
                pos_rate = (w * ys).sum() / w_sum
                n_pos_eff = (w * ys).sum()
            else:
                pos_rate = np.nan
                n_pos_eff = 0
            basis_info[k] = {'n_eff': float(w_sum), 'n_pos_eff': float(n_pos_eff), 'pos_rate': float(pos_rate)}
        all_results[src_name] = {
            'basis_info': basis_info, 'M': M, 'N': len(ys), 'overall_pos': float(ys.mean()),
        }
    return all_results


@torch.no_grad()
def extract_tp_tn_with_tabular(model, loader, device, dataset_name):
    model.eval()
    X_orig, y_orig = load_original_tabular(dataset_name)
    all_pi, all_y, all_prob, all_sidx = [], [], [], []
    for batch in loader:
        batch_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model.predict(batch_t, return_all=True)
        global_pred = out[0] if isinstance(out, tuple) else out
        pi = model.graph_quantizer.last_pi
        prob = torch.sigmoid(global_pred).squeeze(-1)
        all_pi.append(pi.cpu().numpy())
        all_y.append(batch_t['y'].cpu().view(-1).numpy())
        all_prob.append(prob.cpu().view(-1).numpy())
        if 's_idx' in batch_t:
            sidx = batch_t['s_idx']
            if isinstance(sidx, torch.Tensor):
                all_sidx.append(sidx.cpu().view(-1).numpy())
            else:
                all_sidx.append(np.array([sidx] * pi.shape[0]))
        else:
            start = sum(len(s) for s in all_sidx)
            all_sidx.append(np.arange(start, start + pi.shape[0]))
    pis = np.concatenate(all_pi)
    ys = np.concatenate(all_y).astype(int)
    probs = np.concatenate(all_prob)
    sidxs = np.concatenate(all_sidx).astype(int)
    preds = (probs >= 0.5).astype(int)
    valid = sidxs < len(X_orig)
    if not valid.all():
        pis, ys, probs, sidxs, preds = pis[valid], ys[valid], probs[valid], sidxs[valid], preds[valid]
    X_mapped = X_orig.iloc[sidxs].reset_index(drop=True)
    return {
        'dataset_name': dataset_name, 'X': X_mapped, 'pis': pis, 'ys': ys,
        'probs': probs, 'preds': preds, 'argmax': pis.argmax(axis=1),
        'tp_mask': (ys == 1) & (preds == 1),
        'tn_mask': (ys == 0) & (preds == 0),
    }


# ============================================================
# Radar (signed p-value)
# ============================================================

def detect_ftypes(X):
    num_cols, cat_cols = [], []
    for col in X.columns:
        if col.lower() in {f.lower() for f in EXCLUDE_FEATURES}:
            continue
        if X[col].dtype in ['object', 'category'] or X[col].nunique() <= 5:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols


def compute_signed_profile(X, argmax, basis_k, num_cols, cat_cols, ys):
    in_group = (argmax == basis_k)
    out_group = ~in_group
    if in_group.sum() < 3 or out_group.sum() < 3:
        return None, None, None, None

    features, raw_scores, signs, ftypes = [], [], [], []
    pos_mask = (ys == 1)

    for col in num_cols:
        vals_in = X.loc[in_group, col].dropna()
        vals_out = X.loc[out_group, col].dropna()
        if len(vals_in) < 2 or len(vals_out) < 2:
            continue
        try:
            _, p = ttest_ind(vals_in, vals_out, equal_var=False)
            if p < 1e-300: p = 1e-300
            score = -np.log10(p)
            sign = 1.0 if vals_in.mean() > vals_out.mean() else -1.0
            features.append(col); raw_scores.append(score); signs.append(sign); ftypes.append('num')
        except:
            continue

    for col in cat_cols:
        try:
            ct = pd.crosstab(in_group, X[col])
            if ct.shape[0] < 2 or ct.shape[1] < 2: continue
            _, p, _, _ = chi2_contingency(ct)
            if p < 1e-300: p = 1e-300
            score = -np.log10(p)
            pos_vals = X.loc[pos_mask, col].dropna()
            risky_cat = pos_vals.mode().iloc[0] if len(pos_vals) > 0 else X[col].mode().iloc[0]
            in_prop = (X.loc[in_group, col] == risky_cat).mean()
            out_prop = (X.loc[out_group, col] == risky_cat).mean()
            sign = 1.0 if in_prop >= out_prop else -1.0
            features.append(col); raw_scores.append(score); signs.append(sign); ftypes.append('cat')
        except:
            continue

    return features, raw_scores, signs, ftypes


def plot_radar_single(result, onehot_probs, ax, min_n=5, max_features=12):
    src_name = result['dataset_name']
    X = result['X']
    argmax = result['argmax']
    ys = result['ys']
    M = result['pis'].shape[1]
    short_name = DATASET_SHORT_NAMES.get(src_name, src_name)

    num_cols, cat_cols = detect_ftypes(X)
    high_risk_bases = [k for k in range(M) if onehot_probs[k] > 0.5]
    low_risk_bases = [k for k in range(M) if onehot_probs[k] <= 0.5]

    active_h = []
    for k in high_risk_bases:
        n = (argmax == k).sum()
        if n >= min_n:
            active_h.append((k, int(n)))
    active_h.sort(key=lambda x: -x[1])
    active_h = active_h[:5]

    tn_ref, best_tn_n = None, 0
    for k in low_risk_bases:
        n = (argmax == k).sum()
        if n >= min_n and n > best_tn_n:
            tn_ref, best_tn_n = k, int(n)

    draw_list = []
    if tn_ref is not None:
        draw_list.append((tn_ref, 'tn_ref', best_tn_n))
    for k, n in active_h:
        draw_list.append((k, 'high', n))

    if not draw_list:
        ax.text(0.5, 0.5, f'[{short_name}]\nNo active bases', ha='center', va='center', transform=ax.transAxes)
        return

    all_profiles, all_signs = {}, {}
    all_feature_sets, all_ftypes_map = set(), {}

    for k, kind, n in draw_list:
        res = compute_signed_profile(X, argmax, k, num_cols, cat_cols, ys)
        if res[0] is None: continue
        features, raw_scores, signs, ftypes = res
        all_profiles[k] = dict(zip(features, raw_scores))
        all_signs[k] = dict(zip(features, signs))
        all_feature_sets.update(features)
        for f, ft in zip(features, ftypes):
            all_ftypes_map[f] = ft

    if not all_profiles or not all_feature_sets:
        ax.text(0.5, 0.5, f'[{short_name}]\nNo significant features', ha='center', va='center', transform=ax.transAxes)
        return

    feature_max = {f: max(p.get(f, 0) for p in all_profiles.values()) for f in all_feature_sets}
    top_features = sorted(feature_max, key=lambda f: -feature_max[f])[:max_features]

    if len(top_features) < 3:
        ax.text(0.5, 0.5, f'[{short_name}]\nToo few features', ha='center', va='center', transform=ax.transAxes)
        return

    raw_matrix = np.zeros((len(draw_list), len(top_features)))
    sign_matrix = np.ones((len(draw_list), len(top_features)))
    valid_bases = []
    for i, (k, kind, n) in enumerate(draw_list):
        if k not in all_profiles: continue
        valid_bases.append(i)
        for j, f in enumerate(top_features):
            raw_matrix[i, j] = all_profiles[k].get(f, 0)
            sign_matrix[i, j] = all_signs[k].get(f, 1.0)

    all_vals = raw_matrix[valid_bases].flatten()
    cap_val = np.percentile(all_vals, 95)
    if cap_val < 1.0: cap_val = all_vals.max()
    if cap_val < 1e-9:
        norm_matrix = np.zeros_like(raw_matrix)
    else:
        norm_matrix = np.sqrt(np.clip(raw_matrix, 0, cap_val) / cap_val)

    signed_matrix = norm_matrix * sign_matrix

    n_features = len(top_features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    for i, (k, kind, n) in enumerate(draw_list):
        if i not in valid_bases: continue
        color = BASIS_COLORS.get(f'B{k}', '#999999')
        values = signed_matrix[i].tolist() + [signed_matrix[i, 0]]
        if kind == 'tn_ref':
            ls, lw, af = '--', 2.0, 0.05
            label = f'B{k} [L] (TN n={n})'
        else:
            ls, lw, af = '-', 2.5, 0.15
            label = f'B{k} [H] (n={n})'
        ax.plot(angles, values, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill(angles, values, color=color, alpha=af)

    # Feature labels
    labels = []
    pos_mask = (ys == 1)
    for f in top_features:
        ftype = all_ftypes_map.get(f, 'num')
        if ftype == 'num':
            arrow = RISK_DIRECTION.get(f, '?')
            labels.append(f'{f}\n[Num {arrow}]')
        else:
            pos_vals = X.loc[pos_mask, f].dropna()
            if len(pos_vals) > 0:
                mode_val = pos_vals.mode().iloc[0]
                mode_pct = (pos_vals == mode_val).mean() * 100
                labels.append(f'{f}\n[Cat: {mode_val} {mode_pct:.0f}%]')
            else:
                labels.append(f'{f}\n[Cat]')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color='black')
    ax.tick_params(axis='x', pad=20)

    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(['-1.0\n(lower)', '-0.5', '0\n(avg)', '0.5', '1.0\n(higher)'],
                       fontsize=6, color='gray')

    zero_angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(zero_angles, [0] * 100, color='black', linewidth=1.0, linestyle='-', alpha=0.4)

    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.4, 1.1))
    ax.set_title(f'{short_name}', fontsize=12, pad=20)


def save_radar(all_results, onehot_probs, sources, save_dir):
    # Combined: 1x3 x 2 figures
    n_per_row = 3
    n_figs = (len(sources) + n_per_row - 1) // n_per_row
    for fig_idx in range(n_figs):
        start = fig_idx * n_per_row
        end = min(start + n_per_row, len(sources))
        batch = sources[start:end]
        n = len(batch)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), subplot_kw=dict(polar=True))
        if n == 1: axes = [axes]
        for i, src in enumerate(batch):
            if src in all_results:
                plot_radar_single(all_results[src], onehot_probs, axes[i])
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'radar_1x3_part{fig_idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Individual
    for src in sources:
        if src not in all_results: continue
        fig, ax = plt.subplots(1, 1, figsize=(9, 9), subplot_kw=dict(polar=True))
        plot_radar_single(all_results[src], onehot_probs, ax)
        short = src.replace('_Dataset', '').replace('_clinical_records', '').replace('_Health_Dataset', '')
        fig.savefig(os.path.join(save_dir, f'radar_{short}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Matrix Plots (soft WPR-based)
# ============================================================

def plot_coordinate_pr(onehot_probs, source_results, sources, ax):
    M = len(onehot_probs)
    n_src = len(sources)
    short_names = [DATASET_SHORT_NAMES.get(s, s) for s in sources]
    sorted_idx = np.argsort(onehot_probs)
    heatmap_data = np.full((n_src, M), np.nan)
    annot_data = np.full((n_src, M), '', dtype=object)
    for i, src_name in enumerate(sources):
        info = source_results[src_name]['basis_info']
        for j, basis_idx in enumerate(sorted_idx):
            bi = info[basis_idx]
            if bi['n_eff'] > 0.5:
                heatmap_data[i, j] = bi['pos_rate']
                annot_data[i, j] = f"{bi['pos_rate']:.2f}\n(n≈{bi['n_eff']:.0f})"
    y_labels = [f"{sn}\n(pos={source_results[s]['overall_pos']:.2f})" for sn, s in zip(short_names, sources)]
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', xticklabels=[f'B{i}' for i in sorted_idx],
                yticklabels=y_labels, cmap='RdYlBu_r', center=0.5, vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Weighted Positive Rate'})
    ax.set_title('Coordinate-Weighted PR', fontsize=10)
    ax.set_xlabel('')


def plot_pearson(source_results, sources, ax):
    M = max(source_results[s]['M'] for s in sources)
    n_src = len(sources)
    short_names = [DATASET_SHORT_NAMES.get(s, s) for s in sources]
    pos_vectors = np.zeros((n_src, M))
    for i, src_name in enumerate(sources):
        info = source_results[src_name]['basis_info']
        for k in range(M):
            pos_vectors[i, k] = info[k]['pos_rate']
    corr_matrix = np.zeros((n_src, n_src))
    p_matrix = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            r, p = pearsonr(pos_vectors[i], pos_vectors[j])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
    annot = np.full((n_src, n_src), '', dtype=object)
    for i in range(n_src):
        for j in range(n_src):
            r = corr_matrix[i, j]
            p = p_matrix[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            annot[i, j] = f'{r:.3f}{sig}'
    sns.heatmap(corr_matrix, annot=annot, fmt='', xticklabels=short_names, yticklabels=short_names,
                cmap='RdYlBu_r', center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True,
                cbar_kws={'label': 'Pearson r'})
    ax.set_title('Source Pearson Correlation Matrix', fontsize=10)


def save_matrix(onehot_probs, source_results, sources, save_dir):
    # Combined
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_coordinate_pr(onehot_probs, source_results, sources, axes[0])
    plot_pearson(source_results, sources, axes[1])
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'matrix_combined.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Individual
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(onehot_probs) * 1.2), 7))
    plot_coordinate_pr(onehot_probs, source_results, sources, ax)
    fig.savefig(os.path.join(save_dir, 'matrix_wpr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plot_pearson(source_results, sources, ax)
    fig.savefig(os.path.join(save_dir, 'matrix_pearson.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def find_checkpoints(base_dir):
    """Find all best_*.pt files under base_dir/Pre/"""
    pre_dir = os.path.join(base_dir, 'Pre')
    if not os.path.exists(pre_dir):
        print(f"Pre/ directory not found under {base_dir}")
        return []

    found = []
    for root, dirs, files in os.walk(pre_dir):
        for f in files:
            if f.startswith('best_') and f.endswith('.pt'):
                found.append(os.path.join(root, f))
            elif f == 'best_joint.pt':
                found.append(os.path.join(root, f))

    # Deduplicate: if same dir has best_joint.pt and best_*.pt, prefer best_joint.pt
    by_dir = {}
    for p in found:
        d = os.path.dirname(p)
        basename = os.path.basename(p)
        if d not in by_dir or basename == 'best_joint.pt':
            by_dir[d] = p

    return sorted(by_dir.values())


def extract_run_tag(ckpt_path):
    """Extract a readable run tag from checkpoint path."""
    parts = ckpt_path.split('/')
    # Find the part with timestamp (e.g., 20260416_223422)
    for i, p in enumerate(parts):
        if len(p) == 15 and p[8] == '_':  # timestamp format
            if i > 0:
                return f"{parts[i-1]}_seed{parts[i-1]}" if parts[i-1].isdigit() else parts[i]
    return os.path.basename(os.path.dirname(ckpt_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Path to base directory (parent of Pre/)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = find_checkpoints(args.base_dir)
    print(f"Found {len(ckpts)} checkpoints")

    for idx, ckpt_path in enumerate(ckpts):
        run_tag = extract_run_tag(ckpt_path)
        print(f"\n[{idx+1}/{len(ckpts)}] {run_tag}")
        print(f"  Path: {ckpt_path}")

        out_dir = os.path.join(args.base_dir, OUTPUT_DIR_NAME, run_tag)
        if os.path.exists(os.path.join(out_dir, 'radar_1x3_part1.png')):
            print(f"  [SKIP] already analyzed")
            continue
        os.makedirs(out_dir, exist_ok=True)

        try:
            model, model_args = load_model(ckpt_path, device)
            sources, loaders = load_source_loaders(model_args)
            print(f"  Sources: {sources}")

            # Compute
            onehot_probs = compute_onehot_basis_risk(model, device)
            soft_results = compute_soft_pr(model, loaders, sources, device)

            # Extract TP/TN for radar
            all_results = {}
            for src in sources:
                try:
                    all_results[src] = extract_tp_tn_with_tabular(model, loaders[src], device, src)
                except Exception as e:
                    print(f"  [WARN] Radar skip {src}: {e}")

            # Save Radar
            if all_results:
                save_radar(all_results, onehot_probs, sources, out_dir)
                print(f"  Radar saved")

            # Save Matrix
            save_matrix(onehot_probs, soft_results, sources, out_dir)
            print(f"  Matrix saved")

            print(f"  Done -> {out_dir}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

        if 'model' in dir():
            del model
        torch.cuda.empty_cache()

    print(f"\nAll done. Results in {os.path.join(args.base_dir, OUTPUT_DIR_NAME)}")


if __name__ == '__main__':
    main()