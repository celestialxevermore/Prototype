"""
Batch Qualitative Analysis
Usage: python analysis.py /path/to/Pre
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ============================================================
# Setup
# ============================================================

ROOT_DIR = Path('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217')
sys.path.append(str(ROOT_DIR))

from models.TabularFLM_S_ import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed
from torch.utils.data import ConcatDataset, DataLoader

EXCLUDE_FEATURES = {'patientid', 'patient_id', 'id', 'ID', 'time'}

BASE_TABLE_PATH = "/storage/personal/eungyeop/dataset/table/origin_table"

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
    tp = (ys == 1) & (preds == 1)
    tn = (ys == 0) & (preds == 0)
    fp = (ys == 0) & (preds == 1)
    fn = (ys == 1) & (preds == 0)
    X_mapped = X_orig.iloc[sidxs].reset_index(drop=True)
    return {
        'dataset_name': dataset_name, 'X': X_mapped, 'pis': pis, 'ys': ys,
        'probs': probs, 'preds': preds, 'argmax': pis.argmax(axis=1),
        'tp_mask': tp, 'tn_mask': tn, 'fp_mask': fp, 'fn_mask': fn,
    }


# ============================================================
# Radar Plots
# ============================================================

def plot_radar_single(result, onehot_probs, ax, max_features=8, min_tp=3):
    src_name = result['dataset_name']
    X = result['X']
    argmax = result['argmax']
    tp_mask = result['tp_mask']
    ys = result['ys']
    M = result['pis'].shape[1]
    short_name = src_name.replace('_Dataset', '').replace('_clinical_records', '')

    num_cols = [c for c in X.select_dtypes(include=[np.number]).columns
                if c.lower() not in {f.lower() for f in EXCLUDE_FEATURES}][:max_features]
    if len(num_cols) < 3:
        ax.text(0.5, 0.5, f'[{short_name}]\nNot enough features', ha='center', va='center', transform=ax.transAxes)
        return

    X_num = X[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_num)

    high_risk_bases = [k for k in range(M) if onehot_probs[k] > 0.5]
    active_bases = []
    for k in high_risk_bases:
        if ((argmax == k) & tp_mask).sum() >= min_tp:
            active_bases.append(k)

    # Add TP top-3
    tp_counts = sorted([(k, int(((argmax == k) & tp_mask).sum())) for k in range(M)], key=lambda x: -x[1])
    for k, n in tp_counts[:3]:
        if n >= min_tp and k not in active_bases:
            active_bases.append(k)

    # Low-risk TN reference
    low_risk_bases = [k for k in range(M) if onehot_probs[k] <= 0.5]
    for k in sorted(low_risk_bases, key=lambda k: onehot_probs[k]):
        if ((argmax == k) & (ys == 0)).sum() >= 3 and k not in active_bases:
            active_bases.insert(0, k)
            break

    if len(active_bases) < 1:
        ax.text(0.5, 0.5, f'[{short_name}]\nNo active bases', ha='center', va='center', transform=ax.transAxes)
        return

    n_features = len(num_cols)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    for k in active_bases:
        is_high = onehot_probs[k] > 0.5
        if is_high:
            k_mask = (argmax == k) & tp_mask
            label_prefix = "TP"
            linestyle = '-'
        else:
            if ((argmax == k) & tp_mask).sum() >= min_tp:
                k_mask = (argmax == k) & tp_mask
                label_prefix = "TP"
                linestyle = '-'
            else:
                k_mask = (argmax == k) & (ys == 0)
                label_prefix = "TN (ref)"
                linestyle = '--'

        n = k_mask.sum()
        if n < 1:
            continue

        vals = X_num.loc[k_mask].values
        z_median = np.median(scaler.transform(vals), axis=0).tolist()
        z_median += z_median[:1]

        color = BASIS_COLORS.get(f'B{k}', '#999999')
        risk = "H" if is_high else "L"
        linewidth = 2.0 if label_prefix == "TN (ref)" else 2.5
        alpha_fill = 0.05 if label_prefix == "TN (ref)" else 0.15

        ax.plot(angles, z_median, color=color, linewidth=linewidth, linestyle=linestyle,
                label=f'B{k} [{risk}] ({label_prefix} n={n})')
        ax.fill(angles, z_median, color=color, alpha=alpha_fill)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(num_cols, fontsize=7)

    all_vals = []
    for line in ax.get_lines():
        all_vals.extend(line.get_ydata())
    if all_vals:
        vmax = max(2, np.ceil(max(all_vals) + 0.3))
        vmin = min(-2, np.floor(min(all_vals) - 0.3))
    else:
        vmax, vmin = 2, -2
    ax.set_ylim(vmin, vmax)
    yticks = [t for t in range(int(vmin), int(vmax) + 1) if vmin <= t <= vmax]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t:+d}σ' if t != 0 else 'median' for t in yticks], fontsize=6, color='gray')
    ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(f'[{short_name}]', fontsize=9, pad=15)


def save_radar_combined(all_results, onehot_probs, sources, save_path):
    M = len(sources)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 6), subplot_kw=dict(polar=True))
    if M == 1:
        axes = [axes]
    for i, src in enumerate(sources):
        if src in all_results:
            plot_radar_single(all_results[src], onehot_probs, axes[i])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_radar_individual(all_results, onehot_probs, sources, save_dir):
    for src in sources:
        if src not in all_results:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
        plot_radar_single(all_results[src], onehot_probs, ax)
        short = src.replace('_Dataset', '').replace('_clinical_records', '')
        fig.savefig(os.path.join(save_dir, f'radar_{short}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Matrix Plots
# ============================================================

def plot_coordinate_pr(onehot_probs, source_results, sources, ax):
    """Soft coordinate weighted positive rate heatmap"""
    M = len(onehot_probs)
    n_src = len(sources)
    short_names = [s.replace('_Dataset', '').replace('_Health_Dataset', '')
                     .replace('_clinical_records', '').replace('_', ' ')[:25] for s in sources]
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


def plot_ratio_pr(onehot_probs, source_results, sources, ax):
    """Ratio heatmap: basis_pos_rate / source_overall_pos_rate"""
    M = len(onehot_probs)
    n_src = len(sources)
    short_names = [s.replace('_Dataset', '').replace('_Health_Dataset', '')
                     .replace('_clinical_records', '').replace('_', ' ')[:25] for s in sources]
    sorted_idx = np.argsort(onehot_probs)

    heatmap_data = np.full((n_src, M), np.nan)
    annot_data = np.full((n_src, M), '', dtype=object)
    for i, src_name in enumerate(sources):
        info = source_results[src_name]['basis_info']
        overall = source_results[src_name]['overall_pos']
        for j, basis_idx in enumerate(sorted_idx):
            bi = info[basis_idx]
            if bi['n_eff'] > 0.5 and overall > 0:
                ratio = bi['pos_rate'] / overall
                heatmap_data[i, j] = ratio
                annot_data[i, j] = f"{ratio:.2f}\n({bi['pos_rate']:.2f})"
    y_labels = [f"{sn}\n(pos={source_results[s]['overall_pos']:.2f})" for sn, s in zip(short_names, sources)]
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', xticklabels=[f'B{i}' for i in sorted_idx],
                yticklabels=y_labels, cmap='RdYlBu_r', center=1.0, vmin=0, vmax=2.5,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Positive Rate Ratio'})
    ax.set_title('Label-Weighted PR Ratio', fontsize=10)
    ax.set_xlabel('')


def plot_pearson(source_results, sources, ax):
    """Pearson correlation of per-basis positive rate vectors"""
    M = max(source_results[s]['M'] for s in sources)
    n_src = len(sources)
    short_names = [s.replace('_Dataset', '').replace('_Health_Dataset', '')
                     .replace('_clinical_records', '').replace('_', ' ')[:25] for s in sources]

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
    ax.set_title('Source Functional Similarity', fontsize=10)


def save_matrix_combined(onehot_probs, source_results, sources, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    plot_coordinate_pr(onehot_probs, source_results, sources, axes[0])
    plot_ratio_pr(onehot_probs, source_results, sources, axes[1])
    plot_pearson(source_results, sources, axes[2])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_matrix_individual(onehot_probs, source_results, sources, save_dir):
    # 1. Coordinate PR
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(onehot_probs) * 1.2), 7))
    plot_coordinate_pr(onehot_probs, source_results, sources, ax)
    fig.savefig(os.path.join(save_dir, 'matrix_coordinate_pr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Ratio PR
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(onehot_probs) * 1.2), 7))
    plot_ratio_pr(onehot_probs, source_results, sources, ax)
    fig.savefig(os.path.join(save_dir, 'matrix_ratio_pr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Pearson
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plot_pearson(source_results, sources, ax)
    fig.savefig(os.path.join(save_dir, 'matrix_pearson.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def find_checkpoints(pre_dir, ckpt_names=('best_joint.pt', 'best.pt')):
    found = []
    for name in ckpt_names:
        found.extend(glob.glob(os.path.join(pre_dir, '**', name), recursive=True))
    # 같은 폴더에 best_joint.pt 와 best.pt 가 함께 있으면 best_joint.pt 우선
    by_dir = {}
    for p in found:
        d = os.path.dirname(p)
        if d not in by_dir or os.path.basename(p) == 'best_joint.pt':
            by_dir[d] = p
    return sorted(by_dir.values())


def extract_run_tag(ckpt_path):
    return os.path.basename(os.path.dirname(ckpt_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_dir', help='Path to Pre/ directory')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = find_checkpoints(args.pre_dir)
    print(f"Found {len(ckpts)} checkpoints")

    for idx, ckpt_path in enumerate(ckpts):
        run_tag = extract_run_tag(ckpt_path)
        print(f"\n[{idx+1}/{len(ckpts)}] {run_tag}")
        print(f"  Path: {ckpt_path}")

        # Output directory
        out_dir = os.path.join(args.pre_dir, 'analysis_results', run_tag)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Load model
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
                save_radar_combined(all_results, onehot_probs, sources,
                                    os.path.join(out_dir, 'radar_1xM.png'))
                save_radar_individual(all_results, onehot_probs, sources, out_dir)
                print(f"  Radar saved")

            # Save Matrix
            save_matrix_combined(onehot_probs, soft_results, sources,
                                 os.path.join(out_dir, 'matrix_1x3.png'))
            save_matrix_individual(onehot_probs, soft_results, sources, out_dir)
            print(f"  Matrix saved")

            print(f"  Done -> {out_dir}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

        # Free memory
        if 'model' in dir():
            del model
        torch.cuda.empty_cache()

    print(f"\nAll done. Results in {os.path.join(args.pre_dir, 'analysis_results')}")


if __name__ == '__main__':
    main()
