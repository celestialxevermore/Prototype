"""KF (Kidney Function) regression — ProtoLLM scenario3 entry.

Multi-source PT (5 ICU cohorts, 85/15) + per-target FT (80/20 → 85/15).
Output: <experiments_root>/<base_dir>/<target>/args_seed:<seed>/protollm/scenario3_*.json

Mortality main_SS.py 는 무수정. 이 파일은 KF regression 전용 entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _amp_ctx(args):
    if not getattr(args, 'use_amp', False):
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from dataset.data_dataloaders import (  # noqa: E402
    prepare_embedding_dataloaders_kf,
    split_kf_train_pool,
)
from models.TabularFLM_S_ import Model  # noqa: E402

KF_COHORTS = ["mimic_kf", "eicu_kf", "hirid_kf", "sic_kf", "zigong_kf"]
EXPERIMENTS_ROOT = "/storage/personal/eungyeop/experiments/experiments"


def get_args():
    p = argparse.ArgumentParser(description="ProtoLLM KF regression scenario3")
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--source_data', nargs='+', default=KF_COHORTS, choices=KF_COHORTS)
    p.add_argument('--target_data', nargs='+', required=True, choices=KF_COHORTS)
    p.add_argument('--base_dir', type=str, default='dl_baselines_kf_20260505')
    p.add_argument('--batch_size', type=int, default=128)

    # Embedding (gemma-medical d=256 standard)
    p.add_argument('--llm_model', type=str, default='gemma-medical')
    p.add_argument('--embed_type', type=str, default='carte_d256')
    p.add_argument('--input_dim', type=int, default=256)

    # Model dims
    p.add_argument('--hidden_dim', type=int, default=192)
    p.add_argument('--struct_hidden_dim', type=int, default=192)
    p.add_argument('--output_dim', type=int, default=1)
    p.add_argument('--dropout_rate', type=float, default=0.1)
    p.add_argument('--num_classes', type=int, default=1)  # regression marker

    # LCG (default off — vanilla BasisGAT phase 1 only)
    p.add_argument('--use_lcg', action='store_true', default=False)
    p.add_argument('--n_graphs', type=int, default=8)
    p.add_argument('--n_nodes', type=int, default=8)
    p.add_argument('--graph_dim', type=int, default=256)
    p.add_argument('--fgw_alpha', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--eps', type=float, default=0.01)
    p.add_argument('--reg', type=float, default=0.01)
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--soft_tau', type=float, default=0.01)
    p.add_argument('--entropy_reg', type=float, default=0.01)
    p.add_argument('--lcg_div_alpha', type=float, default=10)
    p.add_argument('--vq_beta', type=float, default=0.3)
    p.add_argument('--kl', action='store_true')
    p.add_argument('--kl_gamma', type=float, default=2.0)
    p.add_argument('--additional_FGW', action='store_true')
    p.add_argument('--diversifying_loss', action='store_true')
    p.add_argument('--lcg_diversifying_loss', action='store_true')
    p.add_argument('--lcg_hinge_margin_sq', type=float, default=1.0)
    p.add_argument('--lcg_strategy', type=str, default='hierarchical')
    p.add_argument('--lcg_struct_type', type=str, default='static')
    p.add_argument('--feat_distance', type=str, default='cosine')
    p.add_argument('--orth_reg', type=float, default=0.1)
    p.add_argument('--div_reg', type=float, default=1)
    p.add_argument('--hp_reg', type=float, default=1.0)
    p.add_argument('--hs_reg', type=float, default=0.1)
    p.add_argument('--hs_warmup', type=int, default=30)
    p.add_argument('--ema', action='store_true')

    # Basis GAT
    p.add_argument('--num_shared_layers', type=int, default=2)
    p.add_argument('--num_basis_layers', type=int, default=2)
    p.add_argument('--basis_type', type=str, default='graphormer',
                   choices=['mul', 'ind', 'sdpa', 'graphormer'])
    p.add_argument('--no_self_loop', action='store_true')
    p.add_argument('--attn_type', type=str, default='gat_v1',
                   choices=['gat_v1', 'att', 'gat_v2', 'gate'])
    p.add_argument('--edge_type', type=str, default='mlp',
                   choices=['mlp', 'normal', 'no_use'])
    p.add_argument('--threshold', type=float, default=0.5)
    # Coord (used by some LCG branches even when use_lcg=False — defensive)
    p.add_argument('--coord_softmax_temp', type=float, default=0.5)
    p.add_argument('--coord_reg_lambda', type=float, default=0.2)
    p.add_argument('--coord_target_mode', type=str, default='soft', choices=['soft', 'hard'])
    p.add_argument('--coord_tau', type=float, default=0.3)
    p.add_argument('--few_shot', type=int, default=0)
    p.add_argument('--train_epochs', type=int, default=1000)
    # AMP (mortality d256 production used --use_amp + bf16 to fit 200+ features)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--no_amp', dest='use_amp', action='store_false')
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'])

    # Training
    p.add_argument('--source_lr', type=float, default=1e-4)
    p.add_argument('--source_lr_few', type=float, default=1e-4)
    p.add_argument('--pretrain_epochs', type=int, default=30)
    p.add_argument('--pretrain_patience', type=int, default=5)
    p.add_argument('--finetune_epochs', type=int, default=30)
    p.add_argument('--finetune_patience', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--min_lr_mult', type=float, default=0.10)
    p.add_argument('--weight_decay', type=float, default=1e-5)

    # Misc
    p.add_argument('--use_target_head', action='store_true', default=False)
    p.add_argument('--use_mmap_embeddings', action='store_true', default=False)
    p.add_argument('--source_dataset_name', type=str, default='')
    p.add_argument('--del_feat', nargs='+', default=[])

    args = p.parse_args()
    args.table_path = "/storage/personal/eungyeop/dataset/table/"
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def regression_eval(model, loader, device, src_idx=None, args=None):
    if loader is None:
        return None
    model.eval()
    y_true, y_pred = [], []
    ctx = _amp_ctx(args) if args is not None else nullcontext()
    for batch in loader:
        if src_idx is not None:
            batch['src_idx'] = src_idx
        else:
            batch.pop('src_idx', None)
        with ctx:
            pred = model.predict(batch, return_all=False)
        y_true.extend(batch['y'].cpu().numpy().flatten().tolist())
        y_pred.extend(pred.detach().float().cpu().numpy().flatten().tolist())
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    try:
        pr, _ = pearsonr(y_pred, y_true)
        pr = float(pr) if not np.isnan(pr) else 0.0
    except Exception:
        pr = 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pr, "n": int(len(y_true))}


def train_pt_one_epoch(model, src_train_with_idx, optimizer, args):
    model.train()
    loss_sum, n_sum = 0.0, 0
    for src_idx, (src_name, train_loader) in src_train_with_idx:
        if train_loader is None:
            continue
        for batch in train_loader:
            batch['src_idx'] = src_idx
            optimizer.zero_grad()
            with _amp_ctx(args):
                loss = model(batch, batch['y'])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch['y'])
            n_sum += len(batch['y'])
    return loss_sum / max(n_sum, 1)


def avg_source_val(model, src_val_with_idx, device, args=None):
    metrics = []
    for src_idx, (src_name, val_loader) in src_val_with_idx:
        m = regression_eval(model, val_loader, device, src_idx=src_idx, args=args)
        if m is not None:
            metrics.append(m)
    if not metrics:
        return {"rmse": float('inf'), "mae": float('inf'),
                "r2": -float('inf'), "pearson_r": 0.0, "n": 0}
    keys = ("rmse", "mae", "r2", "pearson_r")
    return {**{k: float(np.mean([m[k] for m in metrics])) for k in keys},
            "n": int(sum(m['n'] for m in metrics))}


def main():
    args = get_args()
    set_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[KF-ProtoLLM] targets={args.target_data} seed={args.random_seed} "
          f"sources={args.source_data}")
    print(f"[KF-ProtoLLM] device={device} use_lcg={args.use_lcg} "
          f"input_dim={args.input_dim} basis={args.basis_type}x{args.num_basis_layers}")

    # (1) Source loaders (one per cohort, same seed)
    src_train_with_idx, src_val_with_idx = [], []
    for i, src_name in enumerate(args.source_data):
        res = prepare_embedding_dataloaders_kf(args, src_name, is_source=True)
        tr, va, _ = res['loaders']
        src_train_with_idx.append((i, (src_name, tr)))
        src_val_with_idx.append((i, (src_name, va)))

    # (2) Build model
    model = Model(
        args,
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        output_dim=args.output_dim, dropout_rate=args.dropout_rate,
        llm_model=args.llm_model,
        experiment_id=f"kf_multi_seed{args.random_seed}",
        mode='Full',
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.source_lr,
                                  weight_decay=args.weight_decay)

    # (3) PT loop with early stopping on avg source-val RMSE
    print(f"[KF-ProtoLLM] === PT start (epochs={args.pretrain_epochs} "
          f"patience={args.pretrain_patience}) ===")
    best_val_rmse = float('inf')
    best_pt_state = None
    bad = 0
    pt_history = []
    for ep in range(1, args.pretrain_epochs + 1):
        t0 = time.time()
        tr_loss = train_pt_one_epoch(model, src_train_with_idx, optimizer, args)
        val_m = avg_source_val(model, src_val_with_idx, device, args=args)
        dt = time.time() - t0
        print(f"[PT ep={ep:02d}] tr_loss={tr_loss:.4f} val_rmse={val_m['rmse']:.4f} "
              f"val_r2={val_m['r2']:.4f} t={dt:.1f}s", flush=True)
        pt_history.append({"epoch": ep, "tr_loss": tr_loss, **val_m, "wall": dt})
        if val_m['rmse'] < best_val_rmse - 1e-4:
            best_val_rmse = val_m['rmse']
            best_pt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.pretrain_patience:
                print(f"[PT] early stop at ep={ep}")
                break

    if best_pt_state is not None:
        model.load_state_dict(best_pt_state)
    print(f"[KF-ProtoLLM] PT done — best avg source val rmse={best_val_rmse:.4f}")

    # (4) Per-target FT loop — restore best PT state for each target
    config_keys = [
        'source_data', 'target_data', 'batch_size', 'llm_model', 'embed_type',
        'input_dim', 'hidden_dim', 'dropout_rate', 'num_classes', 'use_lcg',
        'basis_type', 'num_basis_layers', 'source_lr', 'pretrain_epochs',
        'pretrain_patience', 'finetune_epochs', 'finetune_patience',
        'weight_decay',
    ]
    summary = []
    for tgt_idx, target_name in enumerate(args.target_data):
        # Restore best PT state for fresh FT on each target
        if best_pt_state is not None:
            model.load_state_dict(best_pt_state)

        print(f"\n[KF-ProtoLLM] === FT start on target={target_name} "
              f"({tgt_idx+1}/{len(args.target_data)}) ===")
        res_t = prepare_embedding_dataloaders_kf(args, target_name, is_source=False)
        train_pool = res_t['train_dataset']
        test_loader = res_t['loaders'][2]
        ft_train_loader, ft_val_loader, _, _ = split_kf_train_pool(
            train_pool, args, val_ratio=0.15)
        print(f"[FT] ft_train={len(ft_train_loader.dataset)} "
              f"ft_val={len(ft_val_loader.dataset)} test={len(test_loader.dataset)}")

        optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.source_lr,
                                         weight_decay=args.weight_decay)
        best_val_rmse_t = float('inf')
        best_ft_state = None
        bad_t = 0
        ft_history = []
        for ep in range(1, args.finetune_epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum, n_sum = 0.0, 0
            for batch in ft_train_loader:
                batch.pop('src_idx', None)
                optimizer_ft.zero_grad()
                with _amp_ctx(args):
                    loss = model(batch, batch['y'])
                loss.backward()
                optimizer_ft.step()
                loss_sum += loss.item() * len(batch['y'])
                n_sum += len(batch['y'])
            tr_loss_t = loss_sum / max(n_sum, 1)
            val_m_t = regression_eval(model, ft_val_loader, device, src_idx=None, args=args)
            dt = time.time() - t0
            print(f"[FT {target_name} ep={ep:02d}] tr_loss={tr_loss_t:.4f} "
                  f"val_rmse={val_m_t['rmse']:.4f} val_r2={val_m_t['r2']:.4f} "
                  f"t={dt:.1f}s", flush=True)
            ft_history.append({"epoch": ep, "tr_loss": tr_loss_t, **val_m_t, "wall": dt})
            if val_m_t['rmse'] < best_val_rmse_t - 1e-4:
                best_val_rmse_t = val_m_t['rmse']
                best_ft_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_t = 0
            else:
                bad_t += 1
                if bad_t >= args.finetune_patience:
                    print(f"[FT {target_name}] early stop at ep={ep}")
                    break

        if best_ft_state is not None:
            model.load_state_dict(best_ft_state)

        # (5) Test eval
        test_m = regression_eval(model, test_loader, device, src_idx=None, args=args)
        print(f"[KF-ProtoLLM] TEST {target_name}: rmse={test_m['rmse']:.4f} "
              f"mae={test_m['mae']:.4f} r2={test_m['r2']:.4f} "
              f"pearson={test_m['pearson_r']:.4f}")

        # (6) Save JSON per target
        out_dir = os.path.join(
            EXPERIMENTS_ROOT, args.base_dir, target_name,
            f"args_seed:{args.random_seed}", "protollm",
        )
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = os.path.join(out_dir, f"scenario3_{ts}.json")
        payload = {
            "model": "protollm_kf",
            "scenario": "scenario3",
            "task_type": "regression",
            "cohort": target_name,
            "seed": args.random_seed,
            "test": test_m,
            "best_val_rmse_pt": best_val_rmse,
            "best_val_rmse_ft": best_val_rmse_t,
            "pretrain_history": pt_history,
            "finetune_history": ft_history,
            "config": {k: getattr(args, k) for k in config_keys},
            "embedding": {
                "y_min": res_t.get('y_min'), "y_max": res_t.get('y_max'),
                "y_mean": res_t.get('y_mean'), "y_std": res_t.get('y_std'),
            },
            "timestamp": ts,
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[KF-ProtoLLM] saved -> {out_path}")
        summary.append((target_name, test_m))

    print(f"\n[KF-ProtoLLM] === ALL DONE seed={args.random_seed} ===")
    for tn, tm in summary:
        print(f"  {tn}: rmse={tm['rmse']:.4f} r2={tm['r2']:.4f} "
              f"mae={tm['mae']:.4f} pearson={tm['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
