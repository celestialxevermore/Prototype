# analysis/adjacency_analysis.py
import os, copy, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# 현재 스크립트 파일 위치
current_dir = Path(__file__).resolve().parent
import sys
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리 추가

from models.TabularFLM_S import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed


def extract_deleted_features_from_checkpoint(p):
    import re
    stem = Path(p).stem
    for pat in [r"D:\[([^\]]*)\]", r"D_\[([^\]]*)\]", r"D-\[([^\]]*)\]"]:
        m = re.search(pat, stem)
        if m:
            inside = m.group(1)
            return [s.strip().strip("'\"") for s in inside.split(",") if s.strip()]
    return []


def add_cb(ax, im):
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class MVisualizer:
    def __init__(self, ckpt_path: str, device='cuda', auto_del_feat=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.args = ckpt['args']
        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            print(f"[INFO] Applied del_feat from filename: {auto_del_feat}")

        # 모델 생성
        self.model = Model(self.args, self.args.input_dim, self.args.hidden_dim,
                           self.args.output_dim,
                           self.args.dropout_rate, self.args.llm_model,
                           "viz", "viz").to(self.device)

        # 옛 키(alpha_ema 등)는 제외하고 로드
        sd = ckpt['model_state_dict']
        sd = {k: v for k, v in sd.items() if 'alpha_ema' not in k}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print("[INFO] Missing keys:", missing)
        if unexpected:
            print("[INFO] Unexpected keys:", unexpected)

        self.model.eval()
        self.num_layers = int(getattr(self.args, 'num_basis_layers', 3))
        self.num_heads  = int(getattr(self.args, 'k_basis', 8))

    def _make_loader(self, dataset_name: str):
        args2 = copy.deepcopy(self.args)
        args2.source_data = dataset_name
        fix_seed(args2.random_seed)
        res = prepare_embedding_dataloaders(args2, args2.source_data)
        tr, va, te = res['loaders']
        from torch.utils.data import ConcatDataset, DataLoader
        ds = ConcatDataset([tr.dataset, va.dataset, te.dataset])
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    @torch.no_grad()
    def _forward_collect(self, batch):
        bd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}

        desc_list, nv_list = [], []
        if all(k in bd for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            nv_list.append(bd['cat_name_value_embeddings']); desc_list.append(bd['cat_desc_embeddings'])
        if all(k in bd for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            nv_list.append(bd['num_prompt_embeddings']);     desc_list.append(bd['num_desc_embeddings'])
        desc_list, nv_list = self.model.remove_feature(bd, desc_list, nv_list)

        desc = torch.cat(desc_list, dim=1)   # [B,S,D]
        nv   = torch.cat(nv_list , dim=1)    # [B,S,D]
        B, S, D = nv.shape

        # ---- 슬롯 기반 프라이어: bias_log=log Q, Q ----
        bias_log, Q, _ = self.model.basis_affinity(desc, nv)   # [B,H,S,S], [B,H,S,S]
        slotQ_np = Q[0].detach().cpu().numpy()                 # [H,S,S]

        # ---- 전역 G (모드 무관, 고정) ----
        if hasattr(self.model.basis_affinity, "export_G_numpy"):
            G_np = self.model.basis_affinity.export_G_numpy()  # [H,K,K]
        else:
            G_np = None

        # ---- Basis 레이어 한 바퀴(ATT/ADJ 수집) ----
        Ms, ATTs, ADJs = [], [], []
        cls = self.model.cls.expand(B, 1, D)
        x   = torch.cat([cls, nv], dim=1)  # [B,T,D], T=S+1
        x_basis = x.clone()

        # 학습과 동일: Q를 pre-softmax bias로 사용
        mask_M = bias_log.exp().clamp(1e-6, 1.0 - 1e-6)  # [B,H,S,S]

        for l in range(self.num_layers):
            norm_x = self.model.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.model.basis_layers[l](desc, norm_x, mask_M=mask_M)
            new_adj = self.model.basis_layers[l].new_adjacency  # [B,T,T]
            x_basis = x_basis + basis_outputs.reshape(B, S+1, D)

            # 실제 샘플 attention P의 Var-Var 블록만 저장
            varvar = att[0, :, 1:, 1:].detach().cpu().numpy()  # [H,S,S]
            for h in range(self.num_heads):
                Ms.append(varvar[h])                           # 총 길이 L*H

            ATTs.append(att[0].cpu().numpy())      # [H,T,T] (numpy)
            ADJs.append(new_adj[0].cpu().numpy())  # [T,T]   (numpy)

        feat_names = self.model.extract_feature_names(bd)  # 길이 S
        return Ms, ATTs, ADJs, feat_names, slotQ_np, G_np

    # 공통 그리드
    def _grid_plot(self, mats, names, title, save_path):
        L, H = self.num_layers, self.num_heads
        n = mats[0].shape[0]
        fig, axes = plt.subplots(L, H, figsize=(H*3.2, L*3.2))
        if L == 1: axes = np.expand_dims(axes, 0)
        if H == 1: axes = np.expand_dims(axes, 1)

        # vmin/vmax (대각 제외, 5~95 분위)
        all_vals = []
        for M in mats:
            mask = ~np.eye(n, dtype=bool)
            vals = M[mask]
            vals = vals[np.isfinite(vals)]
            all_vals.extend(vals)
        if all_vals:
            all_vals = np.array(all_vals)
            q05, q95 = np.quantile(all_vals, 0.05), np.quantile(all_vals, 0.95)
            if q05 == q95:
                vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
            else:
                vmin, vmax = float(q05), float(q95)
        else:
            vmin, vmax = 0, 1

        for l in range(L):
            for h in range(H):
                ax = axes[l, h]
                M = mats[l*H + h]
                im = ax.imshow(M, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    # 중심화 그리드
    def _grid_plot_centered(self, mats, names, title, save_path, center=0.0, cmap='RdBu_r'):
        L, H = self.num_layers, self.num_heads
        n = mats[0].shape[0]

        # 대칭 vlim 계산
        max_abs = 0.0
        for M in mats:
            mask = ~np.eye(n, dtype=bool)
            vals = M[mask] - center if np.isscalar(center) else (M[mask] - center[mask])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                max_abs = max(max_abs, np.max(np.abs(vals)))
        vmin, vmax = -max_abs, +max_abs

        fig, axes = plt.subplots(L, H, figsize=(H*3.2, L*3.2))
        if L == 1: axes = np.expand_dims(axes, 0)
        if H == 1: axes = np.expand_dims(axes, 1)

        for l in range(L):
            for h in range(H):
                ax = axes[l, h]
                M = mats[l*H + h]
                data = (M - center) if np.isscalar(center) else (M - center)
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    def visualize_dataset(self, dataset_name: str, role: str, out_root: Path, max_samples=2):
        loader = self._make_loader(dataset_name)
        base = out_root / role / dataset_name
        ensure_dir(base)
        count = 0

        for batch in loader:
            # Ms: 레이어×헤드 Var-Var P, ATTs: 레이어별 전체 P, ADJs: 레이어별 구조 마스크
            Ms, ATTs, ADJs, var_names, slotQ_np, G_np = self._forward_collect(batch)

            sample_dir = base / f"sample_{count}"
            ensure_dir(sample_dir)

            # (1) Var-Var Attention (P) — 레이어×헤드
            grid_M = [Ms[l*self.num_heads + h] for l in range(self.num_layers) for h in range(self.num_heads)]
            self._grid_plot(
                grid_M, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Var-Var Attention (P)",
                sample_dir / "VarVar_P_grid.png"
            )

            # (2) Attention × Adjacency — 레이어×헤드
            names_all = ["CLS"] + var_names
            grid_AX = []
            for l in range(self.num_layers):
                att = ATTs[l]  # [H,T,T]
                adj = ADJs[l]  # [T,T]
                for h in range(self.num_heads):
                    grid_AX.append(att[h] * adj)
            self._grid_plot(
                grid_AX, names_all,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Attention × Adjacency",
                sample_dir / "AttnXAdj_grid.png"
            )

            # (옵션) 구조 마스크 자체 확인 — 레이어×헤드로 복제해 LH 개수 맞춤
            grid_adj = []
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    grid_adj.append(ADJs[l])  # [T,T]를 H번 복제
            self._grid_plot(
                grid_adj, names_all,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Structural Adjacency (binary)",
                sample_dir / "Adjacency_binary_grid.png"
            )

            # (3) 중심화 비교
            S = len(var_names)
            uniform_S = 1.0 / max(S, 1)

            # 마지막 레이어 P 중심화
            alpha_last = ATTs[-1][:, 1:, 1:]   # [H,S,S] (numpy)
            grid_alpha = [alpha_last[h] for h in range(self.num_heads)]
            grid_alpha = grid_alpha * self.num_layers  # L×H 그리드 채우기
            self._grid_plot_centered(
                grid_alpha, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • α (last layer, center=1/S)",
                sample_dir / "Alpha_center_uniform_last.png",
                center=uniform_S, cmap='RdBu_r'
            )

            # Q 중심화 — L×H 복제
            grid_q = [slotQ_np[h] for h in range(self.num_heads)] * self.num_layers
            self._grid_plot_centered(
                grid_q, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Slot Target Q (center=1/S)",
                sample_dir / "SlotTarget_center_uniform.png",
                center=uniform_S, cmap='RdBu_r'
            )

            # Δ = α − Q — 마지막 레이어 기준 (L×H 복제)
            grid_delta = [(alpha_last[h] - slotQ_np[h]) for h in range(self.num_heads)] * self.num_layers
            self._grid_plot_centered(
                grid_delta, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Δ(α − Q) (last layer)",
                sample_dir / "Delta_alpha_minus_Q_last.png",
                center=0.0, cmap='RdBu_r'
            )

            # (4) Global Slot Graph G (전역, 샘플 무관) — 파일명 유지
            if G_np is not None:
                K = G_np.shape[-1]
                slot_names = [f"z{j}" for j in range(K)]
                grid_G = [G_np[h] for h in range(self.num_heads)] * self.num_layers

                self._grid_plot(
                    grid_G, slot_names,
                    f"{role.capitalize()} • {dataset_name} • Sample {count} • Global Slot Graph G",
                    sample_dir / "GlobalSlotGraph_G.png"   # 이름 유지
                )

                uniform_K = 1.0 / max(K, 1)
                self._grid_plot_centered(
                    grid_G, slot_names,
                    f"{role.capitalize()} • {dataset_name} • Sample {count} • Global Slot Graph G (center=1/K)",
                    sample_dir / "GlobalSlotGraph_center_uniform.png",
                    center=uniform_K, cmap='RdBu_r'
                )

            count += 1
            if count >= max_samples:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True, type=str)
    ap.add_argument('--max_samples', type=int, default=2)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    auto_del = extract_deleted_features_from_checkpoint(args.checkpoint_dir)
    if auto_del:
        print(f"[INFO] del_feat from filename: {auto_del}")

    viz = MVisualizer(args.checkpoint_dir, auto_del_feat=auto_del)

    if args.output_dir is None:
        parent = Path(args.checkpoint_dir).parent
        out_root = Path(str(parent).replace("/checkpoints/", "/visualization/")) / "graph_visualization"
    else:
        out_root = Path(args.output_dir)
    ensure_dir(out_root)

    sources = viz.args.source_data if isinstance(viz.args.source_data, (list, tuple)) else [viz.args.source_data]
    target  = getattr(viz.args, 'target_data', None) or 'heart'

    for s in sources:
        viz.visualize_dataset(s, role='source', out_root=out_root, max_samples=args.max_samples)
    viz.visualize_dataset(target, role='target', out_root=out_root, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
