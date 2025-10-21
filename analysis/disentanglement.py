# analysis/adjacency_analysis.py
import os, copy, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 현재 스크립트 파일 위치
current_dir = Path(__file__).resolve().parent
import sys
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리 추가

from utils.affinity import BasisSlotAffinityGAT
from models.TabularFLM_S import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed


# ---------------- utils ----------------
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


# ---------------- visualizer ----------------
class MVisualizer:
    def __init__(self, ckpt_path: str, device='cuda', auto_del_feat=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.args = ckpt['args']

        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            print(f"[INFO] Applied del_feat from filename: {auto_del_feat}")

        self.num_layers = int(self.args.num_basis_layers)
        self.num_heads  = int(self.args.n_heads)
        self.num_slots  = int(self.args.n_slots)
        self.slot_dim   = int(self.args.slot_dim)

        # 모델 생성
        self.model = Model(
            self.args,
            self.args.input_dim,
            self.args.hidden_dim,
            self.args.output_dim,
            self.args.dropout_rate,
            self.args.llm_model,
            "viz",
            "viz"
        ).to(self.device)

        # state_dict 로드
        sd = ckpt['model_state_dict']
        sd = {k: v for k, v in sd.items() if 'alpha_ema' not in k}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:   print("[INFO] Missing keys:", missing)
        if unexpected:print("[INFO] Unexpected keys:", unexpected)

        self.model.eval()

    # -------- dataloader --------
    def _make_loader(self, dataset_name: str):
        args2 = copy.deepcopy(self.args)
        args2.source_data = dataset_name
        fix_seed(args2.random_seed)
        res = prepare_embedding_dataloaders(args2, args2.source_data)
        tr, va, te = res['loaders']
        from torch.utils.data import ConcatDataset, DataLoader
        ds = ConcatDataset([tr.dataset, va.dataset, te.dataset])
        # per-sample 시각화를 위해 batch_size=1 유지
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # -------- forward (한 배치; 시각화에 필요한 모든 것 반환) --------
    @torch.no_grad()
    def _forward_collect(self, batch):
        bd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}

        # ---- gather embeddings ----
        desc_list, nv_list = [], []
        if all(k in bd for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            nv_list.append(bd['cat_name_value_embeddings'])
            desc_list.append(bd['cat_desc_embeddings'])
        if all(k in bd for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            nv_list.append(bd['num_prompt_embeddings'])
            desc_list.append(bd['num_desc_embeddings'])

        desc = torch.cat(desc_list, dim=1)   # [B,S,D]
        nv   = torch.cat(nv_list , dim=1)    # [B,S,D]
        B, S, D = nv.shape

        # ===== Basis Attention =====
        Ms, ATTs, ADJs = [], [], []
        cls = self.model.basis_cls.expand(B, 1, D)
        x_basis = torch.cat([cls, nv], dim=1)
        for l in range(self.num_layers):
            norm_x = self.model.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.model.basis_layers[l](desc, norm_x)
            x_basis = x_basis + basis_outputs.reshape(B, S+1, D)

            # store adjacency and attention
            ADJs.append(self.model.basis_layers[l].new_adjacency[0].cpu().numpy())
            ATTs.append(att[0].cpu().numpy())

            # Var-Var 부분만 잘라서 [H,S,S]
            Ms.extend(att[0, :, 1:, 1:].detach().cpu().numpy())

        # ===== Shared Attention =====
        Shared_ATTs, Shared_ADJs = [], []
        cls = self.model.shared_cls.expand(B, 1, D)
        x_shared = torch.cat([cls, nv], dim=1)
        for l in range(self.model.num_shared_layers):
            norm_x = self.model.shared_layer_norms[l](x_shared)
            out, att = self.model.shared_layers[l](desc, norm_x)
            x_shared = x_shared + out
            Shared_ATTs.append(att[0].cpu().numpy())
            Shared_ADJs.append(self.model.shared_layers[l].new_adjacency[0].cpu().numpy())

        # feature 이름
        feat_names = self.model.extract_feature_names(bd)

        # 간결 반환
        return Ms, ATTs, ADJs, feat_names, Shared_ATTs, Shared_ADJs

    def plot_SharedGAT_heatmap(self, ATTs, var_names, save_path):
        L, H = self.num_layers, self.num_heads
        n = ATTs[0].shape[-1]
        fig, axes = plt.subplots(L, H, figsize=(H*3.2, L*3.2))
        if L == 1: axes = np.expand_dims(axes, 0)
        if H == 1: axes = np.expand_dims(axes, 1)
        for l in range(L):
            for h in range(H):
                ax = axes[l, h]
                M = ATTs[l][h]  # [T,T]
                im = ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max())
                ax.set_title(f"SharedGAT L{l}·H{h}", fontsize=8)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(var_names, rotation=90, fontsize=5)
                ax.set_yticklabels(var_names, fontsize=5)
                add_cb(ax, im)
        plt.suptitle("SharedGAT Attention", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

    def plot_S_heatmap(self, S_np, var_names, save_path):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(S_np, aspect='auto', cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax); cbar.ax.tick_params(labelsize=7)
        ax.set_title("Slot Assignment S (variable → slot distribution)")
        ax.set_xlabel("Slots"); ax.set_ylabel("Variables")
        ax.set_xticks(range(S_np.shape[1])); ax.set_xticklabels([f"z{j}" for j in range(S_np.shape[1])], fontsize=7)
        ax.set_yticks(range(len(var_names))); ax.set_yticklabels(var_names, fontsize=7)
        plt.tight_layout(); plt.savefig(save_path, dpi=250, bbox_inches='tight'); plt.close(fig)

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
            vals = M[mask]; vals = vals[np.isfinite(vals)]
            all_vals.extend(vals)
        if all_vals:
            all_vals = np.array(all_vals)
            q05, q95 = np.quantile(all_vals, 0.05), np.quantile(all_vals, 0.95)
            vmin, vmax = (float(np.min(all_vals)), float(np.max(all_vals))) if q05 == q95 else (float(q05), float(q95))
        else:
            vmin, vmax = 0, 1
        for l in range(L):
            for h in range(H):
                ax = axes[l, h]; M = mats[l*H + h]
                im = ax.imshow(M, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95]); ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight'); plt.close(fig)


    def _grid_plot_permatrix(self, mats, names, title, save_path, cmap='viridis'):
        L, H = self.num_layers, self.num_heads
        n = mats[0].shape[0]
        fig, axes = plt.subplots(L, H, figsize=(H*3.2, L*3.2))
        if L == 1: axes = np.expand_dims(axes, 0)
        if H == 1: axes = np.expand_dims(axes, 1)
        for l in range(L):
            for h in range(H):
                ax = axes[l, h]; M = mats[l*H + h]
                mask = ~np.eye(n, dtype=bool)
                vals = M[mask]; vals = vals[np.isfinite(vals)]
                if vals.size == 0: vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = float(np.min(vals)), float(np.max(vals))
                    if vmin == vmax: vmin, vmax = float(vmin - 1e-6), float(vmax + 1e-6)
                im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95]); ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight'); plt.close(fig)
    def visualize_average_heads(self, all_shared_ATTs, all_basis_ATTs, var_names, save_dir):
        """
        모든 샘플의 Shared / Basis attention 중 '마지막 layer'의 head별 평균 시각화
        """
        ensure_dir(save_dir)
        n_heads = self.num_heads
        n_vars  = len(var_names) + 1  # CLS 포함

        # numpy 배열 형태로 변환
        all_shared_ATTs = np.stack(all_shared_ATTs, axis=0)  # [N, L, H, T, T]
        all_basis_ATTs  = np.stack(all_basis_ATTs, axis=0)   # [N, L, H, T, T]

        # ✅ 마지막 layer만 선택 → [N, H, T, T]
        shared_last = all_shared_ATTs[:, -1, :, :, :]
        basis_last  = all_basis_ATTs[:, -1, :, :, :]

        # ✅ 샘플 평균 → [H, T, T]
        shared_mean = shared_last.mean(axis=0)
        basis_mean  = basis_last.mean(axis=0)

        # 각 head별 heatmap
        for kind, attn_mean in zip(["Shared", "Basis"], [shared_mean, basis_mean]):
            fig, axes = plt.subplots(1, n_heads, figsize=(3.2*n_heads, 3.2))
            if n_heads == 1:
                axes = [axes]

            for h in range(n_heads):
                ax = axes[h]
                M = attn_mean[h]
                im = ax.imshow(M, cmap='viridis', vmin=M.min(), vmax=M.max())
                ax.set_title(f"{kind} Head {h} (last layer)", fontsize=9)
                ax.set_xticks(range(n_vars)); ax.set_yticks(range(n_vars))
                ax.set_xticklabels(["CLS"] + var_names, rotation=90, fontsize=6)
                ax.set_yticklabels(["CLS"] + var_names, fontsize=6)
                add_cb(ax, im)

            plt.suptitle(f"{kind} Attention (Last Layer, all-sample average)", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_dir / f"{kind}_Average_Heads_LastLayer.png", dpi=250, bbox_inches='tight')
            plt.close(fig)

    # -------- per-dataset 시각화(기존) --------
    def visualize_dataset(self, dataset_name: str, role: str, out_root: Path, max_samples=2):
        loader = self._make_loader(dataset_name)
        base = out_root / role / dataset_name
        ensure_dir(base)
        count = 0

        # === 전체 샘플 누적용 리스트 ===
        all_shared_ATTs = []   # [num_samples, num_layers, num_heads, T, T]
        all_basis_ATTs  = []   # [num_samples, num_layers, num_heads, T, T]
        feat_names_ref = None

        for batch in loader:
            Ms, ATTs, ADJs, var_names, Shared_ATTs, Shared_ADJs = self._forward_collect(batch)
            sample_dir = base / f"sample_{count}"
            ensure_dir(sample_dir)

            # --- per-sample 저장 (기존) ---
            grid_M = [Ms[l*self.num_heads + h] for l in range(self.num_layers) for h in range(self.num_heads)]
            self._grid_plot(
                grid_M, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Basis Attention (Var-Var)",
                sample_dir / "Basis_Attention_grid.png"
            )

            names_all = ["CLS"] + var_names
            grid_shared = []
            for l in range(len(Shared_ATTs)):
                att = Shared_ATTs[l]
                for h in range(self.num_heads):
                    grid_shared.append(att[h])
            self._grid_plot(
                grid_shared, names_all,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • SharedGAT Attention",
                sample_dir / "SharedGAT_Attn_grid.png"
            )

            # --- 누적 ---
            all_shared_ATTs.append(np.stack(Shared_ATTs, axis=0))  # [L,H,T,T]
            all_basis_ATTs.append(np.stack(ATTs, axis=0))          # [L,H,T,T]
            feat_names_ref = var_names

            count += 1
            if count >= max_samples:
                break

        # === 모든 샘플 평균 heatmap ===
        if count > 0:
            avg_dir = base / "average"
            self.visualize_average_heads(all_shared_ATTs, all_basis_ATTs, feat_names_ref, avg_dir)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True, type=str)
    ap.add_argument('--max_samples', type=int, default=10)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--alpha_summary_all', action='store_true',
                    help="전체 데이터셋에 대해 α 집계 요약도 생성")
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

    # 1) per-sample 시각화
    for s in sources:
        viz.visualize_dataset(s, role='source', out_root=out_root, max_samples=args.max_samples)
    viz.visualize_dataset(target, role='target', out_root=out_root, max_samples=args.max_samples)


    # for s in sources:
    #     viz.summarize_alpha_dataset(s, role='source', out_root=out_root, max_samples=None)
    # viz.summarize_alpha_dataset(target, role='target', out_root=out_root, max_samples=None)


if __name__ == "__main__":
    main()
