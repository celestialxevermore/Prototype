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
            nv_list.append(bd['cat_name_value_embeddings']); desc_list.append(bd['cat_desc_embeddings'])
        if all(k in bd for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            nv_list.append(bd['num_prompt_embeddings']);     desc_list.append(bd['num_desc_embeddings'])

        desc = torch.cat(desc_list, dim=1)   # [B,S,D]
        nv   = torch.cat(nv_list , dim=1)    # [B,S,D]
        B, S, D = nv.shape
        eps = 1e-8

        # ---- slot prior (bias_log, Q_slot) ----
        bias_log, Q_slot, slot_loss, DG, b = self.model.basis_affinity(desc, nv)   # Q_slot: [B,M,S,S]
        slotQ_np = Q_slot[0].detach().cpu().numpy()
        DG_np    = DG[0].detach().cpu().numpy()
        b_np     = b[0].detach().cpu().numpy()

        # ---- slot assignment S ----
        z = self.model.basis_affinity.fusion_mlp(torch.cat([desc, nv], dim=-1))
        S_logits = self.model.basis_affinity.slot_proj(z)
        S_probs  = torch.softmax(S_logits, dim=-1)           # [B,S,M]
        S_np     = S_probs[0].detach().cpu().numpy()

        # ---- Global slot graph G (numpy) ----
        if hasattr(self.model.basis_affinity, "export_G_numpy"):
            G_np = self.model.basis_affinity.export_G_numpy()
        else:
            G_tensor, _, _ = self.model.basis_affinity._make_G_and_regs(S_probs)
            G_np = G_tensor.detach().cpu().numpy()

        # ---- A_slot (pre-softmax) ----
        A_slot = torch.einsum("bnm,mkl,bjm->bmnj",
                              S_probs,
                              torch.tensor(G_np, device=self.device),
                              S_probs)
        A_slot_np = A_slot[0].detach().cpu().numpy()

        # ---- Build P0, DP ----
        basis_cls     = self.model.basis_cls.expand(B, 1, D)
        x_basis = torch.cat([basis_cls, nv], dim=1)                 # [B,S+1,D]
        new_adj = torch.zeros(B, S+1, S+1, device=nv.device)
        new_adj[:, 1:, 1:] = 1.0
        new_adj[:, 0, 1:]  = 1.0
        new_adj[:, 1:, 0]  = 0.0

        logits_base, _ = self.model.basis_layers[0]._build_base_logits(desc, x_basis, new_adj)
        NEG_INF     = -1e9
        struct_mask = (new_adj.unsqueeze(1) == 0).to(logits_base.dtype) * NEG_INF
        P0          = torch.softmax(logits_base + struct_mask, dim=-1)      # [B,H,T,T]
        P_var       = P0[:, :, 1:, 1:]                                      # [B,H,S,S]
        P_norm      = BasisSlotAffinityGAT.normalize_affinity(P_var, sym=True)
        DP          = BasisSlotAffinityGAT.affinity_to_distance(P_norm)

        a_deg = 0.5 * (P_norm.sum(dim=-1) + P_norm.sum(dim=-2))
        a     = a_deg / a_deg.sum(dim=-1, keepdim=True).clamp_min(eps)

        # ---- GW (no-grad; 분석용) ----
        Pi, gw_val = BasisSlotAffinityGAT._entropic_gw(
            DP, DG, a, b,
            eps=float(getattr(self.args, "gw_eps", 0.05)),
            outer_iters=int(getattr(self.args, "gw_outer_iters", 10)),
            sinkhorn_iters=int(getattr(self.args, "gw_sinkhorn_iters", 30)),
            tol=float(getattr(self.args, "gw_sinkhorn_eps", 1e-6)),
        )

        alpha    = BasisSlotAffinityGAT.alpha_from_gw(gw_val, sigma=0.6)
        Q_hat    = BasisSlotAffinityGAT.sharpen_Q(Q_slot, alpha)

        # numpy로 변환 (0번째 배치만 저장)
        Qhat_np  = Q_hat[0].detach().cpu().numpy()
        gw_np    = gw_val[0].detach().cpu().numpy()
        alpha_np = alpha[0].detach().cpu().numpy()

        # 전체 배치 결과 보존 (시각화/집계용)
        alpha_all_np = alpha.detach().cpu().numpy()   # [B,H,M]
        gw_all_np    = gw_val.detach().cpu().numpy()  # [B,H,M]

        # ---- Basis / Shared attention (기존 유지) ----
        Ms, ATTs, ADJs = [], [], []
        x_basis = torch.cat([basis_cls, nv], dim=1)
        mask_M  = bias_log.exp().clamp(1e-6, 1.0 - 1.0e-6)   # [B,M,S,S]

        for l in range(self.num_layers):
            norm_x = self.model.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.model.basis_layers[l](
                desc, norm_x, prior_Q=mask_M, DG=DG, b=b
            )
            new_adj_l = self.model.basis_layers[l].new_adjacency  # [B,T,T]
            x_basis   = x_basis + basis_outputs.reshape(B, S+1, D)

            varvar = att[0, :, 1:, 1:].detach().cpu().numpy()    # [H,S,S]
            for h in range(self.num_heads):
                Ms.append(varvar[h])

            ATTs.append(att[0].cpu().numpy())
            ADJs.append(new_adj_l[0].cpu().numpy())

        Shared_ATTs, Shared_ADJs = [], []
        shared_cls     = self.model.shared_cls.expand(B, 1, D)
        x_shared = torch.cat([shared_cls, nv], dim=1)
        for l in range(self.model.num_shared_layers):
            norm_x = self.model.shared_layer_norms[l](x_shared)
            out, att = self.model.shared_layers[l](desc, norm_x)
            x_shared = x_shared + out
            Shared_ATTs.append(att[0].cpu().numpy())
            Shared_ADJs.append(self.model.shared_layers[l].new_adjacency[0].cpu().numpy())

        feat_names = self.model.extract_feature_names(bd)

        return (
            Ms, ATTs, ADJs, feat_names,
            slotQ_np, Qhat_np, G_np, S_np,
            Shared_ATTs, Shared_ADJs, A_slot_np,
            DG_np, b_np, alpha_np, gw_np,
            alpha_all_np, gw_all_np
        )

    # ========== NEW: α를 데이터셋 전체로 집계/시각화 ==========
    def _alpha_batch_stats(self, alpha_bhm: np.ndarray):
        """
        alpha_bhm: [H,M] (numpy) — 배치 1개 분량
        return:
          dict with entropy(H), top1, top2, top3, top_idx, mean_alpha(H,M)
        """
        eps = 1e-8
        a = alpha_bhm
        H, M = a.shape
        ent = -(a.clip(eps, 1.0) * np.log(a.clip(eps, 1.0))).sum(-1)        # [H]
        top1 = np.partition(a, -1, axis=1)[:, -1]
        top2 = np.partition(a, -2, axis=1)[:, -2] + top1
        top3 = (np.partition(a, -3, axis=1)[:, -3] + top2) if M >= 3 else top2
        top_idx = a.argmax(axis=1)                                          # [H]
        return {
            "entropy": ent, "top1": top1, "top2": top2, "top3": top3,
            "top_idx": top_idx, "alpha": a
        }

    def summarize_alpha_dataset(self, dataset_name: str, role: str, out_root: Path, max_samples=None):
        """
        데이터셋 전체를 훑어서 α 통계(time-series & 집계)를 저장
        """
        loader = self._make_loader(dataset_name)
        base = out_root / role / dataset_name / "_alpha_summary"
        ensure_dir(base)

        H, M = self.num_heads, self.num_slots
        ent_list, t1_list = [], []
        cov_counts = np.zeros((H, M), dtype=np.int64)   # 헤드별 슬롯 coverage
        alpha_sum  = np.zeros((H, M), dtype=np.float64)
        n_seen = 0

        for bi, batch in enumerate(loader):
            if (max_samples is not None) and (n_seen >= max_samples):
                break
            *_, _, _, _, _, _, _, _, _, _, _, _, _, _, alpha_all_np, _ = self._forward_collect(batch)
            a = alpha_all_np[0]  # [H,M]
            st = self._alpha_batch_stats(a)

            ent_list.append(st["entropy"][None, :])   # [1,H]
            t1_list.append(st["top1"][None, :])       # [1,H]
            for h in range(H):
                cov_counts[h, st["top_idx"][h]] += 1
            alpha_sum += st["alpha"]
            n_seen += 1

        if n_seen == 0:
            print(f"[WARN] No samples found for {role}/{dataset_name}")
            return

        ent_seq = np.concatenate(ent_list, axis=0)     # [T,H]
        t1_seq  = np.concatenate(t1_list, axis=0)      # [T,H]
        alpha_mean = alpha_sum / max(n_seen, 1)

        # --- 저장: 원시 값
        np.savez(base / "alpha_dump.npz",
                 entropy_seq=ent_seq, top1_seq=t1_seq,
                 coverage=cov_counts, alpha_mean=alpha_mean,
                 H=H, M=M, n_seen=n_seen)

        # --- 요약 CSV
        import csv
        with open(base / "alpha_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["head", "entropy_mean", "entropy_std",
                        "top1_mean", "top1_std"])
            for h in range(H):
                w.writerow([
                    h,
                    float(ent_seq[:, h].mean()), float(ent_seq[:, h].std()),
                    float(t1_seq[:, h].mean()),  float(t1_seq[:, h].std())
                ])
        # --- 플롯 3) Mean α heatmap (H×M)
        plt.figure(figsize=(M * 0.6 + 2, H * 0.6 + 1.5))
        im = plt.imshow(alpha_mean, aspect="auto", cmap="viridis",
                        vmin=alpha_mean.min(), vmax=alpha_mean.max())
        plt.colorbar(im)
        plt.xticks(range(M), [f"m{j}" for j in range(M)], rotation=90, fontsize=7)
        plt.yticks(range(H), [f"H{h}" for h in range(H)], fontsize=7)
        plt.title(f"Mean alpha (H×M) • {role}:{dataset_name}")
        plt.tight_layout(); plt.savefig(base / "alpha_mean_heatmap.png", dpi=250); plt.close()

        # --- 플롯 4) Coverage bar (헤드별)
        fig, axes = plt.subplots(1, H, figsize=(max(3, H) * 2.2, 3), sharey=True)
        if H == 1: axes = [axes]
        for h in range(H):
            ax = axes[h]
            ax.bar(np.arange(M), cov_counts[h])
            ax.set_title(f"H{h} coverage(top-1)")
            ax.set_xticks(range(M)); ax.set_xticklabels([f"m{j}" for j in range(M)], rotation=90, fontsize=7)
        fig.suptitle(f"Slot coverage per head • {role}:{dataset_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(base / "alpha_coverage_per_head.png", dpi=250); plt.close()

        print(f"[ALPHA] saved dataset-level α summary → {base}")

    # ---------------- (이하 기존 per-sample 플롯 함수들 그대로) ----------------
    def plot_Q_heatmap(self, Q_np, var_names, save_path):
        L, H = self.num_layers, self.num_heads
        n = Q_np.shape[-1]
        fig, axes = plt.subplots(1, H, figsize=(H*3, 3))
        if H == 1: axes = [axes]
        for h in range(H):
            ax = axes[h]
            im = ax.imshow(Q_np[h], cmap='viridis', vmin=0, vmax=Q_np[h].max())
            ax.set_title(f"Q Slot Prior - H{h}", fontsize=9)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            ax.set_yticklabels(var_names, fontsize=6)
            add_cb(ax, im)
        plt.suptitle("Slot Prior Q", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    def plot_Qhat_heatmap(self, Qhat_np, var_names, save_path):
        H, n, _ = Qhat_np.shape
        fig, axes = plt.subplots(1, H, figsize=(H * 3, 3))
        if H == 1: axes = [axes]
        vmin, vmax = Qhat_np.min(), Qhat_np.max()
        for h in range(H):
            ax = axes[h]
            im = ax.imshow(Qhat_np[h], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(f"Q Sharpened - H{h}", fontsize=9)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            ax.set_yticklabels(var_names, fontsize=6)
            add_cb(ax, im)
        plt.suptitle("Slot Sharpened Q (global scale)", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    def plot_U_per_slot(self, save_path):
        if not hasattr(self.model.basis_affinity, "U_param"):
            print("[WARN] Model has no U_param."); return
        U = torch.nn.functional.softplus(self.model.basis_affinity.U_param).detach().cpu().numpy()
        M, K, R = U.shape
        import seaborn as sns
        fig, axes = plt.subplots(1, M, figsize=(M*3, 3), sharey=True)
        if M == 1: axes = [axes]
        for m in range(M):
            ax = axes[m]; vals = U[m].ravel()
            ax.hist(vals, bins=20, density=True, color="steelblue", alpha=0.5, label="Histogram (density)")
            sns.kdeplot(vals, ax=ax, color="red", linewidth=1.5, label="KDE")
            ax.set_title(f"Slot {m}", fontsize=9); ax.set_xlabel("U values")
            if m == 0: ax.set_ylabel("Probability Density")
            ax.legend(fontsize=6)
        plt.suptitle("U_param distribution per Slot (PDF)", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

    def plot_DG_slotwise(self, DG_np, save_path_prefix):
        M, K, _ = DG_np.shape
        cols = int(np.ceil(np.sqrt(M))); rows = int(np.ceil(M / cols))
        vmin, vmax = DG_np.min(), DG_np.max()
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        for m in range(M):
            ax = axes[m]
            im = ax.imshow(DG_np[m], cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"DG (slot m={m})", fontsize=9)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels([f"s{i}" for i in range(K)], rotation=90, fontsize=6)
            ax.set_yticklabels([f"s{i}" for i in range(K)], fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        for j in range(M, len(axes)): axes[j].axis("off")
        plt.suptitle("Cosine Slot Cost (DG)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{save_path_prefix}", dpi=250, bbox_inches="tight"); plt.close(fig)

    def plot_b_bars(self, b_np, save_path):
        M, K = b_np.shape
        fig, axes = plt.subplots(1, M, figsize=(M*3.2, 3.2))
        if M == 1: axes = [axes]
        for m in range(M):
            ax = axes[m]
            ax.bar(np.arange(K), b_np[m])
            ax.set_title(f"b (slot m={m})", fontsize=9)
            ax.set_xticks(range(K)); ax.set_xticklabels([f"s{j}" for j in range(K)], rotation=90, fontsize=6)
            ax.set_ylim(0, max(1.0, float(b_np[m].max())*1.1))
        plt.suptitle("GW marginals b per slot m", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

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

    def _grid_plot_slotwise(self, mats, names, title, save_path):
        L, M = self.num_layers, len(mats)
        n = mats[0].shape[0]
        fig, axes = plt.subplots(L, M, figsize=(M*3.2, L*3.2))
        if L == 1: axes = np.expand_dims(axes, 0)
        if M == 1: axes = np.expand_dims(axes, 1)
        # 공통 vmin/vmax
        all_vals = []
        for M_ in mats:
            mask = ~np.eye(n, dtype=bool)
            vals = M_[mask]; vals = vals[np.isfinite(vals)]
            all_vals.extend(vals)
        if all_vals:
            q05, q95 = np.quantile(all_vals, 0.05), np.quantile(all_vals, 0.95)
            vmin, vmax = (float(q05), float(q95)) if q05 != q95 else (float(np.min(all_vals)), float(np.max(all_vals)))
        else:
            vmin, vmax = 0, 1
        for l in range(L):
            for m in range(M):
                ax = axes[l, m]; Mmat = mats[m]
                im = ax.imshow(Mmat, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"L{l} · Slot{m}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95]); ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

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

    def plot_SGS_slotwise(self, A_slot_np, Q_np, Qhat_np, var_names, save_path):
        M, N, _ = A_slot_np.shape
        fig, axes = plt.subplots(M, 3, figsize=(9, 3 * M))
        if M == 1: axes = np.expand_dims(axes, 0)
        for m in range(M):
            mats = [A_slot_np[m], Q_np[m], Qhat_np[m]]
            titles = ["A (pre-softmax)", "Q (softmax)", "Q̂ (sharpened)"]
            for j in range(3):
                ax = axes[m, j]
                im = ax.imshow(mats[j], cmap='viridis', interpolation='nearest')
                ax.set_title(f"Slot {m} - {titles[j]}", fontsize=9)
                ax.set_xticks(range(N)); ax.set_yticks(range(N))
                ax.set_xticklabels(var_names, rotation=90, fontsize=6)
                ax.set_yticklabels(var_names, fontsize=6)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        plt.suptitle("Slot Graph Sharpening (SGS) Results", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

    def visualize_gw_val(self, gw_np, save_path=None, title="GW Values"):
        if gw_np.ndim == 2:
            H, M = gw_np.shape; B = 1; gw_np = gw_np[np.newaxis, :, :]
        elif gw_np.ndim == 3:
            B, H, M = gw_np.shape
        else:
            raise ValueError(f"Invalid gw_np shape {gw_np.shape}, expected [H,M] or [B,H,M]")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.hist(gw_np.flatten(), bins=30, color="skyblue", edgecolor="black")
        plt.xlabel("GW value"); plt.ylabel("Frequency"); plt.title(f"{title} - Histogram")
        plt.subplot(1, 2, 2); plt.imshow(gw_np.reshape(B*H, M), aspect="auto", cmap="viridis")
        plt.colorbar(label="GW value"); plt.xlabel("Slot index (M)"); plt.ylabel("Batch×Head index")
        plt.title(f"{title} - Heatmap")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300); print(f"Saved GW visualization to {save_path}")
        else: plt.show()

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

    def visualize_alpha_all_heatmap(self, alpha_all_np: np.ndarray, save_path: Path, sample_idx:int=0):
        H, M = alpha_all_np.shape[1], alpha_all_np.shape[2]
        plt.figure(figsize=(M * 0.6, H * 0.6))
        vmin, vmax = alpha_all_np[sample_idx].min(), alpha_all_np[sample_idx].max()
        im = plt.imshow(alpha_all_np[sample_idx], cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.xticks(range(M), [f"m{j}" for j in range(M)], rotation=90, fontsize=6)
        plt.yticks(range(H), [f"H{h}" for h in range(H)], fontsize=6)
        plt.title(f"Alpha distribution (sample {sample_idx})", fontsize=12)
        plt.tight_layout(); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close()
    
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

    def visualize_gw_all_heatmap(self, gw_all_np: np.ndarray, save_path: Path, sample_idx:int=0):
        H, M = gw_all_np.shape[1], gw_all_np.shape[2]
        plt.figure(figsize=(M * 0.6, H * 0.6))
        im = plt.imshow(gw_all_np[sample_idx], cmap="plasma", aspect="auto")
        plt.colorbar(im)
        plt.xticks(range(M), [f"m{j}" for j in range(M)], rotation=90, fontsize=6)
        plt.yticks(range(H), [f"H{h}" for h in range(H)], fontsize=6)
        plt.title(f"GW values (sample {sample_idx})", fontsize=12)
        plt.tight_layout(); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close()

    def plot_Q_raw_heatmap(self, Q_np, var_names, save_path):
        H, n, _ = Q_np.shape
        fig, axes = plt.subplots(1, H, figsize=(H * 3, 3))
        if H == 1: axes = [axes]
        for h in range(H):
            ax = axes[h]
            im = ax.imshow(Q_np[h], cmap="magma", vmin=Q_np[h].min(), vmax=Q_np[h].max())
            ax.set_title(f"Q Raw - H{h}", fontsize=9)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(var_names, rotation=90, fontsize=6)
            ax.set_yticklabels(var_names, fontsize=6)
            add_cb(ax, im)
        plt.suptitle("Slot Prior Q (raw, pre-softmax)", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

    def plot_Q_comparison(self, Q_raw, Q_sharp, var_names, save_path):
        H, n, _ = Q_raw.shape
        fig, axes = plt.subplots(2, H, figsize=(H*3, 6))
        if H == 1: axes = np.expand_dims(axes, 1)
        for h in range(H):
            ax1 = axes[0, h]
            im1 = ax1.imshow(Q_raw[h], cmap="magma")
            ax1.set_title(f"Raw Q - H{h}", fontsize=8)
            ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
            ax1.set_xticklabels(var_names, rotation=90, fontsize=5)
            ax1.set_yticklabels(var_names, fontsize=5)
            add_cb(ax1, im1)
            ax2 = axes[1, h]
            im2 = ax2.imshow(Q_sharp[h], cmap="viridis", vmin=Q_sharp.min(), vmax=Q_sharp.max())
            ax2.set_title(f"Sharpened Q - H{h}", fontsize=8)
            ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
            ax2.set_xticklabels(var_names, rotation=90, fontsize=5)
            ax2.set_yticklabels(var_names, fontsize=5)
            add_cb(ax2, im2)
        plt.suptitle("Q Raw vs Sharpened Comparison", fontsize=13)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)

    # -------- per-dataset 시각화(기존) --------
    def visualize_dataset(self, dataset_name: str, role: str, out_root: Path, max_samples=2):
        loader = self._make_loader(dataset_name)
        base = out_root / role / dataset_name
        ensure_dir(base)
        count = 0

        for batch in loader:
            (Ms, ATTs, ADJs, var_names,
             slotQ_np, Qhat_np, G_np, S_np,
             Shared_ATTs, Shared_ADJs, A_slot_np,
             DG_np, b_np, alpha_np, gw_np,
             alpha_all_np, gw_all_np) = self._forward_collect(batch)

            sample_dir = base / f"sample_{count}"
            ensure_dir(sample_dir)

            # (1) Var-Var Attention (P) — 레이어×헤드
            grid_M = [Ms[l*self.num_heads + h] for l in range(self.num_layers) for h in range(self.num_heads)]
            self._grid_plot(grid_M, var_names,
                            f"{role.capitalize()} • {dataset_name} • Sample {count} • Var-Var Attention (P)",
                            sample_dir / "VarVar_P_grid.png")

            # (2) Attention × Adjacency — 레이어×헤드
            names_all = ["CLS"] + var_names
            grid_AX = []
            for l in range(self.num_layers):
                att = ATTs[l]  # [H,T,T]
                for h in range(self.num_heads):
                    grid_AX.append(att[h])
            self._grid_plot(grid_AX, names_all,
                            f"{role.capitalize()} • {dataset_name} • Sample {count} • Attention × Adjacency",
                            sample_dir / "AttnXAdj_grid.png")

            # (3) Slotwise A/Q/Q̂
            grid_A = [A_slot_np[m] for m in range(A_slot_np.shape[0])]
            self._grid_plot_slotwise(grid_A, var_names,
                                     f"{role.capitalize()} • {dataset_name} • Sample {count} • A_slot (slotwise)",
                                     sample_dir / "ASlot_slotwise_grid.png")
            self.plot_Q_raw_heatmap(slotQ_np, var_names, sample_dir / "SlotPrior_Q_raw.png")
            self.plot_Q_comparison(slotQ_np, Qhat_np, var_names, sample_dir / "Q_raw_vs_sharpened.png")
            self.plot_Q_heatmap(slotQ_np, var_names, sample_dir / "SlotPrior_Q.png")
            self.plot_Qhat_heatmap(Qhat_np, var_names, sample_dir / "SlotSharpened_Q.png")

            # (4) Global G
            if G_np is not None:
                K = G_np.shape[-1]
                slot_names = [f"z{j}" for j in range(K)]
                grid_G = [G_np[h] for h in range(self.num_heads)] * self.num_layers
                self._grid_plot(grid_G, slot_names,
                                f"{role.capitalize()} • {dataset_name} • Sample {count} • Global Slot Graph G",
                                sample_dir / "GlobalSlotGraph_G.png")
                uniform_K = 1.0 / max(K, 1)
                self._grid_plot_centered(grid_G, slot_names,
                                         f"{role.capitalize()} • {dataset_name} • Sample {count} • Global Slot Graph G (center=1/K)",
                                         sample_dir / "GlobalSlotGraph_center_uniform.png",
                                         center=uniform_K, cmap='RdBu_r')

            self.plot_DG_slotwise(DG_np, sample_dir / "SlotSpace_DG.png")
            self.plot_b_bars(b_np, sample_dir / "SlotMarginal_b.png")
            self.plot_U_per_slot(sample_dir / "Uparam_distribution.png")
            self.visualize_gw_val(gw_np, sample_dir / "GW.png")

            # α / GW heatmap
            self.visualize_alpha_all_heatmap(alpha_all_np, sample_dir / f"Alpha_sample{count}.png", sample_idx=0)
            self.visualize_gw_all_heatmap(gw_all_np, sample_dir / f"GW_sample{count}.png", sample_idx=0)

            # SharedGAT
            names_all = ["CLS"] + var_names
            grid_shared = []
            for l in range(len(Shared_ATTs)):
                att = Shared_ATTs[l]
                for h in range(self.num_heads):
                    grid_shared.append(att[h])
            self._grid_plot(grid_shared, names_all,
                            f"{role.capitalize()} • {dataset_name} • Sample {count} • SharedGAT Attention",
                            sample_dir / "SharedGAT_Attn_grid.png")

            count += 1
            if count >= max_samples:
                break


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


    for s in sources:
        viz.summarize_alpha_dataset(s, role='source', out_root=out_root, max_samples=None)
    viz.summarize_alpha_dataset(target, role='target', out_root=out_root, max_samples=None)


if __name__ == "__main__":
    main()
