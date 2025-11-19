import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd 
from sklearn.decomposition import PCA
import seaborn as sns
# UMAP import
try:
    import umap
    if not hasattr(umap, "UMAP"):
        import umap.umap_ as umap
except ImportError:
    umap = None

# 프로젝트 경로
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from utils.util import fix_seed
from dataset.data_dataloaders import prepare_embedding_dataloaders
from models.TabularFLM_S import Model
# FGWUtils와 LCG 임포트 (argmin 계산용)
from models.LCG import FGWUtils, LatentCompositeGraph


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class LCGVisualizer:
    # ... (init, _make_loader, _forward_collect는 이전과 동일) ...
    def __init__(self, checkpoint_path: str, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 📂 Loading checkpoint: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        args = ckpt["args"]
        fix_seed(getattr(args, "random_seed", 42))

        # 모델 생성 (latent_graph 포함)
        self.model = Model(
            args, args.input_dim, args.hidden_dim, args.output_dim,
            args.dropout_rate, args.llm_model, 
            experiment_id="LCG_VIZ", mode="Full"
        ).to(self.device)
        
        # ✅ 체크포인트 로드
        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"⚠️ Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected[:5]}...")
        
        self.model.eval()
        self.args = args
        
        # ✅ latent_graph 참조
        self.lcg = self.model.latent_graph
        self.M, self.K, self.D = self.lcg.M, self.lcg.K, self.lcg.D
        
        # ✅ 진단: 학습된 가중치 확인 (동일)
        emb = self.lcg.node_embeddings.detach().cpu()
        print(f"\n🔍 LCG 가중치 진단:")
        print(f"  - Shape: {emb.shape} (M={self.M}, K={self.K}, D={self.D})")
        print(f"  - node_embeddings[0,0,:5]: {emb[0,0,:5].numpy()}")
        print(f"  - Min/Max: {emb.min():.4f} / {emb.max():.4f}")
        print(f"  - Mean/Std: {emb.mean():.4f} / {emb.std():.4f}")
        expected_std = np.sqrt(2.0 / (self.K + self.D))
        if abs(emb.std().item() - expected_std) < 0.05:
            print(f"  ⚠️ WARNING: 가중치가 랜덤 초기화 상태일 수 있음!")
            print(f"      (현재 std={emb.std():.4f}, Xavier 예상={expected_std:.4f})")
        else:
            print(f"  ✅ 학습된 가중치로 판단됨")

    def _make_loader(self, dataset_name: str, batch_size=32):
        args2 = self.args
        args2.source_data = dataset_name
        fix_seed(args2.random_seed)
        res = prepare_embedding_dataloaders(args2, dataset_name)
        tr, va, te = res["loaders"]
        from torch.utils.data import ConcatDataset, DataLoader
        ds = ConcatDataset([tr.dataset, va.dataset, te.dataset])
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    @torch.no_grad()
    def _forward_collect(self, batch):
        """
        [수정]
        샘플 그래프의 대표 임베딩으로 'Fx.mean()' 대신
        'x_basis'의 [CLS] 토큰 ([:, 0, :])을 사용합니다.
        """
        bd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
        
        # 1. predict 함수를 실행하여
        #    'self.x_basis'를 포함한 모든 내부 상태를 계산
        _ = self.model.predict(bd) 

        if not hasattr(self.model, "x_basis"):
            print("ERROR: 'self.model'에 'x_basis' 속성이 없습니다.")
            print("     (model.predict()가 self.x_basis = x_basis를 저장하는지 확인하세요.)")
            D_full = self.args.input_dim
            return np.zeros((bd['y'].shape[0], 1, D_full)) # [B, 1, D]

        # 2. [CLS] 토큰 임베딩을 가져옵니다
        # x_basis shape: [B, N+1, D_full]
        x_basis = self.model.x_basis
        
        # [CLS] 토큰은 [:, 0, :]에 있습니다.
        # z_graph_cls shape: [B, D_full]
        z_graph_cls = x_basis[:, 0, :]
        
        # 3. UMAP 플로터와 차원을 맞추기 위해 [B, 1, D_full]로 변경
        z_graph = z_graph_cls.unsqueeze(1) 
        
        return z_graph.detach().cpu().numpy() # [B, 1, D_full]

    @torch.no_grad()
    def _get_assignments_and_plans(self, batch):
        """
        배치를 입력받아, 'argmin' 인덱스 [B, H]와
        전체 Transport Plan Pi [B, H, M, N, K]를 반환합니다.
        """
        bd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
        
        # --- [ 1. 모델에서 Fx, Dx, Fy, Dy 추출 ] ---
        _ = self.model.predict(bd) 

        basis_outputs = self.model.basis_outputs_for_viz
        Fx = basis_outputs[:, 1:, :, :].permute(0, 2, 1, 3)
        B, H, N, D_head = Fx.shape
        
        P_affinity = self.model._last_P_basis 
        Dx = LatentCompositeGraph.normalize_affinity(P_affinity)
        Dx = LatentCompositeGraph.affinity_to_distance(Dx) 
        
        Fy, Dy_affinity, _ = self.lcg() 
        Dy = LatentCompositeGraph.affinity_to_distance(Dy_affinity)

        a = torch.ones(B, H, N, device = Fx.device) / N 
        b = torch.ones(B, self.M, self.K, device=Fx.device) / self.K 

        # --- [ 2. ❗️❗️ 여기가 수정된 지점 ❗️❗️ ] ---
        # 로드된 모델의 GraphQuantizer에서 실제 학습에 사용된 파라미터를 가져옵니다.
        quantizer = self.model.graph_quantizer

        Pi_all , fgw_values = FGWUtils.assign_FGW(
            Fx, Fy, Dx, Dy, a, b, # Dx, Dy 모두 거리
            
            # (수정) quantizer.alpha (e.g., 0.9) 사용
            alpha = quantizer.alpha, 
            
            # (수정) quantizer.eps_assign (e.g., 5.0) 사용
            eps = quantizer.eps_assign, 
            
            # (수정) quantizer에서 iter 값 읽어오기
            outer_iters = quantizer.outer_iters, 
            sinkhorn_iters = quantizer.sinkhorn_iters,
        ) 
        # --- [ 수정 끝 ] ---
        
        assign_idx = torch.argmin(fgw_values, dim=-1) # [B, H]
        
        return assign_idx, Pi_all
    def visualize_transport_plans(self, sources: list, out_root: Path, num_samples=3):
        """
        [수정된 버전]
        입력받은 '모든' 소스(sources: list)에 대해
        Transport Plan (Pi)을 샘플별/헤드별로 시각화합니다.
        """
        print(f"[INFO] 🔄 Visualizing Transport Plans for {sources}...")
        
        # 1. 최상위 저장 폴더 생성
        base_dir = out_root / "transport_plans"
        ensure_dir(base_dir) # (ensure_dir가 lcg.py 어딘가에 정의되어 있어야 함)

        for source_dataset in sources:
            print(f"\n--- Processing Source: {source_dataset} ---")
            
            # 2. 소스별 데이터 로더 생성
            loader = self._make_loader(source_dataset, batch_size=32)
            try:
                batch = next(iter(loader))
            except StopIteration:
                print(f"⚠️ [VIZ] No data in loader for {source_dataset}. Skipping.")
                continue # 다음 소스로 넘어감

            B = batch['y'].shape[0]
            if B == 0:
                print(f"⚠️ [VIZ] Batch has size 0. Skipping.")
                continue
                
            # 3. Pi와 assign_idx 계산 (이 소스에 대해)
            assign_idx, Pi_all = self._get_assignments_and_plans(batch)
            assign_idx_np = assign_idx.cpu().numpy()
            Pi_all_np = Pi_all.cpu().numpy()
            
            H = Pi_all_np.shape[1] # Head 수
            N = Pi_all_np.shape[3] # Source 노드 수 (N)
            K = Pi_all_np.shape[4] # LCG 노드 수 (K)
            
            # 4. 소스별 하위 폴더 생성
            source_specific_dir = base_dir / source_dataset
            ensure_dir(source_specific_dir)

            num_to_viz = min(num_samples, B)
            print(f"[VIZ] 🎨 Plotting {num_to_viz} samples in 1x{H} grids for {source_dataset}...")

            for b in range(num_to_viz):
                fig, axes = plt.subplots(1, H, figsize=(H * 4.5, 4.5), squeeze=False)
                axes = axes.flatten()
                
                vmax = Pi_all_np[b].max() 
                if vmax < 1e-6: vmax = 1.0 
                
                for h in range(H):
                    ax = axes[h]
                    m = assign_idx_np[b, h]
                    pi_sample = Pi_all_np[b, h, m, :, :] # [N, K]
                    
                    sns.heatmap(
                        pi_sample, 
                        ax=ax, 
                        cmap="viridis",
                        cbar=True,
                        vmin=0.0,
                        vmax=vmax,
                        annot=True, fmt=".2f", # 👈 숫자를 소수점 둘째자리까지 표시
                        annot_kws={"size": 6}, # 👈 숫자 폰트 크기
                        cbar_kws={'shrink': 0.75}
                    )
                    
                    ax.set_title(f"Head {h} -> LCG {m}", fontsize=10)
                    ax.set_xlabel(f"LCG Nodes (K={K})", fontsize=8)
                
                axes[0].set_ylabel(f"Source Nodes (N={N})", fontsize=8)
                
                sample_id = batch.get('id', [None]*B)[b]
                title_id = f"ID: {sample_id}" if sample_id is not None else f"Index: {b}"
                
                fig.suptitle(f"Transport Plans (Pi) for Sample ({title_id})\n[Source: {source_dataset}]", fontsize=14, y=1.07)
                plt.tight_layout()
                
                save_path = source_specific_dir / f"plan_sample_{sample_id or b}.png"
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            print(f"[VIZ] ✅ Saved plans for {source_dataset} to: {source_specific_dir}")
    
    def visualize_lcg_only(self, out_root: Path, method="pca"):
        """
        [수정]
        LCG "내부" K개 노드의 분포를 "개별적"으로 PCA 시각화합니다.
        (8개 플롯의 축 스케일을 "통일"하여 분산(흩어짐)을 비교)
        """
        ensure_dir(out_root / "lcg_analysis")
        
        # ✅ 학습된 node_embeddings 가져오기
        lcg_nodes_all = self.lcg.node_embeddings.detach().cpu().numpy()  # [M, K, D]
        M, K, D = lcg_nodes_all.shape
        
        # 1. M=8번 "개별적"으로 PCA 실행 및 좌표 수집
        all_X_2d = []
        for m in range(M):
            X = lcg_nodes_all[m] # [K, D]
            
            # (데이터가 0으로 붕괴했을 경우 PCA 에러 방지)
            if np.std(X) < 1e-6:
                X_2d = np.zeros((K, 2)) # 0으로 채움
            else:
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X) # [K, 2]
            
            all_X_2d.append(X_2d)

        # 2. 8개 PCA 결과의 "글로벌" 스케일(min/max) 찾기
        all_X_2d_np = np.concatenate(all_X_2d) # [M*K, 2]
        x_min, x_max = all_X_2d_np[:, 0].min(), all_X_2d_np[:, 0].max()
        y_min, y_max = all_X_2d_np[:, 1].min(), all_X_2d_np[:, 1].max()
        
        # (패딩 추가)
        padding_x = (x_max - x_min) * 0.1 if (x_max - x_min) > 1e-6 else 0.1
        padding_y = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-6 else 0.1

        # 3. 2x4 그리드 플롯 준비
        ncols = min(M, 4)
        nrows = int(np.ceil(M / ncols))
        fig, axes = plt.subplots(nrows, ncols, 
                                 figsize=(ncols * 4, nrows * 3.5), 
                                 squeeze=False)
        cmap = plt.colormaps.get_cmap("tab10")

        # 4. 동일한 스케일로 8개 플롯 그리기
        for m in range(M):
            ax = axes[m // ncols, m % ncols]
            color = cmap(m % 10)
            X_2d = all_X_2d[m] # [K, 2]
            
            # (1) 8개 노드(점) 그리기
            ax.scatter(X_2d[:, 0], X_2d[:, 1],
                        color=color, s=100, alpha=0.8, 
                        edgecolor='black', linewidth=0.5)
            
            # (2) 중심점
            center = X_2d.mean(axis=0)
            ax.scatter(center[0], center[1], 
                        marker="X", color="black", s=100,
                        edgecolor='white', linewidth=1.5, alpha=0.7)
            
            # (3) ❗️ 스케일 통일 ❗️
            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)
            
            ax.set_title(f"LCG_{m} Internal Node PCA")
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
            ax.grid(alpha=0.2)
        
        # 남는 빈 subplot 숨기기
        for m in range(M, nrows * ncols):
            axes[m // ncols, m % ncols].axis('off')
        
        fig.suptitle(f"Trained LCG Internal Node Distributions (PCA - Unified Scale)", fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = out_root / "lcg_analysis" / f"lcg_only_nodes_pca.png" # 새 이름
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")
        # --- [ 수정 끝 ] ---
        
        # (LCG "그룹" 간의 거리 분석은 UMAP/TSNE와 무관하므로 그대로 둠)
        self._analyze_lcg_distances(lcg_nodes_all, out_root)

    def _analyze_lcg_distances(self, lcg_nodes, out_root):
        """LCG 간 거리 분석"""
        M, K, D = lcg_nodes.shape
        centers = lcg_nodes.mean(axis=1)
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(centers, metric='euclidean'))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        im = ax1.imshow(dist_matrix, cmap='viridis', aspect='auto')
        ax1.set_title("LCG Center-to-Center Distances", fontsize=12)
        ax1.set_xlabel("LCG Index"); ax1.set_ylabel("LCG Index")
        ax1.set_xticks(range(M)); ax1.set_yticks(range(M))
        plt.colorbar(im, ax=ax1, label="Euclidean Distance")
        
        upper_tri = dist_matrix[np.triu_indices(M, k=1)]
        ax2.hist(upper_tri, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(upper_tri.mean(), color='red', linestyle='--', 
                    label=f'Mean: {upper_tri.mean():.3f}')
        ax2.axvline(upper_tri.min(), color='blue', linestyle='--',
                    label=f'Min: {upper_tri.min():.3f}')
        ax2.set_title("Distribution of Inter-LCG Distances", fontsize=12)
        ax2.set_xlabel("Distance"); ax2.set_ylabel("Frequency")
        ax2.legend()
        
        plt.tight_layout()
        save_path = out_root / "lcg_analysis" / "lcg_distance_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")
        
        print(f"\n📊 LCG 거리 통계:")
        print(f"  - 평균 거리: {upper_tri.mean():.4f}")
        print(f"  - 최소 거리: {upper_tri.min():.4f}")
        print(f"  - 최대 거리: {upper_tri.max():.4f}")
        print(f"  - 표준편차: {upper_tri.std():.4f}")
        
        if upper_tri.min() < 0.1:
            print(f"  ⚠️ WARNING: 일부 LCG가 너무 가까움 (min={upper_tri.min():.4f})")
        if upper_tri.std() < 0.5:
            print(f"  ⚠️ WARNING: LCG 분포가 고르지 않음 (std={upper_tri.std():.4f})")

    def visualize_sources_and_lcg(self, sources, out_root: Path, 
                                      max_samples=200, method="umap"):
        """
        [수정]
        Multi-source 데이터 (Head별) + LCG "K개 노드"를 1xH 그리드로 시각화합니다.
        LCG 노드에 'm-k' ID를 표시하고, 시각적 요소를 개선합니다.
        """
        new_dir = out_root / "joint_space_per_head"
        ensure_dir(new_dir)
        
        all_embeds, all_labels = [], []
        print("[INFO] 🔄 Collecting embeddings...")
        
        # 1. 샘플(Fx) 임베딩 수집 ([CLS] 토큰)
        for s in sources:
            loader = self._make_loader(s, batch_size=32)
            temp_embeds = []
            for i, batch in enumerate(loader):
                z_np = self._forward_collect(batch) # [B, 1, D_full]
                temp_embeds.append(z_np)
                if (i * loader.batch_size) >= max_samples:
                    break
            if not temp_embeds: continue
            s_embeds = np.concatenate(temp_embeds, axis=0) 
            all_embeds.append(s_embeds)
            all_labels.extend([s] * len(s_embeds))
        
        if not all_embeds:
            print("[VIZ] ⚠️ No embeddings collected. Skipping UMAP.")
            return

        X_samples = np.concatenate(all_embeds, axis=0) # [Num_Samples, 1, D_full]
        if X_samples.ndim == 3: 
            X_samples = X_samples.reshape(-1, X_samples.shape[-1]) # [Num_Samples, D_full]

        # --- [ D_head 공통 공간 UMAP 로직 (동일) ] ---
        
        # 2. LCG (Fy) "K개 노드" 임베딩 수집 (D_head 차원)
        lcg_nodes = self.lcg.node_embeddings.detach().cpu().numpy().reshape(-1, self.D)
        
        # 3. 샘플 (Fx) 임베딩을 [Num_Samples * H, D_head]로 분리
        H = self.lcg.H 
        D_head = self.D
        
        X_samples_headed = X_samples.reshape(-1, H, D_head) # [Num_Samples, H, D_head]
        X_samples_final = X_samples_headed.reshape(-1, D_head) # [Num_Samples * H, D_head]
        
        # 4. UMAP용 데이터 준비 (모두 D_head 차원)
        X_all = np.concatenate([X_samples_final, lcg_nodes], axis=0)
        X_all_scaled = StandardScaler().fit_transform(X_all)

        # 5. UMAP/t-SNE (공통 공간)
        print(f"    -> 샘플 (per-head, {H}개)과 LCG 노드를 'D_head' 공통 공간에서 변환합니다.")
        reducer = (
            umap.UMAP(n_neighbors=40, min_dist=0.2, metric="euclidean", random_state=42)
            if (method == "umap" and umap is not None)
            else TSNE(n_components=2, perplexity=min(30, X_all.shape[0]-1), random_state=42)
        )
        X_2d_all = reducer.fit_transform(X_all_scaled)
        
        # 6. 좌표 분리
        num_sample_heads = len(X_samples_final)
        X_2d_samples = X_2d_all[:num_sample_heads] # [Num_Samples * H, 2]
        lcg_2d = X_2d_all[num_sample_heads:]       # [M*K, 2]
        
        # --- [ ❗️ 7. 시각화 (방법 B: 1xH 그리드) ❗️ ] ---
        
        fig, axes = plt.subplots(1, H, figsize=(H * 6, 6), squeeze=False) # figsize 약간 키움
        axes = axes.flatten()

        print(f"[VIZ] 🎨 Plotting 1x{H} grid with LCG Node IDs...")

        # LCG 노드 플로팅 준비
        cmap_lcg = plt.colormaps.get_cmap("Set3") 
        lcg_colors = {m: cmap_lcg(m % 12) for m in range(self.M)}
        
        # 소스 샘플 플로팅 준비
        cmap_src = plt.colormaps.get_cmap("tab10") 
        src_names = sorted(list(set(all_labels))) 
        src_colors = {s: cmap_src(i % 10) for i, s in enumerate(src_names)}
        
        head_labels_tiled = np.tile(np.arange(H), len(all_labels)) 
        src_labels_repeated = np.repeat(all_labels, H) 

        for h in range(H):
            ax = axes[h]
            
            # 1. 배경: LCG 노드 (M*K개) 그리기 및 ID 표기
            for m in range(self.M):
                sub_nodes_2d = lcg_2d[m * self.K:(m + 1) * self.K]
                color = lcg_colors[m]
                
                # LCG 그룹을 나타내는 테두리 원 그리기 (선택 사항, 너무 복잡하면 제거)
                # center_m = sub_nodes_2d.mean(axis=0)
                # radius_m = np.max(np.linalg.norm(sub_nodes_2d - center_m, axis=1)) * 1.5
                # circle = plt.Circle(center_m, radius_m, color=color, fill=False, linestyle='--', alpha=0.2, linewidth=1.0)
                # ax.add_artist(circle)
                
                # LCG 노드 (사각형 마커)
                ax.scatter(sub_nodes_2d[:, 0], sub_nodes_2d[:, 1],
                           color=color, s=80, edgecolor='black', 
                           linewidth=0.7, label=f"LCG_{m}" if h == 0 else "", # 첫 번째 플롯에만 범례 추가
                           marker='s', # 'D' 대신 's' (정사각형) 사용
                           alpha=0.7, zorder=3) # zorder로 샘플 위에 표시
                
                # 노드 ID 텍스트 추가
                for k in range(self.K):
                    node_id = f"{m}-{k}"
                    ax.text(sub_nodes_2d[k, 0] + 0.3, sub_nodes_2d[k, 1] + 0.3, # 텍스트 위치 살짝 조정
                            node_id, fontsize=7, color='black', ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.6, ec='none'),
                            zorder=4) # 텍스트가 노드 위에 오도록 zorder 조정

            # 2. 전경: Head 'h'에 해당하는 샘플 그리기
            head_sample_indices = np.where(head_labels_tiled == h)[0] 
            X_2d_head_h = X_2d_samples[head_sample_indices] 
            labels_head_h = src_labels_repeated[head_sample_indices] 
            
            for src_name in src_names:
                src_idx = np.where(labels_head_h == src_name)[0]
                ax.scatter(X_2d_head_h[src_idx, 0], X_2d_head_h[src_idx, 1],
                           c=[src_colors[src_name]], s=15, alpha=0.4, # s=15, alpha=0.4로 조절
                           label=f"{src_name}" if h == 0 else "", # 첫 번째 플롯에만 범례 추가
                           zorder=2) # LCG 노드보다 뒤에 표시

            ax.set_title(f"Head {h} View", fontsize=12)
            ax.grid(alpha=0.3, linestyle=':') # 그리드 스타일 개선
            ax.set_xlabel("Dim-1")
            
        axes[0].set_ylabel("Dim-2")
        
        # 3. 범례 개선 (LCG와 Source 범례 분리)
        # LCG 범례
        lcg_handles = [plt.Line2D([0], [0], marker='s', color='w', 
                                  markerfacecolor=lcg_colors[m], 
                                  markeredgecolor='black',
                                  markersize=8, label=f"LCG {m}") for m in range(self.M)]
        # Source 범례
        src_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=src_colors[s], 
                                  markeredgecolor='none',
                                  markersize=7, label=f"{s}") for s in src_names]
                                  
        # 두 범례를 하나의 레이아웃으로 통합하여 오른쪽에 표시
        # 범례를 두 열로 나누어 배치
        first_col_len = max(len(lcg_handles), len(src_handles)) // 2 + 1 
        
        combined_handles = lcg_handles + src_handles
        combined_labels = [h.get_label() for h in combined_handles]

        fig.legend(combined_handles, combined_labels, 
                   loc='center right', bbox_to_anchor=(1.0 + 0.01 * H, 0.5), # H에 따라 범례 위치 살짝 조정
                   fontsize=9, frameon=True, ncol=1) # 한 열로 깔끔하게

        fig.suptitle(f"D_head ({D_head}D) Common Space UMAP (1x{H} Grid)", fontsize=16, y=1.03)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # 범례 공간 확보
        
        save_path = new_dir / f"sources_and_lcg_per_head_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")

    def visualize_lcg_assignments(self, sources, out_root: Path):
        """
        각 소스 데이터가 어떤 LCG에 할당되는지 누적 막대 그래프로 시각화합니다.
        """
        print("[INFO] 📊 Generating LCG assignment statistics...")
        ensure_dir(out_root / "lcg_analysis")
        
        M = self.M
        H = self.model.args.num_basis_heads
        
        counts = {}
        for s in sources:
            loader = self._make_loader(s, batch_size=32)
            source_counts = np.zeros((M, H))
            
            for i, batch in enumerate(loader):
                assign_idx_tensor, _ = self._get_assignments_and_plans(batch)
                assign_idx_np = assign_idx_tensor.cpu().numpy()
                
                for b in range(assign_idx_np.shape[0]):
                    for h in range(assign_idx_np.shape[1]):
                        lcg_idx = assign_idx_np[b, h]
                        source_counts[lcg_idx, h] += 1
            counts[s] = source_counts

        colors = {s: plt.colormaps.get_cmap("tab10")(i % 10) 
                  for i, s in enumerate(sources)}
        lcg_names = [f"LCG_{i}" for i in range(M)]
        
        for h in range(H):
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            bottom = np.zeros(M)
            
            for s in sources:
                source_counts_per_head = counts[s][:, h]
                ax.bar(lcg_names, source_counts_per_head, 
                       bottom=bottom, label=s, color=colors[s],
                       edgecolor='black', linewidth=0.5)
                bottom += source_counts_per_head
            
            ax.set_ylabel("Assigned Sample Count")
            ax.set_title(f"LCG Assignment Distribution (Head {h})")
            ax.legend(title="Sources", loc='upper right')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            save_path = out_root / "lcg_analysis" / f"lcg_assignments_head_{h}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[VIZ] ✅ Saved: {save_path}")
            
    # [--- 수정: 컬러바 및 제목 Affinity로 변경 ---]
    @torch.no_grad()
    def visualize_lcg_affinities(self, out_root: Path):
        """
        학습된 LCG 8개의 내부 "Affinity" (코사인 유사도, -1 ~ 1)를 시각화합니다.
        """
        print("[INFO] 🎨 Generating LCG internal affinity heatmaps...")
        ensure_dir(out_root / "lcg_analysis")
        
        # 1. self.lcg()로부터 "거리(Distance)" 행렬을 가져옵니다.
        # (반환값 3개 중 2번째가 Dy(거리) 행렬입니다)
        _ , Dy_distance_tensor, _ = self.lcg() 
        
        # ⬇️ --- [ ✨ "거리"를 "유사도(Affinity)"로 역연산 (핵심 수정) ✨ ] --- ⬇️
        # 수식: Dy_distance = (1.0 - Dy_similarity) / 2.0
        # 역연산: Dy_similarity = 1.0 - (2.0 * Dy_distance)
        Dy_affinity_tensor = 1.0 - (2.0 * Dy_distance_tensor)
        # ⬆️ --- [ ✨ 수정 끝 ✨ ] --- ⬆️

        # 2. 이제 "유사도" 텐서를 numpy로 변환합니다.
        Dy_affinity_np = Dy_affinity_tensor.detach().cpu().numpy() # [M, K, K]
        
        M, K, _ = Dy_affinity_np.shape
        ncols = min(M, 4)
        nrows = int(np.ceil(M / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, 
                                 figsize=(ncols * 4, nrows * 3.5), 
                                 squeeze=False)
        
        # 3. "유사도"의 범위(-1 ~ 1)로 플롯합니다. (이 설정은 올바릅니다)
        vmin, vmax = -1, 1
        
        for m in range(M):
            ax = axes[m // ncols, m % ncols]
            matrix = Dy_affinity_np[m]
            
            # vmin/vmax에 따라 대각선(+1.0)이 노란색으로 올바르게 표시됩니다.
            im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax, 
                           interpolation='nearest')
            
            ax.set_title(f"LCG_{m} Internal Affinity")
            ax.set_xlabel("Node Index (K)")
            ax.set_ylabel("Node Index (K)")
        
        for m in range(M, nrows * ncols):
            axes[m // ncols, m % ncols].axis('off')

        # 컬러바 레이블 (이 설정도 올바릅니다)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, 
                     label=f"Cosine similarity (-1 to 1)") 
        
        fig.suptitle("Trained LCG Internal Affinity Structures", fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = out_root / "lcg_analysis" / "lcg_internal_affinities.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")
    # [--- 수정 끝 ---]

    # [--- 수정: visualize_lcg_node_usage 그리드 레이아웃 변경 ---]
    @torch.no_grad()
    def visualize_lcg_node_usage(self, sources, out_root: Path):
        """
        각 LCG가 샘플에 할당될 때, 내부 K개 노드를 얼마나 사용하는지 시각화합니다.
        (즉, Transport Plan Pi의 K-marginal 분포)
        """
        print("[INFO] 📈 Generating LCG internal node usage (K-Marginal)...")
        ensure_dir(out_root / "lcg_analysis")

        M, K, H = self.M, self.K, self.model.args.num_basis_heads
        
        # 1. 집계: k_usage[source_name] = [M, K, H] 크기의 누적 분포
        k_usage = {}
        
        for s in sources:
            loader = self._make_loader(s, batch_size=32)
            source_k_usage = np.zeros((M, K, H))
            
            for i, batch in enumerate(loader):
                assign_idx, Pi_all = self._get_assignments_and_plans(batch)
                B = assign_idx.shape[0]

                for b in range(B):
                    for h in range(H):
                        lcg_idx = assign_idx[b, h]
                        
                        Pi_winner = Pi_all[b, h, lcg_idx, :, :]
                        k_marginal = Pi_winner.sum(dim=0).cpu().numpy()
                        
                        source_k_usage[lcg_idx, :, h] += k_marginal
            
            k_usage[s] = source_k_usage

        cmap = plt.colormaps.get_cmap("tab10")
        colors = {s: cmap(i % 10) for i, s in enumerate(sources)}
        
        # --- [ ❗️❗️ 수정 ❗️❗️ ] ---
        # 그리드 레이아웃을 H행 M열로 변경하여 가로로 길게 그리기
        # H=1인 경우 (1, M) 행렬
        fig, axes = plt.subplots(H, M, # H행 M열
                                 figsize=(M * 3.5, H * 4), # 가로 길이 더 길게
                                 squeeze=False)
        
        for h in range(H): # 먼저 Head를 반복
            for m in range(M): # 그 다음 LCG를 반복
                ax = axes[h, m] # ax 접근 순서도 변경

                bottom = np.zeros(K)
                
                for s in sources:
                    node_usage_k = k_usage[s][m, :, h] # 여전히 m, h 순서
                    
                    ax.bar(range(K), node_usage_k, 
                           bottom=bottom, label=s, color=colors[s],
                           edgecolor='black', linewidth=0.5)
                    
                    bottom += node_usage_k
                
                ax.set_title(f"LCG_{m} / Head_{h}")
                ax.set_xlabel("Node Index (K)")
                ax.set_ylabel("Total Transport Mass")
                ax.set_xticks(range(K))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title="Sources", loc='upper right')
        fig.suptitle("LCG Internal Node Usage (Transport Plan K-Marginal)", fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = out_root / "lcg_analysis" / "lcg_internal_node_usage.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")
    # [--- 수정 끝 ---]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Path to checkpoint file")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--method", type=str, default="umap", 
                    choices=["umap", "tsne"])
    
    # ❗️ [1. Plan 시각화를 위한 인자 추가] ❗️
    ap.add_argument("--num_plan_samples", type=int, default=3,
                    help="Number of samples for Transport Plan viz")
    
    args = ap.parse_args()

    # (LCSVisualizer 클래스와 ensure_dir 함수가 
    #  이 파일 어딘가에 import/정의되어 있어야 함)
    viz = LCGVisualizer(args.checkpoint_dir) 
    out_root = (Path(args.output_dir) if args.output_dir 
                else Path(args.checkpoint_dir).parent / "visualization")
    ensure_dir(out_root)

    # (1) LCG만 시각화
    viz.visualize_lcg_only(out_root, method=args.method)
    
    # (2) Multi-source + LCG 함께 시각화
    sources = (viz.args.source_data 
               if isinstance(viz.args.source_data, (list, tuple)) 
               else [viz.args.source_data])
    target = getattr(viz.args, "target_data", None)
    if target:
        sources.append(target)
    
    viz.visualize_sources_and_lcg(sources, out_root, 
                                  max_samples=args.max_samples,
                                  method=args.method)
    
    # (3) LCG 할당 누적 막대 그래프 시각화
    viz.visualize_lcg_assignments(sources, out_root)
    
    # (4) LCG 내부 "유사도" 시각화 (함수 이름 및 내용 변경)
    viz.visualize_lcg_affinities(out_root) # 함수 이름 변경
    
    # (5) LCG K-노드 사용량(Pi) 시각화 (레이아웃 변경)
    viz.visualize_lcg_node_usage(sources, out_root)
    
    # ❗️ [2. 여기에 Transport Plan 시각화 호출 추가] ❗️
    # (6) Transport Plan (Pi) 히트맵 시각화
    print("\n[INFO] Visualizing Transport Plans (Pi)...")
    viz.visualize_transport_plans(
        sources=sources, # (2)번에서 만든 sources 리스트 재사용
        out_root=out_root,
        num_samples=args.num_plan_samples # 1번에서 추가한 인자 사용
    )
    
    print(f"\n✅ 시각화 완료! 결과: {out_root}")


if __name__ == "__main__":
    main()