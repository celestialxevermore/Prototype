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
        
        _ = self.model.predict(bd) 

        basis_outputs = self.model.basis_outputs_for_viz
        Fx = basis_outputs[:, 1:, :, :].permute(0, 2, 1, 3)
        B, H, N, D_head = Fx.shape
        
        P_affinity = self.model._last_P_basis 
        Dx = LatentCompositeGraph.normalize_affinity(P_affinity)
        Dx = LatentCompositeGraph.affinity_to_distance(Dx) # Dx는 거리
        
        # --- [ ❗️❗️ 여기가 수정된 지점 ❗️❗️ ] ---
        # 3개의 반환값을 모두 받되, diversity_loss는 무시 (시각화에 불필요)
        Fy, Dy_affinity, _ = self.lcg() # lcg()가 3개를 반환
        # --- [ 수정 끝 ] ---
        
        Dy = LatentCompositeGraph.affinity_to_distance(Dy_affinity) # Dy는 거리

        a = torch.ones(B, H, N, device = Fx.device) / N 
        b = torch.ones(B, self.M, self.K, device=Fx.device) / self.K 

        # no_grad()로 argmin 계산
        Pi_all , fgw_values = FGWUtils.assign_FGW(
            Fx, Fy, Dx, Dy, a, b, # Dx, Dy 모두 거리
            alpha = self.args.fgw_alpha, 
            eps = 0.05, 
            outer_iters = 10, 
            sinkhorn_iters = 30
        ) 
        
        assign_idx = torch.argmin(fgw_values, dim=-1) # [B, H]
        
        return assign_idx, Pi_all
    
    # ... (visualize_lcg_only, _analyze_lcg_distances, visualize_sources_and_lcg는 이전과 동일) ...

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
        Multi-source 데이터 + LCG "K개 노드" 64개를 함께 시각화합니다.
        (장식용 다이아몬드 대신 실제 노드 좌표 사용)
        """
        ensure_dir(out_root / "joint_space")
        
        all_embeds, all_labels = [], []
        print("[INFO] 🔄 Collecting embeddings...")
        
        # 1. 샘플(Fx) 임베딩 수집
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
            
        X_samples_scaled = StandardScaler().fit_transform(X_samples)
        
        # 2. LCG (Fy) "K개 노드" 임베딩 수집
        # [M, K, D_head] -> [M*K, D_head]
        lcg_nodes = self.lcg.node_embeddings.detach().cpu().numpy().reshape(-1, self.D)
        
        # 3. UMAP/TSNE (차원 일치 확인)
        # ❗️ Fx(D_full)와 Fy(D_head)의 차원이 다를 수 있습니다.
        if lcg_nodes.shape[1] != X_samples_scaled.shape[1]:
            print(f"⚠️ LCG 노드 차원({lcg_nodes.shape[1]})과 샘플 차원({X_samples_scaled.shape[1]})이 다릅니다!")
            
            
            reducer_lcg = (
                umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
                if (method == "umap" and umap is not None)
                else TSNE(n_components=2, perplexity=min(30, lcg_nodes.shape[0]-1), random_state=42)
            )
            lcg_2d = reducer_lcg.fit_transform(lcg_nodes) # [M*K, 2]
            
            reducer_X = (
                umap.UMAP(n_neighbors=40, min_dist=0.2, metric="euclidean", random_state=42)
                if (method == "umap" and umap is not None)
                else TSNE(n_components=2, perplexity=40, random_state=42)
            )
            X_2d = reducer_X.fit_transform(X_samples_scaled) # [Num_Samples, 2]
            
        else:
            print("    -> 샘플과 LCG를 함께 UMAP/t-SNE 변환합니다.")
            X_all = np.concatenate([X_samples_scaled, lcg_nodes], axis=0)
            reducer = (
                umap.UMAP(n_neighbors=40, min_dist=0.2, metric="euclidean", random_state=42)
                if (method == "umap" and umap is not None)
                else TSNE(n_components=2, perplexity=40, random_state=42)
            )
            X_2d_all = reducer.fit_transform(X_all)
            X_2d = X_2d_all[:len(X_samples_scaled)] # 샘플 2D 좌표
            lcg_2d = X_2d_all[len(X_samples_scaled):] # LCG "K개 노드" 64개의 2D 좌표
        
        # 4. 시각화
        colors = {s: plt.colormaps.get_cmap("tab10")(i % 10) 
                  for i, s in enumerate(sources)}
        plt.figure(figsize=(10, 9))
        
        # Plot Source 데이터 (점)
        for s in sources:
            idx = [i for i, l in enumerate(all_labels) if l == s]
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                        c=[colors[s]], s=15, alpha=0.5, label=f"{s}")
        
        # --- [ ❗️❗️ 여기가 수정된 지점 ❗️❗️ ] ---
        # Plot LCG (K개 노드 64개)
        cmap_lcg = plt.colormaps.get_cmap("Set3")
        for m in range(self.M):
            # lcg_2d ([M*K, 2])에서 K개 노드의 실제 2D 좌표를 가져옴
            sub_nodes_2d = lcg_2d[m * self.K:(m + 1) * self.K]
            color = cmap_lcg(m % 12)
            
            # 8개의 "진짜" 노드 위치를 다이아몬드로 그림
            plt.scatter(sub_nodes_2d[:, 0], sub_nodes_2d[:, 1],
                        color=color, s=100, edgecolor='black', 
                        linewidth=1.0, label=f"LCG_{m}", marker='D',
                        alpha=0.8) # (겹쳐 보이도록 alpha 추가)
            
            # LCG의 평균 중심점에 X자 마커
            center = sub_nodes_2d.mean(axis=0)
            plt.scatter(center[0], center[1], marker="X", 
                        color="black", s=100, linewidth=1.5,
                        alpha=0.5) # (X마커는 검은색으로 통일)
        # --- [ 수정 끝 ] ---

        plt.legend(fontsize=8, frameon=True, loc='best', ncol=2)
        plt.title("Multi-Source Samples + Trained LCG *Nodes* Distributions", fontsize=13) # 제목 수정
        plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        save_path = out_root / "joint_space" / "sources_and_lcg_nodes.png" # 새 이름
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
    args = ap.parse_args()

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
    
    print(f"\n✅ 시각화 완료! 결과: {out_root}")


if __name__ == "__main__":
    main()