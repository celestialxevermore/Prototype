import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse

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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class LCGVisualizer:
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
            print(f"⚠️ Missing keys: {missing[:5]}...")  # 처음 5개만
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected[:5]}...")
        
        self.model.eval()
        self.args = args
        
        # ✅ latent_graph 참조
        self.lcg = self.model.latent_graph
        self.M, self.K, self.D = self.lcg.M, self.lcg.K, self.lcg.D
        
        # ✅ 진단: 학습된 가중치 확인
        emb = self.lcg.node_embeddings.detach().cpu()
        print(f"\n🔍 LCG 가중치 진단:")
        print(f"  - Shape: {emb.shape} (M={self.M}, K={self.K}, D={self.D})")
        print(f"  - node_embeddings[0,0,:5]: {emb[0,0,:5].numpy()}")
        print(f"  - Min/Max: {emb.min():.4f} / {emb.max():.4f}")
        print(f"  - Mean/Std: {emb.mean():.4f} / {emb.std():.4f}")
        
        # 랜덤 초기화 판별 (Xavier: std ≈ sqrt(2/(fan_in+fan_out)))
        expected_std = np.sqrt(2.0 / (self.K + self.D))
        if abs(emb.std().item() - expected_std) < 0.05:
            print(f"  ⚠️ WARNING: 가중치가 랜덤 초기화 상태일 수 있음!")
            print(f"     (현재 std={emb.std():.4f}, Xavier 예상={expected_std:.4f})")
        else:
            print(f"  ✅ 학습된 가중치로 판단됨")

    def _make_loader(self, dataset_name: str):
        args2 = self.args
        args2.source_data = dataset_name
        fix_seed(args2.random_seed)
        res = prepare_embedding_dataloaders(args2, dataset_name)
        tr, va, te = res["loaders"]
        from torch.utils.data import ConcatDataset, DataLoader
        ds = ConcatDataset([tr.dataset, va.dataset, te.dataset])
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    @torch.no_grad()
    def _forward_collect(self, batch):
        bd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
        desc_list, nv_list = [], []
        
        if all(k in bd for k in ["cat_name_value_embeddings", "cat_desc_embeddings"]):
            nv_list.append(bd["cat_name_value_embeddings"])
            desc_list.append(bd["cat_desc_embeddings"])
        if all(k in bd for k in ["num_prompt_embeddings", "num_desc_embeddings"]):
            nv_list.append(bd["num_prompt_embeddings"])
            desc_list.append(bd["num_desc_embeddings"])

        desc = torch.cat(desc_list, dim=1)
        nv = torch.cat(nv_list, dim=1)
        
        if hasattr(self.model, "encode_graph"):
            z_graph = self.model.encode_graph(desc, nv)
        else:
            z_graph = nv.mean(dim=1, keepdim=False).unsqueeze(1)

        return z_graph[0].detach().cpu().numpy()

    def visualize_lcg_only(self, out_root: Path, method="umap"):
        """LCG 노드들만 시각화 (학습 후 분포 확인)"""
        ensure_dir(out_root / "lcg_analysis")
        
        # ✅ 학습된 node_embeddings 가져오기
        lcg_nodes = self.lcg.node_embeddings.detach().cpu().numpy()  # [M, K, D]
        M, K, D = lcg_nodes.shape
        X = lcg_nodes.reshape(M * K, D)  # [M*K, D]
        
        # 차원 축소
        reducer = (
            umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
            if (method == "umap" and umap is not None)
            else TSNE(n_components=2, perplexity=min(30, M*K-1), random_state=42)
        )
        X_2d = reducer.fit_transform(X)
        
        # 시각화
        cmap = plt.colormaps.get_cmap("tab10")
        plt.figure(figsize=(10, 8))
        
        for m in range(M):
            sub_nodes = X_2d[m * K:(m + 1) * K]
            color = cmap(m % 10)
            
            # 노드 scatter
            plt.scatter(sub_nodes[:, 0], sub_nodes[:, 1],
                       color=color, s=80, alpha=0.7, 
                       edgecolor='black', linewidth=0.5,
                       label=f"LCG_{m}")
            
            # 중심점
            center = sub_nodes.mean(axis=0)
            plt.scatter(center[0], center[1], 
                       marker="X", color=color, s=200,
                       edgecolor='black', linewidth=1.5)
            
            # 노드 간 연결 (거리 기반)
            for i in range(K):
                for j in range(i+1, K):
                    dist = np.linalg.norm(sub_nodes[i] - sub_nodes[j])
                    if dist < np.percentile(
                        [np.linalg.norm(sub_nodes[a] - sub_nodes[b]) 
                         for a in range(K) for b in range(a+1, K)], 50
                    ):  # 중간값 이하만 연결
                        plt.plot([sub_nodes[i, 0], sub_nodes[j, 0]],
                                [sub_nodes[i, 1], sub_nodes[j, 1]],
                                color=color, alpha=0.2, linewidth=0.8)
        
        plt.legend(fontsize=9, frameon=True, loc='best')
        plt.title(f"Trained LCG Node Distributions ({method.upper()})", fontsize=14)
        plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        save_path = out_root / "lcg_analysis" / f"lcg_only_{method}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")
        
        # 거리 분석
        self._analyze_lcg_distances(lcg_nodes, out_root)

    def _analyze_lcg_distances(self, lcg_nodes, out_root):
        """LCG 간 거리 분석"""
        M, K, D = lcg_nodes.shape
        
        # 각 LCG의 중심점 계산
        centers = lcg_nodes.mean(axis=1)  # [M, D]
        
        # 중심점 간 거리 행렬
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(centers, metric='euclidean'))
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # (1) 거리 행렬 히트맵
        im = ax1.imshow(dist_matrix, cmap='viridis', aspect='auto')
        ax1.set_title("LCG Center-to-Center Distances", fontsize=12)
        ax1.set_xlabel("LCG Index"); ax1.set_ylabel("LCG Index")
        ax1.set_xticks(range(M)); ax1.set_yticks(range(M))
        plt.colorbar(im, ax=ax1, label="Euclidean Distance")
        
        # (2) 거리 분포 히스토그램
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
        
        # 통계 출력
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
        """Multi-source 데이터 + LCG 함께 시각화"""
        ensure_dir(out_root / "joint_space")
        
        all_embeds, all_labels = [], []
        print("[INFO] 🔄 Collecting embeddings...")
        
        for s in sources:
            loader = self._make_loader(s)
            for i, batch in enumerate(loader):
                z_np = self._forward_collect(batch)
                all_embeds.append(z_np)
                all_labels.append(s)
                if i >= max_samples:
                    break
        
        X = np.concatenate(all_embeds, axis=0)
        X = StandardScaler().fit_transform(X)
        
        # LCG 노드 추가
        lcg_nodes = self.lcg.node_embeddings.detach().cpu().numpy().reshape(-1, self.D)
        X_all = np.concatenate([X, lcg_nodes], axis=0)
        
        # 차원 축소
        reducer = (
            umap.UMAP(n_neighbors=40, min_dist=0.2, metric="euclidean", random_state=42)
            if (method == "umap" and umap is not None)
            else TSNE(n_components=2, perplexity=40, random_state=42)
        )
        X_2d_all = reducer.fit_transform(X_all)
        X_2d, lcg_2d = X_2d_all[:len(X)], X_2d_all[len(X):]
        
        # 시각화
        colors = {s: plt.colormaps.get_cmap("tab10")(i % 10) 
                 for i, s in enumerate(sources)}
        plt.figure(figsize=(10, 9))
        
        # Source 데이터
        for s in sources:
            idx = [i for i, l in enumerate(all_labels) if l == s]
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                       c=[colors[s]], s=15, alpha=0.5, label=f"{s}")
        
        # LCG
        cmap_lcg = plt.colormaps.get_cmap("Set3")
        for m in range(self.M):
            sub_nodes = lcg_2d[m * self.K:(m + 1) * self.K]
            color = cmap_lcg(m % 12)
            
            plt.scatter(sub_nodes[:, 0], sub_nodes[:, 1],
                       color=color, s=100, edgecolor='black', 
                       linewidth=1.0, label=f"LCG_{m}", marker='D')
            
            # 중심
            center = sub_nodes.mean(axis=0)
            plt.scatter(center[0], center[1], marker="X", 
                       color=color, s=250, edgecolor='black', linewidth=2)
        
        plt.legend(fontsize=8, frameon=True, loc='best', ncol=2)
        plt.title("Multi-Source Samples + Trained LCG Distributions", fontsize=13)
        plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        save_path = out_root / "joint_space" / "sources_and_lcg.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] ✅ Saved: {save_path}")


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
    
    print(f"\n✅ 시각화 완료! 결과: {out_root}")


if __name__ == "__main__":
    main()