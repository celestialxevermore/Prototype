# adjacency_analysis.py
import os, copy, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
# 현재 스크립트 파일 위치 (analysis/attentionmap.py)
current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가
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

def robust_vmin_vmax(arr):
    n = arr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(np.nanmin(arr)), float(np.nanmax(arr))
    q05, q95 = np.quantile(vals, 0.05), np.quantile(vals, 0.95)
    if q05 == q95:
        return float(np.min(vals)), float(np.max(vals))
    return float(q05), float(q95)

def add_cb(ax, im):
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

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
                           "viz","viz").to(self.device)

        # 1) 체크포인트에 저장된 alpha_ema를 따로 보관(시각화에 쓸 수 있음)
        sd = ckpt['model_state_dict']
        self.anchor_from_ckpt = sd.get('basis_affinity.alpha_ema', None)

        # 2) alpha_ema 키 제거 후 로드 (사이즈 미스매치 회피)
        sd_wo_anchor = {k: v for k, v in sd.items() if 'alpha_ema' not in k}
        missing, unexpected = self.model.load_state_dict(sd_wo_anchor, strict=False)

        # (선택) 안전 확인: 다른 키가 빠지지 않았는지 로그 정도만
        if any(k != 'basis_affinity.alpha_ema' for k in missing):
            print("[WARN] Missing keys (except alpha_ema):", missing)
        if unexpected:
            print("[WARN] Unexpected keys:", unexpected)

        self.model.eval()

        self.num_layers = int(getattr(self.args,'num_basis_layers',3))
        self.num_heads  = int(getattr(self.args,'k_basis',8))


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
        bd = {k:(v.to(self.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        desc_list, nv_list = [], []
        if all(k in bd for k in ['cat_name_value_embeddings','cat_desc_embeddings']):
            nv_list.append(bd['cat_name_value_embeddings']); desc_list.append(bd['cat_desc_embeddings'])
        if all(k in bd for k in ['num_prompt_embeddings','num_desc_embeddings']):
            nv_list.append(bd['num_prompt_embeddings']);     desc_list.append(bd['num_desc_embeddings'])
        desc_list, nv_list = self.model.remove_feature(bd, desc_list, nv_list)

        desc = torch.cat(desc_list, dim=1)  # [B,S,D]
        nv   = torch.cat(nv_list , dim=1)   # [B,S,D]
        B,S,D = nv.shape
        cls = self.model.cls.expand(B,1,D)
        x = torch.cat([cls, nv], dim=1)     # [B,T,D], T=S+1

        # Shared layers
        # x_shared = x.clone()
        # for l in range(self.model.num_shared_layers):
        #     nx = self.model.shared_layer_norms[l](x_shared)
        #     out, _ = self.model.shared_layers[l](desc, nx)
        #     x_shared = x_shared + out

        # # Get coordinates
        # cls_coord = x_shared[:, 0, :]
        # coordinates = self.model.coordinator(cls_coord)
        bias_log, alpha = self.model.basis_affinity(desc, nv)
        anchor = self.model.basis_affinity.alpha_ema 
        if anchor.numel() == 0 or torch.all(anchor ==0):
            anchor = alpha.mean(dim = 0, keepdim = False)
        alpha_np = alpha[0].detach().cpu().numpy() 
        anchor_np = anchor.detach().cpu().numpy() 

        mask_M = alpha.clamp_(1e-6, 1.0 - 1e-6)


        # Basis layers - 각 레이어마다 새로운 affinity 계산
        Ms, ATTs, ADJs = [], [], []
        x_basis = x.clone()
        
        for l in range(self.num_layers):
            # 각 레이어마다 새로운 affinity 계산
            norm_x = self.model.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.model.basis_layers[l](desc, norm_x, mask_M = mask_M) 
            new_adj = self.model.basis_layers[l].new_adjacency 
            x_basis = x_basis + basis_outputs.reshape(B, S+1, D)
            
            A_np = alpha[0].cpu().numpy() 
            for h in range(self.num_heads):
                Ms.append(A_np[h]) 
            
            ATTs.append(att[0].cpu().numpy())  # [H,T,T] - take first sample
            ADJs.append(new_adj[0].cpu().numpy())  # [T,T] - take first sample

        feat_names = self.model.extract_feature_names(bd)  # length S
        return Ms, ATTs, ADJs, feat_names, alpha_np, anchor_np

    def _grid_plot(self, mats, names, title, save_path):
        L,H = self.num_layers, self.num_heads
        n = mats[0].shape[0]
        fig, axes = plt.subplots(L, H, figsize=(H*3.2, L*3.2))
        if L==1: axes = np.expand_dims(axes,0)
        if H==1: axes = np.expand_dims(axes,1)
        
        # 전체 행렬들에서 통일된 vmin, vmax 계산
        all_vals = []
        for mat in mats:
            mask = ~np.eye(n, dtype=bool)
            vals = mat[mask]
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
                ax = axes[l,h]
                m = mats[l*H+h]
                im = ax.imshow(m, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    def _plot_complete_head_analysis(self, att, adj, var_names, layer_idx, head_idx, sample_idx, base_path):
        """각 헤드별 완전한 분석: 2x2 서브플롯 (Top Edge Weights, Graph Structure, Graph Matrix, Final Matrix)"""
        names_all = ["CLS"] + var_names
        n = len(names_all)
        
        # 최종 그래프 행렬 계산
        final_graph_matrix = att * adj  # 이미 numpy 배열이므로 .numpy() 제거
        mask = (adj == 0)
        final_graph_matrix[mask] = 0.0
        
        # Edge 리스트 수집
        cls_edges_info = []  # CLS에서 나가는 엣지
        var_edges_info = []  # 나머지 엣지
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    w = final_graph_matrix[i, j]
                    if i == 0:
                        cls_edges_info.append((f"{i}->{j}", w))
                    else:
                        var_edges_info.append((f"{i}->{j}", w))
        
        # topK 적용
        top_k = min(10, len(var_edges_info))
        var_edges_info.sort(key=lambda x: x[1], reverse=True)
        var_edges_info = var_edges_info[:top_k]
        
        # 전체 합치기
        edges_info = cls_edges_info + var_edges_info
        edges_info.sort(key=lambda x: x[1], reverse=True)
        edge_labels = [x[0] for x in edges_info]
        edge_weights = [x[1] for x in edges_info]
        
        # CLS 엣지와 일반 엣지 구분을 위한 색상 리스트
        bar_colors = []
        for label in edge_labels:
            if label.startswith("0->"):
                bar_colors.append("crimson")  # CLS 엣지는 빨간색
            else:
                bar_colors.append("cornflowerblue")  # 일반 엣지는 파란색
        
        # 노드 이름 매핑
        node_name_map = {0: "CLS"}
        for i in range(1, n):
            idx_feat = i - 1
            if idx_feat < len(var_names):
                node_name_map[i] = var_names[idx_feat]
            else:
                node_name_map[i] = f"feature_{i}"
                
        # x축 라벨에 사용할 이름 변환
        display_edge_labels = []
        for label in edge_labels:
            i, j = map(int, label.split('->'))
            display_edge_labels.append(f"{node_name_map[i]}->{node_name_map[j]}")
        
        # Figure & 2x2 Subplots 생성
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        ax_bar = axes[0, 0]
        
        # (A) Bar plot
        bars = ax_bar.bar(range(len(edge_weights)), edge_weights, color=bar_colors)
        
        # 각 바 위에 attention score 값 표시
        for i, (weight, label) in enumerate(zip(edge_weights, edge_labels)):
            ax_bar.text(i, weight + 0.01, f"{weight:.3f}", 
                    ha='center', va='bottom', rotation=90, 
                    fontsize=7, color='black')
        
        ax_bar.set_title(f'Top Edge Weights - Layer {layer_idx} · Head {head_idx}', fontsize=12)
        ax_bar.set_xlabel('Edge (i->j)')
        ax_bar.set_ylabel('Attention Weight')
        ax_bar.set_xticks(range(len(edge_labels)))
        ax_bar.set_xticklabels(display_edge_labels, rotation=90, fontsize=8)
        
        # (B) Network Graph
        ax_graph = axes[0, 1]
        import networkx as nx
        G = nx.DiGraph()
        node_labels = {}

        for i in range(n):
            if i == 0:
                node_name = "CLS"
                node_color = "red"
            else:
                idx_feat = i - 1
                if idx_feat < len(var_names):
                    node_name = var_names[idx_feat]
                    node_color = "blue"
                else:
                    node_name = f"feature_{i}"
                    node_color = "blue"

            G.add_node(i, name=node_name, color=node_color)
            node_labels[i] = node_name

        # CLS->Var / Var->Var 구분해서 그리기
        cls_min_edge_weight = 0.001
        min_edge_weight = 0.001
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                w = final_graph_matrix[i, j]
                if i == 0 and j != 0:
                    # CLS->Var
                    if w > cls_min_edge_weight:
                        G.add_edge(i, j, weight=w, cls_to_var=True)
                elif j == 0:
                    # Var->CLS는 표시 안 함
                    continue
                else:
                    if w > min_edge_weight:
                        # Var->Var
                        G.add_edge(i, j, weight=w, cls_to_var=False)

        pos = {}
        pos[0] = np.array([0, 0])
        non_center_nodes = n - 1
        radius = 1.0
        for i_ in range(1, n):
            angle_ = 2 * np.pi * (i_ - 1) / non_center_nodes
            pos[i_] = np.array([radius * np.cos(angle_), radius * np.sin(angle_)])

        # 배경 그리드
        for r_ in [0.25, 0.5, 0.75, 1.0]:
            circle = plt.Circle((0, 0), r_, fill=False, color='lightgray', linestyle='--', alpha=0.5)
            ax_graph.add_patch(circle)
        for i_ in range(1, n):
            angle__ = 2 * np.pi * (i_ - 1) / non_center_nodes
            x_ = 1.1 * np.cos(angle__)
            y_ = 1.1 * np.sin(angle__)
            ax_graph.plot([0, x_], [0, y_], color='lightgray', linestyle='--', alpha=0.5)

        node_colors = [d["color"] for _, d in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax_graph, edgecolors='gray')

        cls_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('cls_to_var')]
        var_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('cls_to_var')]

        cls_weights = [G[u][v]['weight'] for (u, v) in cls_edges]
        var_weights = [G[u][v]['weight'] for (u, v) in var_edges]

        # CLS->Var: 빨강 굵은선
        if cls_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=cls_edges,
                width=[2 + w * 5 for w in cls_weights],
                alpha=0.7,
                edge_color='crimson',
                connectionstyle='arc3,rad=0.1',  
                arrowstyle='-|>',
                arrowsize=15,
                node_size=800,
                ax=ax_graph
            )

        # Var->Var: 파랑 점선
        if var_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=var_edges,
                width=[1 + w * 2 for w in var_weights],
                edge_color='blue',
                style='dashed',
                arrowstyle='-|>',
                arrowsize=30,
                alpha=0.5,
                ax=ax_graph,
                arrows=True
            )

        label_options = {
            "font_size": 9,
            "font_color": "black",
            "bbox": dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        }
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_graph, **label_options)

        ax_graph.set_title(f'Graph Structure - Layer {layer_idx} · Head {head_idx} - Sample {sample_idx}', fontsize=12)
        ax_graph.axis('off')
        ax_graph.set_aspect('equal')
        ax_graph.set_xlim([-1.2, 1.2])
        ax_graph.set_ylim([-1.2, 1.2])

        # (C) Graph matrix heatmap (이진 Structural Adjacency)
        ax_graph_matrix = axes[1, 0]
        graph_matrix_np = adj  # 이미 numpy 배열이므로 .numpy() 제거
        im_graph = ax_graph_matrix.imshow(graph_matrix_np, cmap="Blues", interpolation='nearest')
        ax_graph_matrix.set_title("Graph Matrix (Structural Adjacency)", fontsize=14)
        fig.colorbar(im_graph, ax=ax_graph_matrix)

        ax_graph_matrix.set_xticks(np.arange(len(names_all)))
        ax_graph_matrix.set_yticks(np.arange(len(names_all)))
        ax_graph_matrix.set_xticklabels(names_all, rotation=90, fontsize=8)
        ax_graph_matrix.set_yticklabels(names_all, fontsize=8)

        for i in range(len(names_all)):
            for j in range(len(names_all)):
                ax_graph_matrix.text(j, i, f"{graph_matrix_np[i,j]:.2f}", ha="center", va="center", 
                                color="black" if graph_matrix_np[i,j] < 0.5 else "white", fontsize=8)

        # (D) Final graph matrix (Attention × Adjacency)
        ax_final = axes[1, 1]
        vmax = final_graph_matrix.max()
        vmin = 0.0

        im_final = ax_final.imshow(final_graph_matrix, 
                                cmap='YlOrRd',
                                interpolation='nearest',
                                vmin=vmin, 
                                vmax=vmax)
        ax_final.set_title("Final Graph Matrix (Attention × Adjacency)", fontsize=14)
        fig.colorbar(im_final, ax=ax_final)
        
        ax_final.set_xticks(np.arange(len(names_all)))
        ax_final.set_yticks(np.arange(len(names_all)))
        ax_final.set_xticklabels(names_all, rotation=90, fontsize=8)
        ax_final.set_yticklabels(names_all, fontsize=8)
        
        # 각 셀에 값 표시
        for i in range(len(names_all)):
            for j in range(len(names_all)):
                relative_value = final_graph_matrix[i,j] / vmax if vmax > 0 else 0
                text_color = "black" if relative_value < 0.7 else "white"
                value_text = f"{final_graph_matrix[i,j]:.3f}"
                
                ax_final.text(j, i, value_text, 
                            ha="center", va="center", 
                            color=text_color, 
                            fontsize=7)
        
        # 전체 제목 설정
        fig.suptitle(f'Layer {layer_idx} · Head {head_idx} - Sample {sample_idx}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # 저장
        save_path = base_path / f"sample_{sample_idx}_L{layer_idx}_H{head_idx}_complete.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _grid_plot_centered(self, mats, names, title, save_path, center=0.0, cmap='RdBu_r'):
        L, H = self.num_layers, self.num_heads
        n = mats[0].shape[0]

        # 대칭 vlim 계산 (대각 제외)
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
                data = (M - center) if np.isscalar(center) else (M - center)  # center가 스칼라든 행렬이든 동일 처리
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f"L{l} · H{h}", fontsize=10)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticklabels(names, fontsize=7)
                add_cb(ax, im)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    

    def visualize_dataset(self, dataset_name: str, role: str, out_root: Path, max_samples=2):
        loader = self._make_loader(dataset_name)
        base = out_root/role/dataset_name
        ensure_dir(base)
        count=0
        for batch in loader:
            Ms, ATTs, ADJs, var_names, alpha_np, anchor_np = self._forward_collect(batch)
            
            # 각 샘플별 서브폴더 생성
            sample_dir = base / f"sample_{count}"
            ensure_dir(sample_dir)
            
            # ---------- (1) Var-Var Affinity (BasisAffinityGAT) ----------
            grid_M=[]
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    # Ms는 [head0, head1, ..., head7, head0, head1, ..., head7, ...] 순서로 저장됨
                    # 각 레이어마다 num_heads개씩 저장되어 있음
                    head_idx = l * self.num_heads + h
                    grid_M.append(Ms[head_idx])  # [S,S] - each head's affinity
            self._grid_plot(grid_M, var_names,
                            f"{role.capitalize()} • {dataset_name} • Sample {count} • Var-Var Affinity",
                            sample_dir/f"Affinity_varvar_grid.png")

            # ---------- (2) attention × structural adjacency (CLS 포함) ----------
            names_all = ["CLS"]+var_names
            grid_AX=[]
            for l in range(self.num_layers):
                att = ATTs[l]    # [H,T,T]
                adj = ADJs[l]    # [T,T]
                for h in range(self.num_heads):
                    grid_AX.append(att[h]*adj)     # [T,T]; CLS->Var가 살아있음
            self._grid_plot(grid_AX, names_all,
                            f"{role.capitalize()} • {dataset_name} • Sample {count} • Attention × Adjacency",
                            sample_dir/f"AttnXAdj_grid.png")

            # ---------- (3) 각 헤드별 완전한 분석 (2x2 서브플롯) ----------
            # for l in range(self.num_layers):
            #     att = ATTs[l]    # [H,T,T]
            #     adj = ADJs[l]    # [T,T]
            #     for h in range(self.num_heads):
            #         self._plot_complete_head_analysis(
            #             att[h], adj, var_names, l, h, count, sample_dir
            #         )
            # (A) 샘플 α (균등 1/S를 중심으로)
            uniform = 1.0 / len(var_names)
            grid_alpha = []
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    grid_alpha.append(alpha_np[h])  # [S,S]
            self._grid_plot_centered(
                grid_alpha, var_names,
                f"{role.capitalize()} • {dataset_name} • Sample {count} • Var-Var Affinity (center=1/S)",
                sample_dir / "Affinity_varvar_grid_center_uniform.png",
                center=uniform, cmap='RdBu_r'
            )

            # (B) 전역 앵커 α_ema (동일한 센터/스케일)
            grid_anchor = []
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    grid_anchor.append(anchor_np[h])
            self._grid_plot_centered(
                grid_anchor, var_names,
                f"{role.capitalize()} • {dataset_name} • Global Anchor (EMA) (center=1/S)",
                sample_dir / "Anchor_varvar_grid_center_uniform.png",
                center=uniform, cmap='RdBu_r'
            )

            # (C) 편차 Δ = α - α_ema (center=0)
            grid_delta = []
            for l in range(self.num_layers):
                for h in range(self.num_heads):
                    grid_delta.append(alpha_np[h] - anchor_np[h])
            self._grid_plot_centered(
                grid_delta, var_names,
                f"{role.capitalize()} • {dataset_name} • Δ(α − EMA)",
                sample_dir / "Delta_alpha_minus_anchor.png",
                center=0.0, cmap='RdBu_r'
            )
            count+=1
            if count>=max_samples: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True, type=str)
    ap.add_argument('--max_samples', type=int, default=2)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    auto_del = extract_deleted_features_from_checkpoint(args.checkpoint_dir)
    if auto_del: print(f"[INFO] del_feat from filename: {auto_del}")

    viz = MVisualizer(args.checkpoint_dir, auto_del_feat=auto_del)

    if args.output_dir is None:
        parent = Path(args.checkpoint_dir).parent
        out_root = Path(str(parent).replace("/checkpoints/","/visualization/"))/"graph_visualization"
    else:
        out_root = Path(args.output_dir)
    ensure_dir(out_root)

    sources = viz.args.source_data if isinstance(viz.args.source_data,(list,tuple)) else [viz.args.source_data]
    target  = getattr(viz.args,'target_data', None) or 'heart'

    for s in sources:
        viz.visualize_dataset(s, role='source', out_root=out_root, max_samples=args.max_samples)
    viz.visualize_dataset(target, role='target', out_root=out_root, max_samples=args.max_samples)

if __name__ == "__main__":
    main()