import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pdb
import torch
import logging



import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
logger = logging.getLogger(__name__)
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def visualize_model_structure(model, data_loader, device, args, mode, experiment_id, epoch, max_samples=10):
    """
    모델의 내부 구조(어텐션, 그래프 구조 등)를 시각화하는 함수
    
    Args:
        model: 시각화할 모델
        data_loader: 시각화에 사용할 데이터 로더
        device: 계산에 사용할 디바이스
        args: 실험 설정 인자 (args.viz_heatmap, args.viz_graph 플래그 사용)
        mode: 'train' 또는 'val' 모드
        experiment_id: 현재 실험 ID
        epoch: 현재 에포크
        max_samples: 시각화할 최대 샘플 수
    """

    
    base_viz_dir = os.path.join(f"/storage/personal/eungyeop/experiments/visualization/{args.llm_model}/{args.source_dataset_name}/{mode}/{experiment_id}")
    os.makedirs(base_viz_dir, exist_ok=True)

    # 샘플별 디렉토리 미리 생성
    sample_dirs = []
    for i in range(max_samples):
        # 각 샘플 디렉토리
        sample_dir = os.path.join(base_viz_dir, f'sample_{i}')
        os.makedirs(sample_dir, exist_ok=True)
        sample_dirs.append(sample_dir)
        
        # 각 샘플 내에 heatmap과 graph 폴더 생성
        heatmap_dir = os.path.join(sample_dir, 'heatmap')
        graph_dir = os.path.join(sample_dir, 'graph')
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)
        
        # 각 폴더 내에 레이어별 서브폴더 생성
        for layer_idx in range(len(model.layers)):
            heatmap_layer_dir = os.path.join(heatmap_dir, f'layer_{layer_idx}')
            graph_layer_dir = os.path.join(graph_dir, f'layer_{layer_idx}')
            os.makedirs(heatmap_layer_dir, exist_ok=True)
            os.makedirs(graph_layer_dir, exist_ok=True)

    with torch.no_grad():
        model.eval()
        
        sample_count = 0
        
        for batch_idx, batch in enumerate(data_loader):
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            prediction = model.predict(batch_on_device)
            
            # 배치 크기 확인
            batch_size = model.layers[0].attn_weights.shape[0]
            
            for sample_idx in range(batch_size):
                # 특성 이름 정리 (모든 레이어에서 공통으로 사용)
                feature_names = []
                if 'cat_desc_texts' in batch_on_device:
                    for feature in batch_on_device['cat_desc_texts']:
                        if isinstance(feature, tuple):
                            clean_name = str(feature[0])
                        else:
                            try:
                                clean_name = feature.split("'")[1] if "'" in feature else feature
                                clean_name = clean_name.split(',')[0]
                            except:
                                clean_name = str(feature)
                        feature_names.append(clean_name)

                if 'num_desc_texts' in batch_on_device:
                    for feature in batch_on_device['num_desc_texts']:
                        if isinstance(feature, tuple):
                            clean_name = str(feature[0])
                        else:
                            try:
                                clean_name = feature.split("'")[1] if "'" in feature else feature
                                clean_name = clean_name.split(',')[0]
                            except:
                                clean_name = str(feature)
                        feature_names.append(clean_name)
                        
                # 중복 제거 (순서 유지)
                seen = set()
                unique_features = []
                for feat in feature_names:
                    if feat not in seen:
                        seen.add(feat)
                        unique_features.append(feat)
                feature_names = unique_features
                
                # 1. 히트맵 시각화
                if args.viz_heatmap:
                    for layer_idx in range(len(model.layers)):
                        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
                        
                        # 1. global_sim 히트맵
                        global_sim_np = model.layers[layer_idx].global_sim[sample_idx].cpu().numpy()
                        im1 = axes[0,0].imshow(global_sim_np, cmap='viridis', interpolation='nearest')
                        axes[0,0].set_title('Global Similarity (Cosine)', fontsize=14)
                        fig.colorbar(im1, ax=axes[0,0])
                        
                        for i in range(len(feature_names)):
                            for j in range(len(feature_names)):
                                axes[0,0].text(j, i, f"{global_sim_np[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)

                        # 2. global_topology_A 히트맵
                        global_topology_A = model.layers[layer_idx].global_topology_A[sample_idx].cpu().numpy()
                        im2 = axes[0,1].imshow(global_topology_A, cmap='plasma', interpolation='nearest')
                        axes[0,1].set_title('global_topology_A = torch.sigmoid(self.global_sim + self.topology_bias)', fontsize=14)
                        fig.colorbar(im2, ax=axes[0,1])
                        
                        for i in range(len(feature_names)):
                            for j in range(len(feature_names)):
                                axes[0,1].text(j, i, f"{global_topology_A[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)
                        
                        # 3. Sample_sim 히트맵
                        sample_sim_np = model.layers[layer_idx].sample_sim[sample_idx].cpu().numpy() 
                        im3 = axes[1,0].imshow(sample_sim_np, cmap='viridis', interpolation='nearest')
                        axes[1,0].set_title('Sample Similarity (Self attention)', fontsize=14)
                        fig.colorbar(im3, ax=axes[1,0])

                        for i in range(len(feature_names)):
                            for j in range(len(feature_names)):
                                axes[1,0].text(j, i, f"{sample_sim_np[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)



                        # 4. adjacency 히트맵
                        adjacency_np = model.layers[layer_idx].adjacency[sample_idx].cpu().numpy()
                        im3 = axes[1,1].imshow(adjacency_np, cmap='cividis', interpolation='nearest')
                        axes[1,1].set_title('Final Adjacency self.G = self.global_topology_A * self.sample_sim', fontsize=14)
                        fig.colorbar(im3, ax=axes[1,1])
                        
                        for i in range(len(feature_names)):
                            for j in range(len(feature_names)):
                                axes[1,1].text(j, i, f"{adjacency_np[i,j]:.2f}", ha="center", va="center", color="white", fontsize=5)
                        
                        # 모든 축에 feature_names 적용
                        for row in axes:
                            for ax in row:
                                ax.set_xticks(np.arange(len(feature_names)))
                                ax.set_yticks(np.arange(len(feature_names)))
                                ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
                                ax.set_yticklabels(feature_names, fontsize=8)
                                ax.grid(False)
                        
                        # 전체 타이틀
                        fig.suptitle(f'Graph Construction Process - Epoch {epoch} - Sample {sample_count}', fontsize=16)
                        plt.tight_layout()
                        
                        # cosine_similarity 폴더에 각 샘플별로 저장
                        # cosine_dir = viz_dir.replace('graph_structure', 'cosine_similarity')
                        # sample_cosine_dir = os.path.join(cosine_dir, f'sample_{sample_count}', f'layer_{layer_idx}')
                        # os.makedirs(sample_cosine_dir, exist_ok=True)
                        # sim_viz_path = os.path.join(sample_cosine_dir, f'epoch_{epoch}.png')
                        # fig.savefig(sim_viz_path, dpi=300, bbox_inches='tight')
                        # plt.close(fig)
                        
                        # logger.info(f"Epoch {epoch} - 샘플 {sample_count} 히트맵 저장: {sim_viz_path}")
                        heatmap_path = os.path.join(sample_dirs[sample_count], 'heatmap', f'layer_{layer_idx}', f'epoch_{epoch}.png')
                        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Epoch {epoch} - 샘플 {sample_count} 히트맵 저장: {heatmap_path}")
                # 2. 그래프 구조 시각화
                if args.viz_graph:
                    # 각 레이어별로 시각화 수행
                    for layer_idx in range(len(model.layers)):
                        # 1) Attention 가중치(헤드 평균)
                        attn_weights = model.layers[layer_idx].attn_weights[sample_idx]  # [n_heads, seq, seq]
                        attn_weights_mean = attn_weights.mean(dim=0).cpu()

                        # 원본 adjacency 사용 (히트맵과 일치하는 값)
                        adjacency = model.layers[layer_idx].adjacency[sample_idx].cpu()
                        new_seq = attn_weights_mean.shape[0]
                        graph_matrix = torch.zeros((new_seq, new_seq), device=attn_weights_mean.device)

                        graph_matrix[1:, 1:] = adjacency  # 변수 간 연결은 원본 adjacency 사용
                        graph_matrix[0, 1:] = 1.0  # CLS->변수 연결
                        graph_matrix[1:, 0] = 0.0  # 변수->CLS 연결
                        
                        mask = (graph_matrix == 0)
                        final_graph_matrix = (attn_weights_mean * graph_matrix).numpy()
                        final_graph_matrix[mask.numpy()] = 0.0 
                        n_nodes = final_graph_matrix.shape[0]
                        
                        # 2) Edge 리스트(모든 i->j) 수집 (Barplot용)
                        cls_edges_info = []  # CLS에서 나가는 엣지
                        var_edges_info = []  # 나머지 엣지
                        
                        for i in range(n_nodes):
                            for j in range(n_nodes):
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
                        for i in range(1, n_nodes):
                            idx_feat = i - 1
                            if idx_feat < len(feature_names):
                                node_name_map[i] = feature_names[idx_feat]
                            else:
                                node_name_map[i] = f"feature_{i}"
                                
                        # x축 라벨에 사용할 이름 변환
                        display_edge_labels = []
                        for label in edge_labels:
                            i, j = map(int, label.split('->'))
                            display_edge_labels.append(f"{node_name_map[i]}->{node_name_map[j]}")
                        
                        # Figure & 2 Subplots 생성
                        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
                        ax_bar = axes[0, 0]
                        
                        # -----(A) Left Subplot: Barplot)-----
                        bars = ax_bar.bar(range(len(edge_weights)), edge_weights, color=bar_colors)
                        
                        # 각 바 위에 attention score 값 표시
                        for i, (weight, label) in enumerate(zip(edge_weights, edge_labels)):
                            ax_bar.text(i, weight + 0.01, f"{weight:.3f}", 
                                      ha='center', va='bottom', rotation=45, 
                                      fontsize=7, color='black')
                        
                        ax_bar.set_title(f'Top Edge Weights - Layer {layer_idx}', fontsize=12)
                        ax_bar.set_xlabel('Edge (i->j)')
                        ax_bar.set_ylabel('Attention Weight')
                        # x축 라벨 (너무 많으면 회전)
                        ax_bar.set_xticks(range(len(edge_labels)))
                        ax_bar.set_xticklabels(display_edge_labels, rotation=90, fontsize=8)
                        
                        # -----(B) Right Subplot: Network Graph)-----
                        ax_graph = axes[0, 1]
                        G = nx.DiGraph()
                        node_labels = {}

                        for i in range(n_nodes):
                            if i == 0:
                                node_name = "CLS"
                                node_color = "red"
                            else:
                                idx_feat = i - 1
                                if idx_feat < len(feature_names):
                                    node_name = feature_names[idx_feat]
                                    node_color = "blue"
                                else:
                                    node_name = f"feature_{i}"
                                    node_color = "blue"

                            G.add_node(i, name=node_name, color=node_color)
                            node_labels[i] = node_name

                        # CLS->Var / Var->Var 구분해서 그리기
                        cls_min_edge_weight = 0.001
                        min_edge_weight = 0.001
                        for i in range(n_nodes):
                            for j in range(n_nodes):
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
                        non_center_nodes = n_nodes - 1
                        radius = 1.0
                        for i_ in range(1, n_nodes):
                            angle_ = 2 * np.pi * (i_ - 1) / non_center_nodes
                            pos[i_] = np.array([radius * np.cos(angle_), radius * np.sin(angle_)])

                        # 배경 그리드
                        for r_ in [0.25, 0.5, 0.75, 1.0]:
                            circle = plt.Circle((0, 0), r_, fill=False, color='lightgray', linestyle='--', alpha=0.5)
                            ax_graph.add_patch(circle)
                        for i_ in range(1, n_nodes):
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
                                arrowstyle='-|>',  # 화살표 스타일 변경
                                arrowsize=15,      # 화살표 크기 키우기 (기본값보다 크게)
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

                        ax_graph.set_title(f'Graph Structure - Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=12)
                        ax_graph.axis('off')
                        ax_graph.set_aspect('equal')
                        ax_graph.set_xlim([-1.2, 1.2])
                        ax_graph.set_ylim([-1.2, 1.2])

                        # 3. 확장된 graph matrix heatmap
                        ax_graph_matrix = axes[1, 0]
                        graph_matrix_np = graph_matrix.cpu().numpy() 
                        im_graph = ax_graph_matrix.imshow(graph_matrix_np, cmap="Blues", interpolation='nearest')
                        ax_graph_matrix.set_title("Graph Matrix (with CLS)", fontsize=14)
                        fig.colorbar(im_graph, ax=ax_graph_matrix)

                        all_node_names = ["CLS"] + feature_names 
                        ax_graph_matrix.set_xticks(np.arange(len(all_node_names)))
                        ax_graph_matrix.set_yticks(np.arange(len(all_node_names)))
                        ax_graph_matrix.set_xticklabels(all_node_names, rotation=90, fontsize=8)
                        ax_graph_matrix.set_yticklabels(all_node_names, fontsize=8)

                        for i in range(len(all_node_names)):
                            for j in range(len(all_node_names)):
                                ax_graph_matrix.text(j, i, f"{graph_matrix_np[i,j]:.2f}", ha="center", va="center", color="black" if graph_matrix_np[i,j] < 0.5 else "white", fontsize=8)

                        ax_final = axes[1, 1]
                        vmax = final_graph_matrix.max()
                        vmin = 0.0  # 0부터 시작하도록 설정

                        # 다른 컬러맵 사용 및 범위 조정
                        im_final = ax_final.imshow(final_graph_matrix, 
                                                cmap='YlOrRd',  # 'YlOrRd', 'hot', 'OrRd' 등 시도해볼 수 있음
                                                interpolation='nearest',
                                                vmin=vmin, 
                                                vmax=vmax)
                        ax_final.set_title("Final Graph Matrix (Attention * Graph_matrix)", fontsize=14)
                        fig.colorbar(im_final, ax=ax_final)
                        
                        ax_final.set_xticks(np.arange(len(all_node_names)))
                        ax_final.set_yticks(np.arange(len(all_node_names)))
                        ax_final.set_xticklabels(all_node_names, rotation=90, fontsize=8)
                        ax_final.set_yticklabels(all_node_names, fontsize=8)
                        
                        # 각 셀에 값 표시
                        for i in range(len(all_node_names)):
                            for j in range(len(all_node_names)):
                                # 상대적인 값에 따라 텍스트 색상 결정 (0에 가까울수록 검정, 최대값에 가까울수록 흰색)
                                relative_value = final_graph_matrix[i,j] / vmax if vmax > 0 else 0
                                text_color = "black" if relative_value < 0.7 else "white"
                                
                                # 값이 0일 경우 빈 문자열 표시할 수도 있음
                                value_text = f"{final_graph_matrix[i,j]:.3f}"
                                
                                ax_final.text(j, i, value_text, 
                                            ha="center", va="center", 
                                            color=text_color, 
                                            fontsize=7)
                        
                        # 전체 제목 설정
                        fig.suptitle(f'Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=18)
                        fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # suptitle을 위한 여백 확보
                        
                        # # 레이어별 폴더에 저장
                        # layer_dir = os.path.join(sample_dirs[sample_count], f'layer_{layer_idx}')
                        # graph_path = os.path.join(layer_dir, f'epoch_{epoch}_complete.png')
                        # fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                        # plt.close(fig)
                        
                        # logger.info(f"샘플 {sample_count} - 레이어 {layer_idx} - 에포크 {epoch} 종합 시각화 저장: {graph_path}")
                        graph_path = os.path.join(sample_dirs[sample_count], 'graph', f'layer_{layer_idx}', f'epoch_{epoch}_complete.png')
                        fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"샘플 {sample_count} - 레이어 {layer_idx} - 에포크 {epoch} 그래프 시각화 저장: {graph_path}")
                sample_count += 1
                if sample_count >= max_samples:
                    break

            if sample_count >= max_samples:
                break




def visualize_gmm_clusters(gmm, embeddings, output_dir="visualizations/gmm_clusters", step=None, filename=None):
    """
    t-SNE 시각화만 제공하는 함수입니다.
    
    Args:
        gmm: GMM 또는 GMM2 모델 인스턴스
        embeddings: 입력 임베딩 [batch_size, input_dim]
        output_dir: 결과 저장 디렉토리
        step: 스텝 번호 (파일명용)
        filename: 사용자 지정 파일명 (선택사항)
    """
    try:
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 디바이스 설정
        device = embeddings.device
        
        # GMM으로 클러스터 할당 확률 계산
        if hasattr(gmm, 'forward'):
            with torch.no_grad():
                if 'GMM2' in gmm.__class__.__name__:
                    r, _, _ = gmm(embeddings, is_train=False)
                else:
                    r, _ = gmm(embeddings, is_train=False)
                
                # 가장 확률이 높은 클러스터 할당
                cluster_assignments = torch.argmax(r, dim=1).cpu().numpy()
                # 할당 확률 (신뢰도)
                confidences = torch.max(r, dim=1)[0].cpu().numpy()
                # 프로토타입 가져오기
                prototypes = gmm.running_prototypes.detach().cpu().numpy()
        else:
            # GMM 객체가 아닌 경우 (디버깅용)
            prototypes = np.random.randn(32, embeddings.shape[1])
            cluster_assignments = np.random.randint(0, 32, size=embeddings.shape[0])
            confidences = np.random.rand(embeddings.shape[0])
        
        # 임베딩을 CPU로 이동
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # prototypes가 3차원인 경우 2차원으로 변환
        if prototypes.ndim == 3:  # [1, num_prototypes, input_dim] 형태인 경우
            prototypes = prototypes.squeeze(0)  # [num_prototypes, input_dim] 형태로 변환
        
        # t-SNE를 사용하여 프로토타입과 임베딩을 2D로 투영
        combined_data = np.vstack([prototypes, embeddings_np])
        
        # t-SNE 차원 축소 (perplexity 조정)
        tsne = TSNE(n_components=2, perplexity=min(30, len(combined_data)-1), 
                   random_state=42, learning_rate=200)
        combined_reduced = tsne.fit_transform(combined_data)
        
        # 프로토타입과 임베딩으로 다시 분리
        reduced_prototypes = combined_reduced[:len(prototypes)]
        reduced_embeddings = combined_reduced[len(prototypes):]
        
        # ===== t-SNE 시각화만 생성 =====
        plt.figure(figsize=(12, 10))
        
        # 주요 프로토타입 추적 (시각화용)
        key_prototypes = set(cluster_assignments)
        
        # 클러스터 통계 계산
        cluster_counts = {}
        for c in cluster_assignments:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        
        largest_cluster = max(cluster_counts.items(), key=lambda x: x[1]) if cluster_counts else (-1, 0)
        empty_clusters = len(prototypes) - len(cluster_counts)
        
        # 프로토타입 시각화
        for i, (x, y) in enumerate(reduced_prototypes):
            # 주요 프로토타입 강조 (샘플이 할당된 것들)
            if i in key_prototypes:
                marker_size = 250
                edge_width = 2.5
                zorder = 10
                marker_style = '*'
                plt.scatter(x, y, s=marker_size, c=f'C{i % 10}', marker=marker_style, 
                           edgecolors='black', linewidths=edge_width, alpha=0.9,
                           zorder=zorder, label=f'P{i+1} (Main)')
                
                # 프로토타입 레이블 (눈에 띄게)
                plt.annotate(f'P{i+1}', (x, y), fontsize=14, fontweight='bold',
                            ha='center', va='center', color='white',
                            bbox=dict(boxstyle='round,pad=0.4', fc=f'C{i % 10}', alpha=0.8),
                            zorder=zorder+1)
            else:
                # 비활성 프로토타입 (샘플이 할당되지 않은 것들)
                marker_size = 100
                plt.scatter(x, y, s=marker_size, c=f'C{i % 10}', marker='o', 
                           alpha=0.4, zorder=5)
                plt.annotate(f'P{i+1}', (x, y), fontsize=9, ha='center', va='center', 
                            zorder=6)
        
        # 샘플 시각화 (다이아몬드 모양)
        for j, (x, y) in enumerate(reduced_embeddings):
            cluster_id = cluster_assignments[j]
            confidence = confidences[j]
            
            # 샘플 마커
            plt.scatter(x, y, s=250, c='white', marker='D', edgecolors='black', 
                       linewidths=2, alpha=0.9, zorder=15)
            
            # 샘플 ID
            plt.annotate(f'S{j+1}', (x, y), fontsize=12, fontweight='bold', 
                        ha='center', va='center', zorder=16)
            
            for i, (proto_x, proto_y) in enumerate(reduced_prototypes):
                if i == cluster_id:
                    # 할당 확률에 비례하는 선 두께
                    line_width = 1.5 + 5 * confidence
                    plt.plot([x, proto_x], [y, proto_y], '--', 
                            linewidth=line_width, alpha=0.7, 
                            c=f'C{i % 10}', zorder=1)
                    
                    # 할당 확률 표시
                    mid_x = (x + proto_x) * 0.6 + (proto_x + x) * 0.4
                    mid_y = (y + proto_y) * 0.6 + (proto_y + y) * 0.4
                    plt.annotate(f'{confidence:.2f}', (mid_x, mid_y), 
                                fontsize=11, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9),
                                zorder=7)
        
        # 클러스터링 통계 정보 추가
        info_text = f"Largest cluster: {largest_cluster[1]} samples (P{largest_cluster[0]+1})\n"
        info_text += f"Empty clusters: {empty_clusters}\n"
        
        for c, count in sorted(cluster_counts.items()):
            info_text += f"Cluster {c+1}: {count} samples\n"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                  bbox=dict(boxstyle='round', fc='whitesmoke', alpha=0.9))
        
        # 그래프 제목 및 레이블
        plt.title('t-SNE Visualization of Sample-Prototype Assignments', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if key_prototypes:
            plt.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # 파일 저장
        if filename is None:
            if step is not None:
                filename = f"tsne_only_{step}.png"
            else:
                filename = "tsne_only.png"
                
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to {os.path.join(output_dir, filename)}")
        
        return {
            'cluster_assignments': cluster_assignments,
            'confidences': confidences,
            'prototypes': reduced_prototypes,
            'embeddings': reduced_embeddings
        }
    
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        return None





def visualize_results(args, results, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # few-shot이 8 이상일 때는 few-shot 결과만 시각화
    if args.few_shot > 4:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Few-shot의 Train vs Valid
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Train Loss')
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_val_few_losses"], label='Valid Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.set_title('Few-shot: Train vs Valid Loss')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Train AUC')
        ax4.plot(results["Full_results"]["Ours_few"]["Ours_val_few_auc"], label='Valid AUC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('AUC')
        ax4.set_title('Few-shot: Train vs Valid AUC')
        ax4.legend()
        ax4.grid(True)

    # few-shot이 4일 때는 full과 few-shot 모두 시각화
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Full dataset의 Train vs Valid
        ax1.plot(results["Full_results"]["Ours"]["Ours_train_full_losses"], label='Train Loss')
        ax1.plot(results["Full_results"]["Ours"]["Ours_val_full_losses"], label='Valid Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Full Dataset: Train vs Valid Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(results["Full_results"]["Ours"]["Ours_train_full_auc"], label='Train AUC')
        ax2.plot(results["Full_results"]["Ours"]["Ours_val_full_auc"], label='Valid AUC')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('AUC')
        ax2.set_title('Full Dataset: Train vs Valid AUC')
        ax2.legend()
        ax2.grid(True)
        
        # Few-shot의 Train vs Valid
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Train Loss')
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_val_few_losses"], label='Valid Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.set_title('Few-shot: Train vs Valid Loss')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Train AUC')
        ax4.plot(results["Full_results"]["Ours_few"]["Ours_val_few_auc"], label='Valid AUC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('AUC')
        ax4.set_title('Few-shot: Train vs Valid AUC')
        ax4.legend()
        ax4.grid(True)

    plt.suptitle(f'Training Progress - {args.source_dataset_name} (K={args.few_shot})', y=1.02, fontsize=16)
    plt.tight_layout()
    metrics_plot_path = os.path.join(exp_dir, f"f{args.few_shot}_b{args.batch_size}_l{args.num_layers}_h{args.n_heads}_{timestamp}.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    print(f"Metrics plot saved as {metrics_plot_path}")