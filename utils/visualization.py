import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pdb
import torch




import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_attention_graph(attention_weights, save_path=None, title="Graph"):
    """
    어텐션 가중치 행렬을 사용하여 그래프 구조를 시각화합니다.
    
    Parameters:
    -----------
    attention_weights : torch.Tensor
        어텐션 가중치 행렬, 마지막 두 차원이 노드 간 관계를 나타냄 (batch_size, ...)
    save_path : str, optional
        결과를 저장할 경로, None이면 화면에 표시
    title : str
        그래프 제목
    """
    # 배치에서 첫 번째 항목만 사용
    if isinstance(attention_weights, torch.Tensor):
        if attention_weights.dim() > 3:
            # 여러 헤드가 있는 경우 최대값을 취함
            if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
                attn = attention_weights[0].max(dim=0)[0].cpu().numpy()
            else:
                attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights[0].cpu().numpy()
    else:
        attn = attention_weights
    
    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (CLS 토큰 + 특성 노드들)
    n_nodes = attn.shape[0]
    
    # 노드 이름 설정 (CLS + 특성1, 특성2, ...)
    node_names = ['center'] + [f'feature_{i}' for i in range(1, n_nodes)]
    
    # 노드 추가
    for i, name in enumerate(node_names):
        G.add_node(i, name=name)
    
    # 엣지 추가 (0보다 큰 가중치를 가진 엣지만 - 자기 자신과의 연결 제외)
    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = attn[i, j]
            if weight > 0 and i != j:  # 자기 자신과의 연결 제외
                G.add_edge(i, j, weight=float(weight))
    
    # 그래프 그리기
    plt.figure(figsize=(10, 10))
    
    # 노드 위치 계산 (원형 레이아웃)
    if n_nodes <= 1:
        pos = {0: (0, 0)}
    else:
        # CLS 토큰은 중앙에 배치
        pos = {0: (0, 0)}
        
        # 나머지 노드는 원 위에 배치
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        for i in range(1, n_nodes):
            pos[i] = (np.cos(angles[i]), np.sin(angles[i]))
    
    # 노드 색상 설정 (CLS=빨강, 나머지=파랑)
    node_colors = ['red' if i == 0 else 'blue' for i in range(n_nodes)]
    
    # 엣지 가중치 기반 굵기 계산
    min_width = 1.0
    max_width = 8.0
    
    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        min_weight = min(weights)
        max_weight = max(weights)
        
        # 가중치 정규화 및 선 굵기 계산
        if max_weight > min_weight:
            edge_weights = [min_width + (max_width - min_width) * 
                           (G[u][v]['weight'] - min_weight) / (max_weight - min_weight) 
                           for u, v in G.edges()]
        else:
            edge_weights = [min_width + (max_width - min_width) / 2 for _ in G.edges()]
    else:
        edge_weights = []
    
    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos, labels={i: node_names[i] for i in range(n_nodes)})
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_weights,
        arrows=True, 
        arrowsize=20,
        edge_color='gray',
        alpha=0.7
    )
    
    plt.title(title)
    plt.axis('off')
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_attention_weights(model, batch, feature_names=None, threshold=0.1, save_path=None):
    """
    모델의 어텐션 가중치를 시각화합니다.
    
    Parameters:
    -----------
    model : nn.Module
        시각화할 모델
    batch : dict
        모델에 전달할 입력 배치
    feature_names : list, optional
        특성 이름 목록
    threshold : float
        이 값보다 작은 가중치를 가진 엣지는 표시하지 않음
    save_path : str, optional
        결과를 저장할 경로
    """
    model.eval()
    with torch.no_grad():
        # 모델의 첫 번째 레이어에서 어텐션 가중치 추출
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # 예측 실행
            _ = model.predict(batch)
            
            # 어텐션 가중치 가져오기
            if hasattr(model.layers[0], 'adjacency'):
                attention_weights = model.layers[0].adjacency
            # 특성 이름이 없으면 기본값 사용
            if feature_names is None:
                num_features = attention_weights.shape[1] - 1  # CLS 토큰 제외
                feature_names = ['CLS'] + [f'Feature {i+1}' for i in range(num_features)]
            
            # 그래프 시각화
            title = "Graph"
            visualize_attention_graph(attention_weights, save_path, title)
        else:
            raise ValueError("모델에 layers 속성이 없거나 비어 있습니다.")


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