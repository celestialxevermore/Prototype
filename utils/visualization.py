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

def visualize_cluster_centroids(clustering_info, clustering_dir, epoch, feature_names):
    """
    각 클러스터의 centroid attention map을 히트맵으로 시각화
    클러스터별 폴더 구조로 정리
    """
    if clustering_info['cluster_centroids'] is None:
        return
    
    # visualizations 폴더 생성
    visualizations_dir = os.path.join(clustering_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
        
    for cluster_id, centroid in enumerate(clustering_info['cluster_centroids']):
        # 클러스터별 폴더 생성 (visualizations 하위에)
        cluster_folder = os.path.join(visualizations_dir, f'cluster_{cluster_id}')
        os.makedirs(cluster_folder, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        centroid_np = centroid.detach().cpu().numpy()
        all_node_names = ["CLS"] + feature_names
        
        im = ax.imshow(centroid_np, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Cluster {cluster_id} Centroid - Epoch {epoch}', fontsize=14)
        plt.colorbar(im, ax=ax)
        
        # 축 라벨 설정
        ax.set_xticks(np.arange(len(all_node_names)))
        ax.set_yticks(np.arange(len(all_node_names)))
        ax.set_xticklabels(all_node_names, rotation=90, fontsize=8)
        ax.set_yticklabels(all_node_names, fontsize=8)
        
        # 값 표시
        for i in range(len(all_node_names)):
            for j in range(len(all_node_names)):
                ax.text(j, i, f"{centroid_np[i,j]:.2f}", 
                       ha="center", va="center", 
                       color="white" if centroid_np[i,j] > 0.5 else "black", 
                       fontsize=6)
        
        # visualizations 폴더의 클러스터별 폴더에 저장
        centroid_viz_path = os.path.join(cluster_folder, f'epoch_{epoch}.png')
        fig.savefig(centroid_viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved centroid visualization for cluster {cluster_id}: {centroid_viz_path}")



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
                
                # 1. 히트맵 시각화 (원래 코드 그대로)
                if args.viz_heatmap:
                    # 시각화 시점에서만 클러스터링 리셋 (첫 번째 샘플에서만)
                    if sample_count == 0:
                        model.reset_epoch_clustering()
                        
                        # 현재 에포크의 데이터 수집을 위해 data_loader 순회
                        model.train()  # attention 수집용
                        with torch.no_grad():
                            for batch in data_loader:
                                batch_on_device = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                                _ = model.predict(batch_on_device)  # attention maps 수집
                        
                        model.stop_attention_collection()
                        model.eval()
                    
                    # 수집 완료 후 클러스터링 업데이트
                    clustering_updated = model.update_attention_clustering()
                    
                    # 클러스터링 정보 가져오기
                    clustering_info = model.get_clustering_info()
                    
                    # 1. 기존: 각 레이어별 시각화 (sample_*/heatmap/layer_*/에 저장)
                    for layer_idx in range(len(model.layers)):
                        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                        
                        # 1. Attention Map 히트맵 
                        batch_size = model.layers[layer_idx].attn_weights.shape[0]
                        actual_sample_idx = min(sample_idx, batch_size - 1)  # 배치 크기 초과 방지
                        attn_weights = model.layers[layer_idx].attn_weights[actual_sample_idx]  # [n_heads, seq, seq]
                        attn_weights_mean = attn_weights.mean(dim=0).cpu().numpy()  # 헤드별 평균
                        
                        # CLS 토큰 포함한 feature names
                        all_node_names = ["CLS"] + feature_names 
                        
                        im1 = axes[0].imshow(attn_weights_mean, cmap='viridis', interpolation='nearest')
                        axes[0].set_title(f'Attention Map - Layer {layer_idx}', fontsize=14)
                        fig.colorbar(im1, ax=axes[0])
                        
                        # 축 라벨 설정
                        axes[0].set_xticks(np.arange(len(all_node_names)))
                        axes[0].set_yticks(np.arange(len(all_node_names)))
                        axes[0].set_xticklabels(all_node_names, rotation=90, fontsize=8)
                        axes[0].set_yticklabels(all_node_names, fontsize=8)
                        
                        # 각 셀에 값 표시
                        for i in range(len(all_node_names)):
                            for j in range(len(all_node_names)):
                                axes[0].text(j, i, f"{attn_weights_mean[i,j]:.2f}", 
                                        ha="center", va="center", 
                                        color="white" if attn_weights_mean[i,j] > 0.5 else "black", 
                                        fontsize=7)
                        
                        # 2. 오른쪽: 해당하는 클러스터 centroid 표시
                        if (layer_idx == len(model.layers) - 1 and  # 마지막 레이어(Layer 2)이고
                            clustering_info['cluster_centroids'] is not None and 
                            len(clustering_info['cluster_assignments']) > 0):
                            
                            # 현재 샘플의 attention map이 어느 클러스터에 속하는지 찾기
                            sample_attention = attn_weights_mean  # 현재 샘플의 attention map
                            
                            # 모든 centroid와의 거리 계산
                            min_distance = float('inf')
                            assigned_cluster = 0
                            
                            for cluster_id, centroid in enumerate(clustering_info['cluster_centroids']):
                                centroid_np = centroid.detach().cpu().numpy()
                                # Frobenius norm 거리 계산
                                distance = np.linalg.norm(sample_attention - centroid_np, 'fro')
                                if distance < min_distance:
                                    min_distance = distance
                                    assigned_cluster = cluster_id
                            
                            # 해당 클러스터의 centroid 표시
                            assigned_centroid = clustering_info['cluster_centroids'][assigned_cluster].detach().cpu().numpy()
                            
                            im2 = axes[1].imshow(assigned_centroid, cmap='viridis', interpolation='nearest')
                            axes[1].set_title(f'Closest Cluster Centroid - Cluster {assigned_cluster}\n(Distance: {min_distance:.3f})', fontsize=14)
                            fig.colorbar(im2, ax=axes[1])
                            
                            # 축 라벨 설정
                            axes[1].set_xticks(np.arange(len(all_node_names)))
                            axes[1].set_yticks(np.arange(len(all_node_names)))
                            axes[1].set_xticklabels(all_node_names, rotation=90, fontsize=8)
                            axes[1].set_yticklabels(all_node_names, fontsize=8)
                            
                            # 각 셀에 값 표시
                            for i in range(len(all_node_names)):
                                for j in range(len(all_node_names)):
                                    axes[1].text(j, i, f"{assigned_centroid[i,j]:.2f}", 
                                            ha="center", va="center", 
                                            color="white" if assigned_centroid[i,j] > 0.5 else "black", 
                                            fontsize=7)
                        else:
                            # 마지막 레이어가 아니거나 클러스터링 데이터가 없는 경우 기존 방식
                            axes[1].text(0.5, 0.5, f'Layer {layer_idx} Attention Pattern\n\nFull clustering results\navailable in clustering/ folder\n\nLayer 2 = Final clustering layer', 
                                    ha='center', va='center', transform=axes[1].transAxes, fontsize=14,
                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
                            axes[1].set_title(f'Layer {layer_idx} - See clustering/ for full results', fontsize=14)
                            axes[1].axis('off')
                        
                        # 전체 타이틀
                        fig.suptitle(f'Layer {layer_idx} Attention Analysis - Epoch {epoch} - Sample {sample_count}', fontsize=16)
                        plt.tight_layout()
                        
                        # 기존 경로에 저장
                        heatmap_path = os.path.join(sample_dirs[sample_count], 'heatmap', f'layer_{layer_idx}', f'epoch_{epoch}.png')
                        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Epoch {epoch} - 샘플 {sample_count} 레이어 {layer_idx} 히트맵 저장: {heatmap_path}")

                    # 2. 전체 클러스터링 결과 (clustering/ 폴더에 저장) - 첫 번째 샘플에서만 생성
                    if sample_count == 0:  # 중복 방지: 첫 번째 샘플에서만 클러스터링 시각화 생성
                        # clustering 폴더 생성
                        clustering_dir = os.path.join(base_viz_dir, 'clustering')
                        os.makedirs(clustering_dir, exist_ok=True)
                        model.save_cluster_centroids(clustering_dir, epoch)
                        if clustering_info['cluster_centroids'] is not None:
                            visualize_cluster_centroids(clustering_info, clustering_dir, epoch, feature_names)
    
                        # 🆕 전체 데이터셋 클러스터링 결과 시각화 (Label 정보 포함)
                        if (clustering_info['cluster_centroids'] is not None and 
                            len(clustering_info['cluster_assignments']) > 0):
                            
                            cluster_assignments = clustering_info['cluster_assignments']
                            attention_maps = clustering_info['attention_maps']
                            attention_labels = clustering_info['attention_labels']  # 🆕 label 정보 가져오기
                            
                            if len(cluster_assignments) > 0 and len(attention_maps) > 0:
                                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                                
                                cluster_assignments = np.array(cluster_assignments)
                                attention_labels = np.array(attention_labels)  # 🆕 label 배열로 변환
                                
                                try:
                                    from sklearn.manifold import TSNE
                                    
                                    # attention maps를 numpy로 변환 (시각화용)
                                    attention_np = torch.stack(attention_maps).detach().cpu().numpy()
                                    n_maps, seq_len, seq_len2 = attention_np.shape
                                    
                                    # 평탄화해서 t-SNE 적용
                                    flattened_maps = attention_np.reshape(n_maps, -1)
                                    
                                    if n_maps >= 2:
                                        perplexity = min(30, n_maps-1, max(1, n_maps//3))
                                        
                                        # Centroid 처리 추가
                                        if clustering_info['cluster_centroids'] is not None:
                                            cluster_centroids = clustering_info['cluster_centroids']
                                            
                                            # Centroid를 numpy로 변환 (타입 확인)
                                            if isinstance(cluster_centroids, torch.Tensor):
                                                centroids_np = cluster_centroids.detach().cpu().numpy()
                                            else:
                                                # 리스트인 경우 stack
                                                centroids_np = torch.stack(cluster_centroids).detach().cpu().numpy()
                                            
                                            centroids_flat = centroids_np.reshape(len(centroids_np), -1)
                                            
                                            # 전체 데이터(attention maps + centroids)를 함께 t-SNE 변환
                                            all_data = np.vstack([flattened_maps, centroids_flat])
                                            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                                            tsne_all_embeddings = tsne.fit_transform(all_data)
                                            
                                            # 원본 데이터와 centroid 분리
                                            tsne_embeddings = tsne_all_embeddings[:n_maps]
                                            centroid_embeddings = tsne_all_embeddings[n_maps:]
                                        else:
                                            # Centroid가 없으면 기존 방식
                                            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                                            tsne_embeddings = tsne.fit_transform(flattened_maps)
                                            centroid_embeddings = None
                                        
                                        # 🆕 클러스터별로 다른 색상, label별로 명도 조절
                                        unique_clusters = np.unique(cluster_assignments)
                                        unique_labels = np.unique(attention_labels)
                                        
                                        # 클러스터별 기본 색상 설정
                                        base_colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_clusters), 1)))
                                        
                                        for i, cluster_id in enumerate(unique_clusters):
                                            cluster_mask = cluster_assignments == cluster_id
                                            cluster_points = tsne_embeddings[cluster_mask]
                                            cluster_labels = attention_labels[cluster_mask]
                                            
                                            if len(cluster_points) > 0:
                                                # 🆕 label별로 모양 구분: Label 0=원형, Label 1=네모
                                                for label in unique_labels:
                                                    label_mask = cluster_labels == label
                                                    if np.any(label_mask):
                                                        label_points = cluster_points[label_mask]
                                                        
                                                        # 모양 구분: label 0은 원형, label 1은 네모
                                                        if label == 0:
                                                            marker = 'o'  # 원형
                                                            marker_name = 'Label 0'
                                                        else:
                                                            marker = 's'  # 네모
                                                            marker_name = 'Label 1'
                                                        
                                                        ax.scatter(label_points[:, 0], label_points[:, 1], 
                                                                c=base_colors[i], 
                                                                label=f'Cluster {cluster_id} ({marker_name})', 
                                                                alpha=0.7, s=50, marker=marker)
                                        
                                        # Centroid를 별표로 표시
                                        if centroid_embeddings is not None:
                                            for i, cluster_id in enumerate(unique_clusters):
                                                if i < len(centroid_embeddings):
                                                    ax.scatter(centroid_embeddings[i, 0], centroid_embeddings[i, 1], 
                                                            marker='*', s=300, c='black', 
                                                            edgecolors=base_colors[i], linewidth=3,
                                                            label='Centroids' if i == 0 else "", zorder=5)
                                        
                                        ax.set_title(f'Dataset-wide Final Layer Clustering (Epoch {epoch})\nCumulative Layer 2 Attention Maps', fontsize=16)
                                        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
                                        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
                                        
                                        if len(unique_clusters) > 0:
                                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                        ax.grid(True, alpha=0.3)
                                        
                                        # 🆕 클러스터 및 label 통계 정보
                                        cluster_stats = []
                                        for cluster_id in unique_clusters:
                                            cluster_mask = cluster_assignments == cluster_id
                                            cluster_labels_subset = attention_labels[cluster_mask]
                                            total_count = np.sum(cluster_mask)
                                            total_percentage = (total_count / len(cluster_assignments)) * 100
                                            
                                            label_counts = {}
                                            for label in unique_labels:
                                                count = np.sum(cluster_labels_subset == label)
                                                label_counts[int(label)] = count
                                            
                                            label_str = ", ".join([f"L{k}:{v}" for k, v in label_counts.items()])
                                            cluster_stats.append(f"Cluster {cluster_id}: {total_count} maps ({total_percentage:.1f}%) [{label_str}]")
                                        
                                        if cluster_stats:
                                            stats_text = "\n".join(cluster_stats)
                                            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                                                fontsize=10, verticalalignment='top',
                                                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))
                                        
                                        # 전체 통계
                                        total_samples_processed = n_maps
                                        ax.text(0.02, 0.02, f"Total Layer 2 Maps: {total_samples_processed}\nEpoch: {epoch}\nUpdate Freq: {model.clustering_update_freq}\nCircle=Label 0, Square=Label 1", 
                                            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                                        
                                        plt.tight_layout()
                                        
                                        # clustering 폴더에 저장
                                        clustering_path = os.path.join(clustering_dir, f'epoch_{epoch}.png')
                                        fig.savefig(clustering_path, dpi=300, bbox_inches='tight')
                                        plt.close(fig)
                                        logger.info(f"Epoch {epoch} - 전체 데이터셋 클러스터링 저장: {clustering_path}")
                                        
                                    else:
                                        # 데이터 부족한 경우
                                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                                        ax.text(0.5, 0.5, f'Dataset-wide Clustering (Epoch {epoch})\n\nNeed more data for t-SNE\nCurrent maps: {n_maps}\nMinimum required: 2', 
                                            ha='center', va='center', transform=ax.transAxes, fontsize=14,
                                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                                        ax.set_title(f'Dataset-wide Final Layer Clustering (Epoch {epoch})', fontsize=16)
                                        ax.axis('off')
                                        
                                        clustering_path = os.path.join(clustering_dir, f'epoch_{epoch}.png')
                                        fig.savefig(clustering_path, dpi=300, bbox_inches='tight')
                                        plt.close(fig)
                                        logger.info(f"Epoch {epoch} - 클러스터링 대기 상태 저장: {clustering_path}")
                                        
                                except Exception as e:
                                    logger.error(f"Clustering visualization error: {e}")
                                    
                            else:
                                # 클러스터링 데이터 없는 경우
                                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                                total_maps = clustering_info.get('attention_count', 0)
                                ax.text(0.5, 0.5, f'Dataset-wide Clustering (Epoch {epoch})\n\nNo clustering data available\nTotal maps collected: {total_maps}\nRequired: {model.num_clusters}', 
                                    ha='center', va='center', transform=ax.transAxes, fontsize=14,
                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
                                ax.set_title(f'Dataset-wide Final Layer Clustering (Epoch {epoch})', fontsize=16)
                                ax.axis('off')
                                
                                clustering_path = os.path.join(clustering_dir, f'epoch_{epoch}.png')
                                fig.savefig(clustering_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)
                                logger.info(f"Epoch {epoch} - 클러스터링 없음 상태 저장: {clustering_path}")


                # 2. 그래프 구조 시각화
                if args.viz_graph:
                    # 각 레이어별로 시각화 수행
                    for layer_idx in range(len(model.layers)):
                        # 1) Attention 가중치(헤드 평균)
                        attn_weights = model.layers[layer_idx].attn_weights[sample_idx]  # [n_heads, seq, seq]
                        attn_weights_mean = attn_weights.mean(dim=0).cpu()

                        # 원본 adjacency 사용 (히트맵과 일치하는 값)
                        adjacency = model.layers[layer_idx].adjacency[sample_idx].cpu()
                        # adj_row_sums = adjacency.sum(axis=1, keepdims=True) + 1e-9
                        # adjacency = adjacency / adj_row_sums
                        new_seq = attn_weights_mean.shape[0]
                        graph_matrix = torch.zeros((new_seq, new_seq), device=attn_weights_mean.device, dtype = torch.float)

                        graph_matrix[1:, 1:] = adjacency  # 변수 간 연결은 원본 adjacency 사용
                        graph_matrix[0, 1:] = 1.0  # CLS->변수 연결
                        graph_matrix[1:, 0] = 0.0  # 변수->CLS 연결
                        
                        mask = (graph_matrix == 0)
                        final_graph_matrix = (attn_weights_mean * graph_matrix).numpy()
                        final_graph_matrix[mask.numpy()] = 0.0
                        # row_sums = final_graph_matrix.sum(axis=1, keepdims=True)
                        # final_graph_matrix = final_graph_matrix / (row_sums + 1e-9)  # stability 위해 1e-9 더함 
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