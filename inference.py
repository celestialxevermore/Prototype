"""
Attention Maps Inference 스크립트

학습 완료된 모델을 로드하여 전체 데이터셋에 대해 attention maps를 추출하고
K-means 클러스터링을 수행합니다.

Usage:
    python inference.py --checkpoint_dir /path/to/checkpoint.pt --mode Full
"""

import os
# CUDA deterministic 설정을 가장 먼저 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import argparse
import numpy as np
import logging
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx

# 기존 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionInference:
    def __init__(self, checkpoint_dir, device='cuda'):
        """
        Args:
            checkpoint_dir (str): 체크포인트 파일 경로
            device (str): 'cuda' 또는 'cpu'
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Val AUC: {self.checkpoint['val_auc']:.4f}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
    def _load_model(self):
        """체크포인트에서 모델 로드"""
        # experiment_id와 mode는 체크포인트 경로에서 추출하거나 기본값 사용
        experiment_id = "inference"
        mode = "inference"
        
        self.model = Model(
            self.args, 
            self.args.input_dim, 
            self.args.hidden_dim, 
            self.args.output_dim, 
            self.args.num_layers, 
            self.args.dropout_rate, 
            self.args.llm_model,
            experiment_id,
            mode
        ).to(self.device)
        
        # 모델 가중치 로드
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded and set to evaluation mode")
    
    def _prepare_dataloaders(self):
        """데이터로더 준비"""
        fix_seed(self.args.random_seed)
        
        # 전체 데이터셋 로더 준비
        results = prepare_embedding_dataloaders(self.args, self.args.source_dataset_name)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']
        self.num_classes = results['num_classes']
        
        # Few-shot 로더 준비 (필요한 경우)
        if hasattr(self.args, 'few_shot') and self.args.few_shot > 0:
            self.train_loader_few = get_few_shot_embedding_samples(self.train_loader, self.args)
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_dataset_name}")
    
    def extract_attention_maps(self, data_loader):
        """
        데이터로더에서 attention maps 추출 (전체 데이터)
        
        Args:
            data_loader: 데이터로더
            
        Returns:
            dict: 레이어별 attention maps와 메타데이터
        """
        attention_data = {
            'layer_0': [],
            'layer_1': [], 
            'layer_2': [],
            'labels': [],
            'sample_ids': [],
            'feature_names': None
        }
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 배치를 디바이스로 이동
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 모델 forward (attention weights 추출을 위해 수정된 predict 호출)
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                
                # Feature names 추출 (첫 번째 배치에서만)
                if attention_data['feature_names'] is None:
                    feature_names = self.model.extract_feature_names(batch_on_device)
                    attention_data['feature_names'] = ["CLS"] + feature_names
                
                # 배치 크기 확인
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    # 각 레이어별 attention map 저장
                    for layer_idx, layer_attention in enumerate(attention_weights):
                        # Multi-head attention을 평균내어 단일 attention map으로 변환
                        attention_map = layer_attention[sample_idx].mean(dim=0)  # [seq_len, seq_len]
                        attention_numpy = attention_map.detach().cpu().numpy()
                        attention_data[f'layer_{layer_idx}'].append(attention_numpy)
                    
                    # 라벨과 샘플 ID 저장
                    if 'y' in batch:
                        label = batch['y'][sample_idx].item()
                    else:
                        label = -1  # 라벨이 없는 경우
                    attention_data['labels'].append(label)
                    
                    # 샘플 ID (실제 데이터셋의 인덱스 또는 카운터)
                    if 'sample_ids' in batch:
                        sample_id = batch['sample_ids'][sample_idx]
                    else:
                        sample_id = sample_count
                    attention_data['sample_ids'].append(sample_id)
                    
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        
        logger.info(f"Extracted attention maps for {sample_count} samples (FULL DATASET)")
        return attention_data
    
    def _extract_attention_from_model(self, batch):
        """
        모델에서 attention weights와 예측값을 추출
        (predict 메서드를 수정하지 않고 직접 forward 로직 사용)
        """
        # 모델의 predict 로직을 복사하되 attention_weights도 반환
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        desc_embeddings = [] 
        name_value_embeddings = [] 
        
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            cat_name_value_embeddings = batch['cat_name_value_embeddings'].to(self.device).squeeze(-2)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
            
            name_value_embeddings.append(cat_name_value_embeddings)
            desc_embeddings.append(cat_desc_embeddings)
            
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
            name_value_embeddings.append(num_prompt_embeddings)
            desc_embeddings.append(num_desc_embeddings)
            
        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim = 1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim = 1)
        
        # [CLS] Token 추가
        attention_weights = [] 
        cls_token = self.model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        # Graph Attention Layers
        for i, layer in enumerate(self.model.layers):
            norm_x = self.model.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_weights)
            x = x + attn_output
        
        # 예측값 계산
        pred = x[:, 0, :]
        pred = self.model.predictor(pred)

        return pred, attention_weights

    def visualize_graph_structure(self, data_loader, output_dir, max_samples=10):
        """
        그래프 구조를 시각화 (max_samples 개수만큼만)
        
        Args:
            data_loader: 데이터로더
            output_dir (str): 저장할 디렉토리
            max_samples (int): 그래프 시각화할 최대 샘플 수
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        epoch = self.checkpoint['epoch']  # 체크포인트에서 에포크 정보 가져오기
        
        # 샘플별 디렉토리 생성
        sample_dirs = {}
        for i in range(max_samples):
            sample_dir = output_dir / f'sample_{i}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 각 샘플의 graph 디렉토리와 레이어별 디렉토리 생성
            graph_dir = sample_dir / 'graph'
            graph_dir.mkdir(parents=True, exist_ok=True)
            
            for layer_idx in range(len(self.model.layers)):
                layer_dir = graph_dir / f'layer_{layer_idx}'
                layer_dir.mkdir(parents=True, exist_ok=True)
            
            sample_dirs[i] = sample_dir
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= max_samples:
                    break
                    
                # 배치를 디바이스로 이동
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 모델 forward pass
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                
                # Feature names 추출
                feature_names = self.model.extract_feature_names(batch_on_device)
                
                # 배치 크기 확인
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    # 모델의 각 레이어에서 attention weights와 adjacency 가져오기
                    for layer_idx in range(len(self.model.layers)):
                        # Attention 가중치(헤드 평균)
                        attn_weights = self.model.layers[layer_idx].attn_weights[sample_idx]  # [n_heads, seq, seq]
                        attn_weights_mean = attn_weights.mean(dim=0).cpu()

                        # 원본 adjacency 사용
                        adjacency = self.model.layers[layer_idx].adjacency[sample_idx].cpu()
                        new_seq = attn_weights_mean.shape[0]
                        graph_matrix = torch.zeros((new_seq, new_seq), device=attn_weights_mean.device, dtype=torch.float)

                        graph_matrix[1:, 1:] = adjacency  # 변수 간 연결은 원본 adjacency 사용
                        graph_matrix[0, 1:] = 1.0  # CLS->변수 연결
                        graph_matrix[1:, 0] = 0.0  # 변수->CLS 연결
                        
                        mask = (graph_matrix == 0)
                        final_graph_matrix = (attn_weights_mean * graph_matrix).numpy()
                        final_graph_matrix[mask.numpy()] = 0.0
                        n_nodes = final_graph_matrix.shape[0]
                        
                        # Edge 리스트 수집
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
                        
                        ax_bar.set_title(f'Top Edge Weights - Layer {layer_idx}', fontsize=12)
                        ax_bar.set_xlabel('Edge (i->j)')
                        ax_bar.set_ylabel('Attention Weight')
                        ax_bar.set_xticks(range(len(edge_labels)))
                        ax_bar.set_xticklabels(display_edge_labels, rotation=90, fontsize=8)
                        
                        # (B) Network Graph
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

                        ax_graph.set_title(f'Graph Structure - Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=12)
                        ax_graph.axis('off')
                        ax_graph.set_aspect('equal')
                        ax_graph.set_xlim([-1.2, 1.2])
                        ax_graph.set_ylim([-1.2, 1.2])

                        # (C) Graph matrix heatmap
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
                                ax_graph_matrix.text(j, i, f"{graph_matrix_np[i,j]:.2f}", ha="center", va="center", 
                                                   color="black" if graph_matrix_np[i,j] < 0.5 else "white", fontsize=8)

                        # (D) Final graph matrix
                        ax_final = axes[1, 1]
                        vmax = final_graph_matrix.max()
                        vmin = 0.0

                        im_final = ax_final.imshow(final_graph_matrix, 
                                                cmap='YlOrRd',
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
                                relative_value = final_graph_matrix[i,j] / vmax if vmax > 0 else 0
                                text_color = "black" if relative_value < 0.7 else "white"
                                value_text = f"{final_graph_matrix[i,j]:.3f}"
                                
                                ax_final.text(j, i, value_text, 
                                            ha="center", va="center", 
                                            color=text_color, 
                                            fontsize=7)
                        
                        # 전체 제목 설정
                        fig.suptitle(f'Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=18)
                        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                        
                        # 저장
                        graph_path = sample_dirs[sample_count] / 'graph' / f'layer_{layer_idx}' / '_complete.png'
                        fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        logger.info(f"샘플 {sample_count} - 레이어 {layer_idx} - 에포크 {epoch} 그래프 시각화 저장: {graph_path}")
                    
                    sample_count += 1
                    if sample_count >= max_samples:
                        break

                if sample_count >= max_samples:
                    break
        
        logger.info(f"Graph visualization completed for {sample_count} samples")

    def save_attention_maps_by_cluster(self, attention_data, clustering_results, output_dir, layer_idx):
        """클러스터별로 attention maps 저장"""
        layer_dir = Path(output_dir) / f'layer_{layer_idx}'
        
        # 클러스터별 폴더 생성
        for cluster_id in range(len(np.unique(clustering_results['cluster_assignments']))):
            cluster_dir = layer_dir / f'cluster_{cluster_id}'
            cluster_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 샘플을 해당 클러스터 폴더에 저장
        for sample_idx, cluster_id in enumerate(clustering_results['cluster_assignments']):
            attention_map = attention_data[f'layer_{layer_idx}'][sample_idx]
            sample_id = attention_data['sample_ids'][sample_idx]
            label = attention_data['labels'][sample_idx]
            
            cluster_dir = layer_dir / f'cluster_{cluster_id}'
            filename = f'sample_{sample_id}_label_{label}.npz'
            
            np.savez(cluster_dir / filename,
                    attention_map=attention_map,
                    feature_names=np.array(attention_data['feature_names']),
                    sample_id=sample_id,
                    label=label,
                    cluster_id=cluster_id)

    def save_attention_maps(self, attention_data, output_dir):
        """
        Attention maps를 NPZ 파일들로 저장
        
        Args:
            attention_data (dict): extract_attention_maps에서 반환된 데이터
            output_dir (str): 저장할 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_samples = len(attention_data['labels'])
        num_layers = len([k for k in attention_data.keys() if k.startswith('layer_')])
        
        for sample_idx in range(num_samples):
            sample_id = attention_data['sample_ids'][sample_idx]
            label = attention_data['labels'][sample_idx]
            
            for layer_idx in range(num_layers):
                attention_map = attention_data[f'layer_{layer_idx}'][sample_idx]
                
                filename = f"layer_{layer_idx}_sample_{sample_id}_label_{label}.npz"
                filepath = output_dir / filename
                
                np.savez(filepath,
                        attention_map=attention_map,
                        feature_names=np.array(attention_data['feature_names']),
                        layer_idx=layer_idx,
                        sample_id=sample_id,
                        label=label)
        
        logger.info(f"Saved {num_samples * num_layers} attention map files to {output_dir}")
    
    def perform_clustering(self, attention_data, layer_idx=2, n_clusters=5, output_dir=None):
        """
        특정 레이어의 attention maps에 대해 K-means 클러스터링 수행
        
        Args:
            attention_data (dict): attention maps 데이터
            layer_idx (int): 클러스터링할 레이어 인덱스
            n_clusters (int): 클러스터 수
            output_dir (str, optional): 시각화 결과 저장 디렉토리
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 특정 레이어의 attention maps 추출
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        sample_ids = np.array(attention_data['sample_ids'])
        feature_names = attention_data['feature_names']
        
        logger.info(f"Performing clustering on layer {layer_idx} with {len(attention_maps)} samples")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(flattened_maps)
        
        # 클러스터링 결과 출력
        unique_labels = np.unique(labels)
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_samples = np.sum(cluster_mask)
            
            label_dist = {}
            for label in unique_labels:
                count = np.sum((cluster_assignments == cluster_id) & (labels == label))
                label_dist[f'label_{label}'] = count
            
            logger.info(f"Cluster {cluster_id}: {cluster_samples} samples, distribution: {label_dist}")
        
        # 클러스터링 시각화 (t-SNE)
        if len(flattened_maps) >= 2 and output_dir:
            self._visualize_clustering_distribution(flattened_maps, cluster_assignments, labels, 
                                                  layer_idx, output_dir)
        
        # 클러스터 센트로이드 시각화 (1x5 플롯)
        if output_dir:
            self._visualize_cluster_centroids(kmeans.cluster_centers_, feature_names, 
                                            layer_idx, output_dir, n_clusters)
        
        return {
            'cluster_assignments': cluster_assignments,
            'cluster_centers': kmeans.cluster_centers_,
            'labels': labels,
            'sample_ids': sample_ids,
            'layer_idx': layer_idx
        }
    
    def _visualize_clustering_distribution(self, flattened_maps, cluster_assignments, labels, 
                                 layer_idx, output_dir):
        """클러스터링 분포 시각화 (t-SNE) - 두 번째 코드 스타일 적용"""
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 하나의 큰 플롯
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 클러스터별 색상과 라벨별 마커 조합
        unique_clusters = np.unique(cluster_assignments)
        unique_labels = np.unique(labels)
        
        # 클러스터별 기본 색상 설정 (두 번째 코드 스타일)
        base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
        
        # 클러스터와 라벨 조합으로 시각화 (두 번째 코드 스타일)
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = tsne_embeddings[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            if len(cluster_points) > 0:
                # 라벨별로 모양 구분: Label 0=원형, Label 1=네모 (두 번째 코드와 동일)
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
                                color=base_colors[i], 
                                label=f'Cluster {cluster_id} ({marker_name})', 
                                alpha=0.7, s=50, marker=marker)
        
        # 클러스터 센트로이드 추가 (별표로 표시) - 두 번째 코드 스타일
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            if np.any(cluster_mask):
                # 해당 클러스터 포인트들의 평균 위치 (t-SNE 공간에서)
                centroid_x = np.mean(tsne_embeddings[cluster_mask, 0])
                centroid_y = np.mean(tsne_embeddings[cluster_mask, 1])
                
                ax.scatter(centroid_x, centroid_y, marker='*', s=300, 
                        c='black', edgecolors=base_colors[i], linewidth=3,
                        label='Centroids' if i == 0 else "", zorder=5)
        
        # 제목과 축 라벨 (두 번째 코드 스타일)
        ax.set_title(f'Layer {layer_idx} - Clustering & True Labels\nK-means with t-SNE Visualization', 
                    fontsize=16, pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        # 범례 (두 번째 코드 스타일)
        if len(unique_clusters) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 클러스터 및 label 통계 정보 (두 번째 코드 스타일)
        cluster_stats = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_labels_subset = labels[cluster_mask]
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
        
        # 마커 설명 추가 (두 번째 코드 스타일)
        total_samples_processed = len(flattened_maps)
        ax.text(0.02, 0.02, f"Total Maps: {total_samples_processed}\nCircle=Label 0, Square=Label 1", 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_clustering_distribution.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"✅ Layer {layer_idx} distribution saved!")

    def _visualize_cluster_centroids(self, cluster_centers, feature_names, layer_idx, output_dir, n_clusters):
        """클러스터 센트로이드 히트맵 시각화 (모든 클러스터 지원 + 폴더 정리)"""
        seq_len = len(feature_names)
        centroids_reshaped = cluster_centers.reshape(-1, seq_len, seq_len)
        
        # centroid 폴더 생성
        centroid_dir = output_dir / 'centroid'
        centroid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 전체 overview: 동적 레이아웃 결정 (제한 없음)
        if n_clusters <= 5:
            # 5개 이하: 1행으로 배치
            rows, cols = 1, n_clusters
        elif n_clusters <= 10:
            # 6-10개: 2행으로 배치
            rows, cols = 2, (n_clusters + 1) // 2
        elif n_clusters <= 15:
            # 11-15개: 3행으로 배치
            rows, cols = 3, (n_clusters + 2) // 3
        elif n_clusters <= 20:
            # 16-20개: 4행으로 배치
            rows, cols = 4, (n_clusters + 3) // 4
        else:
            # 20개 초과: 5행으로 배치
            rows, cols = 5, (n_clusters + 4) // 5
        
        # 적절한 크기 계산
        subplot_width = 4
        subplot_height = 4
        figsize = (subplot_width * cols, subplot_height * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # axes를 1차원 배열로 변환
        if n_clusters == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            if hasattr(axes, 'flatten'):
                axes = axes.flatten()
            else:
                axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        else:
            axes = axes.flatten()
        
        # 모든 클러스터 시각화 (제한 없음!)
        for i, centroid in enumerate(centroids_reshaped):
            ax = axes[i]
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Cluster {i} Centroid', fontsize=11)
            
            # 축 라벨 설정
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
            ax.set_yticklabels(feature_names, fontsize=8)
            
            # 숫자는 표시하지 않음 (깔끔한 overview용)
            
            # 컬러바 추가
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_idx} Cluster Centroids (All {n_clusters} clusters)', fontsize=14)
        plt.tight_layout()
        
        # overview는 메인 폴더에 저장
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_centroids_overview.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"All {n_clusters} cluster centroids overview saved: layer_{layer_idx}_cluster_centroids_overview.png")
        
        # 2. 각 센트로이드별로 개별 상세 플롯 생성 (centroid 폴더에 저장)
        for i, centroid in enumerate(centroids_reshaped):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Cluster {i} Centroid - Layer {layer_idx}', fontsize=16, pad=20)
            
            # 축 라벨 설정 (더 큰 폰트)
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=12)
            ax.set_yticklabels(feature_names, fontsize=12)
            
            # 각 셀에 값 표시 (상세 버전)
            for row in range(len(feature_names)):
                for col in range(len(feature_names)):
                    value = centroid[row, col]
                    ax.text(col, row, f"{value:.2f}", 
                        ha="center", va="center", 
                        color="white" if value > 0.15 else "black", 
                        fontsize=10, weight='bold')
            
            # 컬러바 추가
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Attention Weight', fontsize=12)
            
            plt.tight_layout()
            
            # centroid 폴더에 저장
            fig.savefig(centroid_dir / f'cluster_{i}_centroid.png', 
                    dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"All {n_clusters} detailed centroids saved in {centroid_dir}")
        logger.info(f"All cluster centroids saved for layer {layer_idx}")


    # 추가: 클러스터 통계 요약 함수 (옵션)
    def _visualize_cluster_summary(self, cluster_centers, feature_names, layer_idx, output_dir, n_clusters):
        """클러스터 통계 요약 시각화 (선택적 사용)"""
        seq_len = len(feature_names)
        centroids_reshaped = cluster_centers.reshape(-1, seq_len, seq_len)
        
        # 각 센트로이드의 주요 통계 계산
        cluster_stats = []
        for i, centroid in enumerate(centroids_reshaped):
            stats = {
                'cluster_id': i,
                'max_attention': np.max(centroid),
                'mean_attention': np.mean(centroid),
                'std_attention': np.std(centroid),
                'max_position': np.unravel_index(np.argmax(centroid), centroid.shape)
            }
            cluster_stats.append(stats)
        
        # 통계 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 최대 attention 값
        cluster_ids = [s['cluster_id'] for s in cluster_stats]
        max_attentions = [s['max_attention'] for s in cluster_stats]
        axes[0, 0].bar(cluster_ids, max_attentions, color='skyblue')
        axes[0, 0].set_title('Maximum Attention per Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Max Attention')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 평균 attention 값
        mean_attentions = [s['mean_attention'] for s in cluster_stats]
        axes[0, 1].bar(cluster_ids, mean_attentions, color='lightgreen')
        axes[0, 1].set_title('Mean Attention per Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Mean Attention')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 표준편차
        std_attentions = [s['std_attention'] for s in cluster_stats]
        axes[1, 0].bar(cluster_ids, std_attentions, color='lightcoral')
        axes[1, 0].set_title('Attention Std Dev per Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Std Dev')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 요약 텍스트
        axes[1, 1].axis('off')
        summary_text = f"Layer {layer_idx} Cluster Summary\n\n"
        summary_text += f"Total Clusters: {n_clusters}\n"
        summary_text += f"Feature Matrix: {seq_len}x{seq_len}\n\n"
        
        # 상위 3개 클러스터 (최대 attention 기준)
        top_clusters = sorted(cluster_stats, key=lambda x: x['max_attention'], reverse=True)[:min(3, n_clusters)]
        summary_text += "Top Clusters (by max attention):\n"
        for rank, cluster in enumerate(top_clusters, 1):
            row, col = cluster['max_position']
            summary_text += f"{rank}. Cluster {cluster['cluster_id']}: "
            summary_text += f"Max={cluster['max_attention']:.3f}\n"
            summary_text += f"   at {feature_names[row]}→{feature_names[col]}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx} Cluster Analysis Summary', fontsize=16)
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_summary.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Cluster summary saved: layer_{layer_idx}_cluster_summary.png")

def main():
    parser = argparse.ArgumentParser(description='Attention Maps Inference')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                       help='Which model to use (Full or Few)')
    parser.add_argument('--layer_idx', type=int, default=2,
                       help='Layer index for clustering (default: 2)')
    parser.add_argument('--n_clusters', type=int, default=4,
                       help='Number of clusters for K-means')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples for graph visualization ONLY')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--save_attention_maps', action='store_true',
                       help='Save individual attention map files')
    parser.add_argument('--viz_graph', action='store_true',
                       help='Generate graph structure visualization')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정 (visualization 폴더 구조에 맞게)
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        # 체크포인트 경로에서 정보 추출
        path_parts = checkpoint_dir.parts
        
        # checkpoints/.../gpt2_mean/heart/Full/20250606_183958 에서
        # visualization/.../gpt2_mean/heart/Full/20250606_183958 로 변경
        for i, part in enumerate(path_parts):
            if part == 'checkpoints':
                viz_parts = list(path_parts)
                viz_parts[i] = 'visualization'
                viz_path = Path(*viz_parts[:-1])  # best_model_epoch_XX.pt 제외
                args.output_dir = viz_path / f'clustering_{args.n_clusters}'
                break
        
        if args.output_dir is None:
            args.output_dir = checkpoint_dir.parent / 'inference_results'
    
    # Inference 실행
    inference = AttentionInference(args.checkpoint_dir)
    
    # 데이터로더 선택
    if args.mode == 'Full':
        data_loader = inference.train_loader
        logger.info("Using Full dataset loader")
    else:
        data_loader = inference.train_loader_few if hasattr(inference, 'train_loader_few') else inference.test_loader
        logger.info("Using Few-shot dataset loader")
    
    # 그래프 시각화 (max_samples 개수만큼만)
    if args.viz_graph:
        logger.info(f"Generating graph visualizations for {args.max_samples} samples...")
        graph_output_dir = Path(args.output_dir) / 'graph_visualization'
        inference.visualize_graph_structure(data_loader, graph_output_dir, args.max_samples)
    
    # Attention maps 추출 (전체 데이터에 대해 무조건)
    logger.info("Extracting attention maps for clustering (FULL DATASET)...")
    attention_data = inference.extract_attention_maps(data_loader)
    
    # Attention maps 저장 (옵션)
    if args.save_attention_maps:
        save_dir = Path(args.output_dir) / 'attention_maps'
        inference.save_attention_maps(attention_data, save_dir)

    # 모든 레이어에 대해 클러스터링
    for layer_idx in range(len(inference.model.layers)):
        logger.info(f"Performing clustering on layer {layer_idx}...")
        layer_output_dir = Path(args.output_dir) / f'layer_{layer_idx}'
        clustering_results = inference.perform_clustering(
            attention_data, 
            layer_idx=layer_idx,
            n_clusters=args.n_clusters,
            output_dir=layer_output_dir
        )
        logger.info(f"Saving attention maps by cluster for layer {layer_idx}...")
        inference.save_attention_maps_by_cluster(
            attention_data, 
            clustering_results, 
            args.output_dir,  # 메인 output_dir 사용
            layer_idx
    )
    logger.info(f"Inference completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()