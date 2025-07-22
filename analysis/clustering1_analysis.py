"""
Attention Maps Inference 스크립트

학습 완료된 모델을 로드하여 전체 데이터셋에 대해 attention maps를 추출하고
K-means 클러스터링을 수행합니다.

Usage:
    python clustering1.py --checkpoint_dir /path/to/checkpoint.pt --mode Full
"""

import os
# CUDA deterministic 설정을 가장 먼저 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import pickle
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
import pandas as pd
# 현재 스크립트 파일 위치 (analysis/attentionmap.py)
current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_deleted_features_from_checkpoint(checkpoint_path):
    """
    체크포인트 파일명에서 D:[변수명] 패턴을 추출하여 삭제된 변수 리스트 반환
    
    Args:
        checkpoint_path (str): 체크포인트 파일 경로
        
    Returns:
        list: 삭제된 변수 이름 리스트
        str: D:[변수명] 부분 (폴더명용)
    """
    filename = Path(checkpoint_path).stem
    
    # D:[변수명] 패턴 추출 - 여러 형식 지원
    import re
    patterns = [
        r"D:\[([^\]]*)\]",           # D:[Age] 또는 D:['Age'] 형식
        r"D_\[([^\]]*)\]",           # D_[Age] 형식 (백업)
        r"D-\[([^\]]*)\]",           # D-[Age] 형식 (백업)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            deleted_vars_str = match.group(1)  # 대괄호 안의 내용
            d_part = match.group(0)  # D:[...] 전체
            
            if deleted_vars_str:
                # 쉼표로 분리하여 변수 리스트 생성
                # 작은따옴표, 큰따옴표, 공백 모두 제거
                deleted_features = []
                for var in deleted_vars_str.split(','):
                    clean_var = var.strip().strip("'\"")  # 공백과 따옴표 제거
                    if clean_var:  # 빈 문자열이 아닌 경우만 추가
                        deleted_features.append(clean_var)
            else:
                deleted_features = []
                
            logger.info(f"🔥 Auto-detected deleted features from filename: {deleted_features}")
            logger.info(f"🔥 Original D part: {d_part}")
            
            # 폴더명용으로 깔끔하게 변환 (D:[Age] 형식으로 통일)
            if deleted_features:
                clean_d_part = f"D:[{','.join(deleted_features)}]"
                logger.info(f"🔥 Clean D part for folder: {clean_d_part}")
                return deleted_features, clean_d_part
            else:
                return [], ""
    
    logger.info("🔥 No D:[...] pattern found in filename - using all features")
    return [], ""

def extract_checkpoint_config_for_folder(checkpoint_path):
    """체크포인트 파일명에서 설정 정보를 추출해서 폴더명으로 변환 (D:[...], S:42 제외)"""
    filename = Path(checkpoint_path).stem
    
    # 날짜/시간 패턴 제거 (20250617_173832 형태)
    import re
    
    # 🔥 여러 패턴들을 모두 제거하여 기본 설정만 추출
    filename_clean = re.sub(r'_\d{8}_\d{6}', '', filename)
    
    # 여러 패턴 제거: D:[...], S:42, 실험 ID 등
    patterns_to_remove = [
        r'_D:\[[^\]]*\]',        # _D:[...] 형식
        r'_D_\[[^\]]*\]',        # _D_[...] 형식
        r'_D-\[[^\]]*\]',        # _D-[...] 형식
        r'_S:\d+',               # _S:42 형식 (🔥 추가!)
        r'_[a-f0-9-]{36}',       # UUID 형식 제거
        r'_experiment',          # _experiment 제거
        r'_inference',           # _inference 제거
    ]
    
    for pattern in patterns_to_remove:
        filename_clean = re.sub(pattern, '', filename_clean)
    
    # "Embed:carte_desc_Edge:mlp_A:gat_v1" 형태를 파싱
    pattern = r'Embed:([^:_]+(?:_[^:_]+)*?)_Edge:([^:_]+)_A:([^:_]+(?:_[^:_]+)*)'
    match = re.match(pattern, filename_clean)
    
    if match:
        embed_type = match.group(1)  # carte, carte_desc, ours, ours2
        edge_attr = match.group(2)   # mlp, no_use, normal
        attn_type = match.group(3)   # att, gat, gat_v1
        
        # 폴더명 생성: Embed-carte_desc_Edge-mlp_A-gat_v1 (variant까지 포함)
        folder_name = f"Embed-{embed_type}_Edge-{edge_attr}_A-{attn_type}"
        return folder_name
    else:
        # 패턴 매칭 실패시 원본 사용하되 콜론을 대시로 변경
        logger.warning(f"Could not parse config from filename: {filename_clean}")
        return filename_clean.replace(':', '-')

class AttentionInference:
    def __init__(self, checkpoint_dir, device='cuda', auto_del_feat=None):
        """
        Args:
            checkpoint_dir (str): 체크포인트 파일 경로
            device (str): 'cuda' 또는 'cpu'
            auto_del_feat (list): 자동으로 추출된 삭제할 변수 리스트
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 🔥 실제 체크포인트 경로 확인
        logger.info(f"🔥 Attempting to load checkpoint from: {checkpoint_dir}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        # 🔥 자동 추출된 삭제 변수를 args에 적용
        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            logger.info(f"🔥 Applied auto-detected del_feat: {auto_del_feat}")
        
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
        results = prepare_embedding_dataloaders(self.args, self.args.source_data)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']
        self.num_classes = results['num_classes']
        from torch.utils.data import ConcatDataset, DataLoader
        
        combined_dataset = ConcatDataset([
            self.train_loader.dataset,
            self.val_loader.dataset, 
            self.test_loader.dataset
        ])
        
        self.combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=getattr(self.train_loader, 'num_workers', 0)
        )
        # Few-shot 로더 준비 (필요한 경우)
        if hasattr(self.args, 'few_shot') and self.args.few_shot > 0:
            self.train_loader_few = get_few_shot_embedding_samples(self.train_loader, self.args)
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_data}")
    
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
            cat_name_value_embeddings = batch['cat_name_value_embeddings'].to(self.device)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device)
            
            name_value_embeddings.append(cat_name_value_embeddings)
            desc_embeddings.append(cat_desc_embeddings)
            
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device)
            name_value_embeddings.append(num_prompt_embeddings)
            desc_embeddings.append(num_desc_embeddings)
        

        desc_embeddings, name_value_embeddings = self.model.remove_feature(
            batch, desc_embeddings, name_value_embeddings
        )
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

                        ax_graph.set_title(f'Graph Structure - Layer {layer_idx} - Sample {sample_count}', fontsize=12)
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
                        fig.suptitle(f'Layer {layer_idx} - Sample {sample_count}', fontsize=18)
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
        
        logger.info(f"Saved attention maps by cluster for layer {layer_idx}")

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
        
    def _plot_improved_pairwise_distances(self, flattened_maps, optimal_k, layer_idx, output_dir):
        """
        개선된 pairwise distance 시각화 - 3가지 방식으로 표현 (가독성 개선)
        """
        # K-means 클러스터링 수행
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        centroids = kmeans.cluster_centers_
        
        # 클러스터 간 거리 계산
        from sklearn.metrics import pairwise_distances
        distance_matrix = pairwise_distances(centroids, metric='euclidean')
        
        # 클러스터 쌍별 거리 추출
        cluster_pairs = []
        distances = []
        
        for i in range(optimal_k):
            for j in range(i+1, optimal_k):
                cluster_pairs.append(f"C{i}-C{j}")
                distances.append(distance_matrix[i, j])
        
        # 거리 순으로 정렬 (오름차순 - 가까운 거리부터, 세번째 그림과 동일)
        sorted_indices = np.argsort(distances)
        sorted_pairs = [cluster_pairs[i] for i in sorted_indices]
        sorted_distances = [distances[i] for i in sorted_indices]
        
        # 3개 서브플롯 생성 - 크기 및 간격 조정
        fig = plt.figure(figsize=(20, 7))
        
        # ========== 1. Frobenius Distance Matrix ==========
        ax1 = plt.subplot(1, 3, 1)
        im = ax1.imshow(distance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0)
        
        # 값 표시 - 폰트 크기 증가
        for i in range(optimal_k):
            for j in range(optimal_k):
                if i != j:
                    text_color = 'white' if distance_matrix[i, j] > np.median(distance_matrix) else 'black'
                    ax1.text(j, i, f'{distance_matrix[i, j]:.2f}', 
                            ha="center", va="center", color=text_color, 
                            fontsize=max(8, 12-optimal_k//2), fontweight='bold')  # 동적 폰트 크기
        
        ax1.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cluster ID', fontsize=14, fontweight='bold')
        ax1.set_title('Frobenius Distance Matrix', fontsize=16, fontweight='bold', pad=20)
        
        # 틱 설정
        ax1.set_xticks(range(optimal_k))
        ax1.set_yticks(range(optimal_k))
        ax1.set_xticklabels([f'C{i}' for i in range(optimal_k)], fontsize=12)
        ax1.set_yticklabels([f'C{i}' for i in range(optimal_k)], fontsize=12)
        
        # 컬러바 개선
        cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label('Distance', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # ========== 2. All Pairwise Distances (순서 변경) ==========
        ax2 = plt.subplot(1, 3, 2)

        max_pairs_to_show = min(20, len(sorted_distances))
        display_pairs = sorted_pairs[:max_pairs_to_show]
        display_distances = sorted_distances[:max_pairs_to_show]

        # 색상 매핑
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(display_distances)))

        # 바 차트 - 순서 뒤집기 (가장 가까운 것이 위에)
        y_positions = np.arange(len(display_distances))[::-1]  # 순서 뒤집기
        bars = ax2.barh(y_positions, display_distances, color=colors, alpha=0.8, height=0.7)

        # 값 표시
        for i, (bar, distance) in enumerate(zip(bars, display_distances)):
            ax2.text(bar.get_width() + max(display_distances) * 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{distance:.3f}', ha='left', va='center', 
                    fontweight='bold', fontsize=10, color='black')

        # 축 설정
        ax2.set_xlabel('Frobenius Distance', fontsize=14, fontweight='bold')
        ax2.set_title(f'Top {max_pairs_to_show} Pairwise Distances\n(Sorted: Close → Far)', 
                    fontsize=16, fontweight='bold', pad=20)

        # Y축 라벨 - 순서 맞추기
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(display_pairs, fontsize=10, fontweight='bold')

        # 표시하지 않은 쌍이 있다면 알림
        if len(sorted_distances) > max_pairs_to_show:
            ax2.text(0.02, 0.02, f'Showing top {max_pairs_to_show} of {len(sorted_distances)} pairs', 
                    transform=ax2.transAxes, fontsize=9, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax2.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5)
        ax2.set_xlim(0, max(display_distances) * 1.2)
        
        # ========== 3. Top 20 Farthest Pairs ==========
        ax3 = plt.subplot(1, 3, 3)

        # 가장 먼 20개 쌍 선택 (끝에서부터)
        farthest_pairs_to_show = min(20, len(sorted_distances))
        farthest_pairs = sorted_pairs[-farthest_pairs_to_show:]  # 끝에서 20개
        farthest_distances = sorted_distances[-farthest_pairs_to_show:]  # 끝에서 20개

        # 색상 매핑 (먼 거리용 - 빨간색 계열)
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(farthest_distances)))

        # 막대 차트 (거리 순으로 정렬 - 가장 먼 것부터)
        y_positions = np.arange(len(farthest_distances))
        bars = ax3.barh(y_positions, farthest_distances, color=colors, alpha=0.8, height=0.7)

        # 값 표시
        for i, (bar, distance) in enumerate(zip(bars, farthest_distances)):
            ax3.text(bar.get_width() + max(farthest_distances) * 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{distance:.3f}', ha='left', va='center', 
                    fontweight='bold', fontsize=10, color='black')

        # 축 설정
        ax3.set_xlabel('Frobenius Distance', fontsize=14, fontweight='bold')
        ax3.set_title(f'Top {farthest_pairs_to_show} Farthest Pairs\n(Most Different Clusters)', 
                    fontsize=16, fontweight='bold', pad=20)

        # Y축 라벨
        ax3.set_yticks(y_positions)
        ax3.set_yticklabels(farthest_pairs, fontsize=10, fontweight='bold')

        # 그리드
        ax3.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5)
        ax3.set_xlim(0, max(farthest_distances) * 1.2)

        # 통계 정보
        stats_text = f"Total pairs: {len(sorted_distances)} | Range: {min(sorted_distances):.3f}-{max(sorted_distances):.3f}"
        ax3.text(0.02, 0.02, stats_text, transform=ax3.transAxes, fontsize=8, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        fig.suptitle(f'Layer {layer_idx}: Cluster Distance Analysis (k={optimal_k})', 
                    fontsize=18, fontweight='bold', y=0.95)
        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 저장
        plt.savefig(output_dir / f'layer_{layer_idx}_improved_distance_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
        logger.info(f"✅ Improved distance analysis saved for layer {layer_idx}")

    def perform_clustering(self, attention_data, layer_idx=2, n_clusters=5, output_dir=None):
        """
        특정 레이어의 attention maps에 대해 K-means 클러스터링 수행
        (거리 분석 포함)
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 클러스터링 코드...
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        sample_ids = np.array(attention_data['sample_ids'])
        feature_names = attention_data['feature_names']
        
        logger.info(f"Performing clustering on layer {layer_idx} with {len(attention_maps)} samples")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_assignments = kmeans.fit_predict(flattened_maps)
        
        if output_dir:
            # clustering 폴더 경로로 변경
            clustering_dir = str(output_dir).replace('/visualization/', '/clustering/')
            clustering_path = Path(clustering_dir)
            clustering_path.mkdir(parents=True, exist_ok=True)
            
            # KMeans 모델 저장
            kmeans_path = clustering_path / f'layer_{layer_idx}_kmeans_model.pkl'
            with open(kmeans_path, 'wb') as f:
                pickle.dump(kmeans, f)
            logger.info(f"🔥 KMeans model saved to: {kmeans_path}")
            
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
        
        # 개선된 거리 분석 추가
        if output_dir:
            self._plot_improved_pairwise_distances(flattened_maps, n_clusters, layer_idx, output_dir)
        
        # 나머지 기존 시각화들...
        if output_dir:
            self._save_centroids_npy(kmeans.cluster_centers_, feature_names, layer_idx, output_dir, n_clusters)
        
        if len(flattened_maps) >= 2 and output_dir:
            self._visualize_clustering_distribution(flattened_maps, cluster_assignments, labels, 
                                                layer_idx, output_dir)
        
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

    def _save_centroids_npy(self, cluster_centers, feature_names, layer_idx, output_dir, n_clusters):
        """
        클러스터 센트로이드를 NPY 파일로 저장
        """
        seq_len = len(feature_names)
        centroids_reshaped = cluster_centers.reshape(-1, seq_len, seq_len)
        
        # centroid 폴더 생성
        centroid_dir = output_dir / 'centroid'
        centroid_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 클러스터별로 NPY 저장
        for i, centroid in enumerate(centroids_reshaped):
            npy_filename = f'cluster_{i}_centroid.npy'
            npy_path = centroid_dir / npy_filename
            
            np.save(npy_path, centroid)
            logger.info(f"Saved centroid NPY: {npy_path}")
        
        # 메타데이터도 함께 저장
        metadata = {
            'layer_idx': layer_idx,
            'n_clusters': n_clusters,
            'feature_names': feature_names,
            'centroid_shape': [seq_len, seq_len],
            'description': f'Layer {layer_idx} K-means centroids with {n_clusters} clusters'
        }
        
        metadata_path = centroid_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # feature_names는 numpy array일 수 있으므로 리스트로 변환
            metadata_json = metadata.copy()
            metadata_json['feature_names'] = list(feature_names)
            json.dump(metadata_json, f, indent=2)
        
        logger.info(f"Saved centroid metadata: {metadata_path}")
        logger.info(f"✅ All {n_clusters} centroid NPY files saved in {centroid_dir}")

    def generate_centroid_summary(self, main_output_dir, n_clusters):
        """
        모든 레이어의 centroid 결과를 비교 요약하는 함수
        """
        main_output_dir = Path(main_output_dir)
        summary_dir = main_output_dir / 'centroid_summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # 모든 레이어의 centroid 데이터 수집
        layer_data = {}
        feature_names = None
        
        for layer_idx in range(3):  # Layer 0, 1, 2
            layer_dir = main_output_dir / f'layer_{layer_idx}' / 'centroid'
            if not layer_dir.exists():
                logger.warning(f"Centroid directory not found for layer {layer_idx}: {layer_dir}")
                continue
            
            # 메타데이터 로드
            metadata_path = layer_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if feature_names is None:
                        feature_names = metadata['feature_names']
            
            # 각 클러스터의 centroid 로드
            layer_centroids = []
            for cluster_id in range(n_clusters):
                npy_path = layer_dir / f'cluster_{cluster_id}_centroid.npy'
                if npy_path.exists():
                    centroid = np.load(npy_path)
                    layer_centroids.append(centroid)
                else:
                    logger.warning(f"Centroid not found: {npy_path}")
            
            if layer_centroids:
                layer_data[layer_idx] = np.stack(layer_centroids)
                logger.info(f"Loaded {len(layer_centroids)} centroids for layer {layer_idx}")
        
        if not layer_data:
            logger.error("No centroid data found for summary generation")
            return
        
        # 1. 레이어별 클러스터 간 차이 분석
        self._analyze_layer_cluster_differences(layer_data, feature_names, summary_dir, n_clusters)
        
        # 2. 동일 클러스터의 레이어간 진화 분석
        self._analyze_cluster_evolution_across_layers(layer_data, feature_names, summary_dir, n_clusters)
        
        # 3. 전체 요약 통계
        self._generate_overall_summary_statistics(layer_data, feature_names, summary_dir, n_clusters)
        
        logger.info(f"✅ Centroid summary analysis completed! Results saved in {summary_dir}")

    def _analyze_layer_cluster_differences(self, layer_data, feature_names, summary_dir, n_clusters):
        """레이어별 클러스터 간 차이 분석"""
        
        # 각 레이어별로 클러스터 간 차이 계산
        for layer_idx, centroids in layer_data.items():
            differences = []
            pairs = []
            
            # 모든 클러스터 쌍의 차이 계산
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    diff = np.linalg.norm(centroids[i] - centroids[j])
                    differences.append(diff)
                    pairs.append(f"C{i}-C{j}")
            
            # 차이 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            bars = ax1.bar(range(len(differences)), differences, color='skyblue')
            ax1.set_title(f'Layer {layer_idx}: Pairwise Centroid Distances')
            ax1.set_xlabel('Cluster Pairs')
            ax1.set_ylabel('Euclidean Distance')
            ax1.set_xticks(range(len(pairs)))
            ax1.set_xticklabels(pairs, rotation=45)
            
            # 값 표시
            for i, (bar, diff) in enumerate(zip(bars, differences)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{diff:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 히스토그램
            ax2.hist(differences, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_title(f'Layer {layer_idx}: Distance Distribution')
            ax2.set_xlabel('Distance')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(differences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(differences):.3f}')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(summary_dir / f'layer_{layer_idx}_cluster_differences.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 통계 저장
            stats = {
                'layer': layer_idx,
                'mean_distance': float(np.mean(differences)),
                'std_distance': float(np.std(differences)),
                'max_distance': float(np.max(differences)),
                'min_distance': float(np.min(differences)),
                'most_different_pair': pairs[np.argmax(differences)],
                'most_similar_pair': pairs[np.argmin(differences)]
            }
            
            with open(summary_dir / f'layer_{layer_idx}_cluster_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)

    def _analyze_cluster_evolution_across_layers(self, layer_data, feature_names, summary_dir, n_clusters):
        """동일 클러스터의 레이어간 진화 분석"""
        
        if len(layer_data) < 2:
            logger.warning("Need at least 2 layers for evolution analysis")
            return
        
        # 각 클러스터별로 레이어간 변화 추적
        evolution_data = []
        
        for cluster_id in range(n_clusters):
            cluster_evolution = {'cluster_id': cluster_id, 'layer_changes': []}
            
            layer_indices = sorted(layer_data.keys())
            for i in range(len(layer_indices) - 1):
                layer_curr = layer_indices[i]
                layer_next = layer_indices[i + 1]
                
                if cluster_id < len(layer_data[layer_curr]) and cluster_id < len(layer_data[layer_next]):
                    centroid_curr = layer_data[layer_curr][cluster_id]
                    centroid_next = layer_data[layer_next][cluster_id]
                    
                    # 변화량 계산
                    change = np.linalg.norm(centroid_next - centroid_curr)
                    
                    cluster_evolution['layer_changes'].append({
                        'from_layer': layer_curr,
                        'to_layer': layer_next,
                        'change_magnitude': float(change)
                    })
            
            evolution_data.append(cluster_evolution)
        
        # 진화 패턴 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 클러스터별 변화량 추이
        layer_transitions = []
        for cluster in evolution_data:
            changes = [change['change_magnitude'] for change in cluster['layer_changes']]
            transitions = [f"L{change['from_layer']}→L{change['to_layer']}" 
                         for change in cluster['layer_changes']]
            
            if changes:  # 변화가 있는 경우만
                ax1.plot(range(len(changes)), changes, marker='o', 
                        label=f"Cluster {cluster['cluster_id']}", linewidth=2)
                if not layer_transitions:
                    layer_transitions = transitions
        
        ax1.set_title('Cluster Evolution Across Layers')
        ax1.set_xlabel('Layer Transition')
        ax1.set_ylabel('Change Magnitude')
        ax1.set_xticks(range(len(layer_transitions)))
        ax1.set_xticklabels(layer_transitions)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 전체 변화량 분포
        boxplot_data = []
        boxplot_labels = []
        
        for cluster in evolution_data:
            cluster_changes = [change['change_magnitude'] for change in cluster['layer_changes']]
            if cluster_changes:  # 변화가 있는 클러스터만
                boxplot_data.append(cluster_changes)
                boxplot_labels.append(f"C{cluster['cluster_id']}")
        
        if boxplot_data:
            ax2.boxplot(boxplot_data, labels=boxplot_labels)
            ax2.set_title('Change Magnitude Distribution by Cluster')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Change Magnitude')
        else:
            ax2.text(0.5, 0.5, 'No evolution data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Change Magnitude Distribution by Cluster')
        
        plt.tight_layout()
        plt.savefig(summary_dir / 'cluster_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 진화 데이터 저장
        with open(summary_dir / 'cluster_evolution_data.json', 'w') as f:
            json.dump(evolution_data, f, indent=2)

    def _generate_overall_summary_statistics(self, layer_data, feature_names, summary_dir, n_clusters):
        """전체 요약 통계 생성"""
        
        summary_stats = {
            'experiment_info': {
                'n_clusters': n_clusters,
                'n_layers': len(layer_data),
                'n_features': len(feature_names),
                'feature_names': feature_names
            },
            'layer_statistics': {},
            'cross_layer_analysis': {}
        }
        
        # 레이어별 통계
        for layer_idx, centroids in layer_data.items():
            layer_stats = {
                'mean_attention_per_cluster': [],
                'max_attention_per_cluster': [],
                'attention_spread_per_cluster': []
            }
            
            for cluster_id, centroid in enumerate(centroids):
                layer_stats['mean_attention_per_cluster'].append(float(np.mean(centroid)))
                layer_stats['max_attention_per_cluster'].append(float(np.max(centroid)))
                layer_stats['attention_spread_per_cluster'].append(float(np.std(centroid)))
            
            summary_stats['layer_statistics'][f'layer_{layer_idx}'] = layer_stats
        
        # 교차 레이어 분석
        if len(layer_data) > 1:
            # 전체적인 attention 진화 패턴
            layer_means = {}
            for layer_idx, centroids in layer_data.items():
                layer_means[layer_idx] = [float(np.mean(centroid)) for centroid in centroids]
            
            summary_stats['cross_layer_analysis']['attention_evolution'] = layer_means
            
            # 가장 변화가 큰 클러스터 찾기
            max_change_cluster = -1
            max_change_value = 0
            
            for cluster_id in range(n_clusters):
                total_change = 0
                layer_indices = sorted(layer_data.keys())
                
                for i in range(len(layer_indices) - 1):
                    if (cluster_id < len(layer_data[layer_indices[i]]) and 
                        cluster_id < len(layer_data[layer_indices[i+1]])):
                        
                        change = np.linalg.norm(
                            layer_data[layer_indices[i+1]][cluster_id] - 
                            layer_data[layer_indices[i]][cluster_id]
                        )
                        total_change += change
                
                if total_change > max_change_value:
                    max_change_value = total_change
                    max_change_cluster = cluster_id
            
            summary_stats['cross_layer_analysis']['most_dynamic_cluster'] = {
                'cluster_id': max_change_cluster,
                'total_change': float(max_change_value)
            }
        
        # 종합 시각화
        self._create_comprehensive_summary_plot(layer_data, feature_names, summary_dir, summary_stats)
        
        # 요약 통계 저장
        with open(summary_dir / 'comprehensive_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info("✅ Comprehensive summary statistics generated")

    def _create_comprehensive_summary_plot(self, layer_data, feature_names, summary_dir, summary_stats):
        """종합 요약 플롯 생성"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 레이어별 평균 attention
        ax = axes[0, 0]
        for layer_idx in sorted(layer_data.keys()):
            layer_stats = summary_stats['layer_statistics'][f'layer_{layer_idx}']
            means = layer_stats['mean_attention_per_cluster']
            ax.plot(range(len(means)), means, marker='o', label=f'Layer {layer_idx}', linewidth=2)
        
        ax.set_title('Mean Attention per Cluster by Layer')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Mean Attention')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 레이어별 최대 attention
        ax = axes[0, 1]
        for layer_idx in sorted(layer_data.keys()):
            layer_stats = summary_stats['layer_statistics'][f'layer_{layer_idx}']
            maxes = layer_stats['max_attention_per_cluster']
            ax.plot(range(len(maxes)), maxes, marker='s', label=f'Layer {layer_idx}', linewidth=2)
        
        ax.set_title('Max Attention per Cluster by Layer')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Max Attention')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Attention 분산
        ax = axes[0, 2]
        for layer_idx in sorted(layer_data.keys()):
            layer_stats = summary_stats['layer_statistics'][f'layer_{layer_idx}']
            spreads = layer_stats['attention_spread_per_cluster']
            ax.plot(range(len(spreads)), spreads, marker='^', label=f'Layer {layer_idx}', linewidth=2)
        
        ax.set_title('Attention Spread (Std) per Cluster by Layer')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 클러스터 간 거리 히트맵 (Layer 0)
        ax = axes[1, 0]
        if 0 in layer_data:
            centroids = layer_data[0]
            n_clusters = len(centroids)
            distance_matrix = np.zeros((n_clusters, n_clusters))
            
            for i in range(n_clusters):
                for j in range(n_clusters):
                    distance_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            
            im = ax.imshow(distance_matrix, cmap='viridis')
            ax.set_title('Cluster Distance Matrix (Layer 0)')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Cluster ID')
            plt.colorbar(im, ax=ax)
            
            # 값 표시
            for i in range(n_clusters):
                for j in range(n_clusters):
                    ax.text(j, i, f'{distance_matrix[i,j]:.2f}', 
                           ha="center", va="center", color="white" if distance_matrix[i,j] > 0.5 else "black")
        
        # 5. 레이어별 attention 히스토그램
        ax = axes[1, 1]
        for layer_idx in sorted(layer_data.keys()):
            all_attentions = []
            for centroid in layer_data[layer_idx]:
                all_attentions.extend(centroid.flatten())
            ax.hist(all_attentions, bins=30, alpha=0.6, label=f'Layer {layer_idx}', density=True)
        
        ax.set_title('Attention Value Distribution by Layer')
        ax.set_xlabel('Attention Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        # 6. 요약 텍스트
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"Centroid Analysis Summary\n\n"
        summary_text += f"Clusters: {summary_stats['experiment_info']['n_clusters']}\n"
        summary_text += f"Layers: {summary_stats['experiment_info']['n_layers']}\n"
        summary_text += f"Features: {summary_stats['experiment_info']['n_features']}\n\n"
        
        if 'most_dynamic_cluster' in summary_stats['cross_layer_analysis']:
            dynamic_info = summary_stats['cross_layer_analysis']['most_dynamic_cluster']
            summary_text += f"Most Dynamic Cluster: {dynamic_info['cluster_id']}\n"
            summary_text += f"Total Change: {dynamic_info['total_change']:.3f}\n\n"
        
        # 각 레이어별 가장 활성화된 클러스터
        summary_text += "Highest Mean Attention:\n"
        for layer_idx in sorted(layer_data.keys()):
            layer_stats = summary_stats['layer_statistics'][f'layer_{layer_idx}']
            means = layer_stats['mean_attention_per_cluster']
            best_cluster = np.argmax(means)
            summary_text += f"Layer {layer_idx}: Cluster {best_cluster} ({means[best_cluster]:.3f})\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Comprehensive Centroid Analysis Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(summary_dir / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_clustering_distribution(self, flattened_maps, cluster_assignments, labels, 
                             layer_idx, output_dir):
        """클러스터링 분포 시각화 (t-SNE)"""
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 하나의 큰 플롯
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 클러스터별 색상과 라벨별 마커 조합
        unique_clusters = np.unique(cluster_assignments)
        unique_labels = np.unique(labels)
        
        # 클러스터별 기본 색상 설정
        base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
        
        # 레전드 중복 방지를 위한 추적 변수
        legend_added = {
            'clusters': set(),
            'labels': set(),
            'centroid': False
        }
        
        # 클러스터와 라벨 조합으로 시각화
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = tsne_embeddings[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            if len(cluster_points) > 0:
                # 라벨별로 모양 구분: Label 0=원형, Label 1=네모
                for label in unique_labels:
                    label_mask = cluster_labels == label
                    if np.any(label_mask):
                        label_points = cluster_points[label_mask]
                        
                        # 모양 구분
                        if label == 0:
                            marker = 'o'  # 원형
                            marker_name = 'Label 0'
                        else:
                            marker = 's'  # 네모
                            marker_name = 'Label 1'
                        
                        # 레전드 라벨 결정: 중복 방지 로직
                        cluster_key = f'cluster_{cluster_id}'
                        label_key = f'label_{label}'
                        
                        legend_label = None
                        
                        if cluster_key not in legend_added['clusters']:
                            if label_key not in legend_added['labels']:
                                legend_label = f'Cluster {cluster_id} ({marker_name})'
                                legend_added['labels'].add(label_key)
                            else:
                                legend_label = f'Cluster {cluster_id}'
                            legend_added['clusters'].add(cluster_key)
                        elif label_key not in legend_added['labels']:
                            legend_label = f'{marker_name}'
                            legend_added['labels'].add(label_key)
                        
                        ax.scatter(label_points[:, 0], label_points[:, 1], 
                                color=base_colors[i], 
                                label=legend_label,
                                alpha=0.7, s=50, marker=marker)
        
        # 클러스터 센트로이드 추가
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            if np.any(cluster_mask):
                centroid_x = np.mean(tsne_embeddings[cluster_mask, 0])
                centroid_y = np.mean(tsne_embeddings[cluster_mask, 1])
                
                centroid_label = f'C{cluster_id} Centroid'
                
                ax.scatter(centroid_x, centroid_y, marker='*', s=100, 
                        c=base_colors[i], edgecolors='black', linewidth=2,
                        label=centroid_label, zorder=5)
        
        # 제목과 축 라벨
        ax.set_title(f'Layer {layer_idx} - Clustering & True Labels\nK-means with t-SNE Visualization', 
                    fontsize=16, pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        # 범례 정리
        handles, labels_legend = ax.get_legend_handles_labels()
        if handles:
            legend_items = list(zip(handles, labels_legend))
            
            cluster_items = [(h, l) for h, l in legend_items if l and 'Cluster' in l and 'Centroid' not in l]
            label_items = [(h, l) for h, l in legend_items if l and 'Label' in l and 'Cluster' not in l]
            centroid_items = [(h, l) for h, l in legend_items if l and 'Centroid' in l]
            
            cluster_items.sort(key=lambda x: int(x[1].split()[1]) if 'Cluster' in x[1] and x[1].split()[1].isdigit() else 999)
            centroid_items.sort(key=lambda x: int(x[1].split('C')[1].split()[0]) if 'C' in x[1] else 999)
            
            final_items = cluster_items + label_items + centroid_items
            final_handles, final_labels = zip(*final_items) if final_items else ([], [])
            
            ax.legend(final_handles, final_labels, 
                    bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        
        # 클러스터 통계 정보
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
            cluster_stats.append(f"C{cluster_id}: {total_count} ({total_percentage:.1f}%) [{label_str}]")
        
        if cluster_stats:
            stats_text = "\n".join(cluster_stats)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))
        
        # 마커 설명 추가
        total_samples_processed = len(flattened_maps)
        ax.text(0.02, 0.02, f"Total: {total_samples_processed} maps\n○=Label 0, ■=Label 1, ★=Centroid (colored by cluster)", 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_clustering_distribution.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"✅ Layer {layer_idx} distribution saved with clean legend!")

    def _visualize_cluster_centroids(self, cluster_centers, feature_names, layer_idx, output_dir, n_clusters):
        """클러스터 센트로이드 히트맵 시각화"""
        seq_len = len(feature_names)
        centroids_reshaped = cluster_centers.reshape(-1, seq_len, seq_len)
        
        # 전체 클러스터의 공통 스케일 계산
        global_vmin = centroids_reshaped.min()
        global_vmax = centroids_reshaped.max()
        
        # centroid 폴더 생성
        centroid_dir = output_dir / 'centroid'
        centroid_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 전체 overview: 동적 레이아웃 결정
        if n_clusters <= 5:
            rows, cols = 1, n_clusters
        elif n_clusters <= 10:
            rows, cols = 2, (n_clusters + 1) // 2
        elif n_clusters <= 15:
            rows, cols = 3, (n_clusters + 2) // 3
        elif n_clusters <= 20:
            rows, cols = 4, (n_clusters + 3) // 4
        else:
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
        
        # 모든 클러스터에 동일한 스케일 적용
        for i, centroid in enumerate(centroids_reshaped):
            ax = axes[i]
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest',
                        vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f'Cluster {i} Centroid', fontsize=11)
            
            # 축 라벨 설정
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
            ax.set_yticklabels(feature_names, fontsize=8)
            
            # 컬러바 추가
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_idx} Cluster Centroids (All {n_clusters} clusters)\nScale: {global_vmin:.3f} - {global_vmax:.3f}', 
                    fontsize=14)
        plt.tight_layout()
        
        # overview는 메인 폴더에 저장
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_centroids_overview.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"All {n_clusters} cluster centroids overview saved with fixed scale [{global_vmin:.3f}, {global_vmax:.3f}]")
        
        # 2. 각 센트로이드별로 개별 상세 플롯 생성
        for i, centroid in enumerate(centroids_reshaped):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest',
                        vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f'Cluster {i} Centroid - Layer {layer_idx}\nScale: {global_vmin:.3f} - {global_vmax:.3f}', 
                        fontsize=16, pad=20)
            
            # 축 라벨 설정
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=12)
            ax.set_yticklabels(feature_names, fontsize=12)
            
            # 각 셀에 값 표시
            for row in range(len(feature_names)):
                for col in range(len(feature_names)):
                    value = centroid[row, col]
                    threshold = (global_vmin + global_vmax) / 2
                    ax.text(col, row, f"{value:.2f}", 
                        ha="center", va="center", 
                        color="white" if value > threshold else "black", 
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
        
        logger.info(f"All {n_clusters} detailed centroids saved with fixed scale in {centroid_dir}")
        logger.info(f"All cluster centroids saved for layer {layer_idx}")

def extract_seed_from_checkpoint(checkpoint_path):
    """체크포인트 파일명에서 S:42 패턴을 추출하여 시드값 반환"""
    filename = Path(checkpoint_path).stem
    
    import re
    pattern = r'S:(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        seed_value = int(match.group(1))
        logger.info(f"🎲 Auto-detected seed from filename: {seed_value}")
        return seed_value
    else:
        logger.info("🎲 No S:[seed] pattern found in filename")
        return None

def main():
   parser = argparse.ArgumentParser(description='Attention Maps Inference')
   parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Path to model checkpoint')
   parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                      help='Which model to use (Full or Few)')
   parser.add_argument('--layer_idx', type=int, default=2,
                      help='Layer index for clustering (default: 2)')
   parser.add_argument('--del_feat', nargs='+', default=[], help="features to remove (will be overridden by auto-detection)")
   parser.add_argument('--n_clusters', type=int, default=3,
                      help='Number of clusters for K-means')
   parser.add_argument('--max_samples', type=int, default=5,
                      help='Maximum number of samples for graph visualization ONLY')
   parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for results')
   parser.add_argument('--save_attention_maps', action='store_true',
                      help='Save individual attention map files')
   parser.add_argument('--viz_graph', action='store_true',
                      help='Generate graph structure visualization')
   
   args = parser.parse_args()
   
   # 🔥 원본 체크포인트 경로 저장 (쉘 해석 전)
   original_checkpoint_path = args.checkpoint_dir
   logger.info(f"🔥 Original checkpoint path: {original_checkpoint_path}")
   
   # 🔥 체크포인트 파일명에서 자동으로 삭제된 변수 추출
   auto_del_feat, d_part = extract_deleted_features_from_checkpoint(original_checkpoint_path)
   
   # 🎲 체크포인트 파일명에서 시드값 추출
   seed_value = extract_seed_from_checkpoint(original_checkpoint_path)
   
   # 🔥 실제 파일 존재 여부 확인 및 경로 보정
   checkpoint_file = Path(original_checkpoint_path)
   if not checkpoint_file.exists():
       # 파일이 없으면 glob 패턴으로 찾아보기
       parent_dir = checkpoint_file.parent
       filename_pattern = checkpoint_file.name.replace('[', r'\[').replace(']', r'\]')
       
       import glob
       matching_files = glob.glob(str(parent_dir / filename_pattern))
       
       if not matching_files:
           # 다른 패턴들도 시도
           base_name = checkpoint_file.stem
           # D:[Age] -> D:['Age'] 또는 D:["Age"] 패턴 찾기
           possible_patterns = [
               base_name.replace("D:[", "D:['").replace("]", "']") + ".pt",
               base_name.replace("D:[", 'D:["').replace("]", '"]') + ".pt",
           ]
           
           for pattern in possible_patterns:
               potential_path = parent_dir / pattern
               logger.info(f"🔥 Trying pattern: {potential_path}")
               if potential_path.exists():
                   args.checkpoint_dir = str(potential_path)
                   logger.info(f"🔥 Found actual file: {args.checkpoint_dir}")
                   break
           else:
               logger.error(f"❌ Could not find checkpoint file. Tried:")
               logger.error(f"   Original: {original_checkpoint_path}")
               for pattern in possible_patterns:
                   logger.error(f"   Pattern: {parent_dir / pattern}")
               raise FileNotFoundError(f"Checkpoint file not found: {original_checkpoint_path}")
       else:
           args.checkpoint_dir = matching_files[0]
           logger.info(f"🔥 Found file via glob: {args.checkpoint_dir}")
   
   # 🔥 출력 디렉토리 수정: seed 폴더 추가
   if args.output_dir is None:
       # 기본 설정 폴더 (D:[...] 제외) - 원본 경로 사용
       config_folder = extract_checkpoint_config_for_folder(original_checkpoint_path)
       
       # 체크포인트 파일의 부모 디렉토리 경로를 가져와서 변환
       checkpoint_parent_str = str(Path(args.checkpoint_dir).parent)
       
       # checkpoints를 visualization으로 변경
       if '/checkpoints/' in checkpoint_parent_str:
           viz_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/visualization/')
       else:
           # fallback: 직접 경로 구성
           viz_parent_str = '/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full'
       
       # 체크포인트 경로에서 seed 폴더 구조 제거 후 기본 경로만 사용
       base_viz_parent = viz_parent_str.split(f'/{seed_value}')[0] if seed_value else viz_parent_str
       
       # 🎲 시드 폴더 추가
       if seed_value is not None:
           seed_folder = f'seed_{seed_value}'
       else:
           seed_folder = 'seed_unknown'
       
       # 🔥 clustering 폴더명에 D:[...] 정보 추가
       if d_part:
           clustering_folder = f'clustering1_{args.n_clusters}_{d_part}'
       else:
           clustering_folder = f'clustering1_{args.n_clusters}'
       
       # 최종 출력 경로: .../visualization/.../config_folder/seed_42/clustering_{n_clusters}_D:[...]/
       args.output_dir = Path(base_viz_parent) / config_folder / seed_folder / clustering_folder
   
   logger.info(f"📁 Results will be saved to: {args.output_dir}")

   # 🔥 Inference 실행 시 자동 추출된 삭제 변수 적용
   # 보정된 checkpoint_dir 경로 사용
   inference = AttentionInference(args.checkpoint_dir, auto_del_feat=auto_del_feat)
   
   # 데이터로더 선택
   if args.mode == 'Full':
       data_loader = inference.combined_loader
       logger.info("Using Full dataset loader")
   else:
       data_loader = inference.train_loader_few if hasattr(inference, 'train_loader_few') else inference.test_loader
       logger.info("Using Few-shot dataset loader")
   
   # 🔥 클러스터링 결과를 위한 상위 폴더 생성
   clustering_results_dir = Path(args.output_dir) / 'clustering_results'
   clustering_results_dir.mkdir(parents=True, exist_ok=True)
   
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
       # 🔥 레이어별 결과를 clustering_results 폴더 아래에 저장
       layer_output_dir = clustering_results_dir / f'layer_{layer_idx}'
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
           clustering_results_dir,
           layer_idx
       )
   
   # 🔥 전체 센트로이드 요약 분석 - clustering_results 폴더 사용
   logger.info("Generating comprehensive centroid summary...")
   inference.generate_centroid_summary(clustering_results_dir, args.n_clusters)
   
   logger.info(f"Inference completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
   main()