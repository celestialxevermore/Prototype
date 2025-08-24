"""
Head별 Attention Maps 분석 스크립트

BasisGAT의 각 head별 attention maps를 분석하고,
Coordinates를 클러스터링하여 모델의 동작을 이해합니다.

Usage:
    python clustering1_head_analysis.py --checkpoint_dir /path/to/checkpoint.pt --mode Full
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

# 현재 스크립트 파일 위치 (analysis/clustering1_head_analysis.py)
current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 기존 함수들 재사용 (extract_deleted_features_from_checkpoint, extract_checkpoint_config_for_folder 등)
def extract_deleted_features_from_checkpoint(checkpoint_path):
    """체크포인트 파일명에서 D:[변수명] 패턴을 추출하여 삭제된 변수 리스트 반환"""
    filename = Path(checkpoint_path).stem
    
    import re
    patterns = [
        r"D:\[([^\]]*)\]",
        r"D_\[([^\]]*)\]",
        r"D-\[([^\]]*)\]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            deleted_vars_str = match.group(1)
            d_part = match.group(0)
            
            if deleted_vars_str:
                deleted_features = []
                for var in deleted_vars_str.split(','):
                    clean_var = var.strip().strip("'\"")
                    if clean_var:
                        deleted_features.append(clean_var)
            else:
                deleted_features = []
                
            logger.info(f"🔥 Auto-detected deleted features from filename: {deleted_features}")
            logger.info(f"�� Original D part: {d_part}")
            
            if deleted_features:
                clean_d_part = f"D:[{','.join(deleted_features)}]"
                logger.info(f"🔥 Clean D part for folder: {clean_d_part}")
                return deleted_features, clean_d_part
            else:
                return [], ""
    
    logger.info("�� No D:[...] pattern found in filename - using all features")
    return [], ""

def extract_checkpoint_config_for_folder(checkpoint_path):
    """체크포인트 파일명에서 설정 정보를 추출해서 폴더명으로 변환"""
    filename = Path(checkpoint_path).stem
    
    import re
    filename_clean = re.sub(r'_\d{8}_\d{6}', '', filename)
    
    patterns_to_remove = [
        r'_D:\[[^\]]*\]',
        r'_D_\[[^\]]*\]',
        r'_D-\[[^\]]*\]',
        r'_S:\d+',
        r'_[a-f0-9-]{36}',
        r'_experiment',
        r'_inference',
    ]
    
    for pattern in patterns_to_remove:
        filename_clean = re.sub(pattern, '', filename_clean)
    
    pattern = r'Embed:([^:_]+(?:_[^:_]+)*?)_Edge:([^:_]+)_A:([^:_]+(?:_[^:_]+)*)'
    match = re.match(pattern, filename_clean)
    
    if match:
        embed_type = match.group(1)
        edge_attr = match.group(2)
        attn_type = match.group(3)
        
        folder_name = f"Embed-{embed_type}_Edge-{edge_attr}_A-{attn_type}"
        return folder_name
    else:
        logger.warning(f"Could not parse config from filename: {filename_clean}")
        return filename_clean.replace(':', '-')

def extract_seed_from_checkpoint(checkpoint_path):
    """체크포인트 파일명에서 S:42 패턴을 추출하여 시드값 반환"""
    filename = Path(checkpoint_path).stem
    
    import re
    pattern = r'S:(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        seed_value = int(match.group(1))
        logger.info(f"�� Auto-detected seed from filename: {seed_value}")
        return seed_value
    else:
        logger.info("🎲 No S:[seed] pattern found in filename")
        return None

class HeadAttentionInference:
    def __init__(self, checkpoint_dir, device='cuda', auto_del_feat=None):
        """Head별 Attention 분석을 위한 Inference 클래스"""
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"�� Attempting to load checkpoint from: {checkpoint_dir}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
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
        
        if hasattr(self.args, 'few_shot') and self.args.few_shot > 0:
            self.train_loader_few = get_few_shot_embedding_samples(self.train_loader, self.args)
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_data}")
    
    def extract_head_attention_and_coordinates(self, data_loader):
        """각 head별 attention maps와 coordinates 추출"""
        # 🔥 k_basis만 사용 (shared layers 제외)
        total_heads = self.args.k_basis  # 8개만
        
        head_attention_data = {
            'coordinates': [],
            'labels': [],
            'sample_ids': [],
            'feature_names': None
        }
        
        # 🔥 k_basis 개수만큼만 head 키 생성
        for i in range(total_heads):
            head_attention_data[f'head_{i}'] = []
        
        logger.info(f" Using {total_heads} basis heads (k_basis={self.args.k_basis})")
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 모델 forward pass로 head별 attention과 coordinates 추출
                pred, head_attention_weights, coordinates = self._extract_head_attention_from_model(batch_on_device)
                
                # Feature names 추출 (첫 번째 배치에서만)
                if head_attention_data['feature_names'] is None:
                    feature_names = self.model.extract_feature_names(batch_on_device)
                    head_attention_data['feature_names'] = ["CLS"] + feature_names
                
                batch_size = head_attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    # 🔥 BasisGAT의 head만 저장 (shared layers 제외)
                    for head_idx in range(self.args.k_basis):
                        # basis layer의 attention weights만 사용
                        basis_head_idx = head_idx
                        attention_map = head_attention_weights[basis_head_idx][sample_idx]  # [seq_len, seq_len]
                        attention_numpy = attention_map.detach().cpu().numpy()
                        head_attention_data[f'head_{head_idx}'].append(attention_numpy)
                    
                    # Coordinates 저장
                    sample_coordinates = coordinates[sample_idx].detach().cpu().numpy()
                    head_attention_data['coordinates'].append(sample_coordinates)
                    
                    # 라벨과 샘플 ID 저장
                    if 'y' in batch:
                        label = batch['y'][sample_idx].item()
                    else:
                        label = -1
                    head_attention_data['labels'].append(label)
                    
                    if 'sample_ids' in batch:
                        sample_id = batch['sample_ids'][sample_idx]
                    else:
                        sample_id = sample_count
                    head_attention_data['sample_ids'].append(sample_id)
                    
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        
        logger.info(f"Extracted basis head attention maps and coordinates for {sample_count} samples")
        return head_attention_data
    
    def _extract_head_attention_from_model(self, batch):
        """모델에서 head별 attention weights와 coordinates 추출"""
        # 모델의 forward 로직을 복사하되 head별 attention과 coordinates 반환
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

        desc_embeddings = torch.cat(desc_embeddings, dim=1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim=1)
        
        # [CLS] Token 추가
        head_attention_weights = [] 
        cls_token = self.model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        # Shared Graph Attention Layers (attention weights는 저장하지 않음)
        for i, layer in enumerate(self.model.shared_layers):
            norm_x = self.model.shared_layer_norms[i](x)
            attn_output, _ = layer(desc_embeddings, norm_x)  # attention weights는 사용하지 않음
            x = x + attn_output
        
        # BasisGAT Layer에서 head별 attention 추출
        shared_cls = x[:, 0, :]
        coordinates = self.model.coordinator(shared_cls)
        
        norm_x = self.model.basis_layer_norm(x)
        basis_outputs, basis_attention_weights = self.model.basis_layer(desc_embeddings, norm_x)
        
        # 🔥 BasisGAT의 각 head별 attention weights만 저장
        for head_idx in range(self.args.k_basis):  # n_heads 차원
            head_attention = basis_attention_weights[:, head_idx, :, :]  # [batch, seq, seq]
            head_attention_weights.append(head_attention)
        
        # 예측값 계산
        expert_outputs = basis_outputs[:, 0, :, :]  # CLS 토큰만
        expert_predictions = []
        for i in range(self.args.k_basis):
            pred = self.model.expert_predictors[i](expert_outputs[:, i, :])
            expert_predictions.append(pred)
        expert_predictions = torch.stack(expert_predictions, dim=1)
        pred = torch.sum(coordinates.unsqueeze(-1) * expert_predictions, dim=1)

        return pred, head_attention_weights, coordinates

    def visualize_head_attention_distribution(self, data_loader, output_dir, max_samples=10):
        """한 샘플에 대해 모든 head의 attention heatmap을 한 번에 시각화"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= max_samples:
                    break
                    
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 모델 forward pass
                pred, head_attention_weights, coordinates = self._extract_head_attention_from_model(batch_on_device)
                
                # Feature names 추출
                feature_names = self.model.extract_feature_names(batch_on_device)
                
                batch_size = head_attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    # 한 샘플에 대해 모든 head의 attention과 coordinates 시각화
                    self._create_sample_head_visualization(
                        sample_count, head_attention_weights, coordinates, 
                        feature_names, sample_idx, output_dir
                    )
                    
                    sample_count += 1
                    
                if sample_count >= max_samples:
                    break
        
        logger.info(f"Head attention visualization completed for {sample_count} samples")

    def _create_sample_head_visualization(self, sample_idx, head_attention_weights, coordinates, 
                                        feature_names, batch_sample_idx, output_dir):
        """한 샘플의 모든 head attention과 coordinates를 하나의 figure로 시각화"""
        total_heads = len(head_attention_weights)  # 이제 k_basis만큼 (8개)
        
        # 🔥 8개 head에 맞는 레이아웃 (2x4)
        rows, cols = 2, 4
        
        # Figure 생성: Head attention heatmaps + Coordinates
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()
        
        # �� BasisGAT의 8개 head만 시각화
        for head_idx in range(total_heads):
            ax = axes[head_idx]
            
            # BasisGAT layer의 attention weights
            attn_weights = head_attention_weights[head_idx][batch_sample_idx]  # [seq, seq]
            
            # Heatmap 시각화
            im = ax.imshow(attn_weights.cpu().numpy(), cmap='viridis', interpolation='nearest')
            ax.set_title(f'Basis Head {head_idx}', fontsize=10, fontweight='bold')
            
            # 축 라벨 설정
            all_node_names = ["CLS"] + feature_names
            ax.set_xticks(np.arange(len(all_node_names)))
            ax.set_yticks(np.arange(len(all_node_names)))
            ax.set_xticklabels(all_node_names, rotation=90, fontsize=6)
            ax.set_yticklabels(all_node_names, fontsize=6)
            
            # 값 표시 (간단하게)
            attn_np = attn_weights.cpu().numpy()
            for i in range(len(all_node_names)):
                for j in range(len(all_node_names)):
                    value = attn_np[i, j]
                    threshold = (attn_np.min() + attn_np.max()) / 2
                    text_color = "white" if value > threshold else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                           color=text_color, fontsize=5, weight='bold')
        else:
            ax.set_visible(False)
        
        # 🔥 Coordinates bar chart (k_basis 개수만큼만)
        # 별도 subplot으로 생성
        fig_coord, ax_coord = plt.subplots(1, 1, figsize=(8, 4))
        
        # 🔥 coordinates 접근 방식 수정
        # coordinates는 이미 [k_basis] 형태로 저장되어 있음
        if isinstance(coordinates, torch.Tensor):
            if coordinates.dim() == 2:  # [batch_size, k_basis]
                sample_coordinates = coordinates[batch_sample_idx].cpu().numpy()
            else:  # [k_basis] - 이미 해당 샘플의 coordinates
                sample_coordinates = coordinates.cpu().numpy()
        else:
            # numpy array인 경우
            if coordinates.ndim == 2:  # [batch_size, k_basis]
                sample_coordinates = coordinates[batch_sample_idx]
            else:  # [k_basis] - 이미 해당 샘플의 coordinates
                sample_coordinates = coordinates
        
        # 🔥 k_basis 개수만큼만 bar chart 생성
        k_basis = len(sample_coordinates)  # 실제 coordinates 길이
        head_labels = [f'Head {i}' for i in range(k_basis)]
        bars = ax_coord.bar(range(k_basis), sample_coordinates, 
                           color=plt.cm.Set3(np.linspace(0, 1, k_basis)))
        
        ax_coord.set_title('Head Weights (Coordinates)', fontsize=12, fontweight='bold')
        ax_coord.set_xlabel('Head Index', fontsize=10)
        ax_coord.set_ylabel('Weight', fontsize=10)
        ax_coord.set_xticks(range(k_basis))
        ax_coord.set_xticklabels(head_labels, fontsize=9)
        
        # 값 표시
        for i, (bar, coord) in enumerate(zip(bars, sample_coordinates)):
            ax_coord.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{coord:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 전체 제목
        fig.suptitle(f'Sample {sample_idx} - Head Attention Analysis ({total_heads} heads)', 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        # 저장
        output_path = output_dir / f'sample_{sample_idx}_head_attention.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Coordinates도 별도 저장
        coord_output_path = output_dir / f'sample_{sample_idx}_coordinates.png'
        fig_coord.savefig(coord_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_coord)
        
        logger.info(f"Sample {sample_idx} head attention visualization saved: {output_path}")
        logger.info(f"Sample {sample_idx} coordinates visualization saved: {coord_output_path}")

    def analyze_coordinates(self, head_attention_data, output_dir):
        """Coordinates 분석 (K-means 클러스터링 + t-SNE)"""
        coordinates_dir = Path(output_dir) / 'coordinates_analysis'
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        coordinates = np.array(head_attention_data['coordinates'])
        labels = np.array(head_attention_data['labels'])
        sample_ids = np.array(head_attention_data['sample_ids'])
        
        logger.info(f"Analyzing coordinates for {len(coordinates)} samples")
        
        # 1. K-means 클러스터링
        n_clusters = min(5, len(coordinates) // 10)  # 적절한 클러스터 수 결정
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_assignments = kmeans.fit_predict(coordinates)
        
        # 2. t-SNE 시각화 (클러스터링 없이) - 별도로 유지
        self._visualize_coordinates_tsne(coordinates, labels, coordinates_dir)
        
        # 3. Head별 가중치 분포 히스토그램
        self._visualize_head_weight_distributions(coordinates, coordinates_dir)
        
        # 4. 클러스터별 통계
        self._analyze_coordinate_clusters(coordinates, cluster_assignments, labels, 
                                       coordinates_dir, n_clusters)
        
        logger.info(f"Coordinates analysis completed and saved to {coordinates_dir}")

    def _visualize_coordinate_clustering(self, coordinates, cluster_assignments, labels, 
                                  output_dir, n_clusters):
        """Coordinates 클러스터링 결과 시각화 (2x2 plot) - 균형잡힌 레이아웃"""
        # 🔥 2x2 레이아웃으로 변경
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2x2에 맞는 크기 조정
        
        # 1. 좌상단: t-SNE 클러스터링 + 라벨 구분 + Centroid
        ax1 = axes[0, 0]
        perplexity = min(30, len(coordinates)-1, max(1, len(coordinates)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(coordinates)
        
        # 클러스터별 색상
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # 각 클러스터별로 시각화
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = tsne_embeddings[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            # Label 0: 네모, Label 1: 동그라미
            for label in [0, 1]:
                label_mask = cluster_labels == label
                if np.any(label_mask):
                    label_points = cluster_points[label_mask]
                    marker = 's' if label == 0 else 'o'  # 네모 vs 동그라미
                    ax1.scatter(label_points[:, 0], label_points[:, 1], 
                           c=[colors[cluster_id]], marker=marker, 
                           label=f'Cluster {cluster_id} - Label {label}', 
                           alpha=0.7, s=40)
            
            # Centroid (별표) 표시
            centroid = np.mean(cluster_points, axis=0)
            ax1.scatter(centroid[0], centroid[1], 
                   c=[colors[cluster_id]], marker='*', s=150, 
                   edgecolors='black', linewidth=1, 
                   label=f'Cluster {cluster_id} Centroid')
        
        ax1.set_title('Coordinates Clustering + Labels + Centroids (t-SNE)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # 2. 우상단: 클러스터별 평균 coordinates 히트맵
        ax2 = axes[0, 1]
        cluster_means = []
        cluster_labels = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_coords = coordinates[cluster_mask]
            cluster_mean = np.mean(cluster_coords, axis=0)
            cluster_means.append(cluster_mean)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        cluster_means = np.array(cluster_means)
        
        # Heatmap으로 클러스터별 평균 coordinates 표시
        im = ax2.imshow(cluster_means, cmap='viridis', aspect='auto')
        ax2.set_title('Cluster Mean Coordinates', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Head Index', fontsize=10)
        ax2.set_ylabel('Cluster', fontsize=10)
        ax2.set_xticks(range(cluster_means.shape[1]))
        ax2.set_xticklabels([f'H{i}' for i in range(cluster_means.shape[1])])
        ax2.set_yticks(range(len(cluster_labels)))
        ax2.set_yticklabels(cluster_labels)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # 값 표시
        for i in range(len(cluster_labels)):
            for j in range(cluster_means.shape[1]):
                value = cluster_means[i, j]
                threshold = (cluster_means.min() + cluster_means.max()) / 2
                text_color = "white" if value > threshold else "black"
                ax2.text(j, i, f"{value:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=7, weight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        # 3. 좌하단: Label Distribution by Cluster
        ax3 = axes[1, 0]
        
        # 클러스터별 라벨 분포 계산
        label_distributions = []
        cluster_sizes = []
        
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            cluster_labels_subset = labels[cluster_mask]
            unique_labels, counts = np.unique(cluster_labels_subset, return_counts=True)
            
            # Label 0과 1의 개수 (없으면 0)
            label_0_count = counts[unique_labels == 0][0] if 0 in unique_labels else 0
            label_1_count = counts[unique_labels == 1][0] if 1 in unique_labels else 0
            
            label_distributions.append([label_0_count, label_1_count])
            cluster_sizes.append(np.sum(cluster_mask))
        
        label_distributions = np.array(label_distributions)
        
        # Stacked bar chart
        x_pos = np.arange(n_clusters)
        width = 0.6
        
        # Label 0 (네모)
        bars1 = ax3.bar(x_pos, label_distributions[:, 0], width, 
                        label='Label 0', color='darkblue', alpha=0.8)
        
        # Label 1 (동그라미)
        bars2 = ax3.bar(x_pos, label_distributions[:, 1], width, 
                        bottom=label_distributions[:, 0],
                        label='Label 1', color='lightblue', alpha=0.8)
        
        # 각 클러스터 위에 총 샘플 수 표시
        for i, (bar1, bar2, total_size) in enumerate(zip(bars1, bars2, cluster_sizes)):
            height = bar1.get_height() + bar2.get_height()
            ax3.text(bar1.get_x() + bar1.get_width()/2, height + 2,
                    f'n={total_size}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        ax3.set_title('Label Distribution by Cluster', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Cluster', fontsize=10)
        ax3.set_ylabel('Number of Samples', fontsize=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax3.legend(fontsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        
        # 4. 우하단: 클러스터 통계 요약 (새로운 subplot)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 클러스터별 주요 통계 정보
        summary_text = "Cluster Analysis Summary\n\n"
        
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            cluster_coords = coordinates[cluster_mask]
            cluster_labels_subset = labels[cluster_mask]
            
            # 클러스터 크기
            size = np.sum(cluster_mask)
            percentage = (size / len(coordinates)) * 100
            
            # 주요 Head (가장 높은 가중치)
            mean_coords = np.mean(cluster_coords, axis=0)
            dominant_head = np.argmax(mean_coords)
            max_weight = mean_coords[dominant_head]
            
            # 라벨 분포
            unique_labels, counts = np.unique(cluster_labels_subset, return_counts=True)
            label_0_count = counts[unique_labels == 0][0] if 0 in unique_labels else 0
            label_1_count = counts[unique_labels == 1][0] if 1 in unique_labels else 0
            
            summary_text += f"C{i}: {size} samples ({percentage:.1f}%)\n"
            summary_text += f"  → Dominant: H{dominant_head} ({max_weight:.3f})\n"
            summary_text += f"  → Labels: {label_0_count} vs {label_1_count}\n\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 전체 제목
        fig.suptitle('Coordinate Clustering Analysis (2x2 Layout)', fontsize=16, y=0.95)
        
        # 레이아웃 조정
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # suptitle 공간 확보
        
        plt.savefig(output_dir / 'coordinate_clustering_analysis_2x2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("2x2 coordinate clustering visualization saved")

    def _visualize_coordinates_tsne(self, coordinates, labels, output_dir):
        """Coordinates t-SNE 시각화 (클러스터링 없이)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # t-SNE 적용
        perplexity = min(30, len(coordinates)-1, max(1, len(coordinates)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(coordinates)
        
        # 1. 라벨별 색상으로 시각화
        ax1 = axes[0]
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            label_points = tsne_embeddings[label_mask]
            
            ax1.scatter(label_points[:, 0], label_points[:, 1], 
                       c=[colors[i]], label=f'Label {label}', 
                       alpha=0.7, s=50)
        
        ax1.set_title('Coordinates by True Labels (t-SNE)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Coordinates 값의 분포
        ax2 = axes[1]
        coordinates_flat = coordinates.flatten()
        ax2.hist(coordinates_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of All Coordinate Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Coordinate Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.axvline(np.mean(coordinates_flat), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(coordinates_flat):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coordinates_tsne_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Coordinates t-SNE visualization saved")

    def _visualize_head_weight_distributions(self, coordinates, output_dir):
        """각 head별 가중치 분포 히스토그램"""
        n_heads = coordinates.shape[1]
        
        # 동적 레이아웃 결정
        if n_heads <= 4:
            rows, cols = 2, 2
        elif n_heads <= 6:
            rows, cols = 2, 3
        elif n_heads <= 8:
            rows, cols = 2, 4
        else:
            rows, cols = 3, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if n_heads == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            head_weights = coordinates[:, head_idx]
            
            ax.hist(head_weights, bins=20, alpha=0.7, color=f'C{head_idx}', 
                   edgecolor='black')
            ax.set_title(f'Head {head_idx} Weight Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel('Weight Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.axvline(np.mean(head_weights), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(head_weights):.3f}')
            ax.legend(fontsize=9)
        
        # 사용하지 않는 subplot 숨기기
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Head Weight Distributions (n_heads={n_heads})', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(output_dir / 'head_weight_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Head weight distributions visualization saved")

    def _analyze_coordinate_clusters(self, coordinates, cluster_assignments, labels, 
                                   output_dir, n_clusters):
        """클러스터별 통계 분석"""
        cluster_stats = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_coords = coordinates[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            # 클러스터 통계
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'mean_coordinates': cluster_coords.mean(axis=0).tolist(),
                'std_coordinates': cluster_coords.std(axis=0).tolist(),
                'label_distribution': {}
            }
            
            # 라벨 분포
            unique_labels = np.unique(cluster_labels)
            for label in unique_labels:
                count = np.sum(cluster_labels == label)
                cluster_stats[f'cluster_{cluster_id}']['label_distribution'][f'label_{label}'] = int(count)
        
        # 통계 저장
        stats_path = output_dir / 'coordinate_cluster_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(cluster_stats, f, indent=2)
        
        # 통계 요약 시각화
        self._create_coordinate_stats_summary(cluster_stats, output_dir, n_clusters)
        
        logger.info(f"Coordinate cluster statistics saved to {stats_path}")

    def _create_coordinate_stats_summary(self, cluster_stats, output_dir, n_clusters):
        """Coordinates 통계 요약 시각화 (1x2 plot)"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 더 크게!
        
        # 1. 클러스터 크기 + 라벨 분포 통합
        ax1 = axes[0]
        
        # 클러스터별 라벨 분포 데이터 준비
        cluster_sizes = [cluster_stats[f'cluster_{i}']['size'] for i in range(n_clusters)]
        cluster_labels = [f'C{i}' for i in range(n_clusters)]
        
        # Label 분포 계산
        label_distributions = []
        for i in range(n_clusters):
            cluster_data = cluster_stats[f'cluster_{i}']
            label_dist = cluster_data['label_distribution']
            label_distributions.append(list(label_dist.values()))
        
        label_distributions = np.array(label_distributions)
        
        # Stacked bar chart
        x_pos = np.arange(n_clusters)
        width = 0.6
        
        # Label 0과 1 색상
        colors = ['darkblue', 'lightblue']
        labels = ['Label 0', 'Label 1']
        
        bottom = np.zeros(n_clusters)
        for i, (label, color) in enumerate(zip(labels, colors)):
            values = label_distributions[:, i]
            bars = ax1.bar(x_pos, values, width, bottom=bottom, 
                          color=color, label=label, alpha=0.8)
            bottom += values
            
            # 각 막대 위에 값 표시
            for bar, value in zip(bars, values):
                if value > 0:  # 값이 있을 때만 표시
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2,
                            str(int(value)), ha='center', va='center', 
                            fontweight='bold', fontsize=11, color='white')
        
        # 클러스터 크기 표시 (맨 위에)
        for i, (x, total_size) in enumerate(zip(x_pos, cluster_sizes)):
            ax1.text(x, total_size + 5, f'n={total_size}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12, color='red')
        
        ax1.set_title('Cluster Sizes + Label Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Cluster', fontsize=14)
        ax1.set_ylabel('Number of Samples', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cluster_labels)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # 2. 통계 요약 텍스트
        ax2 = axes[1]
        ax2.axis('off')
        
        # 클러스터별 평균 coordinates 정보 추가
        summary_text = "Coordinate Analysis Summary\n\n"
        summary_text += f"Total Clusters: {n_clusters}\n"
        summary_text += f"Total Samples: {sum(cluster_sizes)}\n\n"
        
        for i in range(n_clusters):
            cluster_data = cluster_stats[f'cluster_{i}']
            size = cluster_data['size']
            percentage = (size / sum(cluster_sizes)) * 100
            summary_text += f"C{i}: {size} samples ({percentage:.1f}%)\n"
            
            # 평균 coordinates 정보 추가
            mean_coords = cluster_data['mean_coordinates']
            max_head = np.argmax(mean_coords)
            max_value = mean_coords[max_head]
            summary_text += f"  → Dominant Head: H{max_head} ({max_value:.3f})\n"
        
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Coordinates Analysis Summary', fontsize=18, y=0.95)
        plt.tight_layout()
        plt.savefig(output_dir / 'coordinate_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Coordinate analysis summary visualization saved")

    def visualize_graph_structure(self, data_loader, output_dir, max_samples=10):
        """기존 그래프 구조 시각화 함수 유지 (viz_graph 인자 처리용)"""
        # 기존 clustering1_analysis.py의 visualize_graph_structure 함수와 동일
        # 여기서는 간단한 placeholder로 구현 (필요시 기존 코드 복사)
        logger.info(f"Graph structure visualization requested for {max_samples} samples")
        logger.info(f"Output directory: {output_dir}")
        # 실제 구현은 기존 코드를 참고하여 추가

def main():
    parser = argparse.ArgumentParser(description='Head별 Attention Maps 분석')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                       help='Which model to use (Full or Few)')
    parser.add_argument('--n_clusters', type=int, default=3,
                       help='Number of clusters for K-means (coordinates)')
    parser.add_argument('--max_samples', type=int, default=5,
                       help='Maximum number of samples for head attention visualization')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--viz_graph', action='store_true',
                       help='Generate graph structure visualization')
    
    args = parser.parse_args()
    
    # 원본 체크포인트 경로 저장
    original_checkpoint_path = args.checkpoint_dir
    logger.info(f"🔥 Original checkpoint path: {original_checkpoint_path}")
    
    # 자동으로 삭제된 변수 추출
    auto_del_feat, d_part = extract_deleted_features_from_checkpoint(original_checkpoint_path)
    
    # 시드값 추출
    seed_value = extract_seed_from_checkpoint(original_checkpoint_path)
    
    # 출력 디렉토리 설정 (visualization_head로 구분)
    if args.output_dir is None:
        config_folder = extract_checkpoint_config_for_folder(original_checkpoint_path)
        
        checkpoint_parent_str = str(Path(args.checkpoint_dir).parent)
        
        if '/checkpoints/' in checkpoint_parent_str:
            viz_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/visualization_head/')
        else:
            viz_parent_str = '/storage/personal/eungyeop/experiments/visualization_head/gpt2_mean/heart/Full'
        
        base_viz_parent = viz_parent_str.split(f'/{seed_value}')[0] if seed_value else viz_parent_str
        
        if seed_value is not None:
            seed_folder = f'seed_{seed_value}'
        else:
            seed_folder = 'seed_unknown'
        
        if d_part:
            clustering_folder = f'head_analysis_{d_part}'
        else:
            clustering_folder = 'head_analysis'
        
        args.output_dir = Path(base_viz_parent) / config_folder / seed_folder / clustering_folder
    
    logger.info(f"📁 Results will be saved to: {args.output_dir}")
    
    # Inference 실행
    inference = HeadAttentionInference(args.checkpoint_dir, auto_del_feat=auto_del_feat)
    
    # 데이터로더 선택
    if args.mode == 'Full':
        data_loader = inference.combined_loader
        logger.info("Using Full dataset loader")
    else:
        data_loader = inference.train_loader_few if hasattr(inference, 'train_loader_few') else inference.test_loader
        logger.info("Using Few-shot dataset loader")
    
    # 그래프 시각화 (viz_graph 인자 처리)
    if args.viz_graph:
        logger.info(f"Generating graph visualizations for {args.max_samples} samples...")
        graph_output_dir = Path(args.output_dir) / 'graph_visualization'
        inference.visualize_graph_structure(data_loader, graph_output_dir, args.max_samples)
    
    # Head별 attention maps와 coordinates 추출
    logger.info("Extracting head attention maps and coordinates...")
    head_attention_data = inference.extract_head_attention_and_coordinates(data_loader)
    
    # Head별 attention 분포 시각화
    logger.info(f"Generating head attention visualizations for {args.max_samples} samples...")
    head_viz_dir = Path(args.output_dir) / 'head_attention_visualization'
    inference.visualize_head_attention_distribution(data_loader, head_viz_dir, args.max_samples)
    
    # Coordinates 분석
    logger.info("Analyzing coordinates...")
    inference.analyze_coordinates(head_attention_data, args.output_dir)
    
    logger.info(f"Head analysis completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()