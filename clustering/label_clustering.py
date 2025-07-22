"""
Label-based Clustering Pipeline: 각 label별로 분리하여 클러스터링 수행 + 시각화
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelBasedClusteringInference:
    def __init__(self, checkpoint_dir, device='cuda', auto_del_feat=None):
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
        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            logger.info(f"🔥 Applied auto-detected del_feat: {auto_del_feat}")
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Val AUC: {self.checkpoint['val_auc']:.4f}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
        # 원본 tabular 데이터 로드
        self._load_original_tabular_data()
        
        # Label 정보 분석
        self._analyze_labels()
        
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
        
        # 전체 데이터셋 로더 준비
        results = prepare_embedding_dataloaders(self.args, self.args.source_data)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']
        self.num_classes = results['num_classes']
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_data}")
        logger.info(f"Training data size: {len(self.train_loader.dataset)}")
        logger.info(f"Validation data size: {len(self.val_loader.dataset)}")
        logger.info(f"Test data size: {len(self.test_loader.dataset)}")
    
    def _load_original_tabular_data(self):
        """원본 tabular 데이터 로드 - make_embed_dataset.py의 로직 활용"""
        
        # TabularToEmbeddingDataset 클래스 import 및 초기화
        from make_embed_dataset import TabularToEmbeddingDataset
        
        dataset_loader = TabularToEmbeddingDataset(self.args)
        
        # 원본 CSV 파일 로드 (TabularToEmbeddingDataset와 동일한 경로)
        base_path = "/storage/personal/eungyeop/dataset/table/"
        data_source =  "origin_table"
        csv_path = os.path.join(base_path, data_source, f"{self.args.source_data}.csv")
        
        if os.path.exists(csv_path):
            raw_data = pd.read_csv(csv_path)
            
            # 전처리 적용 (임베딩 생성 시와 동일하게)
            X, y = dataset_loader.preprocessing(raw_data, self.args.source_data)
            
            # X와 y를 합쳐서 전체 데이터 생성
            self.original_data = X.copy()
            self.original_data['target_binary'] = y
            
            logger.info(f"Loaded original tabular data from: {csv_path}")
            logger.info(f"Original data shape: {self.original_data.shape}")
            logger.info(f"Columns: {list(self.original_data.columns)}")
            
        else:
            raise FileNotFoundError(f"Original data not found: {csv_path}")
    
    def _analyze_labels(self):
        """데이터의 라벨 분포 분석"""
        labels = self.original_data['target_binary'].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        self.label_info = {}
        for label, count in zip(unique_labels, counts):
            self.label_info[int(label)] = int(count)
        
        logger.info("=== LABEL DISTRIBUTION ===")
        for label, count in self.label_info.items():
            percentage = (count / len(labels)) * 100
            logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
        
        self.unique_labels = unique_labels
    
    def extract_attention_maps_by_label(self, data_loader, split_name):
        """
        데이터로더에서 attention maps를 라벨별로 분리하여 추출
        
        Args:
            data_loader: 데이터로더
            split_name: 'train', 'valid', 'test'
            
        Returns:
            dict: 라벨별로 분리된 attention maps와 메타데이터
        """
        # 라벨별 데이터 저장 구조
        label_attention_data = {}
        for label in self.unique_labels:
            label_attention_data[int(label)] = {
                'layer_0': [],
                'layer_1': [], 
                'layer_2': [],
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
                
                # 모델 forward (attention weights 추출)
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                
                # Feature names 추출 (첫 번째 배치에서만)
                if label_attention_data[int(self.unique_labels[0])]['feature_names'] is None:
                    feature_names = self._extract_feature_names(batch_on_device)
                    for label in self.unique_labels:
                        label_attention_data[int(label)]['feature_names'] = ["CLS"] + feature_names
                
                # 배치 크기 확인
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    # 라벨 확인
                    if 'y' in batch:
                        label = batch['y'][sample_idx].item()
                    else:
                        continue  # 라벨이 없으면 스킵
                    
                    # 샘플 ID 확인
                    if 's_idx' in batch:
                        sample_id = batch['s_idx'][sample_idx].item()
                    else:
                        sample_id = sample_count
                    
                    # 해당 라벨의 데이터 구조에 저장
                    label_key = int(label)
                    if label_key in label_attention_data:
                        # 각 레이어별 attention map 저장
                        for layer_idx, layer_attention in enumerate(attention_weights):
                            # Multi-head attention을 평균내어 단일 attention map으로 변환
                            attention_map = layer_attention[sample_idx].mean(dim=0)  # [seq_len, seq_len]
                            attention_numpy = attention_map.detach().cpu().numpy()
                            label_attention_data[label_key][f'layer_{layer_idx}'].append(attention_numpy)
                        
                        # 샘플 ID 저장
                        label_attention_data[label_key]['sample_ids'].append(sample_id)
                    
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} {split_name} samples...")
        
        # 라벨별 샘플 수 출력
        logger.info(f"\n=== {split_name.upper()} LABEL-WISE SAMPLE COUNT ===")
        for label in self.unique_labels:
            label_key = int(label)
            count = len(label_attention_data[label_key]['sample_ids'])
            logger.info(f"Label {label}: {count} samples")
        
        return label_attention_data
    
    def _extract_attention_from_model(self, batch):
        """모델에서 attention weights와 예측값을 추출"""
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
    
    def _extract_feature_names(self, batch):
        """실제 feature names 추출"""
        # 원본 데이터에서 target_binary 컬럼 제외한 실제 feature 이름들 가져오기
        feature_columns = [col for col in self.original_data.columns if col != 'target_binary']
        
        # 배치에서 실제 feature 수 확인
        feature_count = 0
        if 'cat_name_value_embeddings' in batch:
            feature_count += batch['cat_name_value_embeddings'].shape[1]
        if 'num_prompt_embeddings' in batch:
            feature_count += batch['num_prompt_embeddings'].shape[1]
        
        # 실제 컬럼 수와 배치 feature 수가 일치하는지 확인
        if len(feature_columns) == feature_count:
            return feature_columns
        else:
            # 불일치하는 경우 가능한 만큼 실제 이름 사용, 나머지는 기본 이름
            actual_names = feature_columns[:feature_count]
            remaining = feature_count - len(actual_names)
            if remaining > 0:
                actual_names.extend([f"feature_{i}" for i in range(len(actual_names), feature_count)])
            return actual_names
    
    def perform_label_wise_clustering(self, layer_idx=2, n_clusters_per_label=4, output_dir=None):
        """
        각 라벨별로 K-means 클러스터링 수행
        
        Args:
            layer_idx: 클러스터링할 레이어 인덱스
            n_clusters_per_label: 각 라벨당 클러스터 수
            output_dir: 결과 저장 디렉토리
            
        Returns:
            dict: 라벨별 클러스터링 결과
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train attention maps를 라벨별로 추출
        train_label_attention = self.extract_attention_maps_by_label(self.train_loader, 'train')
        
        label_clustering_results = {}
        
        for label in self.unique_labels:
            label_key = int(label)
            
            logger.info(f"\n=== CLUSTERING LABEL {label} ===")
            
            # 해당 라벨의 attention maps 확인
            if len(train_label_attention[label_key]['sample_ids']) == 0:
                logger.warning(f"No samples found for label {label}, skipping...")
                continue
            
            # 최소 클러스터 수 조정
            n_samples = len(train_label_attention[label_key]['sample_ids'])
            actual_n_clusters = min(n_clusters_per_label, n_samples)
            
            if actual_n_clusters < 2:
                logger.warning(f"Label {label} has only {n_samples} samples, skipping clustering...")
                continue
            
            logger.info(f"Label {label}: {n_samples} samples, using {actual_n_clusters} clusters")
            
            # 특정 레이어의 attention maps 사용
            attention_maps = np.stack(train_label_attention[label_key][f'layer_{layer_idx}'])
            sample_ids = np.array(train_label_attention[label_key]['sample_ids'])
            feature_names = train_label_attention[label_key]['feature_names']
            
            # 평탄화 (벡터화)
            flattened_maps = attention_maps.reshape(len(attention_maps), -1)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=20)
            cluster_assignments = kmeans.fit_predict(flattened_maps)
            
            # 🔥 클러스터링 결과 저장 - attention_maps와 feature_names 추가
            label_clustering_results[label_key] = {
                'cluster_assignments': cluster_assignments,
                'cluster_centers': kmeans.cluster_centers_,
                'sample_ids': sample_ids,
                'attention_maps': attention_maps,           # ← 추가!
                'feature_names': feature_names,             # ← 추가!
                'flattened_maps': flattened_maps,           # ← t-SNE용 추가!
                'n_clusters': actual_n_clusters,
                'kmeans_model': kmeans
            }
            
            # 클러스터링 결과 출력
            for cluster_id in range(actual_n_clusters):
                cluster_count = np.sum(cluster_assignments == cluster_id)
                logger.info(f"  Label {label} - Cluster {cluster_id}: {cluster_count} samples")
        
        return label_clustering_results
    
    def visualize_label_clustering_results(self, train_results, layer_idx, output_dir):
        """
        라벨별 클러스터링 결과 시각화
        
        Args:
            train_results: perform_label_wise_clustering 결과
            layer_idx: 레이어 인덱스
            output_dir: 출력 디렉토리
        """
        if not output_dir:
            return
            
        # visualization 폴더 생성
        viz_dir = Path(output_dir) / 'visualization'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=== Label-wise Clustering Visualization ===")
        
        # 각 라벨별로 시각화
        for label_key, result in train_results.items():
            label_viz_dir = viz_dir / f'label_{label_key}'
            label_viz_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating visualizations for Label {label_key}...")
            
            # 1. 라벨별 t-SNE 분포
            self._plot_label_tsne_distribution(result, label_key, layer_idx, label_viz_dir)
            
            # 2. 라벨별 centroid heatmap
            self._plot_label_centroids_heatmap(result, label_key, layer_idx, label_viz_dir)
            
            # 3. 라벨별 클러스터 간 거리
            self._plot_label_cluster_distances(result, label_key, layer_idx, label_viz_dir)
        
        # 4. 라벨 간 비교
        self._plot_cross_label_comparison(train_results, layer_idx, viz_dir)
        
        logger.info(f"✅ All visualizations saved in: {viz_dir}")
    
    def _plot_label_tsne_distribution(self, label_result, label_key, layer_idx, output_dir):
        """특정 라벨의 t-SNE 클러스터링 분포 시각화"""
        flattened_maps = label_result['flattened_maps']
        cluster_assignments = label_result['cluster_assignments']
        n_clusters = label_result['n_clusters']
        
        # t-SNE 적용
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 시각화
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 클러스터별 색상
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = tsne_embeddings[cluster_mask]
            
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          color=colors[cluster_id], 
                          label=f'Cluster {cluster_id}',
                          alpha=0.7, s=50, marker='o')
        
        # 클러스터 센트로이드 표시
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            if np.any(cluster_mask):
                centroid_x = np.mean(tsne_embeddings[cluster_mask, 0])
                centroid_y = np.mean(tsne_embeddings[cluster_mask, 1])
                ax.scatter(centroid_x, centroid_y, marker='*', s=200, 
                          c=colors[cluster_id], edgecolors='black', linewidth=2,
                          label=f'C{cluster_id} Centroid', zorder=5)
        
        ax.set_title(f'Label {label_key} - t-SNE Clustering Distribution (Layer {layer_idx})', 
                    fontsize=14, pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 클러스터 통계 정보
        cluster_stats = []
        for cluster_id in range(n_clusters):
            count = np.sum(cluster_assignments == cluster_id)
            percentage = (count / len(cluster_assignments)) * 100
            cluster_stats.append(f"C{cluster_id}: {count} ({percentage:.1f}%)")
        
        stats_text = f"Total: {len(flattened_maps)} samples\n" + "\n".join(cluster_stats)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'label_{label_key}_tsne_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ t-SNE distribution saved for Label {label_key}")
    
    def _plot_label_centroids_heatmap(self, label_result, label_key, layer_idx, output_dir):
        """특정 라벨의 클러스터 centroid들을 heatmap으로 시각화"""
        cluster_centers = label_result['cluster_centers']
        feature_names = label_result['feature_names']
        n_clusters = label_result['n_clusters']
        
        seq_len = len(feature_names)
        centroids_reshaped = cluster_centers.reshape(-1, seq_len, seq_len)
        
        # 공통 스케일 계산
        global_vmin = centroids_reshaped.min()
        global_vmax = centroids_reshaped.max()
        
        # Overview: 모든 클러스터 한번에
        if n_clusters <= 4:
            rows, cols = 1, n_clusters
        else:
            rows, cols = 2, (n_clusters + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if n_clusters == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            if hasattr(axes, 'flatten'):
                axes = axes.flatten()
            else:
                axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        else:
            axes = axes.flatten()
        
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
        
        plt.suptitle(f'Label {label_key} Cluster Centroids (Layer {layer_idx})\nScale: {global_vmin:.3f} - {global_vmax:.3f}', 
                    fontsize=14)
        plt.tight_layout()
        
        plt.savefig(output_dir / f'label_{label_key}_centroids_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 개별 상세 플롯들도 저장
        centroids_dir = output_dir / 'centroids'
        centroids_dir.mkdir(parents=True, exist_ok=True)
        
        for i, centroid in enumerate(centroids_reshaped):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest',
                          vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f'Label {label_key} - Cluster {i} Centroid (Layer {layer_idx})', 
                        fontsize=16, pad=20)
            
            # 축 라벨 설정
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=12)
            ax.set_yticklabels(feature_names, fontsize=12)
            
            # 값 표시
            threshold = (global_vmin + global_vmax) / 2
            for row in range(len(feature_names)):
                for col in range(len(feature_names)):
                    value = centroid[row, col]
                    ax.text(col, row, f"{value:.2f}", 
                           ha="center", va="center", 
                           color="white" if value > threshold else "black", 
                           fontsize=10, weight='bold')
            
            # 컬러바
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Attention Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(centroids_dir / f'label_{label_key}_cluster_{i}_centroid.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"✅ Centroids heatmap saved for Label {label_key}")
    
    def _plot_label_cluster_distances(self, label_result, label_key, layer_idx, output_dir):
        """특정 라벨 내 클러스터 간 거리 분석"""
        cluster_centers = label_result['cluster_centers']
        n_clusters = label_result['n_clusters']
        
        if n_clusters < 2:
            logger.warning(f"Label {label_key} has only {n_clusters} clusters, skipping distance analysis")
            return
        
        # 클러스터 간 거리 계산
        distance_matrix = pairwise_distances(cluster_centers, metric='euclidean')
        
        # 클러스터 쌍별 거리 추출
        cluster_pairs = []
        distances = []
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                cluster_pairs.append(f"C{i}-C{j}")
                distances.append(distance_matrix[i, j])
        
        # 거리 순으로 정렬
        sorted_indices = np.argsort(distances)
        sorted_pairs = [cluster_pairs[i] for i in sorted_indices]
        sorted_distances = [distances[i] for i in sorted_indices]
        
        # 3개 서브플롯 생성
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Distance Matrix
        ax1 = plt.subplot(1, 3, 1)
        im = ax1.imshow(distance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0)
        
        # 값 표시
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    text_color = 'white' if distance_matrix[i, j] > np.median(distance_matrix) else 'black'
                    ax1.text(j, i, f'{distance_matrix[i, j]:.2f}', 
                            ha="center", va="center", color=text_color, 
                            fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('Cluster ID', fontsize=12)
        ax1.set_ylabel('Cluster ID', fontsize=12)
        ax1.set_title(f'Distance Matrix\n(Label {label_key})', fontsize=14)
        ax1.set_xticks(range(n_clusters))
        ax1.set_yticks(range(n_clusters))
        ax1.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax1.set_yticklabels([f'C{i}' for i in range(n_clusters)])
        
        cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar1.set_label('Distance', rotation=270, labelpad=15)
        
        # 2. Pairwise Distances (가까운 순)
        ax2 = plt.subplot(1, 3, 2)
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(sorted_distances)))
        y_positions = np.arange(len(sorted_distances))[::-1]
        bars = ax2.barh(y_positions, sorted_distances, color=colors, alpha=0.8, height=0.7)
        
        # 값 표시
        for i, (bar, distance) in enumerate(zip(bars, sorted_distances)):
            ax2.text(bar.get_width() + max(sorted_distances) * 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{distance:.3f}', ha='left', va='center', 
                    fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('Distance', fontsize=12)
        ax2.set_title(f'Pairwise Distances\n(Label {label_key})', fontsize=14)
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(sorted_pairs, fontsize=10)
        ax2.grid(True, alpha=0.2, axis='x')
        
        # 3. 통계 정보
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')
        
        stats_text = f"Label {label_key} Distance Analysis\n\n"
        stats_text += f"Clusters: {n_clusters}\n"
        stats_text += f"Total pairs: {len(distances)}\n\n"
        stats_text += f"Distance Statistics:\n"
        stats_text += f"Mean: {np.mean(distances):.3f}\n"
        stats_text += f"Std: {np.std(distances):.3f}\n"
        stats_text += f"Min: {np.min(distances):.3f}\n"
        stats_text += f"Max: {np.max(distances):.3f}\n\n"
        stats_text += f"Most similar: {sorted_pairs[0]}\n"
        stats_text += f"Most different: {sorted_pairs[-1]}"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        fig.suptitle(f'Label {label_key}: Cluster Distance Analysis (Layer {layer_idx})', 
                    fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        plt.savefig(output_dir / f'label_{label_key}_cluster_distances.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Cluster distances saved for Label {label_key}")
    
    def _plot_cross_label_comparison(self, train_results, layer_idx, viz_dir):
        """라벨 간 centroid 비교 시각화 - 깔끔한 subplot 구성"""
        comparison_dir = viz_dir / 'cross_label_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        labels = sorted(list(train_results.keys()))  # 라벨 정렬
        if len(labels) < 2:
            logger.warning("Need at least 2 labels for cross-label comparison")
            return
        
        # 1. 라벨별 클러스터 수 정보 수집
        label_cluster_info = {}
        max_clusters = 0
        for label_key, result in train_results.items():
            n_clusters = result['n_clusters']
            label_cluster_info[label_key] = n_clusters
            max_clusters = max(max_clusters, n_clusters)
        
        # 2. 깔끔한 subplot 구성: 세로축(Labels) x 가로축(Clusters)
        fig, axes = plt.subplots(len(labels), max_clusters, 
                                figsize=(4*max_clusters, 4*len(labels)),
                                gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
        
        # axes를 2D 배열로 만들기 (다양한 경우 대비)
        if len(labels) == 1 and max_clusters == 1:
            axes = np.array([[axes]])
        elif len(labels) == 1:
            axes = axes.reshape(1, -1)
        elif max_clusters == 1:
            axes = axes.reshape(-1, 1)
        
        # 전체 데이터의 공통 스케일 계산
        all_centroids_for_scale = []
        for label_key, result in train_results.items():
            cluster_centers = result['cluster_centers']
            feature_names = result['feature_names']
            seq_len = len(feature_names)
            for cluster_id in range(result['n_clusters']):
                centroid = cluster_centers[cluster_id].reshape(seq_len, seq_len)
                all_centroids_for_scale.append(centroid)
        
        if all_centroids_for_scale:
            all_centroids_for_scale = np.stack(all_centroids_for_scale)
            global_vmin = all_centroids_for_scale.min()
            global_vmax = all_centroids_for_scale.max()
        else:
            global_vmin, global_vmax = 0, 1
        
        # 3. 각 위치에 centroid 그리기
        for row, label_key in enumerate(labels):
            result = train_results[label_key]
            cluster_centers = result['cluster_centers']
            feature_names = result['feature_names']
            seq_len = len(feature_names)
            n_clusters = result['n_clusters']
            
            for col in range(max_clusters):
                ax = axes[row, col]
                
                if col < n_clusters:
                    # 해당 클러스터가 존재하는 경우
                    centroid = cluster_centers[col].reshape(seq_len, seq_len)
                    
                    im = ax.imshow(centroid, cmap='viridis', interpolation='nearest',
                                  vmin=global_vmin, vmax=global_vmax)
                    
                    # 깔끔한 제목 (박스 없이)
                    ax.set_title(f'Label {label_key} | Cluster {col}', 
                                fontsize=12, fontweight='bold', pad=10)
                    
                    # 축 설정 - feature가 많으면 샘플링
                    if len(feature_names) <= 15:
                        # 15개 이하면 모든 feature 표시
                        ax.set_xticks(np.arange(len(feature_names)))
                        ax.set_yticks(np.arange(len(feature_names)))
                        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(feature_names, fontsize=8)
                    else:
                        # 15개 초과면 일부만 표시
                        step = max(1, len(feature_names) // 10)
                        tick_positions = np.arange(0, len(feature_names), step)
                        ax.set_xticks(tick_positions)
                        ax.set_yticks(tick_positions)
                        ax.set_xticklabels([feature_names[i] for i in tick_positions], 
                                          rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels([feature_names[i] for i in tick_positions], fontsize=8)
                    
                    # subplot 테두리
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('black')
                    
                else:
                    # 해당 클러스터가 존재하지 않는 경우
                    ax.text(0.5, 0.5, f'Label {label_key}\nNo Cluster {col}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=11, color='gray', style='italic')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # 빈 subplot 테두리
                    for spine in ax.spines.values():
                        spine.set_linewidth(1)
                        spine.set_color('lightgray')
                        spine.set_linestyle('--')
        
        # 4. 전체 제목만 (상단 중앙에 깔끔하게)
        fig.suptitle(f'Cross-Label Centroid Comparison (Layer {layer_idx})\n'
                    f'Attention Weight Scale: {global_vmin:.3f} - {global_vmax:.3f}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 5. 공통 컬러바 (우측에 세로로)
        if 'im' in locals():
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
        
        # 6. 저장
        plt.savefig(comparison_dir / f'all_clusters_cross_label_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 7. 전체 overview 
        self._plot_overall_comparison(train_results, layer_idx, comparison_dir)
        
        logger.info(f"✅ Cross-label comparison (clean grid) saved for {len(labels)} labels x {max_clusters} clusters")
    
    def _plot_overall_comparison(self, train_results, layer_idx, comparison_dir):
        """전체 라벨별 클러스터링 결과 overview"""
        labels = list(train_results.keys())
        
        # 1. 라벨별 클러스터 수 비교
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # (A) 라벨별 클러스터 수와 샘플 수
        ax1 = axes[0, 0]
        label_names = [f'Label {label}' for label in labels]
        cluster_counts = [train_results[label]['n_clusters'] for label in labels]
        sample_counts = [len(train_results[label]['sample_ids']) for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cluster_counts, width, label='Clusters', alpha=0.8)
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, sample_counts, width, label='Samples', 
                            alpha=0.8, color='orange')
        
        ax1.set_xlabel('Labels')
        ax1.set_ylabel('Number of Clusters', color='blue')
        ax1_twin.set_ylabel('Number of Samples', color='orange')
        ax1.set_title('Clusters and Samples per Label')
        ax1.set_xticks(x)
        ax1.set_xticklabels(label_names)
        
        # 값 표시
        for bar, count in zip(bars1, cluster_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        for bar, count in zip(bars2, sample_counts):
            ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                         str(count), ha='center', va='bottom')
        
        # (B) 라벨별 평균 클러스터 간 거리
        ax2 = axes[0, 1]
        avg_distances = []
        for label in labels:
            if train_results[label]['n_clusters'] >= 2:
                cluster_centers = train_results[label]['cluster_centers']
                distance_matrix = pairwise_distances(cluster_centers, metric='euclidean')
                # 대각선 제거하고 평균 계산
                mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
                avg_dist = distance_matrix[mask].mean()
                avg_distances.append(avg_dist)
            else:
                avg_distances.append(0)
        
        bars = ax2.bar(label_names, avg_distances, alpha=0.8, color='lightgreen')
        ax2.set_title('Average Inter-cluster Distance by Label')
        ax2.set_ylabel('Average Distance')
        
        for bar, dist in zip(bars, avg_distances):
            if dist > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_distances)*0.01,
                        f'{dist:.3f}', ha='center', va='bottom')
        
        # (C) 라벨별 클러스터 크기 분포
        ax3 = axes[1, 0]
        cluster_size_data = []
        cluster_size_labels = []
        
        for label in labels:
            cluster_assignments = train_results[label]['cluster_assignments']
            n_clusters = train_results[label]['n_clusters']
            
            sizes = []
            for cluster_id in range(n_clusters):
                size = np.sum(cluster_assignments == cluster_id)
                sizes.append(size)
            
            cluster_size_data.append(sizes)
            cluster_size_labels.append(f'Label {label}')
        
        ax3.boxplot(cluster_size_data, labels=cluster_size_labels)
        ax3.set_title('Cluster Size Distribution by Label')
        ax3.set_ylabel('Cluster Size (# samples)')
        
        # (D) 요약 통계 테이블
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"Label-wise Clustering Summary (Layer {layer_idx})\n\n"
        
        total_samples = sum(len(train_results[label]['sample_ids']) for label in labels)
        total_clusters = sum(train_results[label]['n_clusters'] for label in labels)
        
        summary_text += f"Overall Statistics:\n"
        summary_text += f"Total Labels: {len(labels)}\n"
        summary_text += f"Total Samples: {total_samples}\n"
        summary_text += f"Total Clusters: {total_clusters}\n"
        summary_text += f"Avg Clusters per Label: {total_clusters/len(labels):.1f}\n\n"
        
        summary_text += "Per-Label Details:\n"
        for label in labels:
            n_samples = len(train_results[label]['sample_ids'])
            n_clusters = train_results[label]['n_clusters']
            avg_cluster_size = n_samples / n_clusters
            summary_text += f"Label {label}: {n_samples} samples, {n_clusters} clusters "
            summary_text += f"(avg {avg_cluster_size:.1f} per cluster)\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Label-wise Clustering Overview (Layer {layer_idx})', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(comparison_dir / f'overall_comparison_layer_{layer_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def map_to_label_clusters_by_distance(self, attention_data, train_clustering_results, layer_idx):
        """
        라벨별 Train centroid와의 거리 기반으로 클러스터 할당
        
        Args:
            attention_data: extract_attention_maps_by_label 결과
            train_clustering_results: 라벨별 Train 클러스터링 결과
            layer_idx: 사용할 레이어 인덱스
            
        Returns:
            dict: 라벨별 클러스터 할당 결과
        """
        label_mapping_results = {}
        
        for label in self.unique_labels:
            label_key = int(label)
            
            # Train에서 해당 라벨의 클러스터링 결과가 있는지 확인
            if label_key not in train_clustering_results:
                logger.warning(f"No train clustering result for label {label}, skipping...")
                continue
            
            # Valid/Test에서 해당 라벨의 데이터가 있는지 확인
            if len(attention_data[label_key]['sample_ids']) == 0:
                logger.warning(f"No samples found for label {label} in this split, skipping...")
                continue
            
            attention_maps = np.stack(attention_data[label_key][f'layer_{layer_idx}'])
            sample_ids = np.array(attention_data[label_key]['sample_ids'])
            
            # 평탄화
            flattened_maps = attention_maps.reshape(len(attention_maps), -1)
            
            # Train centroids 가져오기
            train_centroids = train_clustering_results[label_key]['cluster_centers']
            
            # 각 샘플과 train centroids 간의 거리 계산
            distances = pairwise_distances(flattened_maps, train_centroids, metric='euclidean')
            
            # 가장 가까운 클러스터 할당
            cluster_assignments = np.argmin(distances, axis=1)
            
            label_mapping_results[label_key] = {
                'cluster_assignments': cluster_assignments,
                'sample_ids': sample_ids,
                'distances': distances,
                'n_clusters': train_clustering_results[label_key]['n_clusters']
            }
            
            logger.info(f"Label {label}: Mapped {len(sample_ids)} samples to clusters")
        
        return label_mapping_results
    
    def save_label_cluster_csvs(self, clustering_results, output_dir, n_clusters_per_label, split_name):
        """
        라벨별 클러스터별로 원본 tabular 데이터를 CSV 저장
        
        Args:
            clustering_results: 라벨별 클러스터링 결과
            output_dir: 저장할 디렉토리
            n_clusters_per_label: 라벨당 클러스터 수
            split_name: 'train', 'valid', 'test'
        """
        # {split}_label_clustering_{n_clusters} 폴더 생성
        split_dir = output_dir / f'{split_name}_label_clustering_{n_clusters_per_label}'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        total_samples = 0
        summary = {
            'total_samples': 0,
            'n_clusters_per_label': n_clusters_per_label,
            'split': split_name,
            'label_cluster_summary': {}
        }
        
        for label_key, result in clustering_results.items():
            # 라벨별 폴더 생성
            label_dir = split_dir / f'label_{label_key}'
            label_dir.mkdir(parents=True, exist_ok=True)
            
            cluster_assignments = result['cluster_assignments']
            sample_ids = result['sample_ids']
            n_clusters = result['n_clusters']
            
            label_summary = {
                'total_samples': len(sample_ids),
                'n_clusters': n_clusters,
                'cluster_summary': {}
            }
            
            for cluster_id in range(n_clusters):
                # 각 클러스터별 폴더 생성 (라벨 폴더 하위)
                cluster_dir = label_dir / f'cluster_{cluster_id}'
                cluster_dir.mkdir(parents=True, exist_ok=True)
                
                # 해당 클러스터에 속한 샘플들의 인덱스
                cluster_mask = cluster_assignments == cluster_id
                cluster_sample_ids = sample_ids[cluster_mask]
                
                if len(cluster_sample_ids) > 0:
                    # 원본 데이터에서 해당 인덱스의 샘플들 추출
                    cluster_data = self.original_data.iloc[cluster_sample_ids].copy()
                    
                    # 클러스터 정보 추가
                    cluster_data['label'] = label_key
                    cluster_data['cluster_id'] = cluster_id
                    cluster_data['original_index'] = cluster_sample_ids
                    
                    # CSV 저장 (클러스터 폴더 안에)
                    csv_filename = f'label_{label_key}_cluster_{cluster_id}_{split_name}.csv'
                    csv_path = cluster_dir / csv_filename
                    cluster_data.to_csv(csv_path, index=False)
                    
                    logger.info(f"Saved {split_name} label {label_key} cluster {cluster_id}: {len(cluster_data)} samples to {csv_path}")
                    
                    # 요약 정보 업데이트
                    label_summary['cluster_summary'][f'cluster_{cluster_id}'] = {
                        'sample_count': int(len(cluster_sample_ids))
                    }
                    total_samples += len(cluster_sample_ids)
                    
                else:
                    logger.warning(f"{split_name} label {label_key} cluster {cluster_id} has no samples")
                    label_summary['cluster_summary'][f'cluster_{cluster_id}'] = {
                        'sample_count': 0
                    }
            
            summary['label_cluster_summary'][f'label_{label_key}'] = label_summary
        
        summary['total_samples'] = total_samples
        
        # 전체 요약 정보 저장
        summary_path = split_dir / 'label_clustering_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"{split_name} label clustering summary saved to: {summary_path}")
        logger.info(f"✅ All {split_name} label cluster CSVs saved in: {split_dir}")
    
    def save_whole_test_set_by_label(self, test_attention_data, output_dir):
        """
        Test set을 라벨별로 저장 (클러스터 분할 없음)
        
        Args:
            test_attention_data: extract_attention_maps_by_label 결과
            output_dir: 저장할 디렉토리
        """
        test_dir = output_dir / 'test_full_by_label'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_summary = {
            'label_summary': {}
        }
        
        for label_key, attention_data in test_attention_data.items():
            if len(attention_data['sample_ids']) == 0:
                continue
                
            test_sample_ids = np.array(attention_data['sample_ids'])
            
            # Test 데이터 추출
            test_data = self.original_data.iloc[test_sample_ids].copy()
            test_data['label'] = label_key
            test_data['original_index'] = test_sample_ids
            
            # 라벨별 폴더 생성 및 저장
            label_dir = test_dir / f'label_{label_key}'
            label_dir.mkdir(parents=True, exist_ok=True)
            
            test_csv_path = label_dir / f'test_label_{label_key}_full.csv'
            test_data.to_csv(test_csv_path, index=False)
            
            logger.info(f"Saved full test set for label {label_key}: {len(test_data)} samples to {test_csv_path}")
            
            test_summary['label_summary'][f'label_{label_key}'] = {
                'sample_count': int(len(test_sample_ids))
            }
        
        # Test set 요약 정보 저장
        test_summary_path = test_dir / 'test_by_label_summary.json'
        with open(test_summary_path, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        logger.info(f"Test by label summary saved to: {test_summary_path}")
    
    def save_full_population_sets_by_label(self, train_results, valid_results, output_dir):
        """
        전체 population용 train/valid set을 라벨별로 저장 (클러스터 구분 없음)
        
        Args:
            train_results: Train 클러스터링 결과
            valid_results: Valid 클러스터 할당 결과
            output_dir: 저장할 디렉토리
        """
        # Full population 폴더 생성
        full_pop_dir = output_dir / 'full_population_by_label'
        full_pop_dir.mkdir(parents=True, exist_ok=True)
        
        full_pop_summary = {
            'train_by_label': {},
            'valid_by_label': {}
        }
        
        # 1. Full train set을 라벨별로 저장
        for label_key, result in train_results.items():
            train_sample_ids = result['sample_ids']
            train_data = self.original_data.iloc[train_sample_ids].copy()
            train_data['label'] = label_key
            train_data['original_index'] = train_sample_ids
            
            label_dir = full_pop_dir / f'label_{label_key}'
            label_dir.mkdir(parents=True, exist_ok=True)
            
            train_csv_path = label_dir / f'train_label_{label_key}_full.csv'
            train_data.to_csv(train_csv_path, index=False)
            logger.info(f"Saved full train set for label {label_key}: {len(train_data)} samples to {train_csv_path}")
            
            full_pop_summary['train_by_label'][f'label_{label_key}'] = {
                'sample_count': int(len(train_sample_ids))
            }
        
        # 2. Full valid set을 라벨별로 저장
        for label_key, result in valid_results.items():
            valid_sample_ids = result['sample_ids']
            valid_data = self.original_data.iloc[valid_sample_ids].copy()
            valid_data['label'] = label_key
            valid_data['original_index'] = valid_sample_ids
            
            label_dir = full_pop_dir / f'label_{label_key}'
            label_dir.mkdir(parents=True, exist_ok=True)
            
            valid_csv_path = label_dir / f'valid_label_{label_key}_full.csv'
            valid_data.to_csv(valid_csv_path, index=False)
            logger.info(f"Saved full valid set for label {label_key}: {len(valid_data)} samples to {valid_csv_path}")
            
            full_pop_summary['valid_by_label'][f'label_{label_key}'] = {
                'sample_count': int(len(valid_sample_ids))
            }
        
        # 3. 요약 정보 저장
        summary_path = full_pop_dir / 'full_population_by_label_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(full_pop_summary, f, indent=2)
        
        logger.info(f"Full population by label summary saved to: {summary_path}")
        logger.info(f"✅ Full population train/valid sets by label saved in: {full_pop_dir}")
    
    def run_label_wise_clustering_pipeline(self, layer_idx=2, n_clusters_per_label=4, output_dir=None, enable_visualization=True):
        """
        라벨별 클러스터링 전체 파이프라인 실행
        
        Args:
            layer_idx: 클러스터링할 레이어 인덱스
            n_clusters_per_label: 각 라벨당 클러스터 수
            output_dir: 결과 저장 디렉토리
            enable_visualization: 시각화 활성화 여부
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Train 라벨별 클러스터링
        logger.info("=== Step 1: Label-wise Train Clustering ===")
        train_results = self.perform_label_wise_clustering(layer_idx, n_clusters_per_label, output_dir)
        
        # Train centroids를 라벨별로 저장
        centroids_dir = output_dir / 'label_centroids'
        centroids_dir.mkdir(parents=True, exist_ok=True)
        
        for label_key, result in train_results.items():
            centroids_path = centroids_dir / f'label_{label_key}_centroids.npy'
            np.save(centroids_path, result['cluster_centers'])
            logger.info(f"Train centroids for label {label_key} saved to: {centroids_path}")
        
        # 🔥 시각화 추가
        if enable_visualization:
            logger.info("=== Step 1.5: Label-wise Clustering Visualization ===")
            self.visualize_label_clustering_results(train_results, layer_idx, output_dir)
        
        # Train CSV 저장
        self.save_label_cluster_csvs(train_results, output_dir, n_clusters_per_label, 'train')
        
        # 2. Valid attention maps 추출 및 라벨별 클러스터 매핑
        logger.info("=== Step 2: Label-wise Valid Cluster Mapping ===")
        valid_attention_data = self.extract_attention_maps_by_label(self.val_loader, 'valid')
        valid_results = self.map_to_label_clusters_by_distance(
            valid_attention_data, 
            train_results, 
            layer_idx
        )
        
        # Valid CSV 저장
        self.save_label_cluster_csvs(valid_results, output_dir, n_clusters_per_label, 'valid')
        
        # 3. Test attention maps 추출 및 라벨별 저장
        logger.info("=== Step 3: Label-wise Test Set Saving ===")
        test_attention_data = self.extract_attention_maps_by_label(self.test_loader, 'test')
        self.save_whole_test_set_by_label(test_attention_data, output_dir)
        
        # 4. Full population 라벨별 저장
        logger.info("=== Step 4: Saving Full Population Train/Valid Sets by Label ===")
        self.save_full_population_sets_by_label(train_results, valid_results, output_dir)
        
        logger.info("✅ Complete label-wise clustering pipeline finished!")
        
        # 결과 요약
        logger.info("\n=== LABEL-WISE CLUSTERING SUMMARY ===")
        for label_key in train_results.keys():
            train_count = len(train_results[label_key]['sample_ids'])
            valid_count = len(valid_results[label_key]['sample_ids']) if label_key in valid_results else 0
            test_count = len(test_attention_data[label_key]['sample_ids']) if label_key in test_attention_data else 0
            n_clusters = train_results[label_key]['n_clusters']
            
            logger.info(f"Label {label_key}: Train={train_count}, Valid={valid_count}, Test={test_count}, Clusters={n_clusters}")


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
    """체크포인트 파일명에서 설정 정보를 추출해서 폴더명으로 변환 (D:[...] 제외)"""
    filename = Path(checkpoint_path).stem
    
    # 날짜/시간 패턴 제거 (20250617_173832 형태)
    import re
    
    # 🔥 D:[...] 패턴들을 모두 제거하여 기본 설정만 추출
    filename_clean = re.sub(r'_\d{8}_\d{6}', '', filename)
    
    # 여러 D: 패턴 제거
    d_patterns = [
        r'_D:\[[^\]]*\]',        # _D:[...] 형식
        r'_D_\[[^\]]*\]',        # _D_[...] 형식
        r'_D-\[[^\]]*\]',        # _D-[...] 형식
    ]
    
    for pattern in d_patterns:
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


def main():
    parser = argparse.ArgumentParser(description='Label-wise Clustering Pipeline with Visualization')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--layer_idx', type=int, default=2,
                       help='Layer index for clustering (default: 2)')
    parser.add_argument('--n_clusters_per_label', type=int, default=3,
                       help='Number of clusters per label for K-means')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--train_only', action='store_true',
                       help='Only perform train clustering (skip valid/test)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualization for label-wise clustering (default: True)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization (overrides --visualize)')
    
    args = parser.parse_args()
     # 🔥 원본 체크포인트 경로 저장 (쉘 해석 전)
    original_checkpoint_path = args.checkpoint_dir
    logger.info(f"🔥 Original checkpoint path: {original_checkpoint_path}")
    auto_del_feat, d_part = extract_deleted_features_from_checkpoint(original_checkpoint_path)
    
    # 시각화 설정
    enable_visualization = args.visualize and not args.no_visualize
    checkpoint_file = Path(original_checkpoint_path)
    if not checkpoint_file.exists():
        # 파일이 없으면 패턴으로 찾아보기
        parent_dir = checkpoint_file.parent
        
        # D:[Age] -> D:['Age'] 또는 D:["Age"] 패턴 찾기
        base_name = checkpoint_file.stem
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
    
    # 시각화 설정
    enable_visualization = args.visualize and not args.no_visualize
    
    # 출력 디렉토리 설정: label_clustering 폴더로 변경
    if args.output_dir is None:
        # 기본 설정 폴더 (D:[...] 제외) - 원본 경로 사용
        config_folder = extract_checkpoint_config_for_folder(original_checkpoint_path)
        
        # 체크포인트 파일의 부모 디렉토리 경로를 가져와서 변환
        checkpoint_parent_str = str(Path(args.checkpoint_dir).parent)
        
        # checkpoints를 label_clustering으로 변경
        if '/checkpoints/' in checkpoint_parent_str:
            clustering_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/label_clustering/')
        else:
            # fallback: 직접 경로 구성
            clustering_parent_str = '/storage/personal/eungyeop/experiments/label_clustering/gpt2_mean/heart/Full'
        
        # 🔥 clustering 폴더명에 D:[...] 정보 추가
        if d_part:
            clustering_folder = f'label_clustering_{args.n_clusters_per_label}_{d_part}'
        else:
            clustering_folder = f'label_clustering_{args.n_clusters_per_label}'
        
        # 최종 출력 경로: .../label_clustering/.../config_folder/label_clustering_{n_clusters_per_label}_D:[...]/
        args.output_dir = Path(clustering_parent_str) / config_folder / clustering_folder
    
    logger.info(f"📁 Results will be saved to: {args.output_dir}")
    logger.info(f"🎨 Visualization: {'Enabled' if enable_visualization else 'Disabled'}")

    # 🔥 Pipeline 실행 시 자동 추출된 삭제 변수 적용
    pipeline = LabelBasedClusteringInference(args.checkpoint_dir, auto_del_feat=auto_del_feat)
    
    if args.train_only:
        # Train만 수행
        logger.info("Performing label-wise train-only clustering...")
        train_results = pipeline.perform_label_wise_clustering(
            layer_idx=args.layer_idx,
            n_clusters_per_label=args.n_clusters_per_label,
            output_dir=args.output_dir
        )
        
        # Train centroids를 라벨별로 저장
        centroids_dir = args.output_dir / 'label_centroids'
        centroids_dir.mkdir(parents=True, exist_ok=True)
        
        for label_key, result in train_results.items():
            centroids_path = centroids_dir / f'label_{label_key}_centroids.npy'
            np.save(centroids_path, result['cluster_centers'])
            logger.info(f"Train centroids for label {label_key} saved to: {centroids_path}")
        
        # 🔥 시각화 추가 (train_only 모드에서도)
        if enable_visualization:
            logger.info("=== Train-only Visualization ===")
            pipeline.visualize_label_clustering_results(train_results, args.layer_idx, args.output_dir)
        
        # Train CSV 저장
        pipeline.save_label_cluster_csvs(train_results, args.output_dir, args.n_clusters_per_label, 'train')
        
        logger.info(f"Label-wise train clustering completed! Results saved to {args.output_dir}")
    else:
        # 전체 파이프라인 실행
        pipeline.run_label_wise_clustering_pipeline(
            layer_idx=args.layer_idx,
            n_clusters_per_label=args.n_clusters_per_label,
            output_dir=args.output_dir,
            enable_visualization=enable_visualization
        )
        
        logger.info(f"Complete label-wise clustering pipeline completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()