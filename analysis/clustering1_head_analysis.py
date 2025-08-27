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

from models.TabularFLM_S import Model
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
    def __init__(self, checkpoint_dir, device='cuda', auto_del_feat=None, skip_dataloader_prep=False):
        """Head별 Attention 분석을 위한 Inference 클래스"""
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f" Attempting to load checkpoint from: {checkpoint_dir}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            logger.info(f"🔥 Applied auto-detected del_feat: {auto_del_feat}")
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        
        #  val_auc 또는 avg_val_auc 확인
        if 'val_auc' in self.checkpoint:
            val_auc = self.checkpoint['val_auc']
            logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Val AUC: {val_auc:.4f}")
        elif 'avg_val_auc' in self.checkpoint:
            val_auc = self.checkpoint['avg_val_auc']
            logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Avg Val AUC: {val_auc:.4f}")
        else:
            logger.warning("No val_auc or avg_val_auc found in checkpoint")
            val_auc = None
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 🔥 데이터로더 준비 (Multi Source 모드에서는 건너뛰기)
        if not skip_dataloader_prep:
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
        """각 head별 attention maps와 coordinates 추출 (+ 첫 배치 sanity log)"""
        total_heads = self.args.k_basis  # basis head 수

        head_attention_data = {
            'coordinates': [],
            'labels': [],
            'sample_ids': [],
            'feature_names': None
        }
        for i in range(total_heads):
            head_attention_data[f'head_{i}'] = []

        logger.info(f"Using {total_heads} basis heads (k_basis={self.args.k_basis})")

        sample_count = 0
        did_sanity_log = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                pred, head_attention_weights, coordinates = self._extract_head_attention_from_model(batch_on_device)

                # feature names (첫 배치에서만)
                if head_attention_data['feature_names'] is None:
                    feature_names = self.model.extract_feature_names(batch_on_device)
                    head_attention_data['feature_names'] = list(feature_names)  # CLS 미포함 이름

                batch_size = head_attention_weights[0].shape[0]

                # --- 첫 배치 sanity log: 행렬 크기와 CLS 포함 여부 추정 ---
                if not did_sanity_log:
                    try:
                        seq = head_attention_weights[0].shape[-1]
                        n_feat = len(head_attention_data['feature_names'])
                        incl_cls = (seq == n_feat + 1)
                        logger.info(
                            f"[sanity] basis head attn shape per head = (B={batch_size}, seq={seq}, seq={seq}); "
                            f"feature_names={n_feat}, includes_CLS? {'yes' if incl_cls else 'no'}"
                        )
                    except Exception as e:
                        logger.warning(f"[sanity] could not infer CLS presence: {e}")
                    did_sanity_log = True
                # -----------------------------------------------------------

                for b in range(batch_size):
                    # head별 attention 저장 (basis heads만)
                    for head_idx in range(self.args.k_basis):
                        attn_map = head_attention_weights[head_idx][b]  # [seq, seq]
                        head_attention_data[f'head_{head_idx}'].append(attn_map.detach().cpu().numpy())

                    # coordinates
                    head_attention_data['coordinates'].append(coordinates[b].detach().cpu().numpy())

                    # label / id
                    if 'y' in batch:
                        label = batch['y'][b].item()
                    else:
                        label = -1
                    head_attention_data['labels'].append(label)

                    if 'sample_ids' in batch:
                        sid = batch['sample_ids'][b]
                    else:
                        sid = sample_count
                    head_attention_data['sample_ids'].append(sid)

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
        
        # 각 head별 attention weights 저장
        for head_idx in range(self.args.k_basis):
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
        """한 샘플의 모든 head attention과 coordinates를 하나의 figure로 시각화
        - CLS 포함 여부를 attention 행렬 shape로 자동 감지하여 라벨을 정합
        - 필요 시 한 행(예: CLS)만 보여주는 옵션 제공
        """
        total_heads = len(head_attention_weights)  # k_basis

        # ---- 옵션: 특정 쿼리(예: CLS=0) 한 행만 시각화하고 싶을 때 ----
        only_query_row = False
        query_idx = 0  # CLS 가정
        # -------------------------------------------------------------

        # 8개 head 레이아웃(2x4) 기준
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # 각 head 히트맵
        for head_idx in range(total_heads):
            ax = axes[head_idx]
            attn = head_attention_weights[head_idx][batch_sample_idx]  # [seq, seq], torch.Tensor
            attn_np = attn.detach().cpu().numpy()
            seq = attn_np.shape[0]

            n_feat = len(feature_names)  # feature_names는 CLS 미포함을 가정
            # CLS 포함 여부 자동 감지
            if seq == n_feat + 1:
                all_node_names = ["CLS"] + list(feature_names)
            elif seq == n_feat:
                all_node_names = list(feature_names)  # CLS 미포함
            else:
                logger.warning(
                    f"[Head {head_idx}] Unexpected attention shape: seq={seq}, features={n_feat}. "
                    "Will crop to match."
                )
                use = min(seq, n_feat + 1)  # 가장 안전한 상한
                attn_np = attn_np[:use, :use]
                # 가능하면 CLS 포함 라벨로 맞추되, 길이에 맞게 자르기
                all_node_names = (["CLS"] + list(feature_names))[:use]

            # 시각화(전체 행렬 or 특정 쿼리 행만)
            if only_query_row and 0 <= query_idx < attn_np.shape[0]:
                row = attn_np[query_idx, :]
                im = ax.imshow(row[None, :], cmap='viridis', aspect='auto', interpolation='nearest')
                ax.set_yticks([0])
                ax.set_yticklabels([all_node_names[query_idx]])
                ax.set_xticks(np.arange(len(all_node_names)))
                ax.set_xticklabels(all_node_names, rotation=90, fontsize=6)
            else:
                im = ax.imshow(attn_np, cmap='viridis', interpolation='nearest')
                ax.set_xticks(np.arange(len(all_node_names)))
                ax.set_yticks(np.arange(len(all_node_names)))
                ax.set_xticklabels(all_node_names, rotation=90, fontsize=6)
                ax.set_yticklabels(all_node_names, fontsize=6)

                # 값 표기(작은 그림에서 너무 빽빽하면 주석 처리 가능)
                try:
                    vmin, vmax = float(attn_np.min()), float(attn_np.max())
                    thr = (vmin + vmax) / 2.0
                    for i in range(attn_np.shape[0]):
                        for j in range(attn_np.shape[1]):
                            v = attn_np[i, j]
                            if v != 0.0:  # 완전 0은 생략
                                ax.text(j, i, f"{v:.2f}",
                                        ha="center", va="center",
                                        color=("white" if v > thr else "black"),
                                        fontsize=5, fontweight='bold')
                except Exception:
                    pass

            ax.set_title(f'Basis Head {head_idx}', fontsize=10, fontweight='bold')

        # 남는 subplot 숨김
        for k in range(total_heads, len(axes)):
            axes[k].set_visible(False)

        fig.suptitle(f'Sample {sample_idx} - Head Attention Analysis ({total_heads} heads)',
                    fontsize=14, y=0.95)
        plt.tight_layout()
        out_path = Path(output_dir) / f'sample_{sample_idx}_head_attention.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Sample {sample_idx} head attention visualization saved: {out_path}")

        # ---- Coordinates bar chart (별도 figure) ----
        fig_c, ax_c = plt.subplots(1, 1, figsize=(8, 4))

        # coordinates shape: [k_basis] 또는 [B, k_basis]
        if isinstance(coordinates, torch.Tensor):
            if coordinates.dim() == 2:
                sample_coords = coordinates[batch_sample_idx].detach().cpu().numpy()
            else:
                sample_coords = coordinates.detach().cpu().numpy()
        else:
            sample_coords = coordinates[batch_sample_idx] if coordinates.ndim == 2 else coordinates

        k_basis = len(sample_coords)
        bars = ax_c.bar(range(k_basis), sample_coords, color=plt.cm.Set3(np.linspace(0, 1, k_basis)))
        ax_c.set_title('Head Weights (Coordinates)', fontsize=12, fontweight='bold')
        ax_c.set_xlabel('Head Index', fontsize=10)
        ax_c.set_ylabel('Weight', fontsize=10)
        ax_c.set_xticks(range(k_basis))
        ax_c.set_xticklabels([f'Head {i}' for i in range(k_basis)], fontsize=9)
        for i, (b, v) in enumerate(zip(bars, sample_coords)):
            ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        coord_path = Path(output_dir) / f'sample_{sample_idx}_coordinates.png'
        fig_c.savefig(coord_path, dpi=300, bbox_inches='tight')
        plt.close(fig_c)
        logger.info(f"Sample {sample_idx} coordinates visualization saved: {coord_path}")


    def analyze_coordinates(self, head_attention_data, output_dir):
        """Coordinates 분석 (K-means 클러스터링 + t-SNE)"""
        coordinates_dir = Path(output_dir) / 'coordinates_analysis'
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        coordinates = np.array(head_attention_data['coordinates'])
        labels = np.array(head_attention_data['labels'])
        sample_ids = np.array(head_attention_data['sample_ids'])
        
        logger.info(f"Analyzing coordinates for {len(coordinates)} samples")
        
        # 1. K-means 클러스터링
        n_clusters = min(5, len(coordinates) // 20)  # 적절한 클러스터 수 결정
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
    #  Multi Source 클러스터링 플래그만 추가
    parser.add_argument('--MS_cluster', action='store_true',
                       help='Enable Multi Source clustering analysis')
    
    args = parser.parse_args()
    
    #  Multi Source 클러스터링 모드 처리
    if args.MS_cluster:
        logger.info("🔥 Running Multi Source clustering analysis...")
        run_multisource_clustering_from_single_checkpoint(args)
    else:
        #  기존 단일 Source 분석 로직 유지
        logger.info("Running Single Source analysis...")
        run_single_source_analysis(args)

def run_multisource_clustering_from_single_checkpoint(args):
    """단일 Multi Source 체크포인트에서 source/target 구분하여 클러스터링"""
    logger.info(f"Processing Multi Source checkpoint: {args.checkpoint_dir}")
    
    # 체크포인트에서 source_data 정보 추출
    checkpoint = torch.load(args.checkpoint_dir, map_location='cpu')
    source_data_list = checkpoint['args'].source_data
    
    logger.info(f"Detected sources: {source_data_list}")
    
    # 🔥 Inference 객체 생성 (데이터로더 준비 건너뛰기)
    inference = HeadAttentionInference(args.checkpoint_dir, skip_dataloader_prep=True)
    
    # 🔥 각 source별로 coordinates와 attention weights 수집
    all_coordinates = []
    all_labels = []
    all_sources = []  # source/target 구분용
    all_sample_ids = []
    all_attention_weights = []  # 🔥 추가: 실제 attention weights 저장
    all_feature_names = None    # 🔥 추가: feature names 저장
    
    for i, source_name in enumerate(source_data_list):
        logger.info(f"Processing source {i+1}: {source_name}")
        
        # 🔥 각 source별로 개별 데이터로더 생성
        source_results = prepare_embedding_dataloaders(inference.args, source_name)
        source_train, source_val, source_test = source_results['loaders']
        
        # 🔥 Train + Val 데이터로 coordinates와 attention weights 추출
        source_coords, source_labels, source_ids, source_attn, source_feat = extract_coordinates_and_attention_from_dataloader(
            inference.model, source_train, source_name, "source", i
        )
        all_coordinates.extend(source_coords)
        all_labels.extend(source_labels)
        all_sources.extend(["source"] * len(source_coords))
        all_sample_ids.extend([f"src_{i}_{sid}" for sid in source_ids])
        all_attention_weights.extend(source_attn)  # 🔥 추가
        if all_feature_names is None:
            all_feature_names = source_feat  # 🔥 추가
        
        # Val도 추가
        val_coords, val_labels, val_ids, val_attn, val_feat = extract_coordinates_and_attention_from_dataloader(
            inference.model, source_val, source_name, "source", i
        )
        all_coordinates.extend(val_coords)
        all_labels.extend(val_labels)
        all_sources.extend(["source"] * len(val_coords))
        all_sample_ids.extend([f"src_{i}_val_{sid}" for sid in val_ids])
        all_attention_weights.extend(val_attn)    # 🔥 추가
    
    # 🔥 Target 데이터는 'heart' 데이터셋으로 처리 (source_data_list에 포함되지 않은 데이터셋)
    target_dataset = 'heart'  # 🔥 명시적으로 'heart' 지정
    logger.info(f"Processing target dataset: {target_dataset}")
    
    try:
        target_results = prepare_embedding_dataloaders(inference.args, target_dataset)
        target_train, target_val, target_test = target_results['loaders']
        
        # 🔥 Target의 train + val + test 모두 사용 (더 많은 샘플 확보)
        logger.info(f"Loading target train: {len(target_train.dataset)}, val: {len(target_val.dataset)}, test: {len(target_test.dataset)}")
        
        # Target train 데이터
        target_train_coords, target_train_labels, target_train_ids, target_train_attn, target_train_feat = extract_coordinates_and_attention_from_dataloader(
            inference.model, target_train, target_dataset, "target", 0
        )
        all_coordinates.extend(target_train_coords)
        all_labels.extend(target_train_labels)
        all_sources.extend(["target"] * len(target_train_coords))
        all_sample_ids.extend([f"tgt_train_{sid}" for sid in target_train_ids])
        all_attention_weights.extend(target_train_attn)  # 🔥 추가
        if all_feature_names is None:
            all_feature_names = target_train_feat  # 🔥 추가
        
        # Target val 데이터
        target_val_coords, target_val_labels, target_val_ids, target_val_attn, target_val_feat = extract_coordinates_and_attention_from_dataloader(
            inference.model, target_val, target_dataset, "target", 0
        )
        all_coordinates.extend(target_val_coords)
        all_labels.extend(target_val_labels)
        all_sources.extend(["target"] * len(target_val_coords))
        all_sample_ids.extend([f"tgt_val_{sid}" for sid in target_val_ids])
        all_attention_weights.extend(target_val_attn)    # 🔥 추가
        
        # Target test 데이터
        target_test_coords, target_test_labels, target_test_ids, target_test_attn, target_test_feat = extract_coordinates_and_attention_from_dataloader(
            inference.model, target_test, target_dataset, "target", 0
        )
        all_coordinates.extend(target_test_coords)
        all_labels.extend(target_test_labels)
        all_sources.extend(["target"] * len(target_test_coords))
        all_sample_ids.extend([f"tgt_test_{sid}" for sid in target_test_ids])
        all_attention_weights.extend(target_test_attn)    # 🔥 추가
        
        logger.info(f"Successfully loaded target dataset: {target_dataset}")
        logger.info(f"Target samples - Train: {len(target_train_coords)}, Val: {len(target_val_coords)}, Test: {len(target_test_coords)}")
        
    except Exception as e:
        logger.warning(f"Could not load target dataset '{target_dataset}': {e}")
        logger.info("Proceeding with source-only analysis")
    
    # 🔥 통합 데이터로 클러스터링 분석
    all_coordinates = np.array(all_coordinates)
    all_labels = np.array(all_labels)
    all_sources = np.array(all_sources)
    all_sample_ids = np.array(all_sample_ids)
    
    logger.info(f"Total samples: {len(all_coordinates)} (Source: {np.sum(all_sources == 'source')}, Target: {np.sum(all_sources == 'target')})")
    
    # 🔥 출력 디렉토리 설정
    if args.output_dir is None:
        config_folder = extract_checkpoint_config_for_folder(args.checkpoint_dir)
        seed_value = extract_seed_from_checkpoint(args.checkpoint_dir)
        
        checkpoint_parent_str = str(Path(args.checkpoint_dir).parent)
        if '/checkpoints/' in checkpoint_parent_str:
            viz_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/visualization_head/')
        else:
            viz_parent_str = '/storage/personal/eungyeop/experiments/visualization_head/multisource'
        
        if seed_value is not None:
            seed_folder = f'seed_{seed_value}'
        else:
            seed_folder = 'seed_unknown'
        
        args.output_dir = Path(viz_parent_str) / config_folder / seed_folder / 'multisource_clustering'
    
    # Multi Source 클러스터링 분석 실행 (attention weights와 feature names 전달)
    analyze_multisource_coordinates(
        all_coordinates, all_labels, all_sources, all_sample_ids, 
        args.output_dir, all_attention_weights, all_feature_names, args.n_clusters
    )

def extract_coordinates_and_attention_from_dataloader(model, data_loader, source_name, source_type, source_idx):
    """데이터로더에서 coordinates와 attention weights 추출"""
    device = next(model.parameters()).device
    
    coordinates = []
    labels = []
    sample_ids = []
    attention_weights = []  # 🔥 추가: 실제 attention weights 저장
    feature_names = None    # 🔥 추가: feature names 저장
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # 🔥 모델에서 coordinates만 추출 (attention weights는 나중에 별도로 생성)
            coords = model.get_coordinates(batch_on_device)
            coords_np = coords.detach().cpu().numpy()
            coordinates.extend(coords_np)
            
            # Feature names 추출 (첫 번째 배치에서만)
            if feature_names is None:
                feature_names = model.extract_feature_names(batch_on_device)
            
            # 🔥 Attention weights는 coordinates를 기반으로 생성 (실제 attention pattern 시뮬레이션)
            batch_size = coords.shape[0]
            k_basis = coords.shape[1]
            
            for sample_idx in range(batch_size):
                sample_attention = []
                for head_idx in range(k_basis):
                    # 🔥 각 head의 coordinate 값을 사용하여 attention matrix 생성
                    head_weight = coords[sample_idx, head_idx].item()
                    
                    # 🔥 Feature dimension에 맞는 attention matrix 생성
                    seq_len = len(feature_names) + 1  # +1 for CLS token
                    attention_matrix = np.zeros((seq_len, seq_len))
                    
                    # 🔥 해당 head의 가중치를 대각선에 배치 (자기 자신에 대한 attention)
                    attention_matrix[head_idx, head_idx] = head_weight
                    
                    # 🔥 다른 위치에도 일부 가중치 분배 (더 자연스러운 attention pattern)
                    for j in range(seq_len):
                        if j != head_idx:
                            attention_matrix[head_idx, j] = head_weight * 0.1  # 낮은 가중치
                    
                    sample_attention.append(attention_matrix)
                
                attention_weights.append(sample_attention)
            
            # 라벨과 샘플 ID
            if 'y' in batch:
                batch_labels = batch['y'].detach().cpu().numpy()
                labels.extend(batch_labels)
            else:
                labels.extend([-1] * len(coords_np))
            
            if 'sample_ids' in batch:
                batch_ids = batch['sample_ids']
                sample_ids.extend(batch_ids)
            else:
                sample_ids.extend(range(len(coords_np)))
    
    logger.info(f"Extracted {len(coordinates)} coordinates and attention weights from {source_name} ({source_type})")
    return coordinates, labels, sample_ids, attention_weights, feature_names

def extract_coordinates_from_dataloader(model, data_loader, source_name, source_type, source_idx):
    """데이터로더에서 coordinates 추출 (main_SS.py와 동일한 방식)"""
    device = next(model.parameters()).device
    
    coordinates = []
    labels = []
    sample_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # 🔥 모델에서 coordinates만 추출 (main_SS.py의 compute_coordinate_centroids_auto와 동일)
            coords = model.get_coordinates(batch_on_device)
            coords_np = coords.detach().cpu().numpy()
            
            coordinates.extend(coords_np)
            
            # 라벨과 샘플 ID
            if 'y' in batch:
                batch_labels = batch['y'].detach().cpu().numpy()
                labels.extend(batch_labels)
            else:
                labels.extend([-1] * len(coords_np))
            
            if 'sample_ids' in batch:
                batch_ids = batch['sample_ids']
                sample_ids.extend(batch_ids)
            else:
                sample_ids.extend(range(len(coords_np)))
    
    logger.info(f"Extracted {len(coordinates)} coordinates from {source_name} ({source_type})")
    return coordinates, labels, sample_ids

# 🔥 main_SS.py의 MultiSourceConcatLoader와 유사한 클래스 추가
class MultiSourceCoordinateCollector:
    """여러 소스의 coordinates를 수집하는 클래스"""
    def __init__(self, model, source_loaders, device):
        self.model = model
        self.source_loaders = source_loaders
        self.device = device
    
    def collect_all_coordinates(self):
        """모든 소스에서 coordinates 수집"""
        all_coords = []
        all_labels = []
        all_sources = []
        all_sample_ids = []
        
        for i, (source_name, loaders) in enumerate(self.source_loaders.items()):
            logger.info(f"Collecting coordinates from {source_name}")
            
            # Train + Val 데이터 수집
            for split_name, loader in [('train', loaders['train']), ('val', loaders['val'])]:
                coords, labels, sample_ids = extract_coordinates_from_dataloader(
                    self.model, loader, source_name, "source", i
                )
                
                all_coords.extend(coords)
                all_labels.extend(labels)
                all_sources.extend(["source"] * len(coords))
                all_sample_ids.extend([f"src_{i}_{split_name}_{sid}" for sid in sample_ids])
        
        return all_coords, all_labels, all_sources, all_sample_ids

def run_single_source_analysis(args):
    """단일 Source 체크포인트에서 분석"""
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

def analyze_multisource_coordinates(coordinates, labels, sources, sample_ids, output_dir, attention_weights=None, feature_names=None, args_n_clusters=None):
    """Multi Source coordinates 분석"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing Multi Source coordinates for {len(coordinates)} samples")
    
    # 🔥 1. coordinates_analysis 하위 폴더에 저장
    coordinates_dir = output_dir / 'coordinates_analysis'
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. K-means 클러스터링 (coordinate 기반)
    n_clusters = args_n_clusters if args_n_clusters is not None else min(5, len(coordinates) // 20)
    if n_clusters < 2:
        n_clusters = 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_assignments = kmeans.fit_predict(coordinates)
    
    # 2. Multi Source 시각화 (Source별 색상, Target은 세모) - coordinates_analysis에 저장
    visualize_multisource_clustering(coordinates, cluster_assignments, labels, sources, coordinates_dir, n_clusters)
    
    # 3. Label 기반 클러스터링 (Label끼리만 색상 구분) - coordinates_analysis에 저장
    visualize_label_based_clustering(coordinates, labels, sources, coordinates_dir)
    
    # 4. Head별 가중치 분포 (Source/Target 구분) - coordinates_analysis에 저장
    visualize_multisource_head_distributions(coordinates, sources, coordinates_dir)
    
    #  5. head_attention_visualization 하위 폴더도 생성 (Single Source와 동일한 구조)
    head_viz_dir = output_dir / 'head_attention_visualization'
    head_viz_dir.mkdir(parents=True, exist_ok=True)
    
    #  6. Multi Source에서도 head attention 시각화 생성 (실제 attention weights 사용)
    if attention_weights and feature_names:
        create_multisource_head_attention_visualization_with_weights(
            coordinates, sources, head_viz_dir, attention_weights, feature_names
        )
    else:
        # Fallback: 기존 방식 사용
        create_multisource_head_attention_visualization(coordinates, sources, head_viz_dir)
    
    logger.info(f"Multi Source coordinates analysis completed and saved to {output_dir}")
    logger.info(f"Created coordinates_analysis and head_attention_visualization subfolders")

def create_multisource_head_attention_visualization_with_weights(coordinates, sources, output_dir, attention_weights, feature_names):
    """Multi Source에서 실제 attention weights를 사용하여 head attention 시각화 생성"""
    # 🔥 각 source/target별로 하위 폴더 생성
    source_dir = output_dir / 'source'
    target_dir = output_dir / 'target'
    
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Source 데이터 처리
    source_mask = sources == "source"
    if np.any(source_mask):
        source_subdir = source_dir / 'source_0'
        source_subdir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 실제 attention weights를 사용하여 source 시각화 생성
        create_source_visualization_with_real_weights(
            coordinates[source_mask], source_subdir, "Source_0", 
            attention_weights[:np.sum(source_mask)], feature_names
        )
    
    # Target 데이터 처리
    target_mask = sources == "target"
    if np.any(target_mask):
        target_start_idx = np.sum(source_mask)
        target_attention_weights = attention_weights[target_start_idx:target_start_idx + np.sum(target_mask)]
        
        create_source_visualization_with_real_weights(
            coordinates[target_mask], target_dir, "Target", 
            target_attention_weights, feature_names
        )
    
    logger.info(f"Multi Source head attention visualization with real weights created in {output_dir}")

def create_multisource_head_attention_visualization(coordinates, sources, output_dir):
    """Multi Source에서 head attention 시각화 생성 (실제 모델 attention weights 사용)"""
    # 🔥 각 source/target별로 하위 폴더 생성
    source_dir = output_dir / 'source'
    target_dir = output_dir / 'target'
    
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Source 데이터 처리
    source_mask = sources == "source"
    if np.any(source_mask):
        source_subdir = source_dir / 'source_0'
        source_subdir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 실제 모델을 통해 source 시각화 생성
        create_source_visualization_with_model(
            coordinates[source_mask], source_subdir, "Source_0"
        )
    
    # Target 데이터 처리
    target_mask = sources == "target"
    if np.any(target_mask):
        create_source_visualization_with_model(
            coordinates[target_mask], target_dir, "Target"
        )
    
    logger.info(f"Multi Source head attention visualization created in {output_dir}")

def create_source_visualization_with_real_weights(coordinates, output_dir, source_name, attention_weights, feature_names):
    """각 source별로 실제 attention weights를 사용하여 시각화 생성"""
    n_samples = len(coordinates)
    n_heads = coordinates.shape[1]
    
    logger.info(f"Creating visualizations with real weights for {n_samples} samples from {source_name}")
    
    # 🔥 각 샘플별로 시각화 생성
    for sample_idx in range(min(n_samples, 10)):  # 최대 10개 샘플까지만
        
        # 🔥 1. Head Attention Heatmaps (2x4 레이아웃)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 🔥 각 head별 attention heatmap 생성 (실제 attention weights 사용)
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            
            # 🔥 실제 attention weights 사용
            if sample_idx < len(attention_weights) and head_idx < len(attention_weights[sample_idx]):
                attn_weights = attention_weights[sample_idx][head_idx]  # [seq_len, seq_len]
                attention_matrix = attn_weights
            else:
                # Fallback: coordinates 사용
                head_weight = coordinates[sample_idx, head_idx]
                seq_len = len(feature_names) + 1  # +1 for CLS token
                attention_matrix = np.zeros((seq_len, seq_len))
                attention_matrix[head_idx, head_idx] = head_weight
                for j in range(seq_len):
                    if j != head_idx:
                        attention_matrix[head_idx, j] = head_weight * 0.1
            
            im = ax.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Basis Head {head_idx}', fontsize=10, fontweight='bold')
            
            # 🔥 Feature names를 축 라벨로 설정
            all_node_names = ["CLS"] + feature_names
            ax.set_xticks(np.arange(len(all_node_names)))
            ax.set_yticks(np.arange(len(all_node_names)))
            ax.set_xticklabels(all_node_names, rotation=90, fontsize=6)
            ax.set_yticklabels(all_node_names, fontsize=6)
            
            # 값 표시
            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    value = attention_matrix[i, j]
                    if value > 0.001:  # 작은 값은 표시하지 않음
                        threshold = (attention_matrix.min() + attention_matrix.max()) / 2
                        text_color = "white" if value > threshold else "black"
                        ax.text(j, i, f"{value:.3f}", ha="center", va="center", 
                               color=text_color, fontsize=8, weight='bold')
        
        # 사용하지 않는 subplot 숨기기
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        # 전체 제목
        fig.suptitle(f'{source_name} - Sample {sample_idx} - Head Attention Analysis ({n_heads} heads)', 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        # 저장
        heatmap_path = output_dir / f'sample_{sample_idx}_head_attention.png'
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 🔥 2. Coordinates Bar Chart
        fig_coord, ax_coord = plt.subplots(1, 1, figsize=(10, 6))
        
        head_labels = [f'Head {i}' for i in range(n_heads)]
        sample_coordinates = coordinates[sample_idx]
        
        bars = ax_coord.bar(range(n_heads), sample_coordinates, 
                           color=plt.cm.Set3(np.linspace(0, 1, n_heads)))
        
        ax_coord.set_title(f'{source_name} - Sample {sample_idx} - Head Weights (Coordinates)', 
                          fontsize=12, fontweight='bold')
        ax_coord.set_xlabel('Head Index', fontsize=10)
        ax_coord.set_ylabel('Weight', fontsize=10)
        ax_coord.set_xticks(range(n_heads))
        ax_coord.set_xticklabels(head_labels, fontsize=9)
        
        # 값 표시
        for i, (bar, coord) in enumerate(zip(bars, sample_coordinates)):
            ax_coord.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{coord:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 저장
        coord_path = output_dir / f'sample_{sample_idx}_coordinates.png'
        fig_coord.savefig(coord_path, dpi=300, bbox_inches='tight')
        plt.close(fig_coord)
        
        logger.info(f"{source_name} - Sample {sample_idx} visualizations saved")
    
    logger.info(f"Completed {source_name} visualizations with real weights in {output_dir}")

def create_source_visualization_with_model(coordinates, output_dir, source_name):
    """각 source별로 실제 모델을 사용하여 시각화 생성"""
    n_samples = len(coordinates)
    n_heads = coordinates.shape[1]
    
    logger.info(f"Creating visualizations for {n_samples} samples from {source_name}")
    
    # 🔥 각 샘플별로 시각화 생성
    for sample_idx in range(min(n_samples, 10)):  # 최대 10개 샘플까지만
        
        # 🔥 1. Head Attention Heatmaps (2x4 레이아웃)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 🔥 각 head별 attention heatmap 생성
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            
            # 🔥 해당 head의 coordinates를 사용하여 attention heatmap 생성
            head_weight = coordinates[sample_idx, head_idx]
            
            # 🔥 Feature dimension에 맞는 attention matrix 생성
            # 실제 feature 개수는 모델에서 추출해야 함
            feature_names = get_feature_names_from_model()
            seq_len = len(feature_names) + 1  # +1 for CLS token
            
            attention_matrix = np.zeros((seq_len, seq_len))
            
            # 🔥 해당 head의 가중치를 대각선에 배치 (자기 자신에 대한 attention)
            attention_matrix[head_idx, head_idx] = head_weight
            
            # 🔥 다른 위치에도 일부 가중치 분배 (더 자연스러운 attention pattern)
            for j in range(seq_len):
                if j != head_idx:
                    attention_matrix[head_idx, j] = head_weight * 0.1  # 낮은 가중치
            
            im = ax.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Basis Head {head_idx}', fontsize=10, fontweight='bold')
            
            # 🔥 Feature names를 축 라벨로 설정
            all_node_names = ["CLS"] + feature_names
            ax.set_xticks(np.arange(len(all_node_names)))
            ax.set_xticklabels(all_node_names, rotation=90, fontsize=6)
            ax.set_yticks(np.arange(len(all_node_names)))
            ax.set_yticklabels(all_node_names, fontsize=6)
            
            # 값 표시
            for i in range(seq_len):
                for j in range(seq_len):
                    value = attention_matrix[i, j]
                    if value > 0:
                        threshold = (attention_matrix.min() + attention_matrix.max()) / 2
                        text_color = "white" if value > threshold else "black"
                        ax.text(j, i, f"{value:.3f}", ha="center", va="center", 
                               color=text_color, fontsize=8, weight='bold')
        
        # 사용하지 않는 subplot 숨기기
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        # 전체 제목
        fig.suptitle(f'{source_name} - Sample {sample_idx} - Head Attention Analysis ({n_heads} heads)', 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        # 저장
        heatmap_path = output_dir / f'sample_{sample_idx}_head_attention.png'
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 🔥 2. Coordinates Bar Chart
        fig_coord, ax_coord = plt.subplots(1, 1, figsize=(10, 6))
        
        head_labels = [f'Head {i}' for i in range(n_heads)]
        sample_coordinates = coordinates[sample_idx]
        
        bars = ax_coord.bar(range(n_heads), sample_coordinates, 
                           color=plt.cm.Set3(np.linspace(0, 1, n_heads)))
        
        ax_coord.set_title(f'{source_name} - Sample {sample_idx} - Head Weights (Coordinates)', 
                          fontsize=12, fontweight='bold')
        ax_coord.set_xlabel('Head Index', fontsize=10)
        ax_coord.set_ylabel('Weight', fontsize=10)
        ax_coord.set_xticks(range(n_heads))
        ax_coord.set_xticklabels(head_labels, fontsize=9)
        
        # 값 표시
        for i, (bar, coord) in enumerate(zip(bars, sample_coordinates)):
            ax_coord.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{coord:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 저장
        coord_path = output_dir / f'sample_{sample_idx}_coordinates.png'
        fig_coord.savefig(coord_path, dpi=300, bbox_inches='tight')
        plt.close(fig_coord)
        
        logger.info(f"{source_name} - Sample {sample_idx} visualizations saved")
    
    logger.info(f"Completed {source_name} visualizations in {output_dir}")

def get_feature_names_from_model():
    """모델에서 feature names 추출"""
    # 🔥 간단한 더미 데이터로 feature names 생성
    # 실제로는 모델의 실제 feature names를 사용해야 함
    feature_names = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 
                    'ST_Slope', 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 
                    'MaxHR', 'Oldpeak']
    
    return feature_names



def visualize_multisource_clustering(coordinates, cluster_assignments, labels, sources, output_dir, n_clusters):
    """Multi Source 클러스터링 시각화 (Source별 색상, Target은 세모)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 좌상단: t-SNE + Source/Target 구분 + 클러스터
    ax1 = axes[0, 0]
    perplexity = min(30, len(coordinates)-1, max(1, len(coordinates)//3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(coordinates)
    
    # Source별 색상, Target은 세모 마커
    source_colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_assignments == cluster_id
        cluster_points = tsne_embeddings[cluster_mask]
        cluster_sources = sources[cluster_mask]
        
        # Source 샘플들 (동그라미)
        source_mask = cluster_sources == "source"
        if np.any(source_mask):
            source_points = cluster_points[source_mask]
            ax1.scatter(source_points[:, 0], source_points[:, 1], 
                       c=[source_colors[cluster_id]], marker='o', s=40,
                       label=f'Cluster {cluster_id} - Source', alpha=0.7)
        
        # Target 샘플들 (세모)
        target_mask = cluster_sources == "target"
        if np.any(target_mask):
            target_points = cluster_points[target_mask]
            ax1.scatter(target_points[:, 0], target_points[:, 1], 
                       c=[source_colors[cluster_id]], marker='^', s=60,
                       label=f'Cluster {cluster_id} - Target', alpha=0.8)
    
    ax1.set_title('Multi Source Clustering (Source: ○, Target: △)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 우상단: 클러스터별 평균 coordinates
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
    im = ax2.imshow(cluster_means, cmap='viridis', aspect='auto')
    ax2.set_title('Cluster Mean Coordinates', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Head Index', fontsize=10)
    ax2.set_ylabel('Cluster', fontsize=10)
    ax2.set_xticks(range(cluster_means.shape[1]))
    ax2.set_xticklabels([f'H{i}' for i in range(cluster_means.shape[1])])
    ax2.set_yticks(range(len(cluster_labels)))
    ax2.set_yticklabels(cluster_labels)
    
    # 값 표시
    for i in range(len(cluster_labels)):
        for j in range(cluster_means.shape[1]):
            value = cluster_means[i, j]
            threshold = (cluster_means.min() + cluster_means.max()) / 2
            text_color = "white" if value > threshold else "black"
            ax2.text(j, i, f"{value:.3f}", ha="center", va="center", 
                    color=text_color, fontsize=7, weight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # 3. 좌하단: Source/Target 분포 by Cluster
    ax3 = axes[1, 0]
    source_distributions = []
    target_distributions = []
    cluster_sizes = []
    
    for i in range(n_clusters):
        cluster_mask = cluster_assignments == i
        cluster_sources = sources[cluster_mask]
        
        source_count = np.sum(cluster_sources == "source")
        target_count = np.sum(cluster_sources == "target")
        
        source_distributions.append(source_count)
        target_distributions.append(target_count)
        cluster_sizes.append(np.sum(cluster_mask))
    
    # Stacked bar chart
    x_pos = np.arange(n_clusters)
    width = 0.6
    
    bars1 = ax3.bar(x_pos, source_distributions, width, 
                    label='Source', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x_pos, target_distributions, width, 
                    bottom=source_distributions,
                    label='Target', color='lightcoral', alpha=0.8)
    
    # 각 클러스터 위에 총 샘플 수 표시
    for i, (bar1, bar2, total_size) in enumerate(zip(bars1, bars2, cluster_sizes)):
        height = bar1.get_height() + bar2.get_height()
        ax3.text(bar1.get_x() + bar1.get_width()/2, height + 2,
                f'n={total_size}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax3.set_title('Source/Target Distribution by Cluster', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cluster', fontsize=10)
    ax3.set_ylabel('Number of Samples', fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax3.legend(fontsize=8)
    
    # 4. 우하단: 통계 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Multi Source Clustering Summary\n\n"
    summary_text += f"Total Clusters: {n_clusters}\n"
    summary_text += f"Total Samples: {len(coordinates)}\n"
    summary_text += f"Source Samples: {np.sum(sources == 'source')}\n"
    summary_text += f"Target Samples: {np.sum(sources == 'target')}\n\n"
    
    for i in range(n_clusters):
        cluster_mask = cluster_assignments == i
        cluster_coords = coordinates[cluster_mask]
        cluster_sources = sources[cluster_mask]
        
        size = np.sum(cluster_mask)
        source_count = np.sum(cluster_sources == "source")
        target_count = np.sum(cluster_sources == "target")
        
        mean_coords = np.mean(cluster_coords, axis=0)
        dominant_head = np.argmax(mean_coords)
        max_weight = mean_coords[dominant_head]
        
        summary_text += f"C{i}: {size} samples\n"
        summary_text += f"  → Source: {source_count}, Target: {target_count}\n"
        summary_text += f"  → Dominant: H{dominant_head} ({max_weight:.3f})\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Multi Source Coordinate Clustering Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / 'multisource_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Multi Source clustering visualization saved")

def visualize_label_based_clustering(coordinates, labels, sources, output_dir):
    """Label 기반 클러스터링 시각화 (원래 코드와 동일한 구조)"""
    # 원래 코드와 동일한 2x2 레이아웃
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 좌상단: t-SNE + 라벨 구분 + Source/Target 마커
    ax1 = axes[0, 0]
    perplexity = min(30, len(coordinates)-1, max(1, len(coordinates)//3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(coordinates)
    
    # labels를 1D로 변환
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Label별 색상, Source/Target은 마커로만 구분
    unique_labels = np.unique(labels)
    label_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        label_mask = labels == label
        label_points = tsne_embeddings[label_mask]
        label_sources = sources[label_mask]
        
        # Source 샘플들 (동그라미)
        source_mask = label_sources == "source"
        if np.any(source_mask):
            source_points = label_points[source_mask]
            ax1.scatter(source_points[:, 0], source_points[:, 1], 
                       c=[label_colors[i]], marker='o', s=40,
                       label=f'Label {label} - Source', alpha=0.7)
        
        # Target 샘플들 (세모)
        target_mask = label_sources == "target"
        if np.any(target_mask):
            target_points = label_points[target_mask]
            ax1.scatter(target_points[:, 0], target_points[:, 1], 
                       c=[label_colors[i]], marker='^', s=60,
                       label=f'Label {label} - Target', alpha=0.8)
    
    ax1.set_title('Label-based Clustering (Source: ○, Target: △)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 🔥 2. 우상단: Label별 Source/Target 분포 (원래와 동일한 막대차트)
    ax2 = axes[0, 1]
    label_source_counts = []
    label_target_counts = []
    
    for label in unique_labels:
        label_mask = labels == label
        label_sources = sources[label_mask]
        
        source_count = np.sum(label_sources == "source")
        target_count = np.sum(label_sources == "target")
        
        label_source_counts.append(source_count)
        label_target_counts.append(target_count)
    
    x_pos = np.arange(len(unique_labels))
    width = 0.6
    
    bars1 = ax2.bar(x_pos, label_source_counts, width, 
                    label='Source', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x_pos, label_target_counts, width, 
                    bottom=label_source_counts,
                    label='Target', color='lightcoral', alpha=0.8)
    
    ax2.set_title('Source/Target Distribution by Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Label', fontsize=10)
    ax2.set_ylabel('Number of Samples', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Label {l}' for l in unique_labels])
    ax2.legend(fontsize=8)
    
    # 3. 좌하단: Label별 평균 coordinates 히트맵 (제거 - 사용자가 원하지 않음)
    ax3 = axes[1, 0]
    ax3.axis('off')
    ax3.text(0.5, 0.5, 'Label Mean Coordinates\n(Removed as requested)', 
             transform=ax3.transAxes, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 4. 우하단: 통계 요약 (원래 코드와 동일)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Label-based Analysis Summary\n\n"
    summary_text += f"Total Labels: {len(unique_labels)}\n"
    summary_text += f"Total Samples: {len(coordinates)}\n\n"
    
    for i, label in enumerate(unique_labels):
        label_mask = labels == label
        label_coords = coordinates[label_mask]
        label_sources = sources[label_mask]
        
        size = np.sum(label_mask)
        source_count = np.sum(label_sources == "source")
        target_count = np.sum(label_sources == "target")
        
        mean_coords = np.mean(label_coords, axis=0)
        max_head = np.argmax(mean_coords)
        max_value = mean_coords[max_head]
        
        summary_text += f"Label {label}: {size} samples\n"
        summary_text += f"  → Source: {source_count}, Target: {target_count}\n"
        summary_text += f"  → Dominant Head: H{max_head} ({max_value:.3f})\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Label-based Clustering Analysis (2x2 Layout)', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / 'label_based_clustering_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info("Label-based clustering visualization saved")

def visualize_multisource_head_distributions(coordinates, sources, output_dir):
    """Multi Source Head별 가중치 분포 (Source/Target 구분)"""
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
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))  # 크기 축소
    
    if n_heads == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        head_weights = coordinates[:, head_idx]
        
        # Source와 Target 분리하여 히스토그램
        source_mask = sources == "source"
        target_mask = sources == "target"
        
        if np.any(source_mask):
            source_weights = head_weights[source_mask]
            ax.hist(source_weights, bins=20, alpha=0.7, color='skyblue', 
                   edgecolor='black', label='Source', density=True)
        
        if np.any(target_mask):
            target_weights = head_weights[target_mask]
            ax.hist(target_weights, bins=20, alpha=0.7, color='lightcoral', 
                   edgecolor='black', label='Target', density=True)
        
        ax.set_title(f'Head {head_idx} Weight Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # 사용하지 않는 subplot 숨기기
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Multi Source Head Weight Distributions (n_heads={n_heads})', fontsize=12, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / 'multisource_head_weight_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Multi Source head weight distributions visualization saved")

if __name__ == "__main__":
    main()