"""
TabularFLM_Basic 모델용 단일 Source Attention Maps 분석 스크립트

단일 source 환경에서 각 layer별, head별로 attention을 개별적으로 시각화합니다.
"""

import os
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
import psutil
p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# 현재 스크립트 파일 위치
current_dir = Path(__file__).resolve().parent
import sys
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# TabularFLM_Basic 모델 import
from models.TabularFLM_Basic import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                
            logger.info(f"Auto-detected deleted features from filename: {deleted_features}")
            return deleted_features, d_part
    
    logger.info("No D:[...] pattern found in filename - using all features")
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
        logger.info(f"Auto-detected seed from filename: {seed_value}")
        return seed_value
    else:
        logger.info("No S:[seed] pattern found in filename")
        return None

class BasicAttentionInference:
    def __init__(self, checkpoint_dir, device='cuda', auto_del_feat=None):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Attempting to load checkpoint from: {checkpoint_dir}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        # 자동 추출된 삭제 변수를 args에 적용
        if auto_del_feat is not None:
            self.args.del_feat = auto_del_feat
            logger.info(f"Applied auto-detected del_feat: {auto_del_feat}")
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
    def _load_model(self):
        """체크포인트에서 TabularFLM_Basic 모델 로드"""
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
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("TabularFLM_Basic Model loaded and set to evaluation mode")
    
    def _prepare_dataloaders(self):
        """단일 source 환경에서 데이터로더 준비"""
        fix_seed(self.args.random_seed)
        
        # 단일 소스 데이터셋
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
        
        logger.info(f"Dataloader prepared for dataset: {self.args.source_data}")
    
    def extract_attention_maps(self, data_loader):
        """단일 source에서 attention maps 추출 - 각 layer별, head별로 개별 처리"""
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
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # TabularFLM_Basic 모델의 attention weights 추출
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                
                # Feature names 추출 (첫 번째 배치에서만)
                if attention_data['feature_names'] is None:
                    feature_names = self.extract_feature_names(batch_on_device)
                    attention_data['feature_names'] = ["CLS"] + feature_names
                
                # 배치 크기 확인
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    # 각 레이어별 attention map 저장 (head별로 개별 저장)
                    for layer_idx, layer_attention in enumerate(attention_weights):
                        # [n_heads, seq_len, seq_len] 형태로 저장 (평균내지 않음)
                        attention_numpy = layer_attention[sample_idx].detach().cpu().numpy()
                        attention_data[f'layer_{layer_idx}'].append(attention_numpy)
                    
                    # 라벨과 샘플 ID 저장
                    if 'y' in batch:
                        label = batch['y'][sample_idx].item()
                    else:
                        label = -1
                    attention_data['labels'].append(label)
                    
                    if 'sample_ids' in batch:
                        sample_id = batch['sample_ids'][sample_idx]
                    else:
                        sample_id = sample_count
                    attention_data['sample_ids'].append(sample_id)
                    
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        
        logger.info(f"Extracted attention maps for {sample_count} total samples")
        return attention_data
    
    def _extract_attention_from_model(self, batch):
        """TabularFLM_Basic 모델에서 attention weights와 예측값 추출"""
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
        
        # TabularFLM_Basic forward pass
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
        
        pred = x[:, 0, :]
        pred = self.model.predictor(pred)
        
        return pred, attention_weights

    def extract_feature_names(self, batch):
        """배치에서 feature names 추출"""
        try:
            feature_names = self.model.extract_feature_names(batch)
            
            if feature_names and len(feature_names) > 0:
                logger.info(f"Extracted {len(feature_names)} feature names from model: {feature_names[:5]}...")
                return feature_names
            else:
                logger.warning("Model returned empty feature names")
                
        except Exception as e:
            logger.warning(f"Failed to extract feature names from model: {e}")
        
        # Fallback: 기존 로직 사용
        feature_names = []
        
        if 'cat_desc_texts' in batch:
            for feature in batch['cat_desc_texts']:
                if isinstance(feature, tuple):
                    clean_name = str(feature[0])
                else:
                    try:
                        clean_name = feature.split("'")[1] if "'" in feature else feature
                        clean_name = clean_name.split(',')[0]
                    except:
                        clean_name = str(feature)
                feature_names.append(clean_name)

        if 'num_desc_texts' in batch:
            for feature in batch['num_desc_texts']:
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
        
        return unique_features

    def visualize_head_attention_heatmaps(self, data_loader, output_dir, max_samples=5):
        """각 layer별, head별로 개별 attention 히트맵 시각화"""
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
                
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                feature_names = self.extract_feature_names(batch_on_device)
                
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    # 각 layer별, head별로 개별 시각화
                    self._create_head_attention_heatmaps(
                        attention_weights, 
                        feature_names, 
                        output_dir, 
                        sample_count,
                        sample_idx
                    )
                    
                    sample_count += 1
                
                if sample_count >= max_samples:
                    break
        
        logger.info(f"Head attention visualization completed! Total: {sample_count} samples")


    def visualize_attention_score_distribution(self, data_loader, output_dir, max_samples=5):
        """각 Layer의 head별 attention score 분포를 bar graph로 시각화"""
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
                
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                feature_names = self.extract_feature_names(batch_on_device)
                
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    # 각 샘플별로 attention score 분포 시각화
                    self._create_attention_score_bargraphs(
                        attention_weights, 
                        feature_names, 
                        output_dir, 
                        sample_count,
                        sample_idx
                    )
                    
                    sample_count += 1
                
                if sample_count >= max_samples:
                    break
    
        logger.info(f"Attention score distribution visualization completed! Total: {sample_count} samples")

    def _create_attention_score_bargraphs(self, attention_weights, feature_names, save_dir, sample_count, sample_idx):
        """각 Layer의 head별 attention score 분포를 bar graph로 생성 + head 평균"""
        n_layers = len(attention_weights)
        n_heads = attention_weights[0].shape[1]
        seq_len = attention_weights[0].shape[2]
        
        all_node_names = ["CLS"] + feature_names
        
        # 전체 레이어 x (헤드 + 1) 그리드 생성 (맨 오른쪽에 head 평균 추가)
        fig, axes = plt.subplots(n_layers, n_heads + 1, figsize=((n_heads + 1) * 6, n_layers * 5))
        
        # axes 처리
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        if n_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # 각 레이어별, 헤드별로 bar graph 생성
        for layer_idx, layer_attention in enumerate(attention_weights):
            sample_attention = layer_attention[sample_idx]
            
            # 개별 헤드들
            for head_idx in range(n_heads):
                ax = axes[layer_idx, head_idx]
                attention_map = sample_attention[head_idx].cpu().numpy()  # [seq_len, seq_len]
                
                # CLS 토큰이 다른 모든 토큰에 주는 attention scores 추출
                cls_attention_scores = attention_map[0, :]  # CLS -> 모든 토큰
                
                # Bar graph 생성 (정렬하지 않음)
                bars = ax.bar(range(len(cls_attention_scores)), cls_attention_scores, 
                            color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
                
                # 각 bar 위에 값 표시
                for i, (bar, score) in enumerate(zip(bars, cls_attention_scores)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
                
                # 축 설정
                ax.set_title(f'L{layer_idx}H{head_idx} - CLS Attention Scores', 
                            fontsize=10, fontweight='bold', pad=10)
                ax.set_xlabel('Tokens', fontsize=9)
                ax.set_ylabel('Attention Score', fontsize=9)
                
                # X축 라벨 설정 (토큰 이름)
                ax.set_xticks(range(len(all_node_names)))
                ax.set_xticklabels(all_node_names, rotation=90, fontsize=7, ha='right')
                
                # 그리드 추가
                ax.grid(True, alpha=0.3, axis='y')
                
                # Y축 범위 설정 (0부터 시작)
                ax.set_ylim(0, max(cls_attention_scores) * 1.1)
            
            # 맨 오른쪽에 head 평균 bar graph 추가
            ax_avg = axes[layer_idx, n_heads]  # 마지막 열
            
            # 모든 헤드의 CLS attention을 평균
            all_head_cls_scores = []
            for head_idx in range(n_heads):
                attention_map = sample_attention[head_idx].cpu().numpy()
                cls_scores = attention_map[0, :]  # CLS -> 모든 토큰
                all_head_cls_scores.append(cls_scores)
            
            # 헤드별 평균 계산
            avg_cls_scores = np.mean(all_head_cls_scores, axis=0)
            
            # Bar graph 생성 (정렬하지 않음)
            bars = ax_avg.bar(range(len(avg_cls_scores)), avg_cls_scores, 
                            color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=0.5)
            
            # 각 bar 위에 값 표시
            for i, (bar, score) in enumerate(zip(bars, avg_cls_scores)):
                ax_avg.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            # 축 설정
            ax_avg.set_title(f'L{layer_idx}Avg - CLS Attention Scores (Head Average)', 
                            fontsize=10, fontweight='bold', pad=10)
            ax_avg.set_xlabel('Tokens', fontsize=9)
            ax_avg.set_ylabel('Average Attention Score', fontsize=9)
            
            # X축 라벨 설정 (토큰 이름)
            ax_avg.set_xticks(range(len(all_node_names)))
            ax_avg.set_xticklabels(all_node_names, rotation=90, fontsize=7, ha='right')
            
            # 그리드 추가
            ax_avg.grid(True, alpha=0.3, axis='y')
            
            # Y축 범위 설정 (0부터 시작)
            ax_avg.set_ylim(0, max(avg_cls_scores) * 1.1)
        
        # 전체 제목 설정
        plt.suptitle(f'Sample {sample_idx} - Layer x Head CLS Attention Score Distribution\n(Individual Heads + Head Average, Original Order)', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 저장
        attention_path = save_dir / f'sample_{sample_idx}_attention_scores.png'
        fig.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Layer x Head attention score distribution saved: {attention_path}")

    def _create_attention_score_summary(self, attention_weights, feature_names, save_dir, sample_idx):
        """전체 레이어의 attention score 요약 시각화"""
        n_layers = len(attention_weights)
        n_heads = attention_weights[0].shape[1]
        seq_len = attention_weights[0].shape[2]
        
        all_node_names = ["CLS"] + feature_names
        
        # 각 레이어별로 평균 attention score 계산
        layer_avg_scores = []
        layer_names = []
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            sample_attention = layer_attention[sample_idx]  # 첫 번째 샘플만 처리
            
            # 모든 헤드의 CLS attention을 평균
            all_head_cls_scores = []
            for head_idx in range(n_heads):
                attention_map = sample_attention[head_idx].cpu().numpy()
                cls_scores = attention_map[0, :]  # CLS -> 모든 토큰
                all_head_cls_scores.append(cls_scores)
            
            # 헤드별 평균 계산
            avg_cls_scores = np.mean(all_head_cls_scores, axis=0)
            layer_avg_scores.append(avg_cls_scores)
            layer_names.append(f'Layer {layer_idx}')
        
        # 요약 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 레이어별 평균 attention score 비교
        for layer_idx, scores in enumerate(layer_avg_scores):
            # 내림차순 정렬
            score_token_pairs = list(zip(scores, all_node_names))
            score_token_pairs.sort(key=lambda x: x[0], reverse=True)
            sorted_scores = [pair[0] for pair in score_token_pairs]
            sorted_tokens = [pair[1] for pair in score_token_pairs]
            
            ax1.plot(range(len(sorted_scores)), sorted_scores, 
                    marker='o', label=f'Layer {layer_idx}', linewidth=2, markersize=4)
        
        ax1.set_title('Layer-wise Average CLS Attention Scores (Sorted)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tokens (sorted by attention)', fontsize=12)
        ax1.set_ylabel('Average Attention Score', fontsize=12)
        ax1.set_xticks(range(len(all_node_names)))
        ax1.set_xticklabels(sorted_tokens, rotation=90, fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 히트맵으로 레이어별 attention 패턴 비교
        layer_scores_matrix = np.array(layer_avg_scores)  # [n_layers, n_tokens]
        
        im = ax2.imshow(layer_scores_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Layer-wise CLS Attention Pattern Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tokens', fontsize=12)
        ax2.set_ylabel('Layers', fontsize=12)
        ax2.set_xticks(range(len(all_node_names)))
        ax2.set_yticks(range(n_layers))
        ax2.set_xticklabels(all_node_names, rotation=90, fontsize=10)
        ax2.set_yticklabels([f'Layer {i}' for i in range(n_layers)], fontsize=10)
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Attention Score', fontsize=12)
        
        # 각 셀에 값 표시
        for i in range(n_layers):
            for j in range(len(all_node_names)):
                value = layer_scores_matrix[i, j]
                ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white' if value > layer_scores_matrix.max()/2 else 'black', 
                        fontsize=8)
        
        plt.suptitle(f'Sample {sample_idx} - CLS Attention Score Analysis Summary', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 저장
        summary_path = save_dir / f'sample_{sample_idx}_attention_summary.png'
        fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Attention score summary saved: {summary_path}")

    def _create_head_attention_heatmaps(self, attention_weights, feature_names, save_dir, sample_count, sample_idx):
        """각 layer별, head별로 개별 attention 히트맵 생성 + head 평균"""
        n_layers = len(attention_weights)
        n_heads = attention_weights[0].shape[1]
        seq_len = attention_weights[0].shape[2]
        
        all_node_names = ["CLS"] + feature_names
        
        # 전체 레이어 x (헤드 + 1) 그리드 생성 (맨 오른쪽에 평균 추가)
        fig, axes = plt.subplots(n_layers, n_heads + 1, figsize=((n_heads + 1) * 4, n_layers * 4))
        
        # axes 처리
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        if n_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # 전체 attention의 공통 스케일 계산
        all_attention = torch.cat([layer_att[sample_idx] for layer_att in attention_weights], dim=0)
        global_vmin = all_attention.min()
        global_vmax = all_attention.max()
        
        # 각 레이어별, 헤드별로 히트맵 생성
        for layer_idx, layer_attention in enumerate(attention_weights):
            sample_attention = layer_attention[sample_idx]  # 첫 번째 샘플만 처리
            
            # 개별 헤드들
            for head_idx in range(n_heads):
                ax = axes[layer_idx, head_idx]
                attention_map = sample_attention[head_idx].cpu().numpy()
                
                im = ax.imshow(attention_map, cmap='viridis', interpolation='nearest',
                            vmin=global_vmin, vmax=global_vmax)
                
                # 제목을 (Layer, Head) 형태로 표시
                ax.set_title(f'L{layer_idx}H{head_idx}', fontsize=12, fontweight='bold')
                
                # 축 라벨 설정
                ax.set_xticks(np.arange(seq_len))
                ax.set_yticks(np.arange(seq_len))
                
                if seq_len <= len(all_node_names):
                    ax.set_xticklabels(all_node_names[:seq_len], rotation=90, fontsize=8)
                    ax.set_yticklabels(all_node_names[:seq_len], fontsize=8)
                else:
                    ax.set_xticklabels([f'Node {i}' for i in range(seq_len)], rotation=90, fontsize=8)
                    ax.set_yticklabels([f'Node {i}' for i in range(seq_len)], fontsize=8)
            
            # 맨 오른쪽에 head 평균 히트맵 추가
            ax_avg = axes[layer_idx, n_heads]  # 마지막 열
            avg_attention = sample_attention.mean(dim=0).cpu().numpy()  # 모든 head 평균
            
            im_avg = ax_avg.imshow(avg_attention, cmap='viridis', interpolation='nearest',
                                vmin=global_vmin, vmax=global_vmax)
            
            ax_avg.set_title(f'L{layer_idx}Avg', fontsize=12, fontweight='bold')
            
            # 축 라벨 설정
            ax_avg.set_xticks(np.arange(seq_len))
            ax_avg.set_yticks(np.arange(seq_len))
            
            if seq_len <= len(all_node_names):
                ax_avg.set_xticklabels(all_node_names[:seq_len], rotation=90, fontsize=8)
                ax_avg.set_yticklabels(all_node_names[:seq_len], fontsize=8)
            else:
                ax_avg.set_xticklabels([f'Node {i}' for i in range(seq_len)], rotation=90, fontsize=8)
                ax_avg.set_yticklabels([f'Node {i}' for i in range(seq_len)], fontsize=8)
        
        # 컬러바 추가 (마지막 행에만)
        for head_idx in range(n_heads + 1):
            ax = axes[-1, head_idx]
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
        
        # 전체 제목 설정
        plt.suptitle(f'Sample {sample_idx} - Layer x Head Attention Analysis ({n_layers}L x {n_heads}H + Avg, {seq_len}x{seq_len})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 저장
        attention_path = save_dir / f'sample_{sample_idx}_layer_head_attention.png'
        fig.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Layer x Head attention heatmap saved: {attention_path}")


    def _create_single_layer_head_heatmaps(self, layer_attention, all_node_names, save_dir, sample_idx, layer_idx, n_heads):
        """단일 레이어의 모든 head에 대한 히트맵 생성"""
        seq_len = layer_attention.shape[1]
        
        # 동적 그리드 레이아웃 결정
        if n_heads <= 4:
            rows, cols = 2, 2
        elif n_heads <= 6:
            rows, cols = 2, 3
        elif n_heads <= 8:
            rows, cols = 2, 4
        elif n_heads <= 9:
            rows, cols = 3, 3
        elif n_heads <= 12:
            rows, cols = 3, 4
        elif n_heads <= 16:
            rows, cols = 4, 4
        else:
            rows, cols = 4, 4
        
        subplot_width = 4
        subplot_height = 4
        figsize = (subplot_width * cols, subplot_height * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # axes를 1차원 배열로 변환
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # 전체 attention의 공통 스케일 계산
        global_vmin = layer_attention.min()
        global_vmax = layer_attention.max()
        
        for head_idx in range(n_heads):
            if head_idx < len(axes):
                ax = axes[head_idx]
                attention_map = layer_attention[head_idx].cpu().numpy()
                
                im = ax.imshow(attention_map, cmap='viridis', interpolation='nearest',
                              vmin=global_vmin, vmax=global_vmax)
                ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
                
                # 축 라벨 설정
                ax.set_xticks(np.arange(seq_len))
                ax.set_yticks(np.arange(seq_len))
                
                if seq_len <= len(all_node_names):
                    ax.set_xticklabels(all_node_names[:seq_len], rotation=90, fontsize=8)
                    ax.set_yticklabels(all_node_names[:seq_len], fontsize=8)
                else:
                    ax.set_xticklabels([f'Node {i}' for i in range(seq_len)], rotation=90, fontsize=8)
                    ax.set_yticklabels([f'Node {i}' for i in range(seq_len)], fontsize=8)
                
                # 각 셀에 값 표시 (head 수가 적을 때만)
                if n_heads <= 8 and seq_len <= 10:
                    for i in range(seq_len):
                        for j in range(seq_len):
                            value = attention_map[i, j]
                            threshold = (global_vmin + global_vmax) / 2
                            text_color = "white" if value > threshold else "black"
                            ax.text(j, i, f"{value:.2f}", 
                                ha="center", va="center", 
                                color=text_color, fontsize=7, weight='bold')
                
                # 컬러바 (마지막 행에만 표시)
                if head_idx >= len(axes) - cols:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.ax.tick_params(labelsize=8)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Sample {sample_idx} - Layer {layer_idx} Head Attention Analysis ({n_heads} heads, {seq_len}x{seq_len})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 저장
        attention_path = save_dir / f'sample_{sample_idx}_layer_{layer_idx}_head_attention.png'
        fig.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Head attention heatmap saved: {attention_path}")

    def perform_layer_wise_clustering(self, attention_data, n_clusters=5, output_dir=None):
        """각 레이어별로 클러스터링 수행"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        clustering_results = {}
        
        for layer_idx in range(len([k for k in attention_data.keys() if k.startswith('layer_')])):
            layer_key = f'layer_{layer_idx}'
            if layer_key not in attention_data:
                continue
                
            logger.info(f"Performing clustering on layer {layer_idx}")
            
            # 해당 레이어의 attention maps 가져오기
            attention_maps = np.stack(attention_data[layer_key])  # [n_samples, n_heads, seq_len, seq_len]
            labels = np.array(attention_data['labels'])
            sample_ids = np.array(attention_data['sample_ids'])
            feature_names = attention_data['feature_names']
            
            # 각 head별로 클러스터링 수행
            head_clustering_results = {}
            
            for head_idx in range(attention_maps.shape[1]):
                # 해당 head의 attention maps 추출
                head_attention = attention_maps[:, head_idx, :, :]  # [n_samples, seq_len, seq_len]
                
                # 평탄화
                flattened_maps = head_attention.reshape(len(head_attention), -1)
                
                # K-means 클러스터링
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                cluster_assignments = kmeans.fit_predict(flattened_maps)
                
                head_clustering_results[head_idx] = {
                    'cluster_assignments': cluster_assignments,
                    'cluster_centers': kmeans.cluster_centers_,
                    'head_idx': head_idx
                }
                
                logger.info(f"Layer {layer_idx}, Head {head_idx}: {len(flattened_maps)} samples clustered into {n_clusters} clusters")
            
            clustering_results[layer_key] = head_clustering_results
            
            # 시각화
            if output_dir:
                self._visualize_layer_head_clustering(
                    attention_maps, head_clustering_results, feature_names, 
                    layer_idx, output_dir, n_clusters
                )
        
        return clustering_results

    def _visualize_layer_head_clustering(self, attention_maps, head_clustering_results, feature_names, 
                                       layer_idx, output_dir, n_clusters):
        """레이어별, head별 클러스터링 시각화"""
        n_heads = attention_maps.shape[1]
        seq_len = attention_maps.shape[2]
        
        # 각 head별로 t-SNE 시각화
        fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(5 * ((n_heads + 1) // 2), 10))
        if n_heads == 1:
            axes = [axes]
        elif n_heads <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for head_idx in range(n_heads):
            if head_idx >= len(axes):
                break
                
            ax = axes[head_idx]
            
            # 해당 head의 attention maps
            head_attention = attention_maps[:, head_idx, :, :]
            flattened_maps = head_attention.reshape(len(head_attention), -1)
            
            # t-SNE
            perplexity = min(30, len(flattened_maps) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_embeddings = tsne.fit_transform(flattened_maps)
            
            # 클러스터링 결과
            cluster_assignments = head_clustering_results[head_idx]['cluster_assignments']
            
            # 시각화
            unique_clusters = np.unique(cluster_assignments)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_assignments == cluster_id
                ax.scatter(tsne_embeddings[cluster_mask, 0], 
                          tsne_embeddings[cluster_mask, 1],
                          c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7)
            
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_idx} - Head-wise Clustering (t-SNE)', fontsize=16)
        plt.tight_layout()
        
        # 저장
        clustering_path = output_dir / f'layer_{layer_idx}_head_clustering.png'
        fig.savefig(clustering_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Layer {layer_idx} head clustering visualization saved: {clustering_path}")

def main():
    parser = argparse.ArgumentParser(description='TabularFLM_Basic Single Source Attention Maps Analysis')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                        help='Which model to use (Full or Few)')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for K-means')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Maximum number of samples for visualization')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--del_feat', nargs='+', default=[], 
                        help="features to remove")

    args = parser.parse_args()

    # 원본 체크포인트 경로 저장
    original_checkpoint_path = args.checkpoint_dir
    logger.info(f"Original checkpoint path: {original_checkpoint_path}")

    # 체크포인트 파일명에서 자동으로 삭제된 변수 추출
    auto_del_feat, d_part = extract_deleted_features_from_checkpoint(original_checkpoint_path)

    # 체크포인트 파일명에서 시드값 추출
    seed_value = extract_seed_from_checkpoint(original_checkpoint_path)

    # 실제 파일 존재 여부 확인 및 경로 보정
    checkpoint_file = Path(original_checkpoint_path)
    if not checkpoint_file.exists():
        parent_dir = checkpoint_file.parent
        filename_pattern = checkpoint_file.name.replace('[', r'\[').replace(']', r'\]')
        
        import glob
        matching_files = glob.glob(str(parent_dir / filename_pattern))
        
        if not matching_files:
            base_name = checkpoint_file.stem
            possible_patterns = [
                base_name.replace("D:[", "D:['").replace("]", "']") + ".pt",
                base_name.replace("D:[", 'D:["').replace("]", '"]') + ".pt",
            ]
            
            for pattern in possible_patterns:
                potential_path = parent_dir / pattern
                logger.info(f"Trying pattern: {potential_path}")
                if potential_path.exists():
                    args.checkpoint_dir = str(potential_path)
                    logger.info(f"Found actual file: {args.checkpoint_dir}")
                    break
            else:
                logger.error(f"Could not find checkpoint file. Tried:")
                logger.error(f"   Original: {original_checkpoint_path}")
                for pattern in possible_patterns:
                    logger.error(f"   Pattern: {parent_dir / pattern}")
                raise FileNotFoundError(f"Checkpoint file not found: {original_checkpoint_path}")
        else:
            args.checkpoint_dir = matching_files[0]
            logger.info(f"Found file via glob: {args.checkpoint_dir}")

    # 출력 디렉토리 설정
    if args.output_dir is None:
        config_folder = extract_checkpoint_config_for_folder(original_checkpoint_path)
        
        checkpoint_parent_str = str(Path(args.checkpoint_dir).parent)
        
        if '/checkpoints/' in checkpoint_parent_str:
            viz_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/visualization/')
        else:
            viz_parent_str = '/storage/personal/eungyeop/experiments/visualization/gpt2_mean/heart/Full'
        
        base_viz_parent = viz_parent_str.split(f'/{seed_value}')[0] if seed_value else viz_parent_str
        
        if seed_value is not None:
            seed_folder = f'seed_{seed_value}'
        else:
            seed_folder = 'seed_unknown'
        
        if d_part:
            clustering_folder = f'clustering_basic_{args.n_clusters}_{d_part}'
        else:
            clustering_folder = f'clustering_basic_{args.n_clusters}'
        
        args.output_dir = Path(base_viz_parent) / config_folder / seed_folder / clustering_folder

    logger.info(f"Results will be saved to: {args.output_dir}")

    # Inference 실행 시 자동 추출된 삭제 변수 적용
    inference = BasicAttentionInference(args.checkpoint_dir, auto_del_feat=auto_del_feat)

    # 데이터로더 선택
    if args.mode == 'Full':
        data_loader = inference.combined_loader
        logger.info("Using Full dataset loader")
    else:
        data_loader = inference.train_loader_few if hasattr(inference, 'train_loader_few') else inference.test_loader
        logger.info("Using Few-shot dataset loader")

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Head Attention 히트맵 시각화
    logger.info("Generating head attention heatmaps...")
    head_viz_dir = output_dir / 'head_attention_heatmaps'
    inference.visualize_head_attention_heatmaps(data_loader, head_viz_dir, args.max_samples)

    # 2. Attention Score 분포 시각화 (새로 추가)
    logger.info("Generating attention score distributions...")
    score_viz_dir = output_dir / 'attention_scores'
    inference.visualize_attention_score_distribution(data_loader, score_viz_dir, args.max_samples)


    # 2. Attention maps 추출
    logger.info("Extracting attention maps for clustering...")
    attention_data = inference.extract_attention_maps(data_loader)

    # 3. 레이어별, head별 클러스터링 수행
    logger.info("Performing layer-wise head clustering...")
    clustering_results = inference.perform_layer_wise_clustering(
        attention_data, 
        n_clusters=args.n_clusters,
        output_dir=output_dir / 'clustering_results'
    )

    logger.info(f"Single source analysis completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()