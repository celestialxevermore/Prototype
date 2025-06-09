"""
Silhouette Analysis for Optimal Cluster Number

K-means 클러스터링의 최적 클러스터 개수를 실루엣 스코어를 통해 결정합니다.

Usage:
    python silhouette_analysis.py --checkpoint_dir /path/to/checkpoint.pt --mode Full
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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import seaborn as sns

# 기존 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SilhouetteAnalyzer:
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
        experiment_id = "silhouette_analysis"
        mode = "analysis"
        
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
        데이터로더에서 attention maps 추출
        
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
                
                # 모델 forward (attention weights 추출)
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
        
        logger.info(f"Extracted attention maps for {sample_count} samples")
        return attention_data
    
    def _extract_attention_from_model(self, batch):
        """
        모델에서 attention weights와 예측값을 추출
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

    def perform_silhouette_analysis(self, attention_data, layer_idx, k_range=(2, 15), output_dir=None):
        """
        특정 레이어에 대해 실루엣 분석을 수행하여 최적 클러스터 개수를 찾습니다.
        
        Args:
            attention_data (dict): attention maps 데이터
            layer_idx (int): 분석할 레이어 인덱스
            k_range (tuple): 클러스터 개수 범위 (min_k, max_k)
            output_dir (str, optional): 결과 저장 디렉토리
        
        Returns:
            dict: 실루엣 분석 결과
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 특정 레이어의 attention maps 추출
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        feature_names = attention_data['feature_names']
        
        logger.info(f"Performing silhouette analysis on layer {layer_idx} with {len(attention_maps)} samples")
        logger.info(f"Testing cluster range: {k_range[0]} to {k_range[1]}")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        min_k, max_k = k_range
        k_values = list(range(min_k, max_k + 1))
        silhouette_scores = []
        inertias = []
        
        # 각 k값에 대해 클러스터링 및 실루엣 스코어 계산
        for k in k_values:
            logger.info(f"Testing k={k}...")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(flattened_maps)
            
            # 실루엣 스코어 계산
            sil_score = silhouette_score(flattened_maps, cluster_labels)
            silhouette_scores.append(sil_score)
            inertias.append(kmeans.inertia_)
            
            logger.info(f"k={k}: Silhouette Score = {sil_score:.4f}, Inertia = {kmeans.inertia_:.2f}")
        
        # 최적 k 찾기
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_values[best_k_idx]
        best_score = silhouette_scores[best_k_idx]
        
        logger.info(f"🎯 Best k for layer {layer_idx}: {best_k} (Silhouette Score: {best_score:.4f})")
        
        # 결과 저장
        results = {
            'layer_idx': layer_idx,
            'k_values': k_values,
            'silhouette_scores': [float(score) for score in silhouette_scores],  # float 변환
            'inertias': [float(inertia) for inertia in inertias],  # float 변환
            'best_k': int(best_k),  # int 변환
            'best_score': float(best_score),  # float 변환
            'feature_names': feature_names
        }
        
        if output_dir:
            # 1. 실루엣 스코어 플롯
            self._plot_silhouette_scores(results, output_dir)
            
            # 2. 엘보우 플롯
            self._plot_elbow_curve(results, output_dir)
            
            # 3. 최적 k에 대한 상세 실루엣 분석
            self._detailed_silhouette_analysis(flattened_maps, best_k, layer_idx, output_dir)
            
            # 4. t-SNE 시각화 (최적 k)
            self._visualize_optimal_clustering(flattened_maps, labels, best_k, layer_idx, output_dir)
            
            # 5. 결과 JSON 저장
            results_json = results.copy()
            results_json['feature_names'] = list(feature_names)  # numpy array를 list로 변환
            
            with open(output_dir / f'layer_{layer_idx}_silhouette_results.json', 'w') as f:
                json.dump(results_json, f, indent=2)
            
            logger.info(f"✅ Silhouette analysis results saved to {output_dir}")
        
        return results

    def _plot_silhouette_scores(self, results, output_dir):
        """실루엣 스코어 플롯 생성"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        k_values = results['k_values']
        scores = results['silhouette_scores']
        best_k = results['best_k']
        
        # 실루엣 스코어 플롯
        ax.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(y=results['best_score'], color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # 최적 k 포인트 강조
        best_idx = k_values.index(best_k)
        ax.plot(best_k, scores[best_idx], 'ro', markersize=12, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title(f'Layer {results["layer_idx"]}: Silhouette Analysis\nOptimal k = {best_k} (Score: {results["best_score"]:.4f})', 
                    fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        # 점수 값 표시
        for i, (k, score) in enumerate(zip(k_values, scores)):
            ax.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        # 최적 k 텍스트 박스
        ax.text(0.02, 0.98, f'Best k: {best_k}\nBest Score: {results["best_score"]:.4f}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{results["layer_idx"]}_silhouette_scores.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_elbow_curve(self, results, output_dir):
        """엘보우 커브 플롯 생성"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        k_values = results['k_values']
        inertias = results['inertias']
        best_k = results['best_k']
        
        # 엘보우 커브
        ax.plot(k_values, inertias, 'go-', linewidth=2, markersize=8)
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # 최적 k 포인트 강조
        best_idx = k_values.index(best_k)
        ax.plot(best_k, inertias[best_idx], 'ro', markersize=12, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
        ax.set_title(f'Layer {results["layer_idx"]}: Elbow Method\nSilhouette-based Optimal k = {best_k}', 
                    fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{results["layer_idx"]}_elbow_curve.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _detailed_silhouette_analysis(self, flattened_maps, optimal_k, layer_idx, output_dir):
        """최적 k에 대한 상세 실루엣 분석"""
        # 최적 k로 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        
        # 샘플별 실루엣 스코어 계산
        sample_silhouette_values = silhouette_samples(flattened_maps, cluster_labels)
        
        # 실루엣 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 실루엣 플롯
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, optimal_k))
        
        for i in range(optimal_k):
            # i번째 클러스터의 실루엣 스코어들
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            # 클러스터 중앙에 라벨 표시
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient Values')
        ax1.set_ylabel('Cluster Label')
        ax1.set_title(f'Silhouette Plot (k={optimal_k})')
        
        # 평균 실루엣 스코어 라인
        silhouette_avg = silhouette_score(flattened_maps, cluster_labels)
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        ax1.legend()
        
        # 2. 클러스터 크기와 평균 실루엣 스코어
        cluster_sizes = []
        cluster_avg_scores = []
        
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_sizes.append(np.sum(cluster_mask))
            cluster_avg_scores.append(np.mean(sample_silhouette_values[cluster_mask]))
        
        x_pos = np.arange(optimal_k)
        bars = ax2.bar(x_pos, cluster_avg_scores, color=colors, alpha=0.7)
        ax2.axhline(y=silhouette_avg, color="red", linestyle="--", 
                   label=f'Overall Average: {silhouette_avg:.3f}')
        
        # 클러스터 크기 정보 추가
        for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={size}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Average Silhouette Score')
        ax2.set_title('Average Silhouette Score per Cluster')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'C{i}' for i in range(optimal_k)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx}: Detailed Silhouette Analysis (k={optimal_k})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_detailed_silhouette.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_optimal_clustering(self, flattened_maps, true_labels, optimal_k, layer_idx, output_dir):
        """최적 k로 클러스터링 결과를 t-SNE로 시각화"""
        # 최적 k로 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        
        # t-SNE 차원 축소
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 클러스터 결과
        colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
        for i in range(optimal_k):
            mask = cluster_labels == i
            ax1.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        ax1.set_title(f'K-means Clustering (k={optimal_k})')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 실제 라벨 결과
        unique_labels = np.unique(true_labels)
        label_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = true_labels == label
            marker = 'o' if label == 0 else 's'
            ax2.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[label_colors[i]], label=f'Label {int(label)}', 
                       alpha=0.7, s=50, marker=marker)
        
        ax2.set_title('True Labels')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx}: Optimal Clustering Visualization', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_optimal_clustering_viz.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_all_layers(self, data_loader, k_range=(2, 15), output_dir=None):
        """
        모든 레이어에 대해 실루엣 분석을 수행합니다.
        
        Args:
            data_loader: 데이터로더
            k_range (tuple): 클러스터 개수 범위
            output_dir (str, optional): 결과 저장 디렉토리
        
        Returns:
            dict: 모든 레이어의 분석 결과
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Attention maps 추출
        logger.info("Extracting attention maps...")
        attention_data = self.extract_attention_maps(data_loader)
        
        # 모든 레이어 분석
        all_results = {}
        for layer_idx in range(len(self.model.layers)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing Layer {layer_idx}")
            logger.info(f"{'='*50}")
            
            layer_output_dir = output_dir / f'layer_{layer_idx}' if output_dir else None
            results = self.perform_silhouette_analysis(
                attention_data, 
                layer_idx, 
                k_range=k_range,
                output_dir=layer_output_dir
            )
            all_results[f'layer_{layer_idx}'] = results
        
        # 전체 요약 생성
        if output_dir:
            self._generate_summary_comparison(all_results, output_dir)
        
        return all_results

    def _generate_summary_comparison(self, all_results, output_dir):
        """전체 레이어 결과 비교 요약"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        layers = sorted([int(k.split('_')[1]) for k in all_results.keys()])
        
        # 1. 레이어별 최적 k 비교
        ax = axes[0, 0]
        best_ks = [all_results[f'layer_{layer}']['best_k'] for layer in layers]
        best_scores = [all_results[f'layer_{layer}']['best_score'] for layer in layers]
        
        bars = ax.bar(layers, best_ks, color='skyblue', alpha=0.7)
        for i, (layer, k, score) in enumerate(zip(layers, best_ks, best_scores)):
            ax.text(layer, k + 0.1, f'k={k}\n({score:.3f})', 
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Optimal k')
        ax.set_title('Optimal Cluster Number by Layer')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        
        # 2. 레이어별 최고 실루엣 스코어
        ax = axes[0, 1]
        ax.plot(layers, best_scores, 'ro-', linewidth=2, markersize=8)
        for layer, score in zip(layers, best_scores):
            ax.annotate(f'{score:.3f}', (layer, score), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Best Silhouette Score')
        ax.set_title('Best Silhouette Score by Layer')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        
        # 3. 전체 실루엣 스코어 히트맵
        ax = axes[1, 0]
        k_range = all_results[f'layer_{layers[0]}']['k_values']
        silhouette_matrix = []
        
        for layer in layers:
            silhouette_matrix.append(all_results[f'layer_{layer}']['silhouette_scores'])
        
        silhouette_matrix = np.array(silhouette_matrix)
        im = ax.imshow(silhouette_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('k value')
        ax.set_ylabel('Layer')
        ax.set_title('Silhouette Scores Heatmap')
        ax.set_xticks(range(len(k_range)))
        ax.set_xticklabels(k_range)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {layer}' for layer in layers])
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Silhouette Score')
        
        # 최적 k 위치 표시
        for i, layer in enumerate(layers):
            best_k = all_results[f'layer_{layer}']['best_k']
            best_k_idx = k_range.index(best_k)
            ax.plot(best_k_idx, i, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        
        # 4. 요약 통계
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Silhouette Analysis Summary\n\n"
        summary_text += f"Tested k range: {min(k_range)} - {max(k_range)}\n"
        summary_text += f"Total layers: {len(layers)}\n\n"
        
        summary_text += "Optimal k per layer:\n"
        for layer in layers:
            result = all_results[f'layer_{layer}']
            summary_text += f"Layer {layer}: k={result['best_k']} (score: {result['best_score']:.4f})\n"
        
        # 전체 평균
        avg_best_k = np.mean(best_ks)
        avg_best_score = np.mean(best_scores)
        summary_text += f"\nAverage optimal k: {avg_best_k:.1f}\n"
        summary_text += f"Average best score: {avg_best_score:.4f}\n"
        
        # 가장 좋은 레이어
        best_layer_idx = layers[np.argmax(best_scores)]
        best_overall_score = max(best_scores)
        summary_text += f"\nBest performing layer: {best_layer_idx}\n"
        summary_text += f"Best overall score: {best_overall_score:.4f}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Comprehensive Silhouette Analysis Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'silhouette_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 요약 JSON 저장
        summary_data = {
            'analysis_summary': {
                'k_range': [int(k) for k in k_range],  # int 변환
                'layers_analyzed': [int(layer) for layer in layers],  # int 변환
                'optimal_k_per_layer': {f'layer_{layer}': int(all_results[f'layer_{layer}']['best_k']) for layer in layers},
                'best_scores_per_layer': {f'layer_{layer}': float(all_results[f'layer_{layer}']['best_score']) for layer in layers},
                'average_optimal_k': float(avg_best_k),
                'average_best_score': float(avg_best_score),
                'best_performing_layer': int(best_layer_idx),
                'best_overall_score': float(best_overall_score)
            },
            'detailed_results': {}
        }
        
        # 각 레이어의 상세 결과도 포함
        for layer_key, result in all_results.items():
            summary_data['detailed_results'][layer_key] = {
                'best_k': int(result['best_k']),
                'best_score': float(result['best_score']),
                'k_values': [int(k) for k in result['k_values']],
                'silhouette_scores': [float(score) for score in result['silhouette_scores']]
            }
        
        with open(output_dir / 'comprehensive_silhouette_analysis.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info("✅ Comprehensive silhouette analysis summary generated!")
        
        # 추천사항 로그 출력
        logger.info("\n" + "="*60)
        logger.info("🎯 RECOMMENDATIONS")
        logger.info("="*60)
        logger.info(f"📊 Best performing layer: Layer {best_layer_idx} (score: {best_overall_score:.4f})")
        logger.info(f"🔢 Most common optimal k: {max(set(best_ks), key=best_ks.count)}")
        logger.info(f"📈 Layer with highest variability: Layer {layers[np.argmax([max(all_results[f'layer_{layer}']['silhouette_scores']) - min(all_results[f'layer_{layer}']['silhouette_scores']) for layer in layers])]}")
        
        # 안정적인 k 추천
        k_consensus = []
        for k in k_range:
            scores_for_k = [all_results[f'layer_{layer}']['silhouette_scores'][all_results[f'layer_{layer}']['k_values'].index(k)] for layer in layers]
            if min(scores_for_k) > 0.3:  # 모든 레이어에서 0.3 이상
                k_consensus.append((k, np.mean(scores_for_k)))
        
        if k_consensus:
            best_consensus_k = max(k_consensus, key=lambda x: x[1])
            logger.info(f"🌟 Consensus recommendation: k={best_consensus_k[0]} (avg score: {best_consensus_k[1]:.4f})")
        
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Silhouette Analysis for Optimal Cluster Number')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                       help='Which model to use (Full or Few)')
    parser.add_argument('--min_k', type=int, default=2,
                       help='Minimum number of clusters to test (default: 2)')
    parser.add_argument('--max_k', type=int, default=15,
                       help='Maximum number of clusters to test (default: 15)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--layer_idx', type=int, default=None,
                       help='Specific layer to analyze (if not specified, analyze all layers)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        path_parts = checkpoint_dir.parts
        
        # visualization 폴더 구조에 맞게 설정
        for i, part in enumerate(path_parts):
            if part == 'checkpoints':
                viz_parts = list(path_parts)
                viz_parts[i] = 'visualization'
                viz_path = Path(*viz_parts[:-1])  # best_model_epoch_XX.pt 제외
                args.output_dir = viz_path / 'silhouette_analysis'
                break
        
        if args.output_dir is None:
            args.output_dir = checkpoint_dir / 'silhouette_analysis'
    
    # Silhouette Analyzer 초기화
    analyzer = SilhouetteAnalyzer(args.checkpoint_dir)
    
    # 데이터로더 선택
    if args.mode == 'Full':
        data_loader = analyzer.train_loader
        logger.info("Using Full dataset loader")
    else:
        data_loader = analyzer.train_loader_few if hasattr(analyzer, 'train_loader_few') else analyzer.test_loader
        logger.info("Using Few-shot dataset loader")
    
    k_range = (args.min_k, args.max_k)
    
    if args.layer_idx is not None:
        # 특정 레이어만 분석
        logger.info(f"Analyzing only Layer {args.layer_idx}")
        
        # Attention maps 추출
        attention_data = analyzer.extract_attention_maps(data_loader)
        
        # 특정 레이어 분석
        layer_output_dir = Path(args.output_dir) / f'layer_{args.layer_idx}'
        results = analyzer.perform_silhouette_analysis(
            attention_data, 
            args.layer_idx, 
            k_range=k_range,
            output_dir=layer_output_dir
        )
        
        logger.info(f"\n🎯 Results for Layer {args.layer_idx}:")
        logger.info(f"   Optimal k: {results['best_k']}")
        logger.info(f"   Best Silhouette Score: {results['best_score']:.4f}")
        
    else:
        # 모든 레이어 분석
        logger.info("Analyzing all layers")
        all_results = analyzer.analyze_all_layers(
            data_loader, 
            k_range=k_range,
            output_dir=args.output_dir
        )
    
    logger.info(f"\n✅ Silhouette analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()