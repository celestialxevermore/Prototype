"""
Fixed K Clustering Metrics Analysis

기존 clustering_{k} 폴더에서 해당 k값을 기준으로 
다양한 클러스터링 메트릭을 분석합니다.

Usage:
    python analysis3.py --clustering_dir /path/to/clustering_15 --checkpoint_dir /path/to/checkpoint.pt
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
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import cdist
import re

# 기존 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_deterministic():
    """완전한 deterministic 설정"""
    # Random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("✅ Deterministic mode enabled")

class FixedKMetricsAnalyzer:
    def __init__(self, clustering_dir, checkpoint_dir, device='cuda'):
        """
        Args:
            clustering_dir (str): clustering_{k} 폴더 경로
            checkpoint_dir (str): 체크포인트 파일 경로
            device (str): 'cuda' 또는 'cpu'
        """
        # Deterministic 설정 먼저
        ensure_deterministic()
        
        self.clustering_dir = Path(clustering_dir)
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # clustering 폴더에서 k값 추출
        self.k = self._extract_k_from_path(clustering_dir)
        logger.info(f"Extracted k={self.k} from clustering directory: {clustering_dir}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Val AUC: {self.checkpoint['val_auc']:.4f}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
    def _extract_k_from_path(self, clustering_dir):
        """clustering 디렉토리 경로에서 k값 추출"""
        clustering_dir = Path(clustering_dir)
        
        # clustering_{k} 패턴에서 k 추출
        match = re.search(r'clustering_(\d+)', str(clustering_dir))
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract k value from path: {clustering_dir}")
    
    def _load_model(self):
        """체크포인트에서 모델 로드"""
        experiment_id = f"fixed_k_analysis_k{self.k}"
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
        """데이터로더 준비 (deterministic)"""
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
        데이터로더에서 attention maps 추출 (deterministic)
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
        """모델에서 attention weights와 예측값을 추출"""
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

    def calculate_within_between_variance(self, data, cluster_labels):
        """Within-cluster Sum of Squares (WCSS)와 Between-cluster Sum of Squares (BCSS) 계산"""
        n_samples = len(data)
        k = len(np.unique(cluster_labels))
        
        # 전체 데이터의 중심점 (Grand Centroid)
        grand_centroid = np.mean(data, axis=0)
        
        # Total Sum of Squares (TSS)
        tss = np.sum((data - grand_centroid) ** 2)
        
        # Within-cluster Sum of Squares (WCSS)
        wcss = 0
        cluster_centroids = []
        cluster_sizes = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_centroid = np.mean(cluster_data, axis=0)
            cluster_centroids.append(cluster_centroid)
            cluster_sizes.append(len(cluster_data))
            
            # 클러스터 내 분산 계산
            cluster_wcss = np.sum((cluster_data - cluster_centroid) ** 2)
            wcss += cluster_wcss
        
        cluster_centroids = np.array(cluster_centroids)
        cluster_sizes = np.array(cluster_sizes)
        
        # Between-cluster Sum of Squares (BCSS)
        bcss = 0
        for i, centroid in enumerate(cluster_centroids):
            bcss += cluster_sizes[i] * np.sum((centroid - grand_centroid) ** 2)
        
        # 검증: TSS = WCSS + BCSS (반드시 성립해야 함)
        tss_check = wcss + bcss
        
        return {
            'wcss': float(wcss),
            'bcss': float(bcss),
            'tss': float(tss),
            'tss_check': float(tss_check),
            'bcss_wcss_ratio': float(bcss / wcss) if wcss > 0 else float('inf'),
            'explained_variance_ratio': float(bcss / tss) if tss > 0 else 0.0,
            'n_clusters': int(k),
            'n_samples': int(n_samples),
            'cluster_sizes': cluster_sizes.tolist(),
            'verification_error': float(abs(tss - tss_check))
        }

    def _calculate_dunn_index(self, data, cluster_labels):
        """Dunn Index 계산"""
        unique_labels = np.unique(cluster_labels)
        
        # Inter-cluster distances (클러스터 간 최소 거리)
        inter_cluster_distances = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                cluster_i = data[cluster_labels == unique_labels[i]]
                cluster_j = data[cluster_labels == unique_labels[j]]
                
                # 두 클러스터 간 모든 점들의 최소 거리
                distances = cdist(cluster_i, cluster_j, metric='euclidean')
                min_distance = np.min(distances)
                inter_cluster_distances.append(min_distance)
        
        # Intra-cluster distances (클러스터 내 최대 거리)
        intra_cluster_distances = []
        for label in unique_labels:
            cluster_data = data[cluster_labels == label]
            if len(cluster_data) > 1:
                distances = cdist(cluster_data, cluster_data, metric='euclidean')
                # 대각선 제외하고 최대 거리
                np.fill_diagonal(distances, 0)
                max_distance = np.max(distances)
                intra_cluster_distances.append(max_distance)
        
        if len(inter_cluster_distances) == 0 or len(intra_cluster_distances) == 0:
            return 0.0
        
        min_inter = np.min(inter_cluster_distances)
        max_intra = np.max(intra_cluster_distances)
        
        return min_inter / max_intra if max_intra > 0 else 0.0

    def _calculate_cluster_statistics(self, data, cluster_labels):
        """클러스터별 세부 통계 계산"""
        unique_labels = np.unique(cluster_labels)
        cluster_stats = {}
        
        for label in unique_labels:
            cluster_data = data[cluster_labels == label]
            centroid = np.mean(cluster_data, axis=0)
            
            # 클러스터 내 거리들
            if len(cluster_data) > 1:
                distances_to_centroid = np.linalg.norm(cluster_data - centroid, axis=1)
                intra_cluster_distances = cdist(cluster_data, cluster_data, metric='euclidean')
                np.fill_diagonal(intra_cluster_distances, np.inf)  # 자기 자신 제외
                
                cluster_stats[f'cluster_{label}'] = {
                    'size': int(len(cluster_data)),
                    'centroid': centroid.tolist(),
                    'mean_distance_to_centroid': float(np.mean(distances_to_centroid)),
                    'max_distance_to_centroid': float(np.max(distances_to_centroid)),
                    'std_distance_to_centroid': float(np.std(distances_to_centroid)),
                    'min_intra_distance': float(np.min(intra_cluster_distances)),
                    'mean_intra_distance': float(np.mean(intra_cluster_distances[intra_cluster_distances != np.inf])),
                    'max_intra_distance': float(np.max(intra_cluster_distances[intra_cluster_distances != np.inf]))
                }
            else:
                cluster_stats[f'cluster_{label}'] = {
                    'size': 1,
                    'centroid': centroid.tolist(),
                    'mean_distance_to_centroid': 0.0,
                    'max_distance_to_centroid': 0.0,
                    'std_distance_to_centroid': 0.0,
                    'min_intra_distance': 0.0,
                    'mean_intra_distance': 0.0,
                    'max_intra_distance': 0.0
                }
        
        return cluster_stats

    def calculate_comprehensive_metrics(self, data, cluster_labels):
        """모든 클러스터링 메트릭을 종합적으로 계산"""
        metrics = {}
        
        # 1. Silhouette Metrics
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_samples_scores = silhouette_samples(data, cluster_labels)
        
        metrics['silhouette'] = {
            'average_score': float(silhouette_avg),
            'sample_scores': silhouette_samples_scores.tolist(),
            'min_score': float(np.min(silhouette_samples_scores)),
            'max_score': float(np.max(silhouette_samples_scores)),
            'std_score': float(np.std(silhouette_samples_scores))
        }
        
        # 2. Within/Between Variance Analysis
        variance_metrics = self.calculate_within_between_variance(data, cluster_labels)
        metrics['variance_analysis'] = variance_metrics
        
        # 3. Calinski-Harabasz Index (분산비 기준)
        ch_score = calinski_harabasz_score(data, cluster_labels)
        metrics['calinski_harabasz'] = {
            'score': float(ch_score),
            'interpretation': 'higher_is_better'
        }
        
        # 4. Davies-Bouldin Index
        db_score = davies_bouldin_score(data, cluster_labels)
        metrics['davies_bouldin'] = {
            'score': float(db_score),
            'interpretation': 'lower_is_better'
        }
        
        # 5. Inertia (K-means WCSS)
        kmeans_temp = KMeans(n_clusters=len(np.unique(cluster_labels)), random_state=42, n_init=1)
        kmeans_temp.fit(data)
        metrics['inertia'] = {
            'score': float(kmeans_temp.inertia_),
            'interpretation': 'lower_is_better'
        }
        
        # 6. Dunn Index
        dunn_score = self._calculate_dunn_index(data, cluster_labels)
        metrics['dunn_index'] = {
            'score': float(dunn_score),
            'interpretation': 'higher_is_better'
        }
        
        # 7. 클러스터별 통계
        cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
        metrics['cluster_statistics'] = cluster_stats
        
        return metrics

    def analyze_fixed_k_clustering(self, data_loader, mode='Full'):
        """
        지정된 k값으로 모든 레이어에 대해 클러스터링 메트릭 분석
        
        Args:
            data_loader: 데이터로더
            mode: 'Full' 또는 'Few'
        
        Returns:
            dict: 분석 결과
        """
        # 결과 저장을 위한 metrics_analysis 폴더 생성
        metrics_output_dir = self.clustering_dir / 'metrics_analysis'
        metrics_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting fixed k={self.k} analysis for all layers")
        logger.info(f"Results will be saved to: {metrics_output_dir}")
        
        # Attention maps 추출
        logger.info("Extracting attention maps...")
        attention_data = self.extract_attention_maps(data_loader)
        
        # 모든 레이어 분석
        all_results = {
            'analysis_info': {
                'k_value': self.k,
                'mode': mode,
                'clustering_dir': str(self.clustering_dir),
                'checkpoint_dir': str(self.checkpoint_dir),
                'total_samples': len(attention_data['labels']),
                'feature_names': attention_data['feature_names']
            },
            'layer_results': {}
        }
        
        for layer_idx in range(len(self.model.layers)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing Layer {layer_idx} with k={self.k}")
            logger.info(f"{'='*60}")
            
            # 특정 레이어의 attention maps 추출
            attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
            labels = np.array(attention_data['labels'])
            feature_names = attention_data['feature_names']
            
            # 평탄화 (벡터화)
            flattened_maps = attention_maps.reshape(len(attention_maps), -1)
            
            # K-means 클러스터링 (deterministic)
            np.random.seed(42)
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10, algorithm='lloyd')
            cluster_labels = kmeans.fit_predict(flattened_maps)
            
            # 모든 메트릭 계산
            all_metrics = self.calculate_comprehensive_metrics(flattened_maps, cluster_labels)
            
            # 결과 저장
            layer_result = {
                'layer_idx': layer_idx,
                'k_value': self.k,
                'cluster_assignments': cluster_labels.tolist(),
                'metrics': all_metrics,
                'kmeans_centers': kmeans.cluster_centers_.tolist()
            }
            
            all_results['layer_results'][f'layer_{layer_idx}'] = layer_result
            
            # 주요 메트릭 로그 출력
            silhouette = all_metrics['silhouette']['average_score']
            ch_score = all_metrics['calinski_harabasz']['score']
            db_score = all_metrics['davies_bouldin']['score']
            wcss = all_metrics['variance_analysis']['wcss']
            bcss = all_metrics['variance_analysis']['bcss']
            dunn = all_metrics['dunn_index']['score']
            
            logger.info(f"Layer {layer_idx} Results (k={self.k}):")
            logger.info(f"  Silhouette Score: {silhouette:.4f}")
            logger.info(f"  Calinski-Harabasz: {ch_score:.2f}")
            logger.info(f"  Davies-Bouldin: {db_score:.4f}")
            logger.info(f"  WCSS: {wcss:.2f}")
            logger.info(f"  BCSS: {bcss:.2f}")
            logger.info(f"  BCSS/WCSS Ratio: {bcss/wcss:.3f}")
            logger.info(f"  Dunn Index: {dunn:.4f}")
            
            # 레이어별 상세 시각화 생성
            layer_output_dir = metrics_output_dir / f'layer_{layer_idx}'
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            
            self._create_layer_visualizations(
                layer_result, flattened_maps, labels, 
                attention_data['feature_names'], layer_output_dir
            )
        
        # 전체 요약 시각화 생성
        self._create_summary_visualizations(all_results, metrics_output_dir)
        
        # 결과 JSON 저장
        results_json = all_results.copy()
        results_json['analysis_info']['feature_names'] = list(attention_data['feature_names'])
        
        with open(metrics_output_dir / f'k{self.k}_comprehensive_metrics.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # 요약 로그 출력
        self._log_summary_results(all_results)
        
        logger.info(f"\n✅ Fixed k={self.k} analysis completed!")
        logger.info(f"Results saved to: {metrics_output_dir}")
        
        return all_results

    def _create_layer_visualizations(self, layer_result, flattened_maps, labels, feature_names, output_dir):
        """레이어별 상세 시각화 생성"""
        layer_idx = layer_result['layer_idx']
        k = layer_result['k_value']
        cluster_labels = np.array(layer_result['cluster_assignments'])
        metrics = layer_result['metrics']
        
        # 1. 메트릭 요약 차트
        self._plot_metrics_summary(layer_idx, k, metrics, cluster_labels, output_dir)
        
        # 2. Within/Between Variance 분석
        self._plot_variance_breakdown(layer_idx, k, metrics, output_dir)
        
        # 3. 실루엣 분석
        self._plot_silhouette_analysis(layer_idx, k, flattened_maps, cluster_labels, output_dir)
        
        # 4. 클러스터 분포 시각화 (t-SNE)
        self._plot_cluster_distribution(layer_idx, k, flattened_maps, cluster_labels, labels, output_dir)

    def _plot_metrics_summary(self, layer_idx, k, metrics, cluster_labels, output_dir):
        """메트릭 요약 차트"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 메트릭 값 추출
        silhouette = metrics['silhouette']['average_score']
        ch_score = metrics['calinski_harabasz']['score']
        db_score = metrics['davies_bouldin']['score']
        dunn = metrics['dunn_index']['score']
        wcss = metrics['variance_analysis']['wcss']
        bcss = metrics['variance_analysis']['bcss']
        
        # 1. 주요 메트릭 바차트
        ax = axes[0, 0]
        metric_names = ['Silhouette', 'C-H Index', 'Davies-Bouldin', 'Dunn Index']
        metric_values = [silhouette, ch_score/100, 1/db_score if db_score > 0 else 0, dunn]  # 정규화
        colors = ['blue', 'green', 'red', 'orange']
        
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_title(f'Layer {layer_idx}: Normalized Metrics (k={k})')
        ax.set_ylabel('Normalized Scores')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 실제 값 표시
        for i, (bar, val) in enumerate(zip(bars, [silhouette, ch_score, db_score, dunn])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. WCSS vs BCSS
        ax = axes[0, 1]
        variance_names = ['WCSS\n(Within)', 'BCSS\n(Between)']
        variance_values = [wcss, bcss]
        colors = ['red', 'blue']
        
        bars = ax.bar(variance_names, variance_values, color=colors, alpha=0.7)
        ax.set_title(f'Within vs Between Variance')
        ax.set_ylabel('Sum of Squares')
        
        for bar, val in zip(bars, variance_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(variance_values)*0.01,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. 클러스터 크기 분포
        ax = axes[0, 2]
        cluster_sizes = metrics['variance_analysis']['cluster_sizes']
        cluster_ids = list(range(len(cluster_sizes)))
        
        bars = ax.bar(cluster_ids, cluster_sizes, alpha=0.7, color='purple')
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Samples')
        
        for bar, size in zip(bars, cluster_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(size), ha='center', va='bottom', fontsize=9)
        
        # 4. 실루엣 스코어 분포
        ax = axes[1, 0]
        silhouette_scores = metrics['silhouette']['sample_scores']
        ax.hist(silhouette_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(silhouette, color='red', linestyle='--', linewidth=2, label=f'Mean: {silhouette:.3f}')
        ax.set_title('Silhouette Score Distribution')
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 5. 클러스터별 평균 실루엣 스코어
        ax = axes[1, 1]
        cluster_silhouette_means = []
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_scores = np.array(silhouette_scores)[cluster_mask]
                cluster_silhouette_means.append(np.mean(cluster_scores))
            else:
                cluster_silhouette_means.append(0)
        
        bars = ax.bar(range(k), cluster_silhouette_means, alpha=0.7, color='lightgreen')
        ax.axhline(silhouette, color='red', linestyle='--', label=f'Overall: {silhouette:.3f}')
        ax.set_title('Average Silhouette Score per Cluster')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Average Silhouette Score')
        ax.legend()
        
        for bar, score in zip(bars, cluster_silhouette_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. 요약 통계 텍스트
        ax = axes[1, 2]
        ax.axis('off')
        
        # 해석 지침
        interpretation_text = f"Layer {layer_idx} Metrics Interpretation (k={k})\n\n"
        interpretation_text += f"Silhouette Score: {silhouette:.4f}\n"
        if silhouette >= 0.5:
            interpretation_text += "  → Strong cluster structure ✓\n"
        elif silhouette >= 0.25:
            interpretation_text += "  → Reasonable cluster structure\n"
        else:
            interpretation_text += "  → Weak cluster structure ✗\n"
        
        interpretation_text += f"\nCalinski-Harabasz: {ch_score:.2f}\n"
        interpretation_text += "  → Higher is better\n"
        
        interpretation_text += f"\nDavies-Bouldin: {db_score:.4f}\n"
        interpretation_text += "  → Lower is better\n"
        
        interpretation_text += f"\nBCSS/WCSS Ratio: {bcss/wcss:.3f}\n"
        interpretation_text += "  → Higher = better separation\n"
        
        interpretation_text += f"\nExplained Variance: {metrics['variance_analysis']['explained_variance_ratio']:.3f}\n"
        interpretation_text += "  → Higher = better clustering\n"
        
        # 클러스터 품질 평가
        interpretation_text += f"\nOverall Quality: "
        if silhouette >= 0.5 and bcss/wcss >= 1.0:
            interpretation_text += "EXCELLENT ⭐⭐⭐"
        elif silhouette >= 0.3 and bcss/wcss >= 0.5:
            interpretation_text += "GOOD ⭐⭐"
        elif silhouette >= 0.25:
            interpretation_text += "FAIR ⭐"
        else:
            interpretation_text += "POOR"
        
        ax.text(0.1, 0.9, interpretation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx}: Comprehensive Metrics Summary (k={k})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_metrics_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_variance_breakdown(self, layer_idx, k, metrics, output_dir):
        """Within/Between Variance 상세 분석"""
        variance_data = metrics['variance_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Variance 구성요소
        ax = axes[0, 0]
        components = ['WCSS', 'BCSS', 'TSS']
        values = [variance_data['wcss'], variance_data['bcss'], variance_data['tss']]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(components, values, color=colors, alpha=0.7)
        ax.set_title('Variance Components')
        ax.set_ylabel('Sum of Squares')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Variance 분해 (파이 차트)
        ax = axes[0, 1]
        sizes = [variance_data['wcss'], variance_data['bcss']]
        labels = [f'WCSS\n({variance_data["wcss"]:.1f})', f'BCSS\n({variance_data["bcss"]:.1f})']
        colors = ['lightcoral', 'lightblue']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 10})
        ax.set_title('Variance Decomposition')
        
        # 3. 클러스터별 크기와 분산
        ax = axes[1, 0]
        cluster_sizes = variance_data['cluster_sizes']
        cluster_ids = list(range(len(cluster_sizes)))
        
        # 클러스터별 세부 통계에서 분산 정보 추출
        cluster_variances = []
        for i in range(k):
            cluster_key = f'cluster_{i}'
            if cluster_key in metrics['cluster_statistics']:
                cluster_variances.append(metrics['cluster_statistics'][cluster_key]['std_distance_to_centroid'])
            else:
                cluster_variances.append(0)
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cluster_sizes, width, label='Size', alpha=0.7)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, cluster_variances, width, label='Std Dev', alpha=0.7, color='orange')
        
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Cluster Size', color='blue')
        ax2.set_ylabel('Standard Deviation', color='orange')
        ax.set_title('Cluster Size vs Internal Variance')
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_ids)
        
        # 범례
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 4. 메트릭 요약 및 해석
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"Variance Analysis Summary\n\n"
        summary_text += f"k = {k} clusters\n"
        summary_text += f"Total samples = {variance_data['n_samples']}\n\n"
        
        summary_text += f"WCSS = {variance_data['wcss']:.2f}\n"
        summary_text += f"BCSS = {variance_data['bcss']:.2f}\n"
        summary_text += f"TSS = {variance_data['tss']:.2f}\n\n"
        
        summary_text += f"BCSS/WCSS = {variance_data['bcss_wcss_ratio']:.3f}\n"
        summary_text += f"Explained Var = {variance_data['explained_variance_ratio']:.3f}\n\n"
        
        summary_text += f"Verification:\n"
        summary_text += f"TSS = WCSS + BCSS\n"
        summary_text += f"{variance_data['tss']:.2f} = {variance_data['wcss']:.2f} + {variance_data['bcss']:.2f}\n"
        summary_text += f"Error = {variance_data['verification_error']:.6f}\n\n"
        
        # 품질 평가
        ratio = variance_data['bcss_wcss_ratio']
        explained = variance_data['explained_variance_ratio']
        
        summary_text += f"Quality Assessment:\n"
        if ratio >= 2.0 and explained >= 0.7:
            summary_text += "EXCELLENT separation ⭐⭐⭐\n"
        elif ratio >= 1.0 and explained >= 0.5:
            summary_text += "GOOD separation ⭐⭐\n"
        elif ratio >= 0.5 and explained >= 0.3:
            summary_text += "FAIR separation ⭐\n"
        else:
            summary_text += "POOR separation\n"
        
        summary_text += f"\nRecommendation:\n"
        if ratio < 1.0:
            summary_text += "Consider different k or\nfeature engineering"
        else:
            summary_text += "Clustering quality is\nacceptable for analysis"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx}: Within/Between Variance Analysis (k={k})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_variance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_silhouette_analysis(self, layer_idx, k, flattened_maps, cluster_labels, output_dir):
        """실루엣 분석 (기존 스타일과 동일)"""
        # 실루엣 분석
        silhouette_avg = silhouette_score(flattened_maps, cluster_labels)
        sample_silhouette_values = silhouette_samples(flattened_maps, cluster_labels)
        
        # 실루엣 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 실루엣 플롯
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, k))
        
        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient Values')
        ax1.set_ylabel('Cluster Label')
        ax1.set_title(f'Silhouette Plot (k={k})')
        
        # 평균 실루엣 스코어 라인
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        ax1.legend()
        
        # 2. 클러스터별 평균 실루엣 스코어
        cluster_avg_scores = []
        cluster_sizes = []
        
        for i in range(k):
            cluster_mask = cluster_labels == i
            cluster_sizes.append(np.sum(cluster_mask))
            cluster_avg_scores.append(np.mean(sample_silhouette_values[cluster_mask]))
        
        x_pos = np.arange(k)
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
        ax2.set_xticklabels([f'C{i}' for i in range(k)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx}: Detailed Silhouette Analysis (k={k})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_silhouette_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cluster_distribution(self, layer_idx, k, flattened_maps, cluster_labels, labels, output_dir):
        """클러스터 분포 시각화 (t-SNE)"""
        # t-SNE 차원 축소
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 2x1 서브플롯
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 클러스터 결과
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            mask = cluster_labels == i
            ax1.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        ax1.set_title(f'K-means Clustering (k={k})')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        if k <= 15:  # 너무 많은 클러스터는 범례 생략
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 실제 라벨 결과
        unique_labels = np.unique(labels)
        label_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            marker = 'o' if label == 0 else 's'
            ax2.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[label_colors[i]], label=f'True Label {int(label)}', 
                       alpha=0.7, s=50, marker=marker)
        
        ax2.set_title('True Labels')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 클러스터 통계 정보 추가
        cluster_stats = []
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_labels_subset = labels[cluster_mask]
            total_count = np.sum(cluster_mask)
            total_percentage = (total_count / len(cluster_labels)) * 100
            
            label_counts = {}
            for label in unique_labels:
                count = np.sum(cluster_labels_subset == label)
                label_counts[int(label)] = count
            
            label_str = ", ".join([f"L{k}:{v}" for k, v in label_counts.items()])
            cluster_stats.append(f"C{cluster_id}: {total_count} ({total_percentage:.1f}%) [{label_str}]")
        
        if cluster_stats:
            stats_text = "\n".join(cluster_stats)
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))
        
        plt.suptitle(f'Layer {layer_idx}: Cluster Distribution Visualization (k={k})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_cluster_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_visualizations(self, all_results, output_dir):
        """전체 요약 시각화"""
        layers = sorted([int(k.split('_')[1]) for k in all_results['layer_results'].keys()])
        k = all_results['analysis_info']['k_value']
        
        # 레이어별 메트릭 수집
        metrics_by_layer = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'dunn': [],
            'wcss': [],
            'bcss': [],
            'bcss_wcss_ratio': [],
            'explained_variance': []
        }
        
        for layer in layers:
            layer_key = f'layer_{layer}'
            metrics = all_results['layer_results'][layer_key]['metrics']
            
            metrics_by_layer['silhouette'].append(metrics['silhouette']['average_score'])
            metrics_by_layer['calinski_harabasz'].append(metrics['calinski_harabasz']['score'])
            metrics_by_layer['davies_bouldin'].append(metrics['davies_bouldin']['score'])
            metrics_by_layer['dunn'].append(metrics['dunn_index']['score'])
            metrics_by_layer['wcss'].append(metrics['variance_analysis']['wcss'])
            metrics_by_layer['bcss'].append(metrics['variance_analysis']['bcss'])
            metrics_by_layer['bcss_wcss_ratio'].append(metrics['variance_analysis']['bcss_wcss_ratio'])
            metrics_by_layer['explained_variance'].append(metrics['variance_analysis']['explained_variance_ratio'])
        
        # 2x3 서브플롯으로 요약
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 레이어별 주요 메트릭 트렌드
        ax = axes[0, 0]
        ax.plot(layers, metrics_by_layer['silhouette'], 'bo-', label='Silhouette', linewidth=2)
        ax.plot(layers, np.array(metrics_by_layer['calinski_harabasz'])/100, 'go-', label='CH/100', linewidth=2)
        ax.plot(layers, 1/np.array(metrics_by_layer['davies_bouldin']), 'ro-', label='1/DB', linewidth=2)
        ax.set_title('Normalized Metrics Trend Across Layers')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. WCSS vs BCSS
        ax = axes[0, 1]
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, metrics_by_layer['wcss'], width, label='WCSS', alpha=0.7, color='red')
        bars2 = ax.bar(x + width/2, metrics_by_layer['bcss'], width, label='BCSS', alpha=0.7, color='blue')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sum of Squares')
        ax.set_title('WCSS vs BCSS by Layer')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {layer}' for layer in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. BCSS/WCSS 비율
        ax = axes[0, 2]
        bars = ax.bar(layers, metrics_by_layer['bcss_wcss_ratio'], alpha=0.7, color='green')
        ax.set_xlabel('Layer')
        ax.set_ylabel('BCSS/WCSS Ratio')
        ax.set_title('Cluster Separation Quality by Layer')
        ax.grid(True, alpha=0.3)
        
        for bar, ratio in zip(bars, metrics_by_layer['bcss_wcss_ratio']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 모든 메트릭 히트맵
        ax = axes[1, 0]
        
        # 메트릭 정규화 (0-1 스케일)
        metrics_matrix = []
        metric_names = ['Silhouette', 'CH/100', '1/DB', 'Dunn', 'BCSS/WCSS', 'Explained Var']
        
        for layer_idx, layer in enumerate(layers):
            layer_metrics = [
                metrics_by_layer['silhouette'][layer_idx],
                metrics_by_layer['calinski_harabasz'][layer_idx] / 100,
                1 / metrics_by_layer['davies_bouldin'][layer_idx] if metrics_by_layer['davies_bouldin'][layer_idx] > 0 else 0,
                metrics_by_layer['dunn'][layer_idx],
                metrics_by_layer['bcss_wcss_ratio'][layer_idx],
                metrics_by_layer['explained_variance'][layer_idx]
            ]
            metrics_matrix.append(layer_metrics)
        
        metrics_matrix = np.array(metrics_matrix).T  # transpose for heatmap
        
        im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(metric_names)))
        ax.set_xticklabels([f'Layer {layer}' for layer in layers])
        ax.set_yticklabels(metric_names)
        ax.set_title('Metrics Heatmap (Normalized)')
        
        # 값 표시
        for i in range(len(metric_names)):
            for j in range(len(layers)):
                ax.text(j, i, f'{metrics_matrix[i, j]:.2f}', 
                       ha="center", va="center", color="white" if metrics_matrix[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=ax)
        
        # 5. 최고 성능 레이어 식별
        ax = axes[1, 1]
        ax.axis('off')
        
        # 최고 성능 레이어 찾기
        best_silhouette_layer = layers[np.argmax(metrics_by_layer['silhouette'])]
        best_ratio_layer = layers[np.argmax(metrics_by_layer['bcss_wcss_ratio'])]
        best_explained_layer = layers[np.argmax(metrics_by_layer['explained_variance'])]
        
        summary_text = f"Fixed k={k} Analysis Summary\n\n"
        summary_text += f"Layers analyzed: {len(layers)}\n"
        summary_text += f"Total samples: {all_results['analysis_info']['total_samples']}\n\n"
        
        summary_text += f"Best Silhouette: Layer {best_silhouette_layer}\n"
        summary_text += f"  Score: {max(metrics_by_layer['silhouette']):.4f}\n\n"
        
        summary_text += f"Best BCSS/WCSS: Layer {best_ratio_layer}\n"
        summary_text += f"  Ratio: {max(metrics_by_layer['bcss_wcss_ratio']):.3f}\n\n"
        
        summary_text += f"Best Explained Var: Layer {best_explained_layer}\n"
        summary_text += f"  Ratio: {max(metrics_by_layer['explained_variance']):.3f}\n\n"
        
        # 평균 성능
        avg_silhouette = np.mean(metrics_by_layer['silhouette'])
        avg_ratio = np.mean(metrics_by_layer['bcss_wcss_ratio'])
        
        summary_text += f"Average Performance:\n"
        summary_text += f"  Silhouette: {avg_silhouette:.3f}\n"
        summary_text += f"  BCSS/WCSS: {avg_ratio:.3f}\n\n"
        
        # 전체 품질 평가
        summary_text += f"Overall Quality for k={k}:\n"
        if avg_silhouette >= 0.5 and avg_ratio >= 1.0:
            summary_text += "EXCELLENT ⭐⭐⭐\n"
        elif avg_silhouette >= 0.3 and avg_ratio >= 0.5:
            summary_text += "GOOD ⭐⭐\n"
        elif avg_silhouette >= 0.25:
            summary_text += "FAIR ⭐\n"
        else:
            summary_text += "POOR\n"
        
        summary_text += f"\nRecommendation:\n"
        if best_silhouette_layer == best_ratio_layer:
            summary_text += f"Focus on Layer {best_silhouette_layer}\n"
            summary_text += f"(Consistent best performance)"
        else:
            summary_text += f"Layer {best_silhouette_layer}: Best silhouette\n"
            summary_text += f"Layer {best_ratio_layer}: Best separation"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # 6. 레이어 순위
        ax = axes[1, 2]
        
        # 종합 점수 계산 (실루엣 + 정규화된 BCSS/WCSS 비율)
        normalized_ratios = np.array(metrics_by_layer['bcss_wcss_ratio']) / max(metrics_by_layer['bcss_wcss_ratio'])
        composite_scores = np.array(metrics_by_layer['silhouette']) + normalized_ratios
        
        sorted_indices = np.argsort(composite_scores)[::-1]  # 내림차순
        ranked_layers = [layers[i] for i in sorted_indices]
        ranked_scores = [composite_scores[i] for i in sorted_indices]
        
        colors = ['gold', 'silver', '#CD7F32'] + ['lightblue'] * (len(layers) - 3)
        bars = ax.bar(range(len(ranked_layers)), ranked_scores, 
                     color=colors[:len(ranked_layers)], alpha=0.8)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Composite Score')
        ax.set_title('Layer Ranking (Silhouette + BCSS/WCSS)')
        ax.set_xticks(range(len(ranked_layers)))
        ax.set_xticklabels([f'L{layer}' for layer in ranked_layers])
        
        # 점수 표시
        for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 순위 표시
        for i, layer in enumerate(ranked_layers):
            rank_text = f"#{i+1}"
            if i == 0:
                rank_text += " 🥇"
            elif i == 1:
                rank_text += " 🥈"
            elif i == 2:
                rank_text += " 🥉"
            
            ax.text(i, ranked_scores[i] + 0.05, rank_text, 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle(f'Comprehensive Analysis Summary for k={k}', 
                    fontsize=18, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'k{k}_comprehensive_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _log_summary_results(self, all_results):
        """요약 결과 로그 출력"""
        layers = sorted([int(k.split('_')[1]) for k in all_results['layer_results'].keys()])
        k = all_results['analysis_info']['k_value']
        
        logger.info(f"\n" + "="*80)
        logger.info(f"🎯 FIXED k={k} CLUSTERING ANALYSIS SUMMARY")
        logger.info("="*80)
        
        # 레이어별 주요 메트릭 요약
        logger.info(f"\n📊 Layer-wise Performance Summary:")
        logger.info(f"{'Layer':<8} {'Silhouette':<12} {'CH Index':<12} {'DB Index':<12} {'BCSS/WCSS':<12}")
        logger.info("-" * 60)
        
        best_silhouette = 0
        best_silhouette_layer = 0
        
        for layer in layers:
            layer_key = f'layer_{layer}'
            metrics = all_results['layer_results'][layer_key]['metrics']
            
            silhouette = metrics['silhouette']['average_score']
            ch_score = metrics['calinski_harabasz']['score']
            db_score = metrics['davies_bouldin']['score']
            ratio = metrics['variance_analysis']['bcss_wcss_ratio']
            
            logger.info(f"{layer:<8} {silhouette:<12.4f} {ch_score:<12.2f} {db_score:<12.4f} {ratio:<12.3f}")
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_silhouette_layer = layer
        
        logger.info(f"\n🏆 Best performing layer: Layer {best_silhouette_layer} (Silhouette: {best_silhouette:.4f})")
        
        # 전체 통계
        all_silhouettes = [all_results['layer_results'][f'layer_{layer}']['metrics']['silhouette']['average_score'] 
                          for layer in layers]
        avg_silhouette = np.mean(all_silhouettes)
        std_silhouette = np.std(all_silhouettes)
        
        logger.info(f"\n📈 Overall Statistics:")
        logger.info(f"   Average Silhouette: {avg_silhouette:.4f} ± {std_silhouette:.4f}")
        logger.info(f"   Best Silhouette: {max(all_silhouettes):.4f}")
        logger.info(f"   Worst Silhouette: {min(all_silhouettes):.4f}")
        
        # 실루엣 검증 (기존 결과와 비교용)
        logger.info(f"\n✅ SILHOUETTE VERIFICATION for k={k}:")
        for layer in layers:
            sil_score = all_results['layer_results'][f'layer_{layer}']['metrics']['silhouette']['average_score']
            logger.info(f"   Layer {layer}: {sil_score:.4f}")
        
        # 품질 평가
        logger.info(f"\n🔍 Quality Assessment for k={k}:")
        if avg_silhouette >= 0.5:
            quality = "EXCELLENT ⭐⭐⭐"
        elif avg_silhouette >= 0.3:
            quality = "GOOD ⭐⭐"
        elif avg_silhouette >= 0.25:
            quality = "FAIR ⭐"
        else:
            quality = "POOR"
        
        logger.info(f"   Overall Quality: {quality}")
        logger.info(f"   k={k} appears to be {'suitable' if avg_silhouette >= 0.3 else 'questionable'} for this dataset")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Fixed K Clustering Metrics Analysis')
    parser.add_argument('--clustering_dir', type=str, required=True,
                       help='Path to clustering_{k} directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                       help='Which model to use (Full or Few)')
    
    args = parser.parse_args()
    
    # 경로 검증
    clustering_dir = Path(args.clustering_dir)
    if not clustering_dir.exists():
        raise ValueError(f"Clustering directory does not exist: {clustering_dir}")
    
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    # Analyzer 초기화
    analyzer = FixedKMetricsAnalyzer(args.clustering_dir, args.checkpoint_dir)
    
    logger.info(f"Starting fixed k={analyzer.k} analysis")
    logger.info(f"Clustering directory: {args.clustering_dir}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}")
    
    # 데이터로더 선택
    if args.mode == 'Full':
        data_loader = analyzer.train_loader
        logger.info("Using Full dataset loader")
    else:
        data_loader = analyzer.train_loader_few if hasattr(analyzer, 'train_loader_few') else analyzer.test_loader
        logger.info("Using Few-shot dataset loader")
    
    # 분석 수행
    results = analyzer.analyze_fixed_k_clustering(data_loader, mode=args.mode)
    
    logger.info(f"\n✅ Fixed k={analyzer.k} analysis completed!")
    logger.info(f"Results saved to: {analyzer.clustering_dir / 'metrics_analysis'}")
    
    # 추가 정보
    metrics_dir = analyzer.clustering_dir / 'metrics_analysis'
    logger.info(f"\n📁 Generated files:")
    logger.info(f"   📊 Summary: {metrics_dir}/k{analyzer.k}_comprehensive_summary.png")
    logger.info(f"   📋 Data: {metrics_dir}/k{analyzer.k}_comprehensive_metrics.json")
    logger.info(f"   📂 Layer details: {metrics_dir}/layer_*/")


if __name__ == "__main__":
    main()