"""
Comprehensive Clustering Metrics Analysis

기존 실루엣 분석과 동일한 결과를 보장하면서 
within/between variance 등 다양한 클러스터링 메트릭을 분석합니다.

Usage:
    python analysis3.py --checkpoint_dir /path/to/checkpoint.pt --mode Full
"""

import os
# CUDA deterministic 설정을 가장 먼저 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
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

class ComprehensiveClusteringAnalyzer:
    def __init__(self, checkpoint_dir, device='cuda'):
        """
        Args:
            checkpoint_dir (str): 체크포인트 파일 경로
            device (str): 'cuda' 또는 'cpu'
        """
        # Deterministic 설정 먼저
        ensure_deterministic()
        
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
        experiment_id = "comprehensive_analysis"
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
        데이터로더에서 attention maps 추출 (deterministic)
        
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

    def calculate_within_between_variance(self, data, cluster_labels):
        """
        Within-cluster Sum of Squares (WCSS)와 Between-cluster Sum of Squares (BCSS) 계산
        
        Args:
            data: 클러스터링 데이터 [n_samples, n_features]
            cluster_labels: 클러스터 할당 라벨 [n_samples]
            
        Returns:
            dict: WCSS, BCSS, Total SS, 비율 등의 메트릭
        """
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

    def calculate_comprehensive_metrics(self, data, cluster_labels):
        """
        모든 클러스터링 메트릭을 종합적으로 계산
        
        Args:
            data: 클러스터링 데이터
            cluster_labels: 클러스터 할당 라벨
            
        Returns:
            dict: 모든 메트릭 결과
        """
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
        
        # 6. Dunn Index (최소 inter-cluster distance / 최대 intra-cluster distance)
        dunn_score = self._calculate_dunn_index(data, cluster_labels)
        metrics['dunn_index'] = {
            'score': float(dunn_score),
            'interpretation': 'higher_is_better'
        }
        
        # 7. 클러스터별 통계
        cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
        metrics['cluster_statistics'] = cluster_stats
        
        return metrics

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

    def perform_comprehensive_analysis(self, attention_data, layer_idx, k_range=(2, 15), output_dir=None):
        """
        특정 레이어에 대해 종합적인 클러스터링 메트릭 분석 수행
        
        Args:
            attention_data (dict): attention maps 데이터
            layer_idx (int): 분석할 레이어 인덱스
            k_range (tuple): 클러스터 개수 범위 (min_k, max_k)
            output_dir (str, optional): 결과 저장 디렉토리
        
        Returns:
            dict: 종합 분석 결과
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 특정 레이어의 attention maps 추출
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        feature_names = attention_data['feature_names']
        
        logger.info(f"Performing comprehensive analysis on layer {layer_idx} with {len(attention_maps)} samples")
        logger.info(f"Testing cluster range: {k_range[0]} to {k_range[1]}")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        min_k, max_k = k_range
        k_values = list(range(min_k, max_k + 1))
        
        # 각 k값에 대해 모든 메트릭 계산
        comprehensive_results = {
            'layer_idx': layer_idx,
            'k_values': k_values,
            'feature_names': feature_names,
            'results_by_k': {}
        }
        
        for k in k_values:
            logger.info(f"Testing k={k}...")
            
            # Deterministic K-means
            np.random.seed(42)  # KMeans 내부 초기화를 위한 추가 시드
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='lloyd')
            cluster_labels = kmeans.fit_predict(flattened_maps)
            
            # 모든 메트릭 계산
            all_metrics = self.calculate_comprehensive_metrics(flattened_maps, cluster_labels)
            
            # 결과 저장
            comprehensive_results['results_by_k'][k] = {
                'cluster_assignments': cluster_labels.tolist(),
                'metrics': all_metrics,
                'kmeans_centers': kmeans.cluster_centers_.tolist()
            }
            
            # 주요 메트릭 로그 출력
            silhouette = all_metrics['silhouette']['average_score']
            ch_score = all_metrics['calinski_harabasz']['score']
            db_score = all_metrics['davies_bouldin']['score']
            wcss = all_metrics['variance_analysis']['wcss']
            bcss = all_metrics['variance_analysis']['bcss']
            
            logger.info(f"k={k}: Silhouette={silhouette:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}, WCSS={wcss:.2f}, BCSS={bcss:.2f}")
        
        # 최적 k 결정 (실루엣 기준으로 기존과 동일)
        silhouette_scores = [comprehensive_results['results_by_k'][k]['metrics']['silhouette']['average_score'] for k in k_values]
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_values[best_k_idx]
        best_score = silhouette_scores[best_k_idx]
        
        comprehensive_results['best_k_silhouette'] = best_k
        comprehensive_results['best_silhouette_score'] = best_score
        
        # 다른 메트릭 기준 최적 k도 계산
        ch_scores = [comprehensive_results['results_by_k'][k]['metrics']['calinski_harabasz']['score'] for k in k_values]
        db_scores = [comprehensive_results['results_by_k'][k]['metrics']['davies_bouldin']['score'] for k in k_values]
        
        comprehensive_results['best_k_calinski_harabasz'] = k_values[np.argmax(ch_scores)]
        comprehensive_results['best_k_davies_bouldin'] = k_values[np.argmin(db_scores)]
        
        logger.info(f"🎯 Best k for layer {layer_idx}:")
        logger.info(f"   Silhouette: k={comprehensive_results['best_k_silhouette']} (score: {best_score:.4f})")
        logger.info(f"   Calinski-Harabasz: k={comprehensive_results['best_k_calinski_harabasz']} (score: {max(ch_scores):.2f})")
        logger.info(f"   Davies-Bouldin: k={comprehensive_results['best_k_davies_bouldin']} (score: {min(db_scores):.4f})")
        
        # 시각화 생성
        if output_dir:
            self._create_comprehensive_visualizations(comprehensive_results, flattened_maps, labels, output_dir)
            
            # 결과 JSON 저장
            results_json = comprehensive_results.copy()
            results_json['feature_names'] = list(feature_names)  # numpy array를 list로 변환
            
            with open(output_dir / f'layer_{layer_idx}_comprehensive_results.json', 'w') as f:
                json.dump(results_json, f, indent=2)
            
            logger.info(f"✅ Comprehensive analysis results saved to {output_dir}")
        
        return comprehensive_results

    def _create_comprehensive_visualizations(self, results, flattened_maps, labels, output_dir):
        """종합적인 시각화 생성"""
        layer_idx = results['layer_idx']
        k_values = results['k_values']
        
        # 1. 모든 메트릭 비교 플롯
        self._plot_all_metrics_comparison(results, output_dir)
        
        # 2. Within/Between Variance 상세 분석
        self._plot_variance_analysis(results, output_dir)
        
        # 3. 실루엣 분석 상세 (기존과 동일한 스타일)
        self._plot_detailed_silhouette_analysis(results, flattened_maps, output_dir)
        
        # 4. 최적 k에 대한 클러스터 분포 시각화
        self._plot_optimal_clustering_comparison(results, flattened_maps, labels, output_dir)

    def _plot_all_metrics_comparison(self, results, output_dir):
        """모든 메트릭 비교 플롯"""
        k_values = results['k_values']
        layer_idx = results['layer_idx']
        
        # 메트릭 데이터 추출
        silhouette_scores = [results['results_by_k'][k]['metrics']['silhouette']['average_score'] for k in k_values]
        ch_scores = [results['results_by_k'][k]['metrics']['calinski_harabasz']['score'] for k in k_values]
        db_scores = [results['results_by_k'][k]['metrics']['davies_bouldin']['score'] for k in k_values]
        wcss_scores = [results['results_by_k'][k]['metrics']['variance_analysis']['wcss'] for k in k_values]
        bcss_scores = [results['results_by_k'][k]['metrics']['variance_analysis']['bcss'] for k in k_values]
        dunn_scores = [results['results_by_k'][k]['metrics']['dunn_index']['score'] for k in k_values]
        
        # 2x3 서브플롯
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Silhouette Score
        ax = axes[0, 0]
        ax.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        best_k_sil = results['best_k_silhouette']
        best_idx = k_values.index(best_k_sil)
        ax.plot(best_k_sil, silhouette_scores[best_idx], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Silhouette Score')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7)
        
        # 2. Calinski-Harabasz Index
        ax = axes[0, 1]
        ax.plot(k_values, ch_scores, 'go-', linewidth=2, markersize=8)
        best_k_ch = results['best_k_calinski_harabasz']
        best_idx_ch = k_values.index(best_k_ch)
        ax.plot(best_k_ch, ch_scores[best_idx_ch], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Calinski-Harabasz Index')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('CH Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7)
        
        # 3. Davies-Bouldin Index
        ax = axes[0, 2]
        ax.plot(k_values, db_scores, 'mo-', linewidth=2, markersize=8)
        best_k_db = results['best_k_davies_bouldin']
        best_idx_db = k_values.index(best_k_db)
        ax.plot(best_k_db, db_scores[best_idx_db], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Davies-Bouldin Index (Lower is Better)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('DB Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_db, color='red', linestyle='--', alpha=0.7)
        
        # 4. WCSS (Within-cluster Sum of Squares)
        ax = axes[1, 0]
        ax.plot(k_values, wcss_scores, 'co-', linewidth=2, markersize=8)
        ax.set_title('Within-Cluster Sum of Squares (WCSS)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        ax.grid(True, alpha=0.3)
        # WCSS는 항상 감소하므로 특정 최적점 표시 안함
        
        # 5. BCSS (Between-cluster Sum of Squares)
        ax = axes[1, 1]
        ax.plot(k_values, bcss_scores, 'yo-', linewidth=2, markersize=8)
        ax.set_title('Between-Cluster Sum of Squares (BCSS)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('BCSS')
        ax.grid(True, alpha=0.3)
        
        # 6. Dunn Index
        ax = axes[1, 2]
        ax.plot(k_values, dunn_scores, 'ko-', linewidth=2, markersize=8)
        ax.set_title('Dunn Index (Higher is Better)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Dunn Index')
        ax.grid(True, alpha=0.3)
        
        # 전체 제목
        plt.suptitle(f'Layer {layer_idx}: Comprehensive Clustering Metrics Comparison', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_all_metrics_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_variance_analysis(self, results, output_dir):
        """Within/Between Variance 상세 분석 플롯"""
        k_values = results['k_values']
        layer_idx = results['layer_idx']
        
        # 분산 관련 데이터 추출
        wcss_scores = [results['results_by_k'][k]['metrics']['variance_analysis']['wcss'] for k in k_values]
        bcss_scores = [results['results_by_k'][k]['metrics']['variance_analysis']['bcss'] for k in k_values]
        tss_scores = [results['results_by_k'][k]['metrics']['variance_analysis']['tss'] for k in k_values]
        bcss_wcss_ratios = [results['results_by_k'][k]['metrics']['variance_analysis']['bcss_wcss_ratio'] for k in k_values]
        explained_variance_ratios = [results['results_by_k'][k]['metrics']['variance_analysis']['explained_variance_ratio'] for k in k_values]
        
        # 2x2 서브플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. WCSS vs BCSS
        ax = axes[0, 0]
        ax.plot(k_values, wcss_scores, 'b-', label='WCSS (Within)', linewidth=2, marker='o')
        ax.plot(k_values, bcss_scores, 'r-', label='BCSS (Between)', linewidth=2, marker='s')
        ax.plot(k_values, tss_scores, 'g--', label='TSS (Total)', linewidth=2, marker='^')
        ax.set_title('Variance Decomposition: WCSS vs BCSS vs TSS')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Sum of Squares')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. BCSS/WCSS 비율
        ax = axes[0, 1]
        ax.plot(k_values, bcss_wcss_ratios, 'mo-', linewidth=2, markersize=8)
        ax.set_title('BCSS/WCSS Ratio (Higher is Better)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('BCSS/WCSS Ratio')
        ax.grid(True, alpha=0.3)
        
        # 최대 비율 지점 표시
        max_ratio_idx = np.argmax(bcss_wcss_ratios)
        max_ratio_k = k_values[max_ratio_idx]
        ax.plot(max_ratio_k, bcss_wcss_ratios[max_ratio_idx], 'ro', markersize=12)
        ax.axvline(x=max_ratio_k, color='red', linestyle='--', alpha=0.7)
        
        # 3. Explained Variance Ratio
        ax = axes[1, 0]
        ax.plot(k_values, explained_variance_ratios, 'co-', linewidth=2, markersize=8)
        ax.set_title('Explained Variance Ratio (BCSS/TSS)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Explained Variance Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 4. Variance 통계 요약
        ax = axes[1, 1]
        ax.axis('off')
        
        # 통계 텍스트 생성
        best_k_sil = results['best_k_silhouette']
        best_k_idx = k_values.index(best_k_sil)
        
        variance_stats = results['results_by_k'][best_k_sil]['metrics']['variance_analysis']
        
        stats_text = f"Variance Analysis Summary (Best k={best_k_sil})\n\n"
        stats_text += f"WCSS (Within-cluster): {variance_stats['wcss']:.2f}\n"
        stats_text += f"BCSS (Between-cluster): {variance_stats['bcss']:.2f}\n"
        stats_text += f"TSS (Total): {variance_stats['tss']:.2f}\n"
        stats_text += f"BCSS/WCSS Ratio: {variance_stats['bcss_wcss_ratio']:.3f}\n"
        stats_text += f"Explained Variance: {variance_stats['explained_variance_ratio']:.3f}\n\n"
        
        # 다른 k값들과 비교
        max_ratio_k = k_values[np.argmax(bcss_wcss_ratios)]
        max_explained_k = k_values[np.argmax(explained_variance_ratios)]
        
        stats_text += f"Best BCSS/WCSS Ratio: k={max_ratio_k} ({max(bcss_wcss_ratios):.3f})\n"
        stats_text += f"Best Explained Variance: k={max_explained_k} ({max(explained_variance_ratios):.3f})\n\n"
        
        # 해석
        stats_text += "Interpretation:\n"
        stats_text += "• Higher BCSS/WCSS = Better separation\n"
        stats_text += "• Higher Explained Variance = Better clustering\n"
        stats_text += "• WCSS ↓, BCSS ↑ = Ideal clustering"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx}: Within/Between Variance Analysis', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_variance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    # 기존 ComprehensiveClusteringAnalyzer 클래스에 추가할 메서드들

    import numpy as np
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    def calculate_elbow_point(self, k_values, wcss_scores, method='knee'):
        """
        WCSS 값에서 elbow point를 자동으로 찾는 함수
        
        Args:
            k_values: 클러스터 개수 리스트
            wcss_scores: 각 k에 대한 WCSS 값
            method: 'knee', 'derivative', 'variance' 중 선택
        
        Returns:
            dict: elbow point 정보
        """
        k_values = np.array(k_values)
        wcss_scores = np.array(wcss_scores)
        
        if method == 'knee':
            # Knee detection using the "knee/elbow" method
            # 첫 번째 점과 마지막 점을 잇는 직선으로부터의 거리가 최대인 점 찾기
            
            # Normalize data to [0,1] for better knee detection
            k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
            wcss_norm = (wcss_scores - wcss_scores.min()) / (wcss_scores.max() - wcss_scores.min())
            
            # 첫 점과 마지점을 잇는 직선
            line_start = np.array([k_norm[0], wcss_norm[0]])
            line_end = np.array([k_norm[-1], wcss_norm[-1]])
            
            # 각 점에서 직선까지의 거리 계산
            distances = []
            for i in range(len(k_norm)):
                point = np.array([k_norm[i], wcss_norm[i]])
                # 점에서 직선까지의 거리 공식
                distance = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
                distances.append(distance)
            
            elbow_idx = np.argmax(distances)
            elbow_k = k_values[elbow_idx]
            elbow_score = wcss_scores[elbow_idx]
            max_distance = distances[elbow_idx]
            
            return {
                'method': 'knee',
                'elbow_k': int(elbow_k),
                'elbow_wcss': float(elbow_score),
                'confidence': float(max_distance),
                'all_distances': distances
            }
        
        elif method == 'derivative':
            # Second derivative method
            if len(wcss_scores) < 3:
                return {'method': 'derivative', 'elbow_k': k_values[0], 'elbow_wcss': wcss_scores[0], 'confidence': 0.0}
            
            # 1차 미분 (기울기)
            first_derivative = np.diff(wcss_scores)
            # 2차 미분 (기울기의 변화)
            second_derivative = np.diff(first_derivative)
            
            # 2차 미분이 최대인 지점 (가장 급격하게 기울기가 변하는 지점)
            elbow_idx = np.argmax(second_derivative) + 1  # +1 because of diff operations
            elbow_k = k_values[elbow_idx]
            elbow_score = wcss_scores[elbow_idx]
            
            return {
                'method': 'derivative',
                'elbow_k': int(elbow_k),
                'elbow_wcss': float(elbow_score),
                'confidence': float(second_derivative[elbow_idx-1]),
                'first_derivative': first_derivative.tolist(),
                'second_derivative': second_derivative.tolist()
            }
        
        elif method == 'variance':
            # Variance-based method: 기울기의 분산이 줄어드는 지점
            if len(wcss_scores) < 4:
                return {'method': 'variance', 'elbow_k': k_values[0], 'elbow_wcss': wcss_scores[0], 'confidence': 0.0}
            
            # 슬라이딩 윈도우로 기울기의 분산 계산
            window_size = min(3, len(wcss_scores) // 2)
            variances = []
            
            for i in range(window_size, len(wcss_scores) - window_size):
                window_slopes = []
                for j in range(i - window_size, i + window_size):
                    if j < len(wcss_scores) - 1:
                        slope = (wcss_scores[j+1] - wcss_scores[j]) / (k_values[j+1] - k_values[j])
                        window_slopes.append(slope)
                variances.append(np.var(window_slopes))
            
            # 분산이 최소인 지점 (기울기가 안정화되는 지점)
            elbow_idx = np.argmin(variances) + window_size
            elbow_k = k_values[elbow_idx]
            elbow_score = wcss_scores[elbow_idx]
            
            return {
                'method': 'variance',
                'elbow_k': int(elbow_k),
                'elbow_wcss': float(elbow_score),
                'confidence': float(1.0 / (variances[elbow_idx - window_size] + 1e-10)),
                'variances': variances
            }

    def perform_comprehensive_analysis_with_elbow(self, attention_data, layer_idx, k_range=(2, 15), output_dir=None):
        """
        Elbow method가 포함된 종합적인 클러스터링 메트릭 분석 수행
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 특정 레이어의 attention maps 추출
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        feature_names = attention_data['feature_names']
        
        logger.info(f"Performing comprehensive analysis (with Elbow) on layer {layer_idx} with {len(attention_maps)} samples")
        logger.info(f"Testing cluster range: {k_range[0]} to {k_range[1]}")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        min_k, max_k = k_range
        k_values = list(range(min_k, max_k + 1))
        
        # 각 k값에 대해 모든 메트릭 계산
        comprehensive_results = {
            'layer_idx': layer_idx,
            'k_values': k_values,
            'feature_names': feature_names,
            'results_by_k': {}
        }
        
        wcss_scores = []
        
        for k in k_values:
            logger.info(f"Testing k={k}...")
            
            # Deterministic K-means
            np.random.seed(42)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='lloyd')
            cluster_labels = kmeans.fit_predict(flattened_maps)
            
            # 모든 메트릭 계산
            all_metrics = self.calculate_comprehensive_metrics(flattened_maps, cluster_labels)
            
            # WCSS 저장
            wcss_scores.append(all_metrics['variance_analysis']['wcss'])
            
            # 결과 저장
            comprehensive_results['results_by_k'][k] = {
                'cluster_assignments': cluster_labels.tolist(),
                'metrics': all_metrics,
                'kmeans_centers': kmeans.cluster_centers_.tolist()
            }
            
            # 주요 메트릭 로그 출력
            silhouette = all_metrics['silhouette']['average_score']
            ch_score = all_metrics['calinski_harabasz']['score']
            db_score = all_metrics['davies_bouldin']['score']
            wcss = all_metrics['variance_analysis']['wcss']
            bcss = all_metrics['variance_analysis']['bcss']
            
            logger.info(f"k={k}: Silhouette={silhouette:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}, WCSS={wcss:.2f}, BCSS={bcss:.2f}")
        
        # Elbow method 적용
        elbow_knee = self.calculate_elbow_point(k_values, wcss_scores, method='knee')
        elbow_derivative = self.calculate_elbow_point(k_values, wcss_scores, method='derivative')
        elbow_variance = self.calculate_elbow_point(k_values, wcss_scores, method='variance')
        
        comprehensive_results['elbow_analysis'] = {
            'knee_method': elbow_knee,
            'derivative_method': elbow_derivative,
            'variance_method': elbow_variance,
            'wcss_scores': wcss_scores
        }
        
        # 기존 메트릭 기준 최적 k
        silhouette_scores = [comprehensive_results['results_by_k'][k]['metrics']['silhouette']['average_score'] for k in k_values]
        ch_scores = [comprehensive_results['results_by_k'][k]['metrics']['calinski_harabasz']['score'] for k in k_values]
        db_scores = [comprehensive_results['results_by_k'][k]['metrics']['davies_bouldin']['score'] for k in k_values]
        
        comprehensive_results['best_k_silhouette'] = k_values[np.argmax(silhouette_scores)]
        comprehensive_results['best_silhouette_score'] = max(silhouette_scores)
        comprehensive_results['best_k_calinski_harabasz'] = k_values[np.argmax(ch_scores)]
        comprehensive_results['best_k_davies_bouldin'] = k_values[np.argmin(db_scores)]
        
        # Elbow method 기준 최적 k (가장 신뢰도 높은 방법 선택)
        elbow_methods = [elbow_knee, elbow_derivative, elbow_variance]
        best_elbow = max(elbow_methods, key=lambda x: x['confidence'])
        comprehensive_results['best_k_elbow'] = best_elbow['elbow_k']
        comprehensive_results['best_elbow_method'] = best_elbow['method']
        
        logger.info(f"🎯 Best k for layer {layer_idx}:")
        logger.info(f"   Silhouette: k={comprehensive_results['best_k_silhouette']} (score: {max(silhouette_scores):.4f})")
        logger.info(f"   Calinski-Harabasz: k={comprehensive_results['best_k_calinski_harabasz']} (score: {max(ch_scores):.2f})")
        logger.info(f"   Davies-Bouldin: k={comprehensive_results['best_k_davies_bouldin']} (score: {min(db_scores):.4f})")
        logger.info(f"   🔥 Elbow Method: k={comprehensive_results['best_k_elbow']} (method: {comprehensive_results['best_elbow_method']})")
        
        # 시각화 생성
        if output_dir:
            self._create_comprehensive_visualizations_with_elbow(comprehensive_results, flattened_maps, labels, output_dir)
            
            # 결과 JSON 저장
            results_json = comprehensive_results.copy()
            results_json['feature_names'] = list(feature_names)
            
            with open(output_dir / f'layer_{layer_idx}_comprehensive_results_with_elbow.json', 'w') as f:
                json.dump(results_json, f, indent=2)
            
            logger.info(f"✅ Comprehensive analysis (with Elbow) results saved to {output_dir}")
        
        return comprehensive_results

    def _create_comprehensive_visualizations_with_elbow(self, results, flattened_maps, labels, output_dir):
        """Elbow method가 포함된 종합적인 시각화 생성"""
        layer_idx = results['layer_idx']
        
        # 1. 모든 메트릭 + Elbow 비교 플롯
        self._plot_all_metrics_with_elbow_comparison(results, output_dir)
        
        # 2. Elbow method 상세 분석
        self._plot_detailed_elbow_analysis(results, output_dir)
        
        # 3. Within/Between Variance 상세 분석 (기존)
        self._plot_variance_analysis(results, output_dir)
        
        # 4. 모든 최적 k 비교 클러스터링 시각화
        self._plot_all_optimal_clustering_comparison(results, flattened_maps, labels, output_dir)

    def _plot_all_metrics_with_elbow_comparison(self, results, output_dir):
        """모든 메트릭과 Elbow method 비교 플롯"""
        k_values = results['k_values']
        layer_idx = results['layer_idx']
        
        # 메트릭 데이터 추출
        silhouette_scores = [results['results_by_k'][k]['metrics']['silhouette']['average_score'] for k in k_values]
        ch_scores = [results['results_by_k'][k]['metrics']['calinski_harabasz']['score'] for k in k_values]
        db_scores = [results['results_by_k'][k]['metrics']['davies_bouldin']['score'] for k in k_values]
        wcss_scores = results['elbow_analysis']['wcss_scores']
        
        # Elbow 결과
        elbow_knee = results['elbow_analysis']['knee_method']
        elbow_derivative = results['elbow_analysis']['derivative_method']
        elbow_variance = results['elbow_analysis']['variance_method']
        
        # 2x3 서브플롯
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Silhouette Score
        ax = axes[0, 0]
        ax.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        best_k_sil = results['best_k_silhouette']
        best_idx = k_values.index(best_k_sil)
        ax.plot(best_k_sil, silhouette_scores[best_idx], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Silhouette Score')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_sil}')
        ax.legend()
        
        # 2. Calinski-Harabasz Index
        ax = axes[0, 1]
        ax.plot(k_values, ch_scores, 'go-', linewidth=2, markersize=8)
        best_k_ch = results['best_k_calinski_harabasz']
        best_idx_ch = k_values.index(best_k_ch)
        ax.plot(best_k_ch, ch_scores[best_idx_ch], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Calinski-Harabasz Index')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('CH Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_ch}')
        ax.legend()
        
        # 3. Davies-Bouldin Index
        ax = axes[0, 2]
        ax.plot(k_values, db_scores, 'mo-', linewidth=2, markersize=8)
        best_k_db = results['best_k_davies_bouldin']
        best_idx_db = k_values.index(best_k_db)
        ax.plot(best_k_db, db_scores[best_idx_db], 'ro', markersize=12, markerfacecolor='red')
        ax.set_title('Davies-Bouldin Index (Lower is Better)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('DB Score')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_k_db, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_db}')
        ax.legend()
        
        # 4. WCSS with Elbow Points
        ax = axes[1, 0]
        ax.plot(k_values, wcss_scores, 'co-', linewidth=2, markersize=8, label='WCSS')
        
        # Elbow points 표시
        ax.axvline(x=elbow_knee['elbow_k'], color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Knee: k={elbow_knee["elbow_k"]}')
        ax.axvline(x=elbow_derivative['elbow_k'], color='orange', linestyle='--', alpha=0.8, linewidth=2, label=f'Derivative: k={elbow_derivative["elbow_k"]}')
        ax.axvline(x=elbow_variance['elbow_k'], color='purple', linestyle=':', alpha=0.8, linewidth=2, label=f'Variance: k={elbow_variance["elbow_k"]}')
        
        # Best elbow 강조
        best_elbow_k = results['best_k_elbow']
        best_elbow_idx = k_values.index(best_elbow_k)
        ax.plot(best_elbow_k, wcss_scores[best_elbow_idx], 'rs', markersize=15, markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_title('WCSS with Elbow Points')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 모든 최적 k 비교
        ax = axes[1, 1]
        optimal_ks = {
            'Silhouette': results['best_k_silhouette'],
            'Calinski-H': results['best_k_calinski_harabasz'],
            'Davies-B': results['best_k_davies_bouldin'],
            'Elbow': results['best_k_elbow']
        }
        
        methods = list(optimal_ks.keys())
        k_vals = list(optimal_ks.values())
        colors = ['blue', 'green', 'magenta', 'red']
        
        bars = ax.bar(methods, k_vals, color=colors, alpha=0.7)
        for i, (method, k_val) in enumerate(zip(methods, k_vals)):
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + 0.1,
                f'k={k_val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Optimal k by Different Methods')
        ax.set_xlabel('Method')
        ax.set_ylabel('Optimal k')
        ax.grid(True, alpha=0.3)
        
        # 6. Elbow Confidence Scores
        ax = axes[1, 2]
        elbow_methods = ['Knee', 'Derivative', 'Variance']
        elbow_confidences = [elbow_knee['confidence'], elbow_derivative['confidence'], elbow_variance['confidence']]
        elbow_k_vals = [elbow_knee['elbow_k'], elbow_derivative['elbow_k'], elbow_variance['elbow_k']]
        
        bars = ax.bar(elbow_methods, elbow_confidences, color=['red', 'orange', 'purple'], alpha=0.7)
        for i, (method, conf, k_val) in enumerate(zip(elbow_methods, elbow_confidences, elbow_k_vals)):
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + max(elbow_confidences)*0.02,
                f'k={k_val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Elbow Method Confidence Scores')
        ax.set_xlabel('Elbow Method')
        ax.set_ylabel('Confidence Score')
        ax.grid(True, alpha=0.3)
        
        # 최고 신뢰도 방법 강조
        best_method_idx = np.argmax(elbow_confidences)
        bars[best_method_idx].set_edgecolor('black')
        bars[best_method_idx].set_linewidth(3)
        
        plt.suptitle(f'Layer {layer_idx}: Comprehensive Metrics + Elbow Method Comparison', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_all_metrics_with_elbow.png', 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_detailed_elbow_analysis(self, results, output_dir):
        """Elbow method 상세 분석 플롯"""
        k_values = results['k_values']
        layer_idx = results['layer_idx']
        wcss_scores = results['elbow_analysis']['wcss_scores']
        
        elbow_knee = results['elbow_analysis']['knee_method']
        elbow_derivative = results['elbow_analysis']['derivative_method']
        elbow_variance = results['elbow_analysis']['variance_method']
        
        # 2x2 서브플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Knee Method Visualization
        ax = axes[0, 0]
        ax.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        
        # Knee point 강조
        knee_k = elbow_knee['elbow_k']
        knee_idx = k_values.index(knee_k)
        ax.plot(knee_k, wcss_scores[knee_idx], 'rs', markersize=15, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=2, label=f'Knee Point (k={knee_k})')
        
        # 직선 그리기 (첫 점과 마지막 점)
        ax.plot([k_values[0], k_values[-1]], [wcss_scores[0], wcss_scores[-1]], 
                'r--', alpha=0.5, linewidth=2, label='Reference Line')
        
        # 거리 시각화 (knee point에서)
        if 'all_distances' in elbow_knee:
            distances = elbow_knee['all_distances']
            ax2 = ax.twinx()
            ax2.plot(k_values, distances, 'g^-', alpha=0.6, markersize=6, label='Distance to Line')
            ax2.set_ylabel('Distance to Reference Line', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        
        ax.set_title('Knee Method Analysis')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. Derivative Method Visualization
        ax = axes[0, 1]
        ax.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        
        # Derivative elbow point
        deriv_k = elbow_derivative['elbow_k']
        deriv_idx = k_values.index(deriv_k)
        ax.plot(deriv_k, wcss_scores[deriv_idx], 'rs', markersize=15, markerfacecolor='orange', 
                markeredgecolor='darkorange', markeredgewidth=2, label=f'Derivative Elbow (k={deriv_k})')
        
        # 1차, 2차 미분 시각화
        if 'first_derivative' in elbow_derivative and 'second_derivative' in elbow_derivative:
            first_deriv = elbow_derivative['first_derivative']
            second_deriv = elbow_derivative['second_derivative']
            
            ax2 = ax.twinx()
            k_first = k_values[1:]  # 1차 미분은 하나 적음
            k_second = k_values[2:]  # 2차 미분은 둘 적음
            
            ax2.plot(k_first, first_deriv, 'g^-', alpha=0.6, markersize=4, label='1st Derivative')
            ax2.plot(k_second, second_deriv, 'mv-', alpha=0.6, markersize=4, label='2nd Derivative')
            ax2.set_ylabel('Derivative Values', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='lower right')
        
        ax.set_title('Derivative Method Analysis')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 3. Variance Method Visualization
        ax = axes[1, 0]
        ax.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        
        # Variance elbow point
        var_k = elbow_variance['elbow_k']
        var_idx = k_values.index(var_k)
        ax.plot(var_k, wcss_scores[var_idx], 'rs', markersize=15, markerfacecolor='purple', 
                markeredgecolor='indigo', markeredgewidth=2, label=f'Variance Elbow (k={var_k})')
        
        # 분산 시각화
        if 'variances' in elbow_variance:
            variances = elbow_variance['variances']
            window_size = 3  # 기본값 (실제 계산에서 사용된 값과 맞춰야 함)
            k_variance = k_values[window_size:-window_size]
            
            ax2 = ax.twinx()
            ax2.plot(k_variance, variances, 'g^-', alpha=0.6, markersize=6, label='Slope Variance')
            ax2.set_ylabel('Slope Variance', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='lower right')
        
        ax.set_title('Variance Method Analysis')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. Method Comparison Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # 요약 텍스트
        summary_text = f"Elbow Method Analysis Summary\n\n"
        summary_text += f"Knee Method:\n"
        summary_text += f"  Optimal k: {elbow_knee['elbow_k']}\n"
        summary_text += f"  Confidence: {elbow_knee['confidence']:.4f}\n"
        summary_text += f"  WCSS: {elbow_knee['elbow_wcss']:.2f}\n\n"
        
        summary_text += f"Derivative Method:\n"
        summary_text += f"  Optimal k: {elbow_derivative['elbow_k']}\n"
        summary_text += f"  Confidence: {elbow_derivative['confidence']:.4f}\n"
        summary_text += f"  WCSS: {elbow_derivative['elbow_wcss']:.2f}\n\n"
        
        summary_text += f"Variance Method:\n"
        summary_text += f"  Optimal k: {elbow_variance['elbow_k']}\n"
        summary_text += f"  Confidence: {elbow_variance['confidence']:.4f}\n"
        summary_text += f"  WCSS: {elbow_variance['elbow_wcss']:.2f}\n\n"
        
        # 최고 방법
        best_method = results['best_elbow_method']
        best_k = results['best_k_elbow']
        summary_text += f"🏆 BEST METHOD: {best_method.upper()}\n"
        summary_text += f"🎯 RECOMMENDED k: {best_k}\n\n"
        
        summary_text += f"Method Interpretations:\n"
        summary_text += f"• Knee: Maximum distance from reference line\n"
        summary_text += f"• Derivative: Maximum curvature change\n"
        summary_text += f"• Variance: Slope stabilization point\n"
        summary_text += f"• Higher confidence = more reliable"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx}: Detailed Elbow Method Analysis', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_detailed_elbow_analysis.png', 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_all_optimal_clustering_comparison(self, results, flattened_maps, labels, output_dir):
        """모든 최적 k (Silhouette, CH, DB, Elbow)에 대한 클러스터 분포 시각화"""
        layer_idx = results['layer_idx']
        best_k_sil = results['best_k_silhouette']
        best_k_ch = results['best_k_calinski_harabasz']
        best_k_db = results['best_k_davies_bouldin']
        best_k_elbow = results['best_k_elbow']
        
        # t-SNE 차원 축소
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 2x3 서브플롯 (Elbow 추가)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Silhouette 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_sil, 
            f'Silhouette Optimal (k={best_k_sil})', axes[0, 0]
        )
        
        # 2. Calinski-Harabasz 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_ch,
            f'Calinski-Harabasz Optimal (k={best_k_ch})', axes[0, 1]
        )
        
        # 3. Davies-Bouldin 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_db,
            f'Davies-Bouldin Optimal (k={best_k_db})', axes[0, 2]
        )
        
        # 4. Elbow Method 최적 k (NEW!)
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_elbow,
            f'Elbow Method Optimal (k={best_k_elbow})', axes[1, 0]
        )
        
        # 5. 실제 라벨
        ax = axes[1, 1]
        unique_labels = np.unique(labels)
        label_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            marker = 'o' if label == 0 else 's'
            ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                    c=[label_colors[i]], label=f'True Label {int(label)}', 
                    alpha=0.7, s=50, marker=marker)
        
        ax.set_title('True Labels')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 메트릭 비교 요약
        ax = axes[1, 2]
        ax.axis('off')
        
        # 모든 최적 k 비교
        methods_k = {
            'Silhouette': best_k_sil,
            'Calinski-H': best_k_ch,
            'Davies-B': best_k_db,
            'Elbow': best_k_elbow
        }
        
        # 각 k값에 대한 모든 메트릭 스코어 계산
        comparison_text = f"Optimal k Comparison\n\n"
        
        for method, k_val in methods_k.items():
            k_metrics = results['results_by_k'][k_val]['metrics']
            sil_score = k_metrics['silhouette']['average_score']
            ch_score = k_metrics['calinski_harabasz']['score']
            db_score = k_metrics['davies_bouldin']['score']
            wcss = k_metrics['variance_analysis']['wcss']
            
            comparison_text += f"{method} (k={k_val}):\n"
            comparison_text += f"  Silhouette: {sil_score:.3f}\n"
            comparison_text += f"  Calinski-H: {ch_score:.1f}\n"
            comparison_text += f"  Davies-B: {db_score:.3f}\n"
            comparison_text += f"  WCSS: {wcss:.1f}\n\n"
        
        # 메트릭 일치도
        unique_ks = len(set(methods_k.values()))
        comparison_text += f"Method Agreement:\n"
        comparison_text += f"  Unique k values: {unique_ks}/4\n"
        
        if unique_ks == 1:
            comparison_text += f"  🎯 Perfect agreement!\n"
        elif unique_ks == 2:
            comparison_text += f"  ✅ Good agreement\n"
        elif unique_ks == 3:
            comparison_text += f"  ⚠️ Mixed results\n"
        else:
            comparison_text += f"  ❌ No agreement\n"
        
        # 추천사항
        comparison_text += f"\nRecommendation:\n"
        most_common_k = max(set(methods_k.values()), key=list(methods_k.values()).count)
        comparison_text += f"Most common k: {most_common_k}\n"
        comparison_text += f"Elbow suggests: k={best_k_elbow}\n"
        
        # Elbow vs 다른 메트릭 비교
        if best_k_elbow == best_k_sil:
            comparison_text += f"✅ Elbow agrees with Silhouette"
        else:
            comparison_text += f"⚠️ Elbow differs from Silhouette"
        
        ax.text(0.1, 0.9, comparison_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle(f'Layer {layer_idx}: All Optimal Clustering Methods Comparison', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_all_optimal_clustering_comparison.png', 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _create_cross_layer_comparison_with_elbow(self, metrics_comparison, all_results, output_dir):
        """Elbow method가 포함된 레이어간 메트릭 비교 시각화"""
        layers = metrics_comparison['layers']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 메트릭별 최적 k 비교 (Elbow 추가)
        ax = axes[0, 0]
        sil_ks = [metrics_comparison['silhouette'][layer]['best_k'] for layer in layers]
        ch_ks = [metrics_comparison['calinski_harabasz'][layer]['best_k'] for layer in layers]
        db_ks = [metrics_comparison['davies_bouldin'][layer]['best_k'] for layer in layers]
        elbow_ks = [metrics_comparison['elbow'][layer]['best_k'] for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.2
        
        ax.bar(x - 1.5*width, sil_ks, width, label='Silhouette', alpha=0.8)
        ax.bar(x - 0.5*width, ch_ks, width, label='Calinski-Harabasz', alpha=0.8)
        ax.bar(x + 0.5*width, db_ks, width, label='Davies-Bouldin', alpha=0.8)
        ax.bar(x + 1.5*width, elbow_ks, width, label='Elbow', alpha=0.8, color='red')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Optimal k')
        ax.set_title('Optimal k by Different Metrics (Including Elbow)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {layer}' for layer in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Silhouette vs Elbow 상관관계
        ax = axes[0, 1]
        ax.scatter(sil_ks, elbow_ks, s=100, alpha=0.7, c=layers, cmap='viridis')
        for i, layer in enumerate(layers):
            ax.annotate(f'L{layer}', (sil_ks[i], elbow_ks[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Silhouette Optimal k')
        ax.set_ylabel('Elbow Optimal k')
        ax.set_title('Optimal k Correlation: Silhouette vs Elbow')
        ax.grid(True, alpha=0.3)
        
        # 대각선 참조선
        min_k = min(min(sil_ks), min(elbow_ks))
        max_k = max(max(sil_ks), max(elbow_ks))
        ax.plot([min_k, max_k], [min_k, max_k], 'r--', alpha=0.5, label='Perfect Agreement')
        ax.legend()
        
        # 3. Elbow Method Performance by Layer
        ax = axes[0, 2]
        elbow_confidences = [metrics_comparison['elbow'][layer]['confidence'] for layer in layers]
        elbow_methods = [metrics_comparison['elbow'][layer]['method'] for layer in layers]
        
        bars = ax.bar(layers, elbow_confidences, alpha=0.7, color='red')
        for i, (layer, conf, method, k_val) in enumerate(zip(layers, elbow_confidences, elbow_methods, elbow_ks)):
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + max(elbow_confidences)*0.02,
                f'{method}\nk={k_val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Elbow Confidence Score')
        ax.set_title('Elbow Method Confidence by Layer')
        ax.grid(True, alpha=0.3)
        
        # 4. Method Agreement Analysis
        ax = axes[1, 0]
        
        # 각 레이어별 메트릭 일치도 계산
        agreement_scores = []
        for layer in layers:
            layer_ks = [sil_ks[layers.index(layer)], ch_ks[layers.index(layer)], 
                    db_ks[layers.index(layer)], elbow_ks[layers.index(layer)]]
            unique_ks = len(set(layer_ks))
            agreement_score = (4 - unique_ks + 1) / 4  # 1.0 = perfect agreement, 0.25 = no agreement
            agreement_scores.append(agreement_score)
        
        bars = ax.bar(layers, agreement_scores, alpha=0.7, color='green')
        for i, (layer, score) in enumerate(zip(layers, agreement_scores)):
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Method Agreement Score')
        ax.set_title('Cross-Method Agreement by Layer')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 5. Silhouette Score vs Elbow Method
        ax = axes[1, 1]
        sil_scores = [metrics_comparison['silhouette'][layer]['best_score'] for layer in layers]
        
        # 색상은 elbow confidence로
        scatter = ax.scatter(sil_scores, elbow_ks, s=100, c=elbow_confidences, 
                            cmap='Reds', alpha=0.7, edgecolors='black')
        
        for i, layer in enumerate(layers):
            ax.annotate(f'L{layer}', (sil_scores[i], elbow_ks[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Best Silhouette Score')
        ax.set_ylabel('Elbow Optimal k')
        ax.set_title('Silhouette Quality vs Elbow Recommendation')
        ax.grid(True, alpha=0.3)
        
        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elbow Confidence')
        
        # 6. 종합 요약 (Elbow 포함)
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Comprehensive Analysis Summary\n(Including Elbow Method)\n\n"
        summary_text += f"Layers analyzed: {len(layers)}\n"
        summary_text += f"Metrics compared: 4 (Sil, CH, DB, Elbow)\n\n"
        
        # 전체 메트릭 일치도 분석
        all_ks = sil_ks + ch_ks + db_ks + elbow_ks
        complete_agreement = sum(1 for i in range(len(layers)) 
                            if len(set([sil_ks[i], ch_ks[i], db_ks[i], elbow_ks[i]])) == 1)
        
        summary_text += f"Complete agreement: {complete_agreement}/{len(layers)} layers\n"
        summary_text += f"Avg agreement score: {np.mean(agreement_scores):.2f}\n\n"
        
        # Elbow vs Silhouette 일치도
        elbow_sil_agreement = sum(1 for i in range(len(layers)) if sil_ks[i] == elbow_ks[i])
        summary_text += f"Elbow-Silhouette agreement: {elbow_sil_agreement}/{len(layers)}\n\n"
        
        # 최고 성능 레이어들
        best_sil_layer = layers[np.argmax(sil_scores)]
        best_elbow_conf_layer = layers[np.argmax(elbow_confidences)]
        best_agreement_layer = layers[np.argmax(agreement_scores)]
        
        summary_text += f"Best Silhouette: Layer {best_sil_layer}\n"
        summary_text += f"Most confident Elbow: Layer {best_elbow_conf_layer}\n"
        summary_text += f"Best agreement: Layer {best_agreement_layer}\n\n"
        
        # 최종 추천
        most_common_k = max(set(all_ks), key=all_ks.count)
        summary_text += f"🎯 FINAL RECOMMENDATIONS:\n"
        summary_text += f"Most frequent k: {most_common_k}\n"
        
        if elbow_sil_agreement >= len(layers) // 2:
            summary_text += f"✅ Elbow & Silhouette agree\n"
            summary_text += f"Recommended: Use Elbow method"
        else:
            summary_text += f"⚠️ Mixed results across methods\n"
            summary_text += f"Recommended: Layer {best_agreement_layer}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.suptitle('Cross-Layer Comprehensive Analysis (with Elbow Method)', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_cross_layer_analysis_with_elbow.png', 
                dpi=300, bbox_inches='tight')
        plt.close()

    # 기존 _generate_comprehensive_summary 메서드를 수정
    def _generate_comprehensive_summary_with_elbow(self, all_results, output_dir):
        """Elbow method가 포함된 전체 레이어 종합 요약"""
        layers = sorted([int(k.split('_')[1]) for k in all_results.keys()])
        
        # 메트릭별 최적 k 수집 (Elbow 추가)
        metrics_comparison = {
            'silhouette': {},
            'calinski_harabasz': {},
            'davies_bouldin': {},
            'elbow': {},  # NEW!
            'layers': layers
        }
        
        for layer in layers:
            layer_key = f'layer_{layer}'
            result = all_results[layer_key]
            
            metrics_comparison['silhouette'][layer] = {
                'best_k': result['best_k_silhouette'],
                'best_score': result['best_silhouette_score']
            }
            metrics_comparison['calinski_harabasz'][layer] = {
                'best_k': result['best_k_calinski_harabasz'],
                'best_score': max([result['results_by_k'][k]['metrics']['calinski_harabasz']['score'] 
                                for k in result['k_values']])
            }
            metrics_comparison['davies_bouldin'][layer] = {
                'best_k': result['best_k_davies_bouldin'],
                'best_score': min([result['results_by_k'][k]['metrics']['davies_bouldin']['score'] 
                                for k in result['k_values']])
            }
            metrics_comparison['elbow'][layer] = {
                'best_k': result['best_k_elbow'],
                'method': result['best_elbow_method'],
                'confidence': max([result['elbow_analysis']['knee_method']['confidence'],
                                result['elbow_analysis']['derivative_method']['confidence'],
                                result['elbow_analysis']['variance_method']['confidence']])
            }
        
        # 종합 비교 시각화 (Elbow 포함)
        self._create_cross_layer_comparison_with_elbow(metrics_comparison, all_results, output_dir)
        
        # 종합 요약 JSON 저장
        with open(output_dir / 'comprehensive_summary_with_elbow.json', 'w') as f:
            json.dump(metrics_comparison, f, indent=2)
        
        # Elbow 포함 추천사항 로그 출력
        self._log_recommendations_with_elbow(metrics_comparison)

    def _log_recommendations_with_elbow(self, metrics_comparison):
        """Elbow method가 포함된 추천사항 로그 출력"""
        layers = metrics_comparison['layers']
        
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE ANALYSIS RECOMMENDATIONS (WITH ELBOW METHOD)")
        logger.info("="*80)
        
        # 실루엣 기준 최고 성능 레이어
        sil_scores = [metrics_comparison['silhouette'][layer]['best_score'] for layer in layers]
        best_sil_layer = layers[np.argmax(sil_scores)]
        best_sil_score = max(sil_scores)
        
        # Elbow 신뢰도 기준 최고 레이어
        elbow_confidences = [metrics_comparison['elbow'][layer]['confidence'] for layer in layers]
        best_elbow_layer = layers[np.argmax(elbow_confidences)]
        best_elbow_confidence = max(elbow_confidences)
        
        logger.info(f"📊 Best performing layer (Silhouette): Layer {best_sil_layer} (score: {best_sil_score:.4f})")
        logger.info(f"🔥 Most confident Elbow layer: Layer {best_elbow_layer} (confidence: {best_elbow_confidence:.4f})")
        
        # 메트릭별 최적 k 일치도 분석
        sil_ks = [metrics_comparison['silhouette'][layer]['best_k'] for layer in layers]
        ch_ks = [metrics_comparison['calinski_harabasz'][layer]['best_k'] for layer in layers]
        db_ks = [metrics_comparison['davies_bouldin'][layer]['best_k'] for layer in layers]
        elbow_ks = [metrics_comparison['elbow'][layer]['best_k'] for layer in layers]
        
        # 완전 일치 (모든 메트릭이 같은 k)
        complete_agreement = sum(1 for i in range(len(layers)) 
                            if len(set([sil_ks[i], ch_ks[i], db_ks[i], elbow_ks[i]])) == 1)
        
        # Elbow-Silhouette 일치
        elbow_sil_agreement = sum(1 for i in range(len(layers)) if sil_ks[i] == elbow_ks[i])
        
        logger.info(f"🔍 Complete metric agreement: {complete_agreement}/{len(layers)} layers")
        logger.info(f"🤝 Elbow-Silhouette agreement: {elbow_sil_agreement}/{len(layers)} layers")
        
        # 가장 일반적인 최적 k
        all_ks = sil_ks + ch_ks + db_ks + elbow_ks
        most_common_k = max(set(all_ks), key=all_ks.count)
        logger.info(f"🔢 Most frequent optimal k across all metrics: {most_common_k}")
        
        # 메트릭별 최적 k 요약
        logger.info(f"\n📈 Optimal k by metric:")
        for layer in layers:
            sil_k = metrics_comparison['silhouette'][layer]['best_k']
            ch_k = metrics_comparison['calinski_harabasz'][layer]['best_k']
            db_k = metrics_comparison['davies_bouldin'][layer]['best_k']
            elbow_k = metrics_comparison['elbow'][layer]['best_k']
            elbow_method = metrics_comparison['elbow'][layer]['method']
            logger.info(f"   Layer {layer}: Silhouette={sil_k}, CH={ch_k}, DB={db_k}, Elbow={elbow_k}({elbow_method})")
        
        # Elbow method 상세 분석
        logger.info(f"\n🔥 ELBOW METHOD ANALYSIS:")
        for layer in layers:
            elbow_k = metrics_comparison['elbow'][layer]['best_k']
            elbow_method = metrics_comparison['elbow'][layer]['method']
            elbow_confidence = metrics_comparison['elbow'][layer]['confidence']
            logger.info(f"   Layer {layer}: k={elbow_k}, method={elbow_method}, confidence={elbow_confidence:.4f}")
        
        # 최종 추천사항
        logger.info(f"\n🎯 FINAL RECOMMENDATIONS:")
        
        if complete_agreement > 0:
            perfect_layers = [i for i in range(len(layers)) 
                            if len(set([sil_ks[i], ch_ks[i], db_ks[i], elbow_ks[i]])) == 1]
            logger.info(f"✅ Layers with perfect agreement: {[layers[i] for i in perfect_layers]}")
            logger.info(f"✅ Use k={sil_ks[perfect_layers[0]]} for these layers")
        
        if elbow_sil_agreement >= len(layers) // 2:
            logger.info(f"✅ Elbow method mostly agrees with Silhouette - RELIABLE RESULTS")
            logger.info(f"🎯 PRIMARY RECOMMENDATION: Use Elbow method results")
        else:
            logger.info(f"⚠️ Mixed results between Elbow and Silhouette methods")
            logger.info(f"🎯 RECOMMENDATION: Focus on Layer {best_sil_layer} (best Silhouette)")
        
        # 특별한 케이스들
        if best_sil_layer == best_elbow_layer:
            logger.info(f"🌟 Layer {best_sil_layer} shows both best Silhouette AND most confident Elbow!")
        
        logger.info(f"🔢 Conservative choice: k={most_common_k} (most frequent across all methods)")
        logger.info("="*80)

    # 메인 분석 함수도 수정 필요
    def analyze_all_layers_with_elbow(self, data_loader, k_range=(2, 15), output_dir=None):
        """
        모든 레이어에 대해 Elbow method가 포함된 종합적인 메트릭 분석을 수행합니다.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Attention maps 추출
        logger.info("Extracting attention maps...")
        attention_data = self.extract_attention_maps(data_loader)
        
        # 모든 레이어 분석 (Elbow 포함)
        all_results = {}
        for layer_idx in range(len(self.model.layers)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing Layer {layer_idx} (WITH ELBOW METHOD)")
            logger.info(f"{'='*60}")
            
            layer_output_dir = output_dir / f'layer_{layer_idx}' if output_dir else None
            results = self.perform_comprehensive_analysis_with_elbow(  # 새로운 함수 사용
                attention_data, 
                layer_idx, 
                k_range=k_range,
                output_dir=layer_output_dir
            )
            all_results[f'layer_{layer_idx}'] = results
        
        # 전체 요약 생성 (Elbow 포함)
        if output_dir:
            self._generate_comprehensive_summary_with_elbow(all_results, output_dir)
        
        return all_results
    def _plot_detailed_silhouette_analysis(self, results, flattened_maps, output_dir):
        """상세 실루엣 분석 (기존 스타일과 동일)"""
        layer_idx = results['layer_idx']
        best_k = results['best_k_silhouette']
        
        # 최적 k로 클러스터링
        np.random.seed(42)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        
        # 실루엣 분석
        silhouette_avg = silhouette_score(flattened_maps, cluster_labels)
        sample_silhouette_values = silhouette_samples(flattened_maps, cluster_labels)
        
        # 실루엣 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 실루엣 플롯
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, best_k))
        
        for i in range(best_k):
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
        ax1.set_title(f'Silhouette Plot (k={best_k})')
        
        # 평균 실루엣 스코어 라인
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        ax1.legend()
        
        # 2. 클러스터별 평균 실루엣 스코어
        cluster_avg_scores = []
        cluster_sizes = []
        
        for i in range(best_k):
            cluster_mask = cluster_labels == i
            cluster_sizes.append(np.sum(cluster_mask))
            cluster_avg_scores.append(np.mean(sample_silhouette_values[cluster_mask]))
        
        x_pos = np.arange(best_k)
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
        ax2.set_xticklabels([f'C{i}' for i in range(best_k)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx}: Detailed Silhouette Analysis (k={best_k})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_detailed_silhouette.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_optimal_clustering_comparison(self, results, flattened_maps, labels, output_dir):
        """최적 k에 대한 클러스터 분포 시각화"""
        layer_idx = results['layer_idx']
        best_k_sil = results['best_k_silhouette']
        best_k_ch = results['best_k_calinski_harabasz']
        best_k_db = results['best_k_davies_bouldin']
        
        # t-SNE 차원 축소
        perplexity = min(30, len(flattened_maps)-1, max(1, len(flattened_maps)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(flattened_maps)
        
        # 각 메트릭의 최적 k로 클러스터링
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Silhouette 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_sil, 
            f'Silhouette Optimal (k={best_k_sil})', axes[0, 0]
        )
        
        # 2. Calinski-Harabasz 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_ch,
            f'Calinski-Harabasz Optimal (k={best_k_ch})', axes[0, 1]
        )
        
        # 3. Davies-Bouldin 최적 k
        self._plot_single_clustering_result(
            tsne_embeddings, flattened_maps, labels, best_k_db,
            f'Davies-Bouldin Optimal (k={best_k_db})', axes[1, 0]
        )
        
        # 4. 실제 라벨
        ax = axes[1, 1]
        unique_labels = np.unique(labels)
        label_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            marker = 'o' if label == 0 else 's'
            ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[label_colors[i]], label=f'True Label {int(label)}', 
                       alpha=0.7, s=50, marker=marker)
        
        ax.set_title('True Labels')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx}: Optimal Clustering Comparison', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_optimal_clustering_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_single_clustering_result(self, tsne_embeddings, flattened_maps, labels, k, title, ax):
        """단일 클러스터링 결과 플롯"""
        # 클러스터링
        np.random.seed(42)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        
        # 시각화
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            mask = cluster_labels == i
            ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        ax.set_title(title)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        if k <= 10:  # 너무 많은 클러스터는 범례 생략
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def analyze_all_layers(self, data_loader, k_range=(2, 15), output_dir=None):
        """
        모든 레이어에 대해 종합적인 메트릭 분석을 수행합니다.
        
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
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing Layer {layer_idx}")
            logger.info(f"{'='*60}")
            
            layer_output_dir = output_dir / f'layer_{layer_idx}' if output_dir else None
            results = self.perform_comprehensive_analysis(
                attention_data, 
                layer_idx, 
                k_range=k_range,
                output_dir=layer_output_dir
            )
            all_results[f'layer_{layer_idx}'] = results
        
        # 전체 요약 생성
        if output_dir:
            self._generate_comprehensive_summary(all_results, output_dir)
        
        return all_results

    def _generate_comprehensive_summary(self, all_results, output_dir):
        """전체 레이어 종합 요약"""
        layers = sorted([int(k.split('_')[1]) for k in all_results.keys()])
        
        # 메트릭별 최적 k 수집
        metrics_comparison = {
            'silhouette': {},
            'calinski_harabasz': {},
            'davies_bouldin': {},
            'layers': layers
        }
        
        for layer in layers:
            layer_key = f'layer_{layer}'
            result = all_results[layer_key]
            
            metrics_comparison['silhouette'][layer] = {
                'best_k': result['best_k_silhouette'],
                'best_score': result['best_silhouette_score']
            }
            metrics_comparison['calinski_harabasz'][layer] = {
                'best_k': result['best_k_calinski_harabasz'],
                'best_score': max([result['results_by_k'][k]['metrics']['calinski_harabasz']['score'] 
                                 for k in result['k_values']])
            }
            metrics_comparison['davies_bouldin'][layer] = {
                'best_k': result['best_k_davies_bouldin'],
                'best_score': min([result['results_by_k'][k]['metrics']['davies_bouldin']['score'] 
                                 for k in result['k_values']])
            }
        
        # 종합 비교 시각화
        self._create_cross_layer_comparison_with_elbow(metrics_comparison, all_results, output_dir)
        
        # 종합 요약 JSON 저장
        with open(output_dir / 'comprehensive_summary.json', 'w') as f:
            json.dump(metrics_comparison, f, indent=2)
        
        # 추천사항 로그 출력
        self._log_recommendations(metrics_comparison)

    def _create_cross_layer_comparison(self, metrics_comparison, all_results, output_dir):
        """레이어간 메트릭 비교 시각화"""
        layers = metrics_comparison['layers']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 메트릭별 최적 k 비교
        ax = axes[0, 0]
        sil_ks = [metrics_comparison['silhouette'][layer]['best_k'] for layer in layers]
        ch_ks = [metrics_comparison['calinski_harabasz'][layer]['best_k'] for layer in layers]
        db_ks = [metrics_comparison['davies_bouldin'][layer]['best_k'] for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.25
        
        ax.bar(x - width, sil_ks, width, label='Silhouette', alpha=0.8)
        ax.bar(x, ch_ks, width, label='Calinski-Harabasz', alpha=0.8)
        ax.bar(x + width, db_ks, width, label='Davies-Bouldin', alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Optimal k')
        ax.set_title('Optimal k by Different Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {layer}' for layer in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Silhouette 스코어 비교
        ax = axes[0, 1]
        sil_scores = [metrics_comparison['silhouette'][layer]['best_score'] for layer in layers]
        ax.plot(layers, sil_scores, 'bo-', linewidth=2, markersize=8)
        for layer, score in zip(layers, sil_scores):
            ax.annotate(f'{score:.3f}', (layer, score), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Best Silhouette Score')
        ax.set_title('Best Silhouette Score by Layer')
        ax.grid(True, alpha=0.3)
        
        # 3. Within/Between Variance 비교
        ax = axes[0, 2]
        for layer in layers:
            layer_key = f'layer_{layer}'
            best_k = all_results[layer_key]['best_k_silhouette']
            variance_data = all_results[layer_key]['results_by_k'][best_k]['metrics']['variance_analysis']
            
            wcss = variance_data['wcss']
            bcss = variance_data['bcss']
            ratio = variance_data['bcss_wcss_ratio']
            
            ax.bar(layer - 0.2, wcss, 0.4, label='WCSS' if layer == layers[0] else "", alpha=0.7, color='red')
            ax.bar(layer + 0.2, bcss, 0.4, label='BCSS' if layer == layers[0] else "", alpha=0.7, color='blue')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sum of Squares')
        ax.set_title('WCSS vs BCSS by Layer (at optimal k)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 메트릭 상관관계
        ax = axes[1, 0]
        
        # 실루엣 vs Calinski-Harabasz 최적 k 비교
        ax.scatter(sil_ks, ch_ks, s=100, alpha=0.7, c=layers, cmap='viridis')
        for i, layer in enumerate(layers):
            ax.annotate(f'L{layer}', (sil_ks[i], ch_ks[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Silhouette Optimal k')
        ax.set_ylabel('Calinski-Harabasz Optimal k')
        ax.set_title('Optimal k Correlation: Silhouette vs CH')
        ax.grid(True, alpha=0.3)
        
        # 대각선 참조선
        min_k = min(min(sil_ks), min(ch_ks))
        max_k = max(max(sil_ks), max(ch_ks))
        ax.plot([min_k, max_k], [min_k, max_k], 'r--', alpha=0.5, label='Perfect Agreement')
        ax.legend()
        
        # 5. BCSS/WCSS 비율 비교
        ax = axes[1, 1]
        ratios = []
        for layer in layers:
            layer_key = f'layer_{layer}'
            best_k = all_results[layer_key]['best_k_silhouette']
            ratio = all_results[layer_key]['results_by_k'][best_k]['metrics']['variance_analysis']['bcss_wcss_ratio']
            ratios.append(ratio)
        
        bars = ax.bar(layers, ratios, alpha=0.7, color='green')
        for i, (layer, ratio) in enumerate(zip(layers, ratios)):
            ax.text(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + 0.01,
                   f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('BCSS/WCSS Ratio')
        ax.set_title('Cluster Separation Quality by Layer')
        ax.grid(True, alpha=0.3)
        
        # 6. 요약 텍스트
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Comprehensive Analysis Summary\n\n"
        summary_text += f"Layers analyzed: {len(layers)}\n"
        summary_text += f"Metrics compared: 3 (Silhouette, CH, DB)\n\n"
        
        # 메트릭 일치도 분석
        agreement_count = sum(1 for i in range(len(layers)) 
                            if sil_ks[i] == ch_ks[i] == db_ks[i])
        summary_text += f"Complete metric agreement: {agreement_count}/{len(layers)} layers\n\n"
        
        # 최고 성능 레이어
        best_sil_layer = layers[np.argmax(sil_scores)]
        best_ratio_layer = layers[np.argmax(ratios)]
        
        summary_text += f"Best Silhouette: Layer {best_sil_layer} ({max(sil_scores):.3f})\n"
        summary_text += f"Best BCSS/WCSS: Layer {best_ratio_layer} ({max(ratios):.3f})\n\n"
        
        # 추천사항
        summary_text += "Recommendations:\n"
        if agreement_count > len(layers) / 2:
            summary_text += "• High metric agreement - reliable clustering\n"
        else:
            summary_text += "• Mixed metric results - consider ensemble\n"
        
        most_common_k = max(set(sil_ks), key=sil_ks.count)
        summary_text += f"• Most common optimal k: {most_common_k}\n"
        summary_text += f"• Focus on Layer {best_sil_layer} for best clustering"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.suptitle('Cross-Layer Comprehensive Clustering Analysis', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_cross_layer_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _log_recommendations(self, metrics_comparison):
        """추천사항 로그 출력"""
        layers = metrics_comparison['layers']
        
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE ANALYSIS RECOMMENDATIONS")
        logger.info("="*80)
        
        # 실루엣 기준 최고 성능 레이어
        sil_scores = [metrics_comparison['silhouette'][layer]['best_score'] for layer in layers]
        best_sil_layer = layers[np.argmax(sil_scores)]
        best_sil_score = max(sil_scores)
        
        logger.info(f"📊 Best performing layer (Silhouette): Layer {best_sil_layer} (score: {best_sil_score:.4f})")
        
        # 메트릭별 최적 k 일치도 분석
        sil_ks = [metrics_comparison['silhouette'][layer]['best_k'] for layer in layers]
        ch_ks = [metrics_comparison['calinski_harabasz'][layer]['best_k'] for layer in layers]
        db_ks = [metrics_comparison['davies_bouldin'][layer]['best_k'] for layer in layers]
        
        agreement_count = sum(1 for i in range(len(layers)) if sil_ks[i] == ch_ks[i] == db_ks[i])
        logger.info(f"🔍 Metric agreement: {agreement_count}/{len(layers)} layers show consistent optimal k")
        
        # 가장 일반적인 최적 k
        all_ks = sil_ks + ch_ks + db_ks
        most_common_k = max(set(all_ks), key=all_ks.count)
        logger.info(f"🔢 Most frequent optimal k across all metrics: {most_common_k}")
        
        # 메트릭별 최적 k 요약
        logger.info(f"\n📈 Optimal k by metric:")
        for layer in layers:
            sil_k = metrics_comparison['silhouette'][layer]['best_k']
            ch_k = metrics_comparison['calinski_harabasz'][layer]['best_k']
            db_k = metrics_comparison['davies_bouldin'][layer]['best_k']
            logger.info(f"   Layer {layer}: Silhouette={sil_k}, CH={ch_k}, DB={db_k}")
        
        # 실루엣 검증
        logger.info(f"\n✅ SILHOUETTE VERIFICATION:")
        for layer in layers:
            sil_score = metrics_comparison['silhouette'][layer]['best_score']
            sil_k = metrics_comparison['silhouette'][layer]['best_k']
            logger.info(f"   Layer {layer}: k={sil_k}, score={sil_score:.4f}")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Clustering Metrics Analysis')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['Full', 'Few'], default='Full',
                       help='Which model to use (Full or Few)')
    parser.add_argument('--min_k', type=int, default=2,
                       help='Minimum number of clusters to test (default: 2)')
    parser.add_argument('--max_k', type=int, default=50,
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
                args.output_dir = viz_path / 'comprehensive_metrics_analysis'
                break
        
        if args.output_dir is None:
            args.output_dir = checkpoint_dir / 'comprehensive_metrics_analysis'
    
    # Comprehensive Analyzer 초기화
    analyzer = ComprehensiveClusteringAnalyzer(args.checkpoint_dir)
    
    # 데이터로더 선택
    if args.mode == 'Full':
        #data_loader = analyzer.train_loader
        data_loader = analyzer.combined_loader
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
        results = analyzer.perform_comprehensive_analysis(
            attention_data, 
            args.layer_idx, 
            k_range=k_range,
            output_dir=layer_output_dir
        )
        
        logger.info(f"\n🎯 Results for Layer {args.layer_idx}:")
        logger.info(f"   Silhouette optimal k: {results['best_k_silhouette']} (score: {results['best_silhouette_score']:.4f})")
        logger.info(f"   Calinski-Harabasz optimal k: {results['best_k_calinski_harabasz']}")
        logger.info(f"   Davies-Bouldin optimal k: {results['best_k_davies_bouldin']}")
        
    else:
        # 모든 레이어 분석
        logger.info("Analyzing all layers with comprehensive metrics + ELBOW METHOD")
        all_results = analyzer.analyze_all_layers_with_elbow(
            data_loader, 
            k_range=k_range,
            output_dir=args.output_dir
        )
    
    logger.info(f"\n✅ Comprehensive metrics analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()