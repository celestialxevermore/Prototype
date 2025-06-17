"""
Comprehensive Clustering Metrics Analysis (Using Saved Attention Maps)

저장된 attention maps를 로드하여 클러스터링 메트릭을 분석합니다.

Usage:
    python analysis4.py --attention_map_dir /path/to/attention_map/dir
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
import glob
import re

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

class SavedAttentionMapAnalyzer:
    def __init__(self, attention_map_dir):
        """
        Args:
            attention_map_dir (str): 저장된 attention maps 디렉토리 경로
        """
        ensure_deterministic()
        
        self.attention_map_dir = Path(attention_map_dir)
        
        # 메타데이터 로드
        self.metadata_path = self.attention_map_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['feature_names']
        self.total_samples = self.metadata['total_samples']
        self.num_layers = self.metadata['num_layers']
        
        logger.info(f"Loaded metadata from {self.metadata_path}")
        logger.info(f"Total samples: {self.total_samples}")
        logger.info(f"Number of layers: {self.num_layers}")
        logger.info(f"Feature names: {len(self.feature_names)} features")
        
    def load_attention_maps(self):
        """
        저장된 attention maps를 모두 로드
        
        Returns:
            dict: 레이어별 attention maps와 메타데이터
        """
        logger.info("Loading saved attention maps...")
        
        # 샘플 파일들 찾기
        sample_files = list(self.attention_map_dir.glob("sample_*_label_*.npz"))
        sample_files.sort(key=lambda x: int(re.search(r'sample_(\d+)_', x.name).group(1)))
        
        if len(sample_files) != self.total_samples:
            logger.warning(f"Expected {self.total_samples} sample files, found {len(sample_files)}")
        
        # 동적으로 레이어 수에 맞춰 저장용 딕셔너리 생성
        attention_data = {
            'labels': [],
            'sample_ids': [],
            'feature_names': self.feature_names
        }
        
        # 레이어별 빈 리스트 초기화
        for layer_idx in range(self.num_layers):
            attention_data[f'layer_{layer_idx}'] = []
        
        # 각 샘플 파일 로드
        for sample_file in sample_files:
            sample_data = np.load(sample_file)
            
            # 각 레이어별 attention map 저장
            for layer_idx in range(self.num_layers):
                attention_map = sample_data[f'layer_{layer_idx}']
                attention_data[f'layer_{layer_idx}'].append(attention_map)
            
            # 라벨과 샘플 ID 저장
            attention_data['labels'].append(sample_data['label'])
            attention_data['sample_ids'].append(sample_data['sample_id'])
        
        # NumPy 배열로 변환
        for layer_idx in range(self.num_layers):
            attention_data[f'layer_{layer_idx}'] = np.stack(attention_data[f'layer_{layer_idx}'])
        
        attention_data['labels'] = np.array(attention_data['labels'])
        attention_data['sample_ids'] = np.array(attention_data['sample_ids'])
        
        logger.info(f"✅ Loaded {len(sample_files)} attention maps")
        logger.info(f"   Shape per layer: {attention_data['layer_0'].shape}")
        
        return attention_data

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
        attention_maps = attention_data[f'layer_{layer_idx}']
        labels = attention_data['labels']
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
        """Elbow method 포함 종합 시각화 생성"""
        layer_idx = results['layer_idx']
        k_values = results['k_values']
        
        # 1. Elbow Method 시각화
        plt.figure(figsize=(15, 12))
        
        # WCSS Elbow Plot
        plt.subplot(2, 3, 1)
        wcss_scores = results['elbow_analysis']['wcss_scores']
        plt.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8)
        
        # Elbow 포인트들 표시
        knee_k = results['elbow_analysis']['knee_method']['elbow_k']
        derivative_k = results['elbow_analysis']['derivative_method']['elbow_k']
        variance_k = results['elbow_analysis']['variance_method']['elbow_k']
        
        plt.axvline(x=knee_k, color='red', linestyle='--', alpha=0.7, label=f'Knee Method (k={knee_k})')
        plt.axvline(x=derivative_k, color='green', linestyle='--', alpha=0.7, label=f'Derivative Method (k={derivative_k})')
        plt.axvline(x=variance_k, color='orange', linestyle='--', alpha=0.7, label=f'Variance Method (k={variance_k})')
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title(f'Layer {layer_idx}: Elbow Method Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Silhouette Score vs k
        plt.subplot(2, 3, 2)
        silhouette_scores = [results['results_by_k'][k]['metrics']['silhouette']['average_score'] for k in k_values]
        plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        best_k_sil = results['best_k_silhouette']
        plt.axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_sil}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Layer {layer_idx}: Silhouette Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Calinski-Harabasz Index vs k
        plt.subplot(2, 3, 3)
        ch_scores = [results['results_by_k'][k]['metrics']['calinski_harabasz']['score'] for k in k_values]
        plt.plot(k_values, ch_scores, 'mo-', linewidth=2, markersize=8)
        best_k_ch = results['best_k_calinski_harabasz']
        plt.axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_ch}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title(f'Layer {layer_idx}: Calinski-Harabasz Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Davies-Bouldin Index vs k
        plt.subplot(2, 3, 4)
        db_scores = [results['results_by_k'][k]['metrics']['davies_bouldin']['score'] for k in k_values]
        plt.plot(k_values, db_scores, 'co-', linewidth=2, markersize=8)
        best_k_db = results['best_k_davies_bouldin']
        plt.axvline(x=best_k_db, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_db}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title(f'Layer {layer_idx}: Davies-Bouldin Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 모든 메트릭 정규화 비교
        plt.subplot(2, 3, 5)
        # 정규화 (0-1 범위로)
        sil_norm = (np.array(silhouette_scores) - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        ch_norm = (np.array(ch_scores) - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores))
        db_norm = 1 - (np.array(db_scores) - np.min(db_scores)) / (np.max(db_scores) - np.min(db_scores))  # DB는 낮을수록 좋음
        wcss_norm = 1 - (np.array(wcss_scores) - np.min(wcss_scores)) / (np.max(wcss_scores) - np.min(wcss_scores))  # WCSS도 낮을수록 좋음
        
        plt.plot(k_values, sil_norm, 'g-', label='Silhouette (normalized)', linewidth=2)
        plt.plot(k_values, ch_norm, 'm-', label='Calinski-Harabasz (normalized)', linewidth=2)
        plt.plot(k_values, db_norm, 'c-', label='Davies-Bouldin (inverted & normalized)', linewidth=2)
        plt.plot(k_values, wcss_norm, 'r-', label='WCSS (inverted & normalized)', linewidth=2)
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Normalized Score')
        plt.title(f'Layer {layer_idx}: All Metrics Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. t-SNE 시각화 (최적 k로)
        plt.subplot(2, 3, 6)
        best_k = results['best_k_elbow']
        best_cluster_labels = results['results_by_k'][best_k]['cluster_assignments']
        
        # t-SNE 차원 축소 (샘플이 많으면 subset 사용)
        if len(flattened_maps) > 1000:
            sample_indices = np.random.choice(len(flattened_maps), 1000, replace=False)
            tsne_data = flattened_maps[sample_indices]
            tsne_labels = np.array(best_cluster_labels)[sample_indices]
        else:
            tsne_data = flattened_maps
            tsne_labels = np.array(best_cluster_labels)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)-1))
        tsne_result = tsne.fit_transform(tsne_data)
        
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=tsne_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f'Layer {layer_idx}: t-SNE Visualization (k={best_k})')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_comprehensive_analysis_with_elbow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 개별 Elbow 메서드 세부 시각화
        self._create_elbow_detail_plots(results, output_dir)
        
        logger.info(f"✅ Visualizations saved for layer {layer_idx}")

    def _create_elbow_detail_plots(self, results, output_dir):
        """Elbow 메서드 세부 분석 플롯"""
        layer_idx = results['layer_idx']
        k_values = results['k_values']
        wcss_scores = results['elbow_analysis']['wcss_scores']
        
        plt.figure(figsize=(18, 6))
        
        # 1. Knee Method 세부
        plt.subplot(1, 3, 1)
        knee_data = results['elbow_analysis']['knee_method']
        distances = knee_data['all_distances']
        
        plt.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        plt.axvline(x=knee_data['elbow_k'], color='red', linestyle='--', linewidth=2, 
                   label=f"Knee Point (k={knee_data['elbow_k']})")
        
        # 거리 그래프 (보조 y축)
        ax2 = plt.gca().twinx()
        ax2.plot(k_values, distances, 'ro-', alpha=0.7, label='Distance to Line')
        ax2.set_ylabel('Distance to Line', color='red')
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title(f'Layer {layer_idx}: Knee Method Detail\n(Confidence: {knee_data["confidence"]:.4f})')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 2. Derivative Method 세부
        plt.subplot(1, 3, 2)
        derivative_data = results['elbow_analysis']['derivative_method']
        
        plt.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        plt.axvline(x=derivative_data['elbow_k'], color='green', linestyle='--', linewidth=2,
                   label=f"Derivative Point (k={derivative_data['elbow_k']})")
        
        # 1차 미분 그래프 (보조 y축)
        if 'first_derivative' in derivative_data:
            ax2 = plt.gca().twinx()
            first_deriv = derivative_data['first_derivative']
            ax2.plot(k_values[1:], first_deriv, 'go-', alpha=0.7, label='1st Derivative')
            ax2.set_ylabel('1st Derivative', color='green')
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title(f'Layer {layer_idx}: Derivative Method Detail\n(Confidence: {derivative_data["confidence"]:.4f})')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 3. Variance Method 세부
        plt.subplot(1, 3, 3)
        variance_data = results['elbow_analysis']['variance_method']
        
        plt.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8, label='WCSS')
        plt.axvline(x=variance_data['elbow_k'], color='orange', linestyle='--', linewidth=2,
                   label=f"Variance Point (k={variance_data['elbow_k']})")
        
        # 분산 그래프 (보조 y축)
        if 'variances' in variance_data and len(variance_data['variances']) > 0:
            ax2 = plt.gca().twinx()
            variances = variance_data['variances']
            
            # variance method에서 사용된 실제 window_size 계산
            # calculate_elbow_point의 variance method 로직과 동일하게
            window_size = min(3, len(k_values) // 2)
            
            # variance 값들에 대응하는 k 값들 (window_size만큼 앞뒤 제외)
            if len(variances) > 0:
                var_k_start = window_size
                var_k_end = len(k_values) - window_size
                var_k_values = k_values[var_k_start:var_k_end]
                
                # 길이 맞추기 (안전장치)
                min_len = min(len(var_k_values), len(variances))
                var_k_values = var_k_values[:min_len]
                variances_plot = variances[:min_len]
                
                if len(var_k_values) > 0 and len(variances_plot) > 0:
                    ax2.plot(var_k_values, variances_plot, 'o-', color='orange', alpha=0.7, label='Slope Variance')
                    ax2.set_ylabel('Slope Variance', color='orange')
                else:
                    # variance 데이터가 없는 경우 텍스트로 표시
                    ax2.text(0.5, 0.5, 'No variance data available', 
                            ha='center', va='center', transform=ax2.transAxes, color='orange')
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.title(f'Layer {layer_idx}: Variance Method Detail\n(Confidence: {variance_data["confidence"]:.4f})')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_elbow_methods_detail.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_all_layers_with_elbow(self, k_range=(2, 15), output_dir=None):
        """
        모든 레이어에 대해 Elbow method가 포함된 종합적인 메트릭 분석을 수행합니다.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장된 Attention maps 로드
        logger.info("Loading saved attention maps...")
        attention_data = self.load_attention_maps()
        
        # 모든 레이어 분석 (Elbow 포함)
        all_results = {}
        for layer_idx in range(self.num_layers):
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing Layer {layer_idx} (WITH ELBOW METHOD)")
            logger.info(f"{'='*60}")
            
            layer_output_dir = output_dir / f'layer_{layer_idx}' if output_dir else None
            results = self.perform_comprehensive_analysis_with_elbow(
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

    def _generate_comprehensive_summary_with_elbow(self, all_results, output_dir):
        """Elbow method 포함 종합 요약 생성"""
        logger.info("Generating comprehensive summary with elbow method...")
        
        # 레이어별 최적 k 요약
        summary = {
            'analysis_timestamp': str(np.datetime64('now')),
            'total_layers': len(all_results),
            'layer_summary': {},
            'cross_layer_analysis': {}
        }
        
        # 각 레이어별 요약
        for layer_key, layer_results in all_results.items():
            layer_idx = layer_results['layer_idx']
            
            summary['layer_summary'][layer_key] = {
                'best_k_silhouette': layer_results['best_k_silhouette'],
                'best_silhouette_score': layer_results['best_silhouette_score'],
                'best_k_calinski_harabasz': layer_results['best_k_calinski_harabasz'],
                'best_k_davies_bouldin': layer_results['best_k_davies_bouldin'],
                'best_k_elbow': layer_results['best_k_elbow'],
                'best_elbow_method': layer_results['best_elbow_method'],
                'elbow_methods': {
                    'knee': layer_results['elbow_analysis']['knee_method']['elbow_k'],
                    'derivative': layer_results['elbow_analysis']['derivative_method']['elbow_k'],
                    'variance': layer_results['elbow_analysis']['variance_method']['elbow_k']
                }
            }
        
        # 레이어 간 비교 분석
        silhouette_bests = [summary['layer_summary'][f'layer_{i}']['best_k_silhouette'] for i in range(len(all_results))]
        elbow_bests = [summary['layer_summary'][f'layer_{i}']['best_k_elbow'] for i in range(len(all_results))]
        
        summary['cross_layer_analysis'] = {
            'silhouette_consistency': len(set(silhouette_bests)) == 1,
            'elbow_consistency': len(set(elbow_bests)) == 1,
            'most_common_k_silhouette': int(np.bincount(silhouette_bests).argmax()),
            'most_common_k_elbow': int(np.bincount(elbow_bests).argmax()),
            'k_variance_silhouette': float(np.var(silhouette_bests)),
            'k_variance_elbow': float(np.var(elbow_bests))
        }
        
        # JSON 저장
        with open(output_dir / 'comprehensive_summary_with_elbow.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 요약 시각화
        self._create_summary_visualizations_with_elbow(all_results, summary, output_dir)
        
        logger.info(f"✅ Comprehensive summary with elbow method saved to {output_dir}")
        
        # 콘솔 요약 출력
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE ANALYSIS SUMMARY (WITH ELBOW METHOD)")
        logger.info("="*80)
        
        for layer_key in summary['layer_summary']:
            layer_summary = summary['layer_summary'][layer_key]
            logger.info(f"\n🔍 {layer_key.upper()}:")
            logger.info(f"   Silhouette Best k: {layer_summary['best_k_silhouette']} (score: {layer_summary['best_silhouette_score']:.4f})")
            logger.info(f"   Elbow Best k: {layer_summary['best_k_elbow']} (method: {layer_summary['best_elbow_method']})")
            logger.info(f"   Calinski-Harabasz Best k: {layer_summary['best_k_calinski_harabasz']}")
            logger.info(f"   Davies-Bouldin Best k: {layer_summary['best_k_davies_bouldin']}")
        
        cross_analysis = summary['cross_layer_analysis']
        logger.info(f"\n🔄 CROSS-LAYER ANALYSIS:")
        logger.info(f"   Silhouette Consistency: {'✅' if cross_analysis['silhouette_consistency'] else '❌'}")
        logger.info(f"   Elbow Consistency: {'✅' if cross_analysis['elbow_consistency'] else '❌'}")
        logger.info(f"   Most Common k (Silhouette): {cross_analysis['most_common_k_silhouette']}")
        logger.info(f"   Most Common k (Elbow): {cross_analysis['most_common_k_elbow']}")
        
        return summary

    def _create_summary_visualizations_with_elbow(self, all_results, summary, output_dir):
        """요약 시각화 생성 - 첫 번째 그림과 완전히 동일한 스타일로"""
        layers = list(range(len(all_results)))
        
        # 2x3 레이아웃
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 메트릭별 최적 k 비교 (바 차트 - 첫 번째 그림과 동일)
        ax = axes[0, 0]
        silhouette_ks = [summary['layer_summary'][f'layer_{i}']['best_k_silhouette'] for i in layers]
        ch_ks = [summary['layer_summary'][f'layer_{i}']['best_k_calinski_harabasz'] for i in layers]
        db_ks = [summary['layer_summary'][f'layer_{i}']['best_k_davies_bouldin'] for i in layers]
        elbow_ks = [summary['layer_summary'][f'layer_{i}']['best_k_elbow'] for i in layers]
        
        x = np.arange(len(layers))
        width = 0.2
        
        ax.bar(x - 1.5*width, silhouette_ks, width, label='Silhouette', alpha=0.8, color='tab:blue')
        ax.bar(x - 0.5*width, ch_ks, width, label='Calinski-Harabasz', alpha=0.8, color='tab:orange')
        ax.bar(x + 0.5*width, db_ks, width, label='Davies-Bouldin', alpha=0.8, color='tab:green')
        ax.bar(x + 1.5*width, elbow_ks, width, label='Elbow', alpha=0.8, color='tab:red')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Optimal k')
        ax.set_title('Optimal k by Different Metrics (Including Elbow)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {layer}' for layer in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Silhouette vs Elbow 상관관계
        ax = axes[0, 1]
        ax.scatter(silhouette_ks, elbow_ks, s=100, alpha=0.7, c=layers, cmap='viridis')
        for i, layer in enumerate(layers):
            ax.annotate(f'L{layer}', (silhouette_ks[i], elbow_ks[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Silhouette Optimal k')
        ax.set_ylabel('Elbow Optimal k')
        ax.set_title('Optimal k Correlation: Silhouette vs Elbow')
        ax.grid(True, alpha=0.3)
        
        # 대각선 참조선
        min_k = min(min(silhouette_ks), min(elbow_ks))
        max_k = max(max(silhouette_ks), max(elbow_ks))
        ax.plot([min_k, max_k], [min_k, max_k], 'r--', alpha=0.5, label='Perfect Agreement')
        ax.legend()
        
        # 3. Elbow Method Confidence by Layer
        ax = axes[0, 2]
        elbow_confidences = []
        elbow_methods = []
        for i in layers:
            layer_key = f'layer_{i}'
            layer_results = all_results[layer_key]
            knee_conf = layer_results['elbow_analysis']['knee_method']['confidence']
            deriv_conf = layer_results['elbow_analysis']['derivative_method']['confidence']
            var_conf = layer_results['elbow_analysis']['variance_method']['confidence']
            
            best_conf = max(knee_conf, deriv_conf, var_conf)
            elbow_confidences.append(best_conf)
            
            if best_conf == knee_conf:
                method = 'knee'
            elif best_conf == deriv_conf:
                method = 'derivative'
            else:
                method = 'variance'
            elbow_methods.append(method)
        
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
            layer_ks = [silhouette_ks[layer], ch_ks[layer], db_ks[layer], elbow_ks[layer]]
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
        
        # 5. Silhouette Quality vs Elbow Recommendation
        ax = axes[1, 1]
        silhouette_scores = [summary['layer_summary'][f'layer_{i}']['best_silhouette_score'] for i in layers]
        
        # 색상은 elbow confidence로
        scatter = ax.scatter(silhouette_scores, elbow_ks, s=100, c=elbow_confidences, 
                            cmap='Reds', alpha=0.7, edgecolors='black')
        
        for i, layer in enumerate(layers):
            ax.annotate(f'L{layer}', (silhouette_scores[i], elbow_ks[i]), xytext=(5, 5), 
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
        all_ks = silhouette_ks + ch_ks + db_ks + elbow_ks
        complete_agreement = sum(1 for i in range(len(layers)) 
                            if len(set([silhouette_ks[i], ch_ks[i], db_ks[i], elbow_ks[i]])) == 1)
        
        summary_text += f"Complete agreement: {complete_agreement}/{len(layers)} layers\n"
        summary_text += f"Avg agreement score: {np.mean(agreement_scores):.2f}\n\n"
        
        # Elbow vs Silhouette 일치도
        elbow_sil_agreement = sum(1 for i in range(len(layers)) if silhouette_ks[i] == elbow_ks[i])
        summary_text += f"Elbow-Silhouette agreement: {elbow_sil_agreement}/{len(layers)}\n\n"
        
        # 최고 성능 레이어들
        best_sil_layer = layers[np.argmax(silhouette_scores)]
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
        plt.savefig(output_dir / 'comprehensive_summary_with_elbow.png', dpi=300, bbox_inches='tight')
        plt.close()


def extract_config_from_attention_path(attention_map_dir):
    """Attention map 디렉토리 경로에서 설정 정보 추출"""
    path_parts = Path(attention_map_dir).parts
    
    # /attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att 형태에서
    # 마지막 부분이 config
    config_str = path_parts[-1]
    
    return config_str

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Clustering Metrics Analysis with Elbow Method')
    parser.add_argument('--attention_map_dir', type=str, required=True,
                       help='Directory containing saved attention maps')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results (default: same as attention_map_dir + "_analysis")')
    parser.add_argument('--k_min', type=int, default=2,
                       help='Minimum number of clusters to test (default: 2)')
    parser.add_argument('--k_max', type=int, default=100,
                       help='Maximum number of clusters to test (default: 15)')
    
    args = parser.parse_args()
    
    # 경로 검증
    attention_map_dir = Path(args.attention_map_dir)
    if not attention_map_dir.exists():
        logger.error(f"Attention map directory not found: {attention_map_dir}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        # attention_map 경로를 visualization 경로로 변환
        config_str = extract_config_from_attention_path(attention_map_dir)
        
        # /experiments/attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att
        # → /experiments/visualization/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att_analysis
        output_dir = Path(str(attention_map_dir).replace('/attention_map/', '/visualization/') + '_analysis')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 정보 추출 및 로깅
    config_str = extract_config_from_attention_path(attention_map_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 COMPREHENSIVE CLUSTERING ANALYSIS WITH ELBOW METHOD")
    logger.info(f"{'='*80}")
    logger.info(f"📁 Input: {attention_map_dir}")
    logger.info(f"📊 Config: {config_str}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"🎯 Cluster range: {args.k_min} to {args.k_max}")
    logger.info(f"{'='*80}")
    
    try:
        # 분석기 초기화
        analyzer = SavedAttentionMapAnalyzer(attention_map_dir)
        
        # 모든 레이어에 대해 종합 분석 수행
        all_results = analyzer.analyze_all_layers_with_elbow(
            k_range=(args.k_min, args.k_max),
            output_dir=output_dir
        )
        
        logger.info(f"\n🎉 Analysis completed successfully!")
        logger.info(f"📊 Results saved to: {output_dir}")
        
        # 최종 요약 출력
        logger.info(f"\n📋 FINAL SUMMARY:")
        for layer_key, layer_results in all_results.items():
            best_k_elbow = layer_results['best_k_elbow']
            best_k_silhouette = layer_results['best_k_silhouette']
            best_silhouette_score = layer_results['best_silhouette_score']
            best_elbow_method = layer_results['best_elbow_method']
            
            logger.info(f"   {layer_key}: Elbow k={best_k_elbow} ({best_elbow_method}), Silhouette k={best_k_silhouette} (score={best_silhouette_score:.4f})")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())