"""
Attention Maps Clustering Analysis 스크립트

저장된 attention maps를 로드하여 K-means 클러스터링을 수행합니다.

Usage:
    python clustering_inference.py --attention_map_dir /path/to/attention_maps --n_clusters 8
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
import seaborn as sns
from scipy.spatial.distance import cdist
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

class AttentionMapClusteringAnalyzer:
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
        
        # 거리 순으로 정렬 (오름차순 - 가까운 거리부터)
        sorted_indices = np.argsort(distances)
        sorted_pairs = [cluster_pairs[i] for i in sorted_indices]
        sorted_distances = [distances[i] for i in sorted_indices]
        
        # 3개 서브플롯 생성
        fig = plt.figure(figsize=(20, 7))
        
        # 1. Frobenius Distance Matrix
        ax1 = plt.subplot(1, 3, 1)
        im = ax1.imshow(distance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0)
        
        # 값 표시
        for i in range(optimal_k):
            for j in range(optimal_k):
                if i != j:
                    text_color = 'white' if distance_matrix[i, j] > np.median(distance_matrix) else 'black'
                    ax1.text(j, i, f'{distance_matrix[i, j]:.2f}', 
                            ha="center", va="center", color=text_color, 
                            fontsize=max(8, 12-optimal_k//2), fontweight='bold')
        
        ax1.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cluster ID', fontsize=14, fontweight='bold')
        ax1.set_title('Frobenius Distance Matrix', fontsize=16, fontweight='bold', pad=20)
        
        ax1.set_xticks(range(optimal_k))
        ax1.set_yticks(range(optimal_k))
        ax1.set_xticklabels([f'C{i}' for i in range(optimal_k)], fontsize=12)
        ax1.set_yticklabels([f'C{i}' for i in range(optimal_k)], fontsize=12)
        
        cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label('Distance', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # 2. All Pairwise Distances (가까운 순서)
        ax2 = plt.subplot(1, 3, 2)
        max_pairs_to_show = min(20, len(sorted_distances))
        display_pairs = sorted_pairs[:max_pairs_to_show]
        display_distances = sorted_distances[:max_pairs_to_show]

        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(display_distances)))
        y_positions = np.arange(len(display_distances))[::-1]
        bars = ax2.barh(y_positions, display_distances, color=colors, alpha=0.8, height=0.7)

        for i, (bar, distance) in enumerate(zip(bars, display_distances)):
            ax2.text(bar.get_width() + max(display_distances) * 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{distance:.3f}', ha='left', va='center', 
                    fontweight='bold', fontsize=10, color='black')

        ax2.set_xlabel('Frobenius Distance', fontsize=14, fontweight='bold')
        ax2.set_title(f'Top {max_pairs_to_show} Pairwise Distances\n(Sorted: Close → Far)', 
                    fontsize=16, fontweight='bold', pad=20)

        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(display_pairs, fontsize=10, fontweight='bold')

        if len(sorted_distances) > max_pairs_to_show:
            ax2.text(0.02, 0.02, f'Showing top {max_pairs_to_show} of {len(sorted_distances)} pairs', 
                    transform=ax2.transAxes, fontsize=9, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax2.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5)
        ax2.set_xlim(0, max(display_distances) * 1.2)
        
        # 3. Top 20 Farthest Pairs
        ax3 = plt.subplot(1, 3, 3)
        farthest_pairs_to_show = min(20, len(sorted_distances))
        farthest_pairs = sorted_pairs[-farthest_pairs_to_show:]
        farthest_distances = sorted_distances[-farthest_pairs_to_show:]

        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(farthest_distances)))
        y_positions = np.arange(len(farthest_distances))
        bars = ax3.barh(y_positions, farthest_distances, color=colors, alpha=0.8, height=0.7)

        for i, (bar, distance) in enumerate(zip(bars, farthest_distances)):
            ax3.text(bar.get_width() + max(farthest_distances) * 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{distance:.3f}', ha='left', va='center', 
                    fontweight='bold', fontsize=10, color='black')

        ax3.set_xlabel('Frobenius Distance', fontsize=14, fontweight='bold')
        ax3.set_title(f'Top {farthest_pairs_to_show} Farthest Pairs\n(Most Different Clusters)', 
                    fontsize=16, fontweight='bold', pad=20)

        ax3.set_yticks(y_positions)
        ax3.set_yticklabels(farthest_pairs, fontsize=10, fontweight='bold')

        ax3.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5)
        ax3.set_xlim(0, max(farthest_distances) * 1.2)

        stats_text = f"Total pairs: {len(sorted_distances)} | Range: {min(sorted_distances):.3f}-{max(sorted_distances):.3f}"
        ax3.text(0.02, 0.02, stats_text, transform=ax3.transAxes, fontsize=8, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        fig.suptitle(f'Layer {layer_idx}: Cluster Distance Analysis (k={optimal_k})', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
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
        
        for layer_idx in range(self.num_layers):
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
        
        for layer_idx, centroids in layer_data.items():
            differences = []
            pairs = []
            
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
                    
                    change = np.linalg.norm(centroid_next - centroid_curr)
                    
                    cluster_evolution['layer_changes'].append({
                        'from_layer': layer_curr,
                        'to_layer': layer_next,
                        'change_magnitude': float(change)
                    })
            
            evolution_data.append(cluster_evolution)
        
        # 진화 패턴 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        layer_transitions = []
        for cluster in evolution_data:
            changes = [change['change_magnitude'] for change in cluster['layer_changes']]
            transitions = [f"L{change['from_layer']}→L{change['to_layer']}" 
                         for change in cluster['layer_changes']]
            
            if changes:
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
            if cluster_changes:
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
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        unique_clusters = np.unique(cluster_assignments)
        unique_labels = np.unique(labels)
        
        base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
        
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
                for label in unique_labels:
                    label_mask = cluster_labels == label
                    if np.any(label_mask):
                        label_points = cluster_points[label_mask]
                        
                        if label == 0:
                            marker = 'o'
                            marker_name = 'Label 0'
                        else:
                            marker = 's'
                            marker_name = 'Label 1'
                        
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
        
        ax.set_title(f'Layer {layer_idx} - Clustering & True Labels\nK-means with t-SNE Visualization', 
                    fontsize=16, pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
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
        
        # centroid 폴더 생성
        centroid_dir = output_dir / 'centroid'
        centroid_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 overview: 동적 레이아웃 결정
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
        
        # 모든 클러스터 시각화
        for i, centroid in enumerate(centroids_reshaped):
            ax = axes[i]
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Cluster {i} Centroid', fontsize=11)
            
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
            ax.set_yticklabels(feature_names, fontsize=8)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_idx} Cluster Centroids (All {n_clusters} clusters)', fontsize=14)
        plt.tight_layout()
        
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_centroids_overview.png', 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"All {n_clusters} cluster centroids overview saved: layer_{layer_idx}_cluster_centroids_overview.png")
        
        # 각 센트로이드별로 개별 상세 플롯 생성
        for i, centroid in enumerate(centroids_reshaped):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax.imshow(centroid, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Cluster {i} Centroid - Layer {layer_idx}', fontsize=16, pad=20)
            
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=12)
            ax.set_yticklabels(feature_names, fontsize=12)
            
            # 각 셀에 값 표시
            for row in range(len(feature_names)):
                for col in range(len(feature_names)):
                    value = centroid[row, col]
                    ax.text(col, row, f"{value:.2f}", 
                        ha="center", va="center", 
                        color="white" if value > 0.15 else "black", 
                        fontsize=10, weight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Attention Weight', fontsize=12)
            
            plt.tight_layout()
            
            fig.savefig(centroid_dir / f'cluster_{i}_centroid.png', 
                    dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"All {n_clusters} detailed centroids saved in {centroid_dir}")
        logger.info(f"All cluster centroids saved for layer {layer_idx}")

def extract_config_from_attention_path(attention_map_dir):
    """Attention map 디렉토리 경로에서 설정 정보 추출"""
    path_parts = Path(attention_map_dir).parts
    # /attention_map/gpt2_mean/heart/Full/Embed-carte_desc_Edge-False_A-att 형태에서
    # 마지막 부분이 config
    config_str = path_parts[-1]
    return config_str

def main():
    parser = argparse.ArgumentParser(description='Attention Maps Clustering Analysis')
    parser.add_argument('--attention_map_dir', type=str, required=True,
                       help='Directory containing saved attention maps')
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='Number of clusters for K-means')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 경로 검증
    attention_map_dir = Path(args.attention_map_dir)
    if not attention_map_dir.exists():
        logger.error(f"Attention map directory not found: {attention_map_dir}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        config_str = extract_config_from_attention_path(attention_map_dir)
        # attention_map을 visualization으로 변경
        path_parts = list(attention_map_dir.parts)
        for i, part in enumerate(path_parts):
            if part == 'attention_map':
                path_parts[i] = 'visualization'
                break
        
        viz_path = Path(*path_parts[:-1])  # config 폴더 제거
        args.output_dir = viz_path / config_str / f'clustering_{args.n_clusters}'
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 정보 추출 및 로깅
    config_str = extract_config_from_attention_path(attention_map_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 ATTENTION MAPS CLUSTERING ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"📁 Input: {attention_map_dir}")
    logger.info(f"📊 Config: {config_str}")
    logger.info(f"🎯 Clusters: {args.n_clusters}")
    logger.info(f"💾 Output: {args.output_dir}")
    logger.info(f"{'='*80}")
    
    try:
        # 분석기 초기화
        analyzer = AttentionMapClusteringAnalyzer(attention_map_dir)
        
        # Attention maps 로드
        logger.info("Loading attention maps...")
        attention_data = analyzer.load_attention_maps()
        
        # 클러스터링 결과를 위한 상위 폴더 생성
        clustering_results_dir = args.output_dir / 'clustering_results'
        clustering_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 모든 레이어에 대해 클러스터링
        for layer_idx in range(analyzer.num_layers):
            logger.info(f"Performing clustering on layer {layer_idx}...")
            
            layer_output_dir = clustering_results_dir / f'layer_{layer_idx}'
            clustering_results = analyzer.perform_clustering(
                attention_data, 
                layer_idx=layer_idx,
                n_clusters=args.n_clusters,
                output_dir=layer_output_dir
            )
            
            logger.info(f"Saving attention maps by cluster for layer {layer_idx}...")
            analyzer.save_attention_maps_by_cluster(
                attention_data, 
                clustering_results, 
                clustering_results_dir,
                layer_idx
            )
        
        # 전체 센트로이드 요약 분석
        logger.info("Generating comprehensive centroid summary...")
        analyzer.generate_centroid_summary(clustering_results_dir, args.n_clusters)
        
        logger.info(f"🎉 Clustering analysis completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}")
        
        # 최종 요약 출력
        logger.info(f"\n📋 FINAL SUMMARY:")
        logger.info(f"   Total layers analyzed: {analyzer.num_layers}")
        logger.info(f"   Clusters per layer: {args.n_clusters}")
        logger.info(f"   Total samples: {analyzer.total_samples}")
        logger.info(f"   Results directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())