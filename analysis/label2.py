"""
Attention Map 기반 클러스터 분석 스크립트 (개선된 CLSI 적용)

저장된 attention maps를 로드하여 클러스터링을 수행하고
클러스터별 라벨 분포와 비율을 시각화합니다.
개선된 CLSI (Cluster Label Specialization Index)를 사용합니다.

Usage:
   python analysis_attention_map.py --attention_map_dir /path/to/attention_maps
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
import warnings
import re
import glob

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionMapClusterAnalyzer:
    def __init__(self, attention_map_dir):
        """
        Args:
            attention_map_dir (str): attention maps 디렉토리 경로
        """
        self.attention_map_dir = Path(attention_map_dir)
        self.attention_data = None
        self.layer_results = {}
        
        # 메타데이터 및 attention maps 로드
        self._load_attention_maps()
        
        logger.info(f"AttentionMapClusterAnalyzer initialized for {self.num_layers} layers")
    
    def _load_attention_maps(self):
        """저장된 attention maps를 로드"""
        # 메타데이터 로드
        metadata_path = self.attention_map_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['feature_names']
        self.total_samples = self.metadata['total_samples']
        self.num_layers = self.metadata['num_layers']
        
        logger.info(f"Loading attention maps from {self.attention_map_dir}")
        logger.info(f"Total samples: {self.total_samples}, Layers: {self.num_layers}")
        
        # 샘플 파일들 찾기
        sample_files = list(self.attention_map_dir.glob("sample_*_label_*.npz"))
        sample_files.sort(key=lambda x: int(re.search(r'sample_(\d+)_', x.name).group(1)))
        
        if len(sample_files) != self.total_samples:
            logger.warning(f"Expected {self.total_samples} sample files, found {len(sample_files)}")
        
        # 동적으로 레이어 수에 맞춰 저장용 딕셔너리 생성
        self.attention_data = {
            'labels': [],
            'sample_ids': [],
            'feature_names': self.feature_names
        }
        
        # 레이어별 빈 리스트 초기화
        for layer_idx in range(self.num_layers):
            self.attention_data[f'layer_{layer_idx}'] = []
        
        # 각 샘플 파일 로드
        for sample_file in sample_files:
            sample_data = np.load(sample_file)
            
            # 각 레이어별 attention map 저장
            for layer_idx in range(self.num_layers):
                attention_map = sample_data[f'layer_{layer_idx}']
                self.attention_data[f'layer_{layer_idx}'].append(attention_map)
            
            # 라벨과 샘플 ID 저장
            self.attention_data['labels'].append(sample_data['label'])
            self.attention_data['sample_ids'].append(sample_data['sample_id'])
        
        # NumPy 배열로 변환
        for layer_idx in range(self.num_layers):
            self.attention_data[f'layer_{layer_idx}'] = np.stack(self.attention_data[f'layer_{layer_idx}'])
        
        self.attention_data['labels'] = np.array(self.attention_data['labels'])
        self.attention_data['sample_ids'] = np.array(self.attention_data['sample_ids'])
        
        logger.info(f"✅ Loaded {len(sample_files)} attention maps")
    
    def perform_clustering_for_layer(self, layer_idx, k=None, k_range=(2, 10)):
        """특정 레이어에 대해 클러스터링 수행"""
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer {layer_idx} not available. Max layer: {self.num_layers-1}")
        
        # Attention maps 추출 및 평탄화
        attention_maps = self.attention_data[f'layer_{layer_idx}']
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        labels = self.attention_data['labels']
        
        logger.info(f"Performing clustering for layer {layer_idx}")
        logger.info(f"Data shape: {flattened_maps.shape}")
        
        # k 값 결정 (silhouette score로 최적화)
        if k is None:
            from sklearn.metrics import silhouette_score
            best_k = 2
            best_score = -1
            
            for test_k in range(k_range[0], k_range[1] + 1):
                if test_k >= len(flattened_maps):
                    break
                    
                kmeans = KMeans(n_clusters=test_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(flattened_maps)
                
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(flattened_maps, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = test_k
            
            k = best_k
            logger.info(f"Selected k={k} with silhouette score={best_score:.4f}")
        
        # 최종 클러스터링
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flattened_maps)
        
        # 클러스터별 데이터 구성
        cluster_data = {}
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = []
            
            for idx in np.where(cluster_mask)[0]:
                sample_info = {
                    'sample_id': int(self.attention_data['sample_ids'][idx]),
                    'label': int(labels[idx]),
                    'cluster_id': int(cluster_id)
                }
                cluster_samples.append(sample_info)
            
            cluster_data[cluster_id] = cluster_samples
        
        # 결과 저장
        self.layer_results[layer_idx] = {
            'clusters': cluster_data,
            'total_samples': len(flattened_maps),
            'n_clusters': k,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans
        }
        
        logger.info(f"Layer {layer_idx} clustering completed: {k} clusters")
        return cluster_data

    def analyze_layer_label_distribution(self, layer_idx, output_dir, k=None):
        """특정 레이어의 라벨 분포 분석 및 시각화"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 클러스터링 수행 (아직 안 했다면)
        if layer_idx not in self.layer_results:
            self.perform_clustering_for_layer(layer_idx, k=k)
        
        layer_data = self.layer_results[layer_idx]
        clusters = layer_data['clusters']
        
        logger.info(f"Analyzing label distribution for layer {layer_idx}...")
        
        # 클러스터별 라벨 분포 수집
        cluster_label_data = {}
        all_labels = set()
        
        for cluster_id, samples in clusters.items():
            labels = [sample['label'] for sample in samples]
            cluster_label_data[cluster_id] = labels
            all_labels.update(labels)
        
        all_labels = sorted(list(all_labels))
        
        # 통계적 검정 수행
        stats_results = self._perform_chi_square_test(cluster_label_data, all_labels, layer_idx)
        
        # 시각화
        self._plot_label_distribution_simple(cluster_label_data, all_labels, stats_results, layer_idx, output_dir)
        
        # 결과 저장
        self._save_layer_analysis_results(cluster_label_data, all_labels, stats_results, layer_idx, output_dir)
        
        return cluster_label_data, stats_results
    
    def _perform_chi_square_test(self, cluster_label_data, all_labels, layer_idx):
        """Chi-square 검정 수행"""
        stats_results = {
            'layer_idx': layer_idx,
            'label_chi2': None
        }
        
        cluster_ids = sorted(list(cluster_label_data.keys()))
        
        # 라벨 분포 contingency table 생성
        label_contingency = []
        for cluster_id in cluster_ids:
            labels = cluster_label_data[cluster_id]
            label_counts = {label: 0 for label in all_labels}
            
            for label in labels:
                label_counts[label] += 1
            
            label_row = [label_counts[label] for label in all_labels]
            label_contingency.append(label_row)
        
        # Chi-square 검정
        if len(label_contingency) > 1 and all(sum(row) > 0 for row in label_contingency):
            try:
                chi2_stat, chi2_p, _, _ = chi2_contingency(label_contingency)
                stats_results['label_chi2'] = {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_p),
                    'significant': chi2_p < 0.05,
                    'contingency_table': label_contingency
                }
                logger.info(f"Chi-square test - Statistic: {chi2_stat:.4f}, p-value: {chi2_p:.4f}")
            except Exception as e:
                logger.warning(f"Chi-square test failed: {e}")
        
        return stats_results
    
    def _plot_label_distribution_simple(self, cluster_label_data, all_labels, stats_results, layer_idx, output_dir):
        """라벨 분포 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 클러스터 ID를 숫자순으로 정렬
        cluster_ids = sorted(list(cluster_label_data.keys()))
        
        # 데이터 준비
        labels_data = []
        for cluster_id in cluster_ids:
            labels = cluster_label_data[cluster_id]
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            for label, count in zip(unique_labels, counts):
                labels_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Label': f'Label {int(label)}',
                    'Count': count,
                    'cluster_id': cluster_id
                })
        
        df_labels = pd.DataFrame(labels_data)
        df_labels = df_labels.sort_values('cluster_id')
        
        # 스택 바 차트 (개수)
        pivot_df = df_labels.pivot(index='Cluster', columns='Label', values='Count').fillna(0)
        cluster_order = [f'Cluster {cid}' for cid in cluster_ids]
        pivot_df = pivot_df.reindex(cluster_order)
        
        bars = pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title(f'Layer {layer_idx} - Label Distribution by Cluster\n(Count)')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Count')
        ax1.legend(title='Label')
        ax1.tick_params(axis='x', rotation=45)
        
        # 각 바 위에 숫자 표시
        for container in ax1.containers:
            ax1.bar_label(container, label_type='center', fontsize=10, color='white', weight='bold')
        
        # 스택 바 차트 (비율)
        pivot_df_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        bars2 = pivot_df_norm.plot(kind='bar', stacked=True, ax=ax2, color=['skyblue', 'lightcoral'])
        ax2.set_title(f'Layer {layer_idx} - Label Proportion by Cluster\n(Ratio)')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Proportion')
        ax2.legend(title='Label')
        ax2.tick_params(axis='x', rotation=45)
        
        # 비율 표시
        for container in ax2.containers:
            ax2.bar_label(container, labels=[f'{v:.2f}' if v > 0.05 else '' for v in container.datavalues], 
                        label_type='center', fontsize=9, color='white', weight='bold')
        
        # Chi-square 결과 표시
        if stats_results['label_chi2']:
            chi2_result = stats_results['label_chi2']
            fig.suptitle(f'Chi-square test: p={chi2_result["p_value"]:.4f} ' + 
                        ('(Significant)' if chi2_result['significant'] else '(Not Significant)'), 
                        fontsize=14)
        
        # 통계 정보 표시
        total_samples = sum(len(labels) for labels in cluster_label_data.values())
        stats_text = f"Total samples: {total_samples}\n"
        stats_text += f"Clusters: {len(cluster_ids)}\n"
        stats_text += f"Labels: {len(all_labels)}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_label_distribution_only.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Label distribution plot saved for layer {layer_idx}")
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 JSON 직렬화 가능한 Python 타입으로 변환"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    def _save_layer_analysis_results(self, cluster_label_data, all_labels, stats_results, layer_idx, output_dir):
        """레이어 분석 결과를 JSON으로 저장"""
        results = {
            'layer_idx': layer_idx,
            'total_samples': sum(len(labels) for labels in cluster_label_data.values()),
            'n_clusters': len(cluster_label_data),
            'n_labels': len(all_labels),
            'cluster_summary': {},
            'statistical_tests': stats_results
        }
        
        # 클러스터별 요약 정보
        for cluster_id, labels in cluster_label_data.items():
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_distribution = {}
            for label, count in zip(unique_labels, counts):
                label_distribution[f'label_{int(label)}'] = int(count)
            
            results['cluster_summary'][f'cluster_{cluster_id}'] = {
                'n_samples': len(labels),
                'sample_percentage': (len(labels) / results['total_samples']) * 100,
                'label_distribution': label_distribution
            }
        
        # numpy 타입 변환
        results = self._convert_numpy_types(results)
        
        # JSON 저장
        results_file = output_dir / f'layer_{layer_idx}_label_distribution_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Label distribution results saved to {results_file}")

    def cluster_entropy_metric(self, cluster_label_data):
        """클러스터별 엔트로피 기반 메트릭 계산"""
        if len(cluster_label_data) == 0:
            return {
                'mean_entropy': 0.0,
                'weighted_mean_entropy': 0.0,
                'entropy_score': 1.0,
                'cluster_entropies': []
            }
        
        cluster_entropies = []
        cluster_sizes = []
        total_samples = 0
        
        for cluster_id, labels in cluster_label_data.items():
            if len(labels) == 0:
                continue
                
            # 라벨 비율 계산
            label_0_count = labels.count(0)
            label_1_count = labels.count(1)
            total = len(labels)
            total_samples += total
            
            label_0_ratio = label_0_count / total
            label_1_ratio = label_1_count / total
            
            # 엔트로피 계산 (비트 단위)
            entropy = 0.0
            if label_0_ratio > 0:
                entropy -= label_0_ratio * np.log2(label_0_ratio)
            if label_1_ratio > 0:
                entropy -= label_1_ratio * np.log2(label_1_ratio)
            
            cluster_entropies.append({
                'cluster_id': cluster_id,
                'entropy': entropy,
                'size': total,
                'label_0_ratio': label_0_ratio,
                'label_1_ratio': label_1_ratio,
                'purity': max(label_0_ratio, label_1_ratio)
            })
            cluster_sizes.append(total)
        
        if len(cluster_entropies) == 0:
            return {
                'mean_entropy': 0.0,
                'weighted_mean_entropy': 0.0,
                'entropy_score': 1.0,
                'cluster_entropies': []
            }
        
        # 평균 엔트로피 (단순 평균)
        entropies = [c['entropy'] for c in cluster_entropies]
        mean_entropy = np.mean(entropies)
        
        # 가중 평균 엔트로피 (클러스터 크기로 가중)
        weighted_entropy = sum(c['entropy'] * c['size'] for c in cluster_entropies) / total_samples
        
        # 엔트로피 점수 (0=최악, 1=최고)
        entropy_score = 1.0 - weighted_entropy
        
        return {
            'mean_entropy': mean_entropy,
            'weighted_mean_entropy': weighted_entropy,
            'entropy_score': entropy_score,
            'n_clusters': len(cluster_entropies),
            'cluster_entropies': cluster_entropies
        }

    def cluster_label_specialization_index(self, cluster_label_data):
        """Cluster Label Specialization Index (CLSI) 계산"""
        if len(cluster_label_data) == 0:
            return {
                'clsi': 0.0,
                'mean_bias': 0.0,
                'diversity': 0.0,
                'cluster_details': []
            }
        
        cluster_biases = []
        label_0_ratios = []
        cluster_details = []
        
        for cluster_id, labels in cluster_label_data.items():
            if len(labels) == 0:
                continue
                
            # 라벨 비율 계산
            label_0_count = labels.count(0)
            label_1_count = labels.count(1)
            total = len(labels)
            
            label_0_ratio = label_0_count / total
            label_1_ratio = label_1_count / total
            
            # 편향도: 0.5에서 얼마나 멀리 떨어져 있는지 (0~1 범위)
            bias = abs(label_0_ratio - 0.5) * 2
            cluster_biases.append(bias)
            label_0_ratios.append(label_0_ratio)
            
            cluster_details.append({
                'cluster_id': cluster_id,
                'size': total,
                'label_0_ratio': label_0_ratio,
                'label_1_ratio': label_1_ratio,
                'bias': bias,
                'dominant_label': 0 if label_0_ratio > 0.5 else 1,
                'confidence': max(label_0_ratio, label_1_ratio)
            })
        
        if len(cluster_biases) == 0:
            return {
                'clsi': 0.0,
                'mean_bias': 0.0,
                'diversity': 0.0,
                'cluster_details': []
            }
        
        # 평균 편향도
        mean_bias = np.mean(cluster_biases)
        
        # 특화 다양성
        if len(label_0_ratios) <= 1:
            diversity = 0.0
        else:
            diversity = np.var(label_0_ratios) * 4
            diversity = min(diversity, 1.0)
        
        # CLSI 계산
        clsi = mean_bias * (1 + diversity / 2)
        
        return {
            'clsi': clsi,
            'mean_bias': mean_bias,
            'diversity': diversity,
            'n_clusters': len(cluster_biases),
            'cluster_details': cluster_details
        }

    def analyze_layer_specialization(self, output_base_dir, k_values=None):
        """모든 레이어의 CLSI 및 엔트로피 계산 및 시각화"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        clsi_results = {}
        entropy_results = {}
        detailed_results = {}
        
        logger.info("Starting CLSI and Entropy analysis...")
        
        for layer_idx in range(self.num_layers):
            # 클러스터링 수행 (아직 안 했다면)
            k = k_values[layer_idx] if k_values else None
            if layer_idx not in self.layer_results:
                self.perform_clustering_for_layer(layer_idx, k=k)
            
            layer_data = self.layer_results[layer_idx]
            clusters = layer_data['clusters']
            
            # 클러스터별 라벨 데이터 준비
            cluster_label_data = {}
            for cluster_id, samples in clusters.items():
                labels = [sample['label'] for sample in samples]
                cluster_label_data[cluster_id] = labels
            
            # CLSI 계산
            clsi_result = self.cluster_label_specialization_index(cluster_label_data)
            clsi_results[layer_idx] = clsi_result['clsi']
            
            # 엔트로피 계산
            entropy_result = self.cluster_entropy_metric(cluster_label_data)
            entropy_results[layer_idx] = entropy_result
            
            # 결합 결과 저장
            detailed_results[layer_idx] = {
                'clsi': clsi_result,
                'entropy': entropy_result
            }
            
            logger.info(f"Layer {layer_idx}: CLSI = {clsi_result['clsi']:.4f}, "
                       f"Entropy Score = {entropy_result['entropy_score']:.4f}")
        
        # 시각화
        self._plot_clsi_entropy_progression(clsi_results, entropy_results, detailed_results, output_base_dir)
        
        # 상세 리포트 저장
        self._save_clsi_entropy_detailed_report(detailed_results, output_base_dir)
        
        return clsi_results, entropy_results, detailed_results

    def _plot_clsi_entropy_progression(self, clsi_results, entropy_results, detailed_results, output_dir):
        """레이어별 CLSI와 엔트로피 진행 상황 시각화"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        layers = sorted(clsi_results.keys())
        clsi_scores = [clsi_results[layer] for layer in layers]
        entropy_scores = [entropy_results[layer]['entropy_score'] for layer in layers]
        mean_entropies = [entropy_results[layer]['mean_entropy'] for layer in layers]
        
        # 1. 레이어별 CLSI 점수
        ax = axes[0, 0]
        colors = ['red', 'orange', 'green'] if len(layers) == 3 else plt.cm.viridis(np.linspace(0, 1, len(layers)))
        bars = ax.bar(layers, clsi_scores, color=colors, alpha=0.7)
        ax.set_title('CLSI Score by Layer', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('CLSI Score')
        ax.set_ylim(0, max(1.5, max(clsi_scores) * 1.1))
        ax.grid(True, alpha=0.3)
        
        # 점수 표시
        for bar, score in zip(bars, clsi_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 2. 레이어별 엔트로피 점수
        ax = axes[0, 1]
        bars2 = ax.bar(layers, entropy_scores, color='purple', alpha=0.7)
        ax.set_title('Entropy Score by Layer', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy Score (1-entropy)')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 점수 표시
        for bar, score in zip(bars2, entropy_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 3. 클러스터 순도(Purity) 분포
        ax = axes[0, 2]
        for i, layer_idx in enumerate(layers):
            entropy_data = detailed_results[layer_idx]['entropy']
            purities = [c['purity'] for c in entropy_data['cluster_entropies']]
            if purities:
                ax.hist(purities, alpha=0.6, label=f"Layer {layer_idx}", bins=10, density=True)
        
        ax.set_xlabel('Cluster Purity (max label ratio)')
        ax.set_ylabel('Density')
        ax.set_title('Cluster Purity Distribution by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 1.0)
        
        # 4. 평균 엔트로피 값
        ax = axes[1, 0]
        bars5 = ax.bar(layers, mean_entropies, color='orange', alpha=0.7)
        ax.set_title('Mean Entropy by Layer', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy (bits)')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 엔트로피 값 표시
        for bar, entropy in zip(bars5, mean_entropies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 5. 클러스터별 엔트로피 분포
        ax = axes[1, 1]
        for i, layer_idx in enumerate(layers):
            entropy_data = detailed_results[layer_idx]['entropy']
            entropies = [c['entropy'] for c in entropy_data['cluster_entropies']]
            
            if entropies:
                ax.hist(entropies, alpha=0.6, label=f'Layer {layer_idx}', bins=10, density=True)
        
        ax.set_xlabel('Cluster Entropy (bits)')
        ax.set_ylabel('Density')
        ax.set_title('Cluster Entropy Distribution by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 요약 정보
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Analysis Summary\n\n"
        summary_text += f"Layers analyzed: {len(layers)}\n\n"
        
        summary_text += "CLSI Results:\n"
        summary_text += f"Range: {min(clsi_scores):.3f} - {max(clsi_scores):.3f}\n"
        for layer_idx in layers:
            summary_text += f"Layer {layer_idx}: {clsi_results[layer_idx]:.3f}\n"
        
        summary_text += "\nEntropy Results:\n"
        summary_text += f"Score Range: {min(entropy_scores):.3f} - {max(entropy_scores):.3f}\n"
        for layer_idx in layers:
            entropy_score = entropy_results[layer_idx]['entropy_score']
            mean_entropy = entropy_results[layer_idx]['mean_entropy']
            summary_text += f"Layer {layer_idx}: {entropy_score:.3f} (H={mean_entropy:.3f})\n"
        
        if len(layers) > 1:
            clsi_improvement = clsi_scores[-1] - clsi_scores[0]
            entropy_improvement = entropy_scores[-1] - entropy_scores[0]
            summary_text += f"\nOverall Improvements:\n"
            summary_text += f"CLSI: {clsi_improvement:+.3f}\n"
            summary_text += f"Entropy: {entropy_improvement:+.3f}\n"
        
        # 해석 가이드
        summary_text += "\nInterpretation:\n"
        summary_text += "CLSI: Higher = Better specialization\n"
        summary_text += "Entropy Score: Higher = Lower uncertainty\n"
        summary_text += "Mean Entropy: Lower = More specialized\n"
        summary_text += "Purity: Higher = More label-specific\n\n"
        summary_text += "Perfect cluster: CLSI=1.5, Entropy=0.0, Purity=1.0"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('CLSI and Entropy Analysis\n(GAT Layer Evolution)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'clsi_entropy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("CLSI and Entropy analysis plot saved")

    def _save_clsi_entropy_detailed_report(self, detailed_results, output_dir):
        """CLSI와 엔트로피 상세 리포트를 JSON으로 저장"""
        
        # numpy 타입 변환
        detailed_results_json = self._convert_numpy_types(detailed_results)
        
        # 요약 추가
        layers = sorted(detailed_results.keys())
        clsi_scores = [detailed_results[layer]['clsi']['clsi'] for layer in layers]
        entropy_scores = [detailed_results[layer]['entropy']['entropy_score'] for layer in layers]
        mean_entropies = [detailed_results[layer]['entropy']['mean_entropy'] for layer in layers]
        
        summary = {
            'analysis_summary': {
                'layers_analyzed': layers,
                'clsi_scores': dict(zip(layers, clsi_scores)),
                'entropy_scores': dict(zip(layers, entropy_scores)),
                'mean_entropies': dict(zip(layers, mean_entropies)),
                'clsi_range': [min(clsi_scores), max(clsi_scores)],
                'entropy_score_range': [min(entropy_scores), max(entropy_scores)],
                'overall_clsi_improvement': clsi_scores[-1] - clsi_scores[0] if len(clsi_scores) > 1 else 0.0,
                'overall_entropy_improvement': entropy_scores[-1] - entropy_scores[0] if len(entropy_scores) > 1 else 0.0
            },
            'detailed_results': detailed_results_json
        }
        
        # JSON 저장
        results_file = output_dir / 'clsi_entropy_detailed_report.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"CLSI and Entropy detailed report saved to {results_file}")

    def analyze_all_layers(self, output_base_dir, k_values=None):
        """모든 레이어에 대해 라벨 분포 분석 수행"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for layer_idx in range(self.num_layers):
            logger.info(f"Starting label distribution analysis for layer {layer_idx}...")
            
            # 레이어별 출력 디렉토리
            layer_output_dir = output_base_dir / f'layer_{layer_idx}_analysis'
            
            # k 값 설정
            k = k_values[layer_idx] if k_values else None
            
            # 분석 수행
            cluster_data, stats_results = self.analyze_layer_label_distribution(layer_idx, layer_output_dir, k=k)
            all_results[layer_idx] = {
                'cluster_data': cluster_data,
                'stats_results': stats_results
            }
        
        # 전체 레이어 비교 시각화
        self._create_cross_layer_comparison(all_results, output_base_dir)
        
        logger.info("All layer label distribution analysis completed!")
        return all_results

    def _create_cross_layer_comparison(self, all_results, output_dir):
        """레이어 간 라벨 분포 비교 시각화"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        layers = sorted(all_results.keys())
        chi2_pvalues = []
        n_clusters_list = []
        clsi_scores = []
        
        for layer_idx in layers:
            stats = all_results[layer_idx]['stats_results']
            cluster_data = all_results[layer_idx]['cluster_data']
            
            # Chi-square p-value
            if stats['label_chi2']:
                p_val = max(stats['label_chi2']['p_value'], 1e-50)
                chi2_pvalues.append(p_val)
            else:
                chi2_pvalues.append(1.0)
            
            # 클러스터 수
            n_clusters_list.append(len(cluster_data))
            
            # CLSI 계산
            clsi_result = self.cluster_label_specialization_index(cluster_data)
            clsi_scores.append(clsi_result['clsi'])
        
        # 1. Chi-square p-value 시각화
        bars1 = ax1.bar([f'Layer {l}' for l in layers], chi2_pvalues, 
                    color=['red' if p < 0.05 else 'lightcoral' for p in chi2_pvalues])
        ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='alpha=0.05')
        ax1.set_title('Chi-square Test p-values\n(Label Distribution Differences)')
        ax1.set_ylabel('p-value')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-50, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # p < 0.05인 경우 별표 표시
        for i, p in enumerate(chi2_pvalues):
            if p < 0.05:
                ax1.text(i, max(p, 1e-45), '*', ha='center', va='bottom', fontsize=16, color='white')
        
        # 2. 레이어별 클러스터 수
        bars2 = ax2.bar([f'Layer {l}' for l in layers], n_clusters_list, color='lightblue')
        ax2.set_title('Number of Clusters by Layer')
        ax2.set_ylabel('Number of Clusters')
        ax2.grid(True, alpha=0.3)
        
        # 클러스터 수 표시
        for bar, count in zip(bars2, n_clusters_list):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # 3. CLSI 점수 비교
        bars3 = ax3.bar([f'Layer {l}' for l in layers], clsi_scores, color='darkgreen', alpha=0.8)
        ax3.set_title('CLSI Score by Layer')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('CLSI Score')
        ax3.grid(True, alpha=0.3)
        
        # CLSI 점수 표시
        for bar, score in zip(bars3, clsi_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. 클러스터 편향도 분포 (모든 레이어)
        for layer_idx in layers:
            cluster_data = all_results[layer_idx]['cluster_data']
            clsi_result = self.cluster_label_specialization_index(cluster_data)
            biases = [cluster['bias'] for cluster in clsi_result['cluster_details']]
            
            if biases:
                ax4.hist(biases, alpha=0.6, label=f'Layer {layer_idx}', bins=10, density=True)
        
        ax4.set_xlabel('Cluster Bias (0=balanced, 1=completely biased)')
        ax4.set_ylabel('Density')
        ax4.set_title('Cluster Bias Distribution by Layer')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Layer Label Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / 'cross_layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Cross-layer analysis plot saved")

    def print_cluster_summary(self, layer_idx=None, include_clsi=True):
        """클러스터 요약 정보를 콘솔에 출력 (CLSI 포함)"""
        if layer_idx is not None:
            layers_to_print = [layer_idx] if layer_idx < self.num_layers else []
        else:
            layers_to_print = list(range(self.num_layers))
        
        for layer_idx in layers_to_print:
            if layer_idx not in self.layer_results:
                print(f"Layer {layer_idx}: No clustering results available")
                continue
                
            layer_data = self.layer_results[layer_idx]
            clusters = layer_data['clusters']
            
            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx} SUMMARY")
            print(f"{'='*60}")
            print(f"Total samples: {layer_data['total_samples']}")
            print(f"Number of clusters: {layer_data['n_clusters']}")
            
            # CLSI 계산 및 출력
            if include_clsi:
                cluster_label_data = {}
                for cluster_id, samples in clusters.items():
                    labels = [sample['label'] for sample in samples]
                    cluster_label_data[cluster_id] = labels
                
                clsi_result = self.cluster_label_specialization_index(cluster_label_data)
                print(f"\nCLSI: {clsi_result['clsi']:.4f}")
                print(f"Mean Bias: {clsi_result['mean_bias']:.4f}")
                print(f"Diversity: {clsi_result['diversity']:.4f}")
            
            print(f"\nCluster Details:")
            for cluster_id, samples in clusters.items():
                labels = [sample['label'] for sample in samples]
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                print(f"\nCluster {cluster_id}: {len(samples)} samples")
                for label, count in zip(unique_labels, counts):
                    percentage = (count / len(samples)) * 100
                    print(f"  Label {label}: {count:3d} samples ({percentage:5.1f}%)")
                
                # 클러스터 편향도 표시
                if len(samples) > 0:
                    label_0_ratio = labels.count(0) / len(labels)
                    bias = abs(label_0_ratio - 0.5) * 2
                    dominant_label = 0 if label_0_ratio > 0.5 else 1
                    confidence = max(label_0_ratio, 1 - label_0_ratio)
                    print(f"  Bias: {bias:.3f}, Dominant: Label {dominant_label} ({confidence:.1%})")

def extract_config_from_attention_path(attention_map_dir):
    """Attention map 디렉토리 경로에서 설정 정보 추출"""
    path_parts = Path(attention_map_dir).parts
    config_str = path_parts[-1]
    return config_str

def main():
    parser = argparse.ArgumentParser(description='Attention Map Based Cluster Analysis with Improved CLSI')
    parser.add_argument('--attention_map_dir', type=str, required=True,
                       help='Directory containing saved attention maps')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--layer_idx', type=int, default=None,
                       help='Specific layer to analyze (default: analyze all layers)')
    parser.add_argument('--k_values', type=str, default=None,
                       help='Comma-separated k values for each layer (e.g., "3,4,5")')
    parser.add_argument('--print_summary', action='store_true',
                       help='Print cluster summary to console')
    parser.add_argument('--clsi_analysis', action='store_true',
                       help='Perform CLSI analysis')
    
    args = parser.parse_args()
    
    # 경로 검증
    attention_map_dir = Path(args.attention_map_dir)
    if not attention_map_dir.exists():
        logger.error(f"Attention map directory not found: {attention_map_dir}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        config_str = extract_config_from_attention_path(attention_map_dir)
        output_dir = Path(str(attention_map_dir).replace('/attention_map/', '/visualization/') + '_cluster_analysis')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # k 값 파싱
    k_values = None
    if args.k_values:
        try:
            k_values = [int(k.strip()) for k in args.k_values.split(',')]
            logger.info(f"Using specified k values: {k_values}")
        except ValueError:
            logger.error("Invalid k_values format. Use comma-separated integers (e.g., '3,4,5')")
            return
    
    # 설정 정보 추출 및 로깅
    config_str = extract_config_from_attention_path(attention_map_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 ATTENTION MAP BASED CLUSTER ANALYSIS WITH CLSI")
    logger.info(f"{'='*80}")
    logger.info(f"📁 Input: {attention_map_dir}")
    logger.info(f"📊 Config: {config_str}")
    logger.info(f"💾 Output: {output_dir}")
    if k_values:
        logger.info(f"🎯 K values: {k_values}")
    logger.info(f"{'='*80}")
    
    try:
        # 분석기 초기화
        analyzer = AttentionMapClusterAnalyzer(attention_map_dir)
        
        # 요약 정보 출력 (옵션)
        if args.print_summary:
            analyzer.print_cluster_summary(args.layer_idx, include_clsi=True)
        
        if args.layer_idx is not None:
            # 특정 레이어만 분석
            logger.info(f"Analyzing layer {args.layer_idx}...")
            k = k_values[args.layer_idx] if k_values and args.layer_idx < len(k_values) else None
            layer_output_dir = output_dir / f'layer_{args.layer_idx}_analysis'
            analyzer.analyze_layer_label_distribution(args.layer_idx, layer_output_dir, k=k)
        else:
            # 모든 레이어 분석
            logger.info("Analyzing all layers...")
            analyzer.analyze_all_layers(output_dir, k_values=k_values)

        # CLSI 분석 (기본적으로 수행)
        if args.clsi_analysis or args.layer_idx is None:
            logger.info("Starting CLSI analysis...")
            clsi_results, entropy_results, detailed_results = analyzer.analyze_layer_specialization(output_dir, k_values=k_values)
            
            print("\n" + "="*70)
            print("CLSI (Cluster Label Specialization Index) Results")
            print("="*70)
            print("Formula: CLSI = bias × (1 + diversity/2)")
            print("Range: 0.0 (no specialization) ~ 1.5 (perfect specialization)")
            print("-" * 70)
            
            layers = sorted(clsi_results.keys())
            for layer in layers:
                result = detailed_results[layer]
                clsi_score = result['clsi']['clsi']
                bias = detailed_results[layer]['clsi']['mean_bias']
                diversity = detailed_results[layer]['clsi']['diversity']
                
                print(f"Layer {layer:2d}: CLSI={clsi_score:.3f} | Bias={bias:.3f}, Diversity={diversity:.3f}")
            
            if len(clsi_results) > 1:
                overall_improvement = clsi_results[layers[-1]] - clsi_results[layers[0]]
                print(f"\nOverall CLSI progression: {overall_improvement:+.4f}")
                
                if overall_improvement > 0.3:
                    print("Strong specialization progression detected!")
                    print("GAT layers are effectively learning label-specific patterns")
                elif overall_improvement > 0.15:
                    print("Moderate specialization progression detected.")
                    print("Some improvement in label specialization across layers")
                else:
                    print("Limited specialization progression.")
                    print("Consider adjusting model architecture or training strategy")

        logger.info(f"🎉 Cluster analysis completed successfully!")
        logger.info(f"📊 Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())