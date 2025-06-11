"""
간소화된 클러스터 분석 스크립트

inference.py에서 생성된 클러스터링 결과를 분석하여
저장된 NPZ 파일만을 참조하여 클러스터별 라벨 분포와 비율을 시각화합니다.
예측값이나 성능 메트릭 분석은 포함하지 않습니다.

Usage:
   python analysis1.py --clustering_dir /path/to/clustering/results
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
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedClusterAnalyzer:
   def __init__(self, clustering_dir):
       """
       Args:
           clustering_dir (str): clustering results 디렉토리 경로
       """
       self.clustering_dir = Path(clustering_dir)
       self.layer_results = {}
       
       # 클러스터링 결과 로드
       self._load_clustering_results()
       
       logger.info(f"SimplifiedClusterAnalyzer initialized for {len(self.layer_results)} layers")
   
   def _load_clustering_results(self):
       """클러스터링 결과 디렉토리에서 NPZ 데이터만 로드"""
       
       # clustering_results 폴더 확인
       clustering_results_dir = self.clustering_dir / 'clustering_results'
       if not clustering_results_dir.exists():
           clustering_results_dir = self.clustering_dir
           logger.info("Using legacy clustering directory structure")
       else:
           logger.info("Using new clustering_results directory structure")
       
       # 각 레이어별 결과 로드
       for layer_dir in clustering_results_dir.glob('layer_*'):
           if not layer_dir.is_dir():
               continue
               
           layer_idx = int(layer_dir.name.split('_')[1])
           logger.info(f"Loading clustering results for layer {layer_idx}...")
           
           # 클러스터별 샘플 정보 수집
           cluster_data = {}
           total_samples = 0
           
           for cluster_dir in layer_dir.glob('cluster_*'):
               if not cluster_dir.is_dir():
                   continue
                   
               cluster_id = int(cluster_dir.name.split('_')[1])
               samples = []
               
               # 클러스터 내 모든 샘플 로드 (NPZ 파일명에서 직접 정보 추출)
               for sample_file in cluster_dir.glob('sample_*.npz'):
                   # 파일명에서 정보 추출: sample_{id}_label_{label}.npz
                   filename = sample_file.stem  # .npz 제거
                   parts = filename.split('_')
                   
                   try:
                       sample_id = int(parts[1])  # sample_부분 다음
                       label = int(parts[3])      # label_부분 다음
                       
                       # NPZ 파일도 로드해서 추가 정보 확인
                       data = np.load(sample_file)
                       
                       sample_info = {
                           'sample_id': sample_id,
                           'label': label,
                           'cluster_id': cluster_id,
                           'filename': sample_file.name,
                           'attention_map_shape': data['attention_map'].shape if 'attention_map' in data else None
                       }
                       samples.append(sample_info)
                       total_samples += 1
                       
                   except (ValueError, IndexError) as e:
                       logger.warning(f"Failed to parse filename {sample_file}: {e}")
                       continue
               
               cluster_data[cluster_id] = samples
               logger.info(f"  Cluster {cluster_id}: {len(samples)} samples")
           
           self.layer_results[layer_idx] = {
               'clusters': cluster_data,
               'total_samples': total_samples,
               'n_clusters': len(cluster_data)
           }
           
           logger.info(f"Layer {layer_idx}: {total_samples} samples in {len(cluster_data)} clusters")

   def analyze_layer_label_distribution(self, layer_idx, output_dir):
       """특정 레이어의 라벨 분포 분석 및 시각화 (라벨 분포만 분석)"""
       if layer_idx not in self.layer_results:
           logger.error(f"Layer {layer_idx} not found in clustering results")
           return
       
       output_dir = Path(output_dir)
       output_dir.mkdir(parents=True, exist_ok=True)
       
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
       
       # 통계적 검정 수행 (라벨 분포 차이만)
       stats_results = self._perform_chi_square_test(cluster_label_data, all_labels, layer_idx)
       
       # 시각화 (라벨 분포만)
       self._plot_label_distribution_simple(cluster_label_data, all_labels, stats_results, layer_idx, output_dir)
       
       # 결과 저장
       self._save_layer_analysis_results(cluster_label_data, all_labels, stats_results, layer_idx, output_dir)
       
       return cluster_label_data, stats_results
   
   def _perform_chi_square_test(self, cluster_label_data, all_labels, layer_idx):
       """Chi-square 검정 수행 (라벨 분포 차이만 검정)"""
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
       
       # Chi-square 검정 (라벨 분포의 클러스터 간 차이)
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
       """라벨 분포 시각화 (NPZ 데이터만 사용)"""
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

   def cluster_label_specialization_index(self, cluster_label_data):
       """
       Cluster Label Specialization Index (CLSI) 계산
       
       클러스터들이 라벨에 대해 얼마나 특화되어 있는지 측정
       0.0 = 모든 클러스터가 50:50 분포 (특화 안됨)
       1.0 = 클러스터들이 뚜렷하게 서로 다른 라벨로 특화됨
       
       Args:
           cluster_label_data: {cluster_id: [labels]} 형태의 딕셔너리
           
       Returns:
           float: CLSI 점수 (0.0 ~ 1.0)
       """
       if len(cluster_label_data) <= 1:
           return 0.0
       
       cluster_biases = []
       label_0_ratios = []
       
       for cluster_id, labels in cluster_label_data.items():
           if len(labels) == 0:
               continue
               
           # 라벨 비율 계산
           label_0_count = labels.count(0)
           label_1_count = labels.count(1)
           total = len(labels)
           
           label_0_ratio = label_0_count / total
           label_1_ratio = label_1_count / total
           
           # 편향도: 0.5에서 얼마나 멀리 떨어져 있는지
           bias = abs(label_0_ratio - 0.5) * 2
           cluster_biases.append(bias)
           label_0_ratios.append(label_0_ratio)
       
       if len(cluster_biases) == 0:
           return 0.0
       
       # 평균 편향도 (개별 클러스터들이 얼마나 특화되었는지)
       mean_bias = np.mean(cluster_biases)
       
       # 특화 다양성 (클러스터들이 서로 다른 방향으로 특화되었는지)
       if len(label_0_ratios) <= 1:
           diversity = 0.0
       else:
           # 라벨 비율의 분산 (0~0.25 범위를 0~1로 정규화)
           diversity = np.var(label_0_ratios) * 4
           diversity = min(diversity, 1.0)  # 1.0 초과 방지
       
       # 종합 점수 (편향도와 다양성의 조화 평균)
       if mean_bias + diversity == 0:
           clsi = 0.0
       else:
           clsi = 2 * (mean_bias * diversity) / (mean_bias + diversity)
       
       return clsi

   def analyze_layer_specialization(self, output_base_dir):
       """모든 레이어의 CLSI 계산 및 시각화"""
       output_base_dir = Path(output_base_dir)
       output_base_dir.mkdir(parents=True, exist_ok=True)
       
       clsi_results = {}
       detailed_results = {}
       
       logger.info("Starting CLSI (Cluster Label Specialization Index) analysis...")
       
       for layer_idx in sorted(self.layer_results.keys()):
           layer_data = self.layer_results[layer_idx]
           clusters = layer_data['clusters']
           
           # 클러스터별 라벨 데이터 준비
           cluster_label_data = {}
           for cluster_id, samples in clusters.items():
               labels = [sample['label'] for sample in samples]
               cluster_label_data[cluster_id] = labels
           
           # CLSI 계산
           clsi_score = self.cluster_label_specialization_index(cluster_label_data)
           clsi_results[layer_idx] = clsi_score
           
           # 상세 정보 수집
           cluster_details = []
           for cluster_id, labels in cluster_label_data.items():
               if len(labels) > 0:
                   label_0_count = labels.count(0)
                   label_1_count = labels.count(1)
                   total = len(labels)
                   label_0_ratio = label_0_count / total
                   bias = abs(label_0_ratio - 0.5) * 2
                   
                   cluster_details.append({
                       'cluster_id': cluster_id,
                       'size': total,
                       'label_0_ratio': label_0_ratio,
                       'label_1_ratio': 1 - label_0_ratio,
                       'bias': bias
                   })
           
           detailed_results[layer_idx] = {
               'clsi_score': clsi_score,
               'n_clusters': len(cluster_label_data),
               'cluster_details': cluster_details
           }
           
           logger.info(f"Layer {layer_idx}: CLSI = {clsi_score:.4f}")
       
       # 시각화
       self._plot_clsi_progression(clsi_results, detailed_results, output_base_dir)
       
       # 상세 리포트 저장
       self._save_clsi_detailed_report(detailed_results, output_base_dir)
       
       return clsi_results, detailed_results

   def _plot_clsi_progression(self, clsi_results, detailed_results, output_dir):
       """레이어별 CLSI 진행 상황 시각화"""
       
       fig, axes = plt.subplots(2, 2, figsize=(16, 12))
       
       layers = sorted(clsi_results.keys())
       scores = [clsi_results[layer] for layer in layers]
       
       # 1. 레이어별 CLSI 점수
       ax = axes[0, 0]
       colors = ['red', 'orange', 'green'] if len(layers) == 3 else plt.cm.viridis(np.linspace(0, 1, len(layers)))
       bars = ax.bar(layers, scores, color=colors, alpha=0.7)
       ax.set_title('Cluster Label Specialization Index by Layer', fontsize=14, fontweight='bold')
       ax.set_xlabel('Layer')
       ax.set_ylabel('CLSI Score')
       ax.set_ylim(0, 1)
       ax.grid(True, alpha=0.3)
       
       # 점수 표시
       for bar, score in zip(bars, scores):
           ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
       
       # 2. 레이어 간 개선도
       ax = axes[0, 1]
       if len(layers) > 1:
           improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
           transition_labels = [f'L{layers[i]}→L{layers[i+1]}' for i in range(len(layers)-1)]
           
           colors = ['green' if imp > 0 else 'red' for imp in improvements]
           bars2 = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
           ax.set_title('CLSI Improvement Between Layers', fontsize=14, fontweight='bold')
           ax.set_xlabel('Layer Transition')
           ax.set_ylabel('CLSI Improvement')
           ax.set_xticks(range(len(improvements)))
           ax.set_xticklabels(transition_labels)
           ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
           ax.grid(True, alpha=0.3)
           
           # 개선도 값 표시
           for bar, imp in zip(bars2, improvements):
               ax.text(bar.get_x() + bar.get_width()/2, 
                      bar.get_height() + (0.01 if imp > 0 else -0.02),
                      f'{imp:+.3f}', ha='center', 
                      va='bottom' if imp > 0 else 'top', fontweight='bold')
       else:
           ax.text(0.5, 0.5, 'Need at least 2 layers\nfor improvement analysis', 
                  ha='center', va='center', transform=ax.transAxes, fontsize=12)
           ax.set_title('CLSI Improvement Between Layers')
       
       # 3. 클러스터 편향도 분포
       ax = axes[1, 0]
       for i, layer_idx in enumerate(layers):
           layer_details = detailed_results[layer_idx]
           biases = [cluster['bias'] for cluster in layer_details['cluster_details']]
           
           if biases:
               ax.hist(biases, alpha=0.6, label=f'Layer {layer_idx}', bins=10, density=True)
       
       ax.set_xlabel('Cluster Bias (0=균등, 1=완전편향)')
       ax.set_ylabel('Density')
       ax.set_title('Cluster Bias Distribution by Layer', fontsize=14, fontweight='bold')
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       # 4. 요약 정보
       ax = axes[1, 1]
       ax.axis('off')
       
       summary_text = "CLSI Analysis Summary\n\n"
       summary_text += f"Layers analyzed: {len(layers)}\n"
       summary_text += f"CLSI Range: {min(scores):.3f} - {max(scores):.3f}\n\n"
       
       summary_text += "Layer-wise CLSI:\n"
       for layer_idx in layers:
           score = clsi_results[layer_idx]
           n_clusters = detailed_results[layer_idx]['n_clusters']
           summary_text += f"Layer {layer_idx}: {score:.3f} ({n_clusters} clusters)\n"
       
       if len(layers) > 1:
           overall_improvement = scores[-1] - scores[0]
           summary_text += f"\nOverall improvement: {overall_improvement:+.3f}\n"
           
           if overall_improvement > 0.1:
               summary_text += "✅ Strong specialization progression"
           elif overall_improvement > 0.05:
               summary_text += "🔄 Moderate specialization progression"
           else:
               summary_text += "⚠️ Limited specialization progression"
       
       # 해석 가이드
       summary_text += "\n\nCLSI Interpretation:\n"
       summary_text += "0.0-0.3: Low specialization\n"
       summary_text += "0.3-0.6: Moderate specialization\n"
       summary_text += "0.6-1.0: High specialization"
       
       ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
       
       plt.suptitle('Cluster Label Specialization Analysis\n(GAT Layer Evolution)', 
                   fontsize=16, fontweight='bold', y=0.98)
       plt.tight_layout()
       plt.savefig(output_dir / 'clsi_analysis.png', dpi=300, bbox_inches='tight')
       plt.close(fig)
       
       logger.info("CLSI analysis plot saved")

   def _save_clsi_detailed_report(self, detailed_results, output_dir):
       """CLSI 상세 리포트를 JSON으로 저장"""
       
       # numpy 타입 변환
       detailed_results_json = self._convert_numpy_types(detailed_results)
       
       # CLSI 요약 추가
       layers = sorted(detailed_results.keys())
       scores = [detailed_results[layer]['clsi_score'] for layer in layers]
       
       summary = {
           'clsi_summary': {
               'layers_analyzed': layers,
               'clsi_scores': dict(zip(layers, scores)),
               'min_clsi': min(scores),
               'max_clsi': max(scores),
               'clsi_range': max(scores) - min(scores),
               'overall_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0.0
           },
           'detailed_results': detailed_results_json
       }
       
       # JSON 저장
       results_file = output_dir / 'clsi_detailed_report.json'
       with open(results_file, 'w') as f:
           json.dump(summary, f, indent=2)
       
       logger.info(f"CLSI detailed report saved to {results_file}")
   
   def analyze_all_layers(self, output_base_dir):
       """모든 레이어에 대해 라벨 분포 분석 수행"""
       output_base_dir = Path(output_base_dir)
       output_base_dir.mkdir(parents=True, exist_ok=True)
       
       all_results = {}
       
       for layer_idx in sorted(self.layer_results.keys()):
           logger.info(f"Starting label distribution analysis for layer {layer_idx}...")
           
           # 레이어별 출력 디렉토리
           layer_output_dir = output_base_dir / f'layer_{layer_idx}_simple'
           
           # 분석 수행
           cluster_data, stats_results = self.analyze_layer_label_distribution(layer_idx, layer_output_dir)
           all_results[layer_idx] = {
               'cluster_data': cluster_data,
               'stats_results': stats_results
           }
       
       # 전체 레이어 비교 시각화
       self._create_cross_layer_comparison_simple(all_results, output_base_dir)
       
       logger.info("All layer label distribution analysis completed!")
       return all_results
   
   def _create_cross_layer_comparison_simple(self, all_results, output_dir):
       """레이어 간 라벨 분포 비교 시각화"""
       
       # 레이어별 Chi-square p-value 비교
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
       layers = sorted(all_results.keys())
       chi2_pvalues = []
       n_clusters_list = []
       
       for layer_idx in layers:
           stats = all_results[layer_idx]['stats_results']
           
           if stats['label_chi2']:
               p_val = max(stats['label_chi2']['p_value'], 1e-50)
               chi2_pvalues.append(p_val)
           else:
               chi2_pvalues.append(1.0)
           
           # 클러스터 수 추가
           n_clusters_list.append(len(all_results[layer_idx]['cluster_data']))
       
       # Chi-square p-value 시각화
       bars1 = ax1.bar([f'Layer {l}' for l in layers], chi2_pvalues, 
                   color=['red' if p < 0.05 else 'lightcoral' for p in chi2_pvalues])
       ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α=0.05')
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
       
       ## 레이어별 클러스터 수
       bars2 = ax2.bar([f'Layer {l}' for l in layers], n_clusters_list, color='lightblue')
       ax2.set_title('Number of Clusters by Layer')
       ax2.set_ylabel('Number of Clusters')
       ax2.grid(True, alpha=0.3)
       
       # 클러스터 수 표시
       for bar, count in zip(bars2, n_clusters_list):
           ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
       
       plt.suptitle('Cross-Layer Label Distribution Analysis', fontsize=16)
       plt.tight_layout()
       fig.savefig(output_dir / 'cross_layer_label_distribution_analysis.png', dpi=300, bbox_inches='tight')
       plt.close(fig)
       
       logger.info("Cross-layer label distribution analysis plot saved")
   
   def print_cluster_summary(self, layer_idx=None):
       """클러스터 요약 정보를 콘솔에 출력"""
       if layer_idx is not None:
           layers_to_print = [layer_idx] if layer_idx in self.layer_results else []
       else:
           layers_to_print = sorted(self.layer_results.keys())
       
       for layer_idx in layers_to_print:
           layer_data = self.layer_results[layer_idx]
           clusters = layer_data['clusters']
           
           print(f"\n{'='*50}")
           print(f"LAYER {layer_idx} SUMMARY")
           print(f"{'='*50}")
           print(f"Total samples: {layer_data['total_samples']}")
           print(f"Number of clusters: {layer_data['n_clusters']}")
           
           for cluster_id, samples in clusters.items():
               labels = [sample['label'] for sample in samples]
               unique_labels, counts = np.unique(labels, return_counts=True)
               
               print(f"\nCluster {cluster_id}: {len(samples)} samples")
               for label, count in zip(unique_labels, counts):
                   percentage = (count / len(samples)) * 100
                   print(f"  Label {label}: {count} samples ({percentage:.1f}%)")

def main():
   parser = argparse.ArgumentParser(description='Simplified Cluster Analysis')
   parser.add_argument('--clustering_dir', type=str, required=True,
                      help='Directory containing clustering results (e.g., clustering_7 folder)')
   parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for analysis results')
   parser.add_argument('--layer_idx', type=int, default=None,
                      help='Specific layer to analyze (default: analyze all layers)')
   parser.add_argument('--print_summary', action='store_true',
                      help='Print cluster summary to console')
   parser.add_argument('--clsi_analysis', action='store_true',
                      help='Perform CLSI (Cluster Label Specialization Index) analysis')
   
   args = parser.parse_args()
   
   # 출력 디렉토리 설정
   if args.output_dir is None:
       clustering_dir = Path(args.clustering_dir)
       args.output_dir = clustering_dir / 'label_analysis'
   
   # 분석기 초기화
   analyzer = SimplifiedClusterAnalyzer(args.clustering_dir)
   
   # 요약 정보 출력 (옵션)
   if args.print_summary:
       analyzer.print_cluster_summary(args.layer_idx)
   
   if args.layer_idx is not None:
       # 특정 레이어만 분석
       logger.info(f"Analyzing layer {args.layer_idx}...")
       output_dir = Path(args.output_dir) / f'layer_{args.layer_idx}_simple'
       analyzer.analyze_layer_label_distribution(args.layer_idx, output_dir)
   else:
       # 모든 레이어 분석
       logger.info("Analyzing all layers...")
       analyzer.analyze_all_layers(args.output_dir)

   # CLSI 분석 (옵션)
   if args.clsi_analysis:
       logger.info("Starting CLSI analysis...")
       clsi_results, detailed_results = analyzer.analyze_layer_specialization(args.output_dir)
       
       print("\n" + "="*60)
       print("CLSI (Cluster Label Specialization Index) Results")
       print("="*60)
       for layer, score in clsi_results.items():
           print(f"Layer {layer}: {score:.4f}")
       
       if len(clsi_results) > 1:
           layers = sorted(clsi_results.keys())
           improvement = clsi_results[layers[-1]] - clsi_results[layers[0]]
           print(f"\nOverall improvement: {improvement:+.4f}")
           
           if improvement > 0.1:
               print("✅ Strong specialization progression detected!")
           elif improvement > 0.05:
               print("🔄 Moderate specialization progression detected.")
           else:
               print("⚠️ Limited specialization progression.")

   logger.info(f"Label distribution analysis completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
   main()