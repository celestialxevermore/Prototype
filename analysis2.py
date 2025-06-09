"""
클러스터 성능 영향 분석 스크립트

cluster_analysis.py에서 생성된 기본 분석 결과를 바탕으로
각 클러스터가 모델 성능에 미치는 실질적 영향을 심층 분석합니다.

Features:
- 클러스터별 예측 기여도 분석
- 오분류 패턴 상세 분석  
- 모델 신뢰도 및 calibration 분석
- What-if 시뮬레이션 (클러스터 제거 영향)
- Attention 패턴 분석

Usage:
    python cluster_performance_impact.py --clustering_dir /path/to/clustering/results --checkpoint_dir /path/to/checkpoint.pt
    python cluster_performance_impact.py --clustering_dir /path/to/clustering/results --checkpoint_dir /path/to/checkpoint.pt --layer_idx 5
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# inference.py에서 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterPerformanceAnalyzer:
    def __init__(self, clustering_dir, checkpoint_dir, device='cuda'):
        """
        Args:
            clustering_dir (str): inference.py 결과 디렉토리 경로
            checkpoint_dir (str): 모델 체크포인트 경로
            device (str): 'cuda' 또는 'cpu'
        """
        self.clustering_dir = Path(clustering_dir)
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트와 모델 로드
        self._load_model_and_data()
        
        # 클러스터링 결과 로드
        self._load_clustering_results()
        
        logger.info(f"ClusterPerformanceAnalyzer initialized for {len(self.layer_results)} layers")
    
    def _load_model_and_data(self):
        """모델과 데이터 로드"""
        self.checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        self.model = Model(
            self.args, 
            self.args.input_dim, 
            self.args.hidden_dim, 
            self.args.output_dim, 
            self.args.num_layers, 
            self.args.dropout_rate, 
            self.args.llm_model,
            "analysis",
            "analysis"
        ).to(self.device)
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        fix_seed(self.args.random_seed)
        results = prepare_embedding_dataloaders(self.args, self.args.source_dataset_name)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']

        logger.info("Model and data loaded successfully")
    
    def _load_clustering_results(self):
        """클러스터링 결과 디렉토리에서 데이터 로드 - 새로운 경로 구조에 맞게 수정"""
        self.layer_results = {}
        
        # clustering_results 폴더 확인
        clustering_results_dir = self.clustering_dir / 'clustering_results'
        if not clustering_results_dir.exists():
            # 구버전 호환성을 위해 clustering_dir 직접 확인
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
                
                # 클러스터 내 모든 샘플 로드
                for sample_file in cluster_dir.glob('sample_*.npz'):
                    data = np.load(sample_file)
                    sample_info = {
                        'sample_id': int(data['sample_id']),
                        'label': int(data['label']),
                        'cluster_id': cluster_id,
                        'attention_map': data['attention_map'],
                        'feature_names': data['feature_names'].tolist()
                    }
                    samples.append(sample_info)
                    total_samples += 1
                
                cluster_data[cluster_id] = samples
            
            self.layer_results[layer_idx] = {
                'clusters': cluster_data,
                'total_samples': total_samples,
                'n_clusters': len(cluster_data)
            }
            
            logger.info(f"Layer {layer_idx}: {total_samples} samples in {len(cluster_data)} clusters")
    
    def extract_predictions_and_labels(self, data_loader):
        """모든 샘플의 예측값과 라벨 추출"""
        predictions = {}
        labels = {}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                pred, _ = self._get_model_predictions(batch_on_device)
                batch_size = pred.shape[0]
                
                for sample_idx in range(batch_size):
                    if pred.shape[1] == 1:  # binary classification
                        pred_prob = torch.sigmoid(pred[sample_idx, 0]).cpu().item()
                    else:  # multi-class
                        pred_prob = torch.softmax(pred[sample_idx], dim=0).cpu().numpy()
                    
                    predictions[sample_count] = pred_prob
                    
                    if 'y' in batch:
                        labels[sample_count] = batch['y'][sample_idx].item()
                    
                    sample_count += 1
        
        logger.info(f"Extracted predictions for {sample_count} samples")
        return predictions, labels
    
    def _get_model_predictions(self, batch):
        """모델에서 예측값 추출"""
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

        desc_embeddings = torch.cat(desc_embeddings, dim=1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim=1)
        
        cls_token = self.model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        for i, layer in enumerate(self.model.layers):
            norm_x = self.model.layer_norms[i](x)
            attn_output, _ = layer(desc_embeddings, norm_x)
            x = x + attn_output
        
        pred = x[:, 0, :]
        pred = self.model.predictor(pred)
        return pred, None

    def analyze_cluster_performance_impact(self, layer_idx, output_dir):
        """특정 레이어의 클러스터 성능 영향 분석"""
        if layer_idx not in self.layer_results:
            logger.error(f"Layer {layer_idx} not found")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing performance impact for layer {layer_idx}...")
        
        # 예측값과 라벨 추출
        predictions, labels = self.extract_predictions_and_labels(self.train_loader)
        
        # 클러스터링 결과 구성
        clustering_results = self._prepare_clustering_data(layer_idx, predictions, labels)
        
        # 1. 예측 기여도 분석
        contribution_analysis = self._analyze_prediction_contribution(clustering_results)
        
        # 2. 오분류 패턴 분석
        misclassification_analysis = self._analyze_misclassification_patterns(clustering_results)
        
        # 3. 모델 신뢰도 분석
        confidence_analysis = self._analyze_model_confidence(clustering_results)
        
        # 4. 클러스터 제거 영향 시뮬레이션
        removal_impact = self._simulate_cluster_removal(clustering_results)
        
        # 5. Attention 패턴 분석
        attention_analysis = self._analyze_attention_patterns(layer_idx)
        
        # 6. 종합 분석 결과 생성
        comprehensive_results = {
            'layer_idx': layer_idx,
            'contribution_analysis': contribution_analysis,
            'misclassification_analysis': misclassification_analysis,
            'confidence_analysis': confidence_analysis,
            'removal_impact': removal_impact,
            'attention_analysis': attention_analysis,
            'summary': self._generate_performance_summary(
                contribution_analysis, misclassification_analysis, 
                confidence_analysis, removal_impact
            )
        }
        
        # 7. 시각화
        self._create_performance_visualizations(comprehensive_results, output_dir)
        
        # 8. 결과 저장
        self._save_results(comprehensive_results, output_dir)
        
        logger.info(f"Performance impact analysis completed for layer {layer_idx}")
        return comprehensive_results
    
    def _prepare_clustering_data(self, layer_idx, predictions, labels):
        """클러스터링 데이터 준비"""
        clusters = self.layer_results[layer_idx]['clusters']
        
        cluster_assignments = []
        cluster_sample_ids = []
        cluster_labels = []
        
        for cluster_id, samples in clusters.items():
            for sample in samples:
                sample_id = sample['sample_id']
                if sample_id in predictions:
                    cluster_assignments.append(cluster_id)
                    cluster_sample_ids.append(sample_id)
                    cluster_labels.append(labels.get(sample_id, sample['label']))
        
        return {
            'cluster_assignments': np.array(cluster_assignments),
            'sample_ids': np.array(cluster_sample_ids),
            'labels': np.array(cluster_labels),
            'predictions': predictions
        }
    
    def _analyze_prediction_contribution(self, clustering_results):
        """클러스터별 예측 기여도 분석"""
        logger.info("Analyzing prediction contribution...")
        
        cluster_assignments = clustering_results['cluster_assignments']
        sample_ids = clustering_results['sample_ids']
        labels = clustering_results['labels']
        predictions = clustering_results['predictions']
        
        contribution_results = {}
        
        # 전체 성능 계산
        all_preds = [predictions[sid] for sid in sample_ids]
        all_labels = labels
        overall_accuracy = np.mean((np.array(all_preds) > 0.5) == all_labels)
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            correct_predictions = 0
            total_predictions = len(cluster_indices)
            
            for idx in cluster_indices:
                sid = sample_ids[idx]
                pred = predictions[sid] > 0.5
                true_label = labels[idx]
                if pred == true_label:
                    correct_predictions += 1
            
            cluster_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            contribution_ratio = correct_predictions / len(all_preds) if len(all_preds) > 0 else 0
            weighted_contribution = (total_predictions / len(all_preds)) * cluster_accuracy if len(all_preds) > 0 else 0
            
            contribution_results[cluster_id] = {
                'cluster_accuracy': cluster_accuracy,
                'contribution_to_overall': contribution_ratio,
                'weighted_contribution': weighted_contribution,
                'size_ratio': total_predictions / len(all_preds),
                'efficiency': contribution_ratio / max(total_predictions / len(all_preds), 0.001),
                'sample_count': total_predictions,
                'correct_count': correct_predictions
            }
        
        return contribution_results
    
    def _analyze_misclassification_patterns(self, clustering_results):
        """클러스터별 오분류 패턴 분석"""
        logger.info("Analyzing misclassification patterns...")
        
        cluster_assignments = clustering_results['cluster_assignments']
        sample_ids = clustering_results['sample_ids']
        labels = clustering_results['labels']
        predictions = clustering_results['predictions']
        
        misclass_results = {}
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            fp_count = fn_count = tp_count = tn_count = 0
            fp_confidences = []
            fn_confidences = []
            
            for idx in cluster_indices:
                sid = sample_ids[idx]
                pred_prob = predictions[sid]
                pred_label = pred_prob > 0.5
                true_label = labels[idx]
                
                if pred_label == 1 and true_label == 0:
                    fp_count += 1
                    fp_confidences.append(pred_prob)
                elif pred_label == 0 and true_label == 1:
                    fn_count += 1
                    fn_confidences.append(pred_prob)
                elif pred_label == 1 and true_label == 1:
                    tp_count += 1
                else:
                    tn_count += 1
            
            total = fp_count + fn_count + tp_count + tn_count
            error_rate = (fp_count + fn_count) / total if total > 0 else 0
            
            misclass_results[cluster_id] = {
                'false_positive_count': fp_count,
                'false_negative_count': fn_count,
                'true_positive_count': tp_count,
                'true_negative_count': tn_count,
                'error_rate': error_rate,
                'fp_avg_confidence': np.mean(fp_confidences) if fp_confidences else 0,
                'fn_avg_confidence': np.mean(fn_confidences) if fn_confidences else 0,
                'precision': tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0,
                'recall': tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            }
        
        return misclass_results
    
    def _analyze_model_confidence(self, clustering_results):
        """클러스터별 모델 신뢰도 분석"""
        logger.info("Analyzing model confidence...")
        
        cluster_assignments = clustering_results['cluster_assignments']
        sample_ids = clustering_results['sample_ids']
        labels = clustering_results['labels']
        predictions = clustering_results['predictions']
        
        confidence_results = {}
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_predictions = []
            cluster_labels = []
            
            for idx in cluster_indices:
                sid = sample_ids[idx]
                cluster_predictions.append(predictions[sid])
                cluster_labels.append(labels[idx])
            
            if not cluster_predictions:
                continue
                
            cluster_predictions = np.array(cluster_predictions)
            cluster_labels = np.array(cluster_labels)
            
            # ECE (Expected Calibration Error) 계산
            ece = self._calculate_ece(cluster_predictions, cluster_labels)
            
            # Brier Score 계산
            brier_score = np.mean((cluster_predictions - cluster_labels) ** 2)
            
            # 신뢰도 구간별 분석
            high_conf_mask = (cluster_predictions > 0.8) | (cluster_predictions < 0.2)
            low_conf_mask = (cluster_predictions >= 0.2) & (cluster_predictions <= 0.8)
            
            high_conf_accuracy = 0
            low_conf_accuracy = 0
            
            if np.any(high_conf_mask):
                high_conf_preds = (cluster_predictions[high_conf_mask] > 0.5).astype(int)
                high_conf_labels = cluster_labels[high_conf_mask]
                high_conf_accuracy = np.mean(high_conf_preds == high_conf_labels)
            
            if np.any(low_conf_mask):
                low_conf_preds = (cluster_predictions[low_conf_mask] > 0.5).astype(int)
                low_conf_labels = cluster_labels[low_conf_mask]
                low_conf_accuracy = np.mean(low_conf_preds == low_conf_labels)
            
            confidence_results[cluster_id] = {
                'ece': ece,
                'brier_score': brier_score,
                'avg_confidence': np.mean(cluster_predictions),
                'confidence_std': np.std(cluster_predictions),
                'high_confidence_accuracy': high_conf_accuracy,
                'low_confidence_accuracy': low_conf_accuracy,
                'high_confidence_ratio': np.mean(high_conf_mask),
                'overconfident_ratio': np.mean(cluster_predictions > 0.8),
                'underconfident_ratio': np.mean(cluster_predictions < 0.2)
            }
        
        return confidence_results
    
    def _calculate_ece(self, predictions, labels, n_bins=10):
        """Expected Calibration Error 계산"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (labels[in_bin] == (predictions[in_bin] > 0.5)).mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _simulate_cluster_removal(self, clustering_results):
        """클러스터 제거 시뮬레이션"""
        logger.info("Simulating cluster removal impact...")
        
        cluster_assignments = clustering_results['cluster_assignments']
        sample_ids = clustering_results['sample_ids']
        labels = clustering_results['labels']
        predictions = clustering_results['predictions']
        
        # 전체 성능
        all_preds = [predictions[sid] for sid in sample_ids]
        overall_accuracy = np.mean((np.array(all_preds) > 0.5) == labels)
        
        removal_results = {}
        
        for cluster_id in np.unique(cluster_assignments):
            # 해당 클러스터 제외
            remaining_mask = cluster_assignments != cluster_id
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) == 0:
                continue
            
            remaining_preds = [predictions[sample_ids[idx]] for idx in remaining_indices]
            remaining_labels = labels[remaining_indices]
            
            remaining_accuracy = np.mean((np.array(remaining_preds) > 0.5) == remaining_labels)
            impact = remaining_accuracy - overall_accuracy
            
            removal_results[cluster_id] = {
                'accuracy_without_cluster': remaining_accuracy,
                'impact_on_accuracy': impact,
                'samples_removed': len(sample_ids) - len(remaining_indices),
                'relative_impact': impact / overall_accuracy if overall_accuracy > 0 else 0,
                'is_critical': impact < -0.05  # 5% 이상 성능 저하시 critical
            }
        
        return removal_results
    
    def _analyze_attention_patterns(self, layer_idx):
        """클러스터별 attention 패턴 분석"""
        logger.info("Analyzing attention patterns...")
        
        clusters = self.layer_results[layer_idx]['clusters']
        attention_results = {}
        
        for cluster_id, samples in clusters.items():
            attention_maps = []
            
            for sample in samples:
                if 'attention_map' in sample:
                    attention_maps.append(sample['attention_map'])
            
            if attention_maps:
                attention_maps = np.array(attention_maps)
                
                mean_attention = np.mean(attention_maps, axis=0)
                std_attention = np.std(attention_maps, axis=0)
                
                # Top-k 중요 features
                k = min(5, len(mean_attention))
                top_features = np.argsort(mean_attention)[-k:][::-1]
                
                # Attention 일관성
                attention_consistency = 1 - np.mean(std_attention / (mean_attention + 1e-8))
                
                attention_results[cluster_id] = {
                    'mean_attention': mean_attention.tolist(),
                    'top_features': top_features.tolist(),
                    'top_attention_values': mean_attention[top_features].tolist(),
                    'attention_consistency': attention_consistency,
                    'attention_entropy': self._calculate_attention_entropy(mean_attention)
                }
        
        return attention_results
    
    def _calculate_attention_entropy(self, attention_weights):
        """Attention 가중치의 엔트로피 계산"""
        # 정규화
        attention_weights = attention_weights / (np.sum(attention_weights) + 1e-8)
        # 엔트로피 계산
        entropy = -np.sum(attention_weights * np.log2(attention_weights + 1e-8))
        return entropy
    
    def _generate_performance_summary(self, contribution_analysis, misclassification_analysis, 
                                    confidence_analysis, removal_impact):
        """성능 분석 요약 생성"""
        cluster_ids = list(contribution_analysis.keys())
        
        # 최고 성능 클러스터들
        best_accuracy = max(cluster_ids, key=lambda x: contribution_analysis[x]['cluster_accuracy'])
        best_contributor = max(cluster_ids, key=lambda x: contribution_analysis[x]['weighted_contribution'])
        most_efficient = max(cluster_ids, key=lambda x: contribution_analysis[x]['efficiency'])
        
        # 문제 클러스터들
        highest_error = max(cluster_ids, key=lambda x: misclassification_analysis[x]['error_rate'])
        worst_calibrated = max(cluster_ids, key=lambda x: confidence_analysis[x]['ece'])
        critical_clusters = [cid for cid, data in removal_impact.items() if data['is_critical']]
        
        # 주요 인사이트
        insights = []
        
        # 기여도 관련 인사이트
        total_weighted_contrib = sum(data['weighted_contribution'] for data in contribution_analysis.values())
        top_contrib_ratio = contribution_analysis[best_contributor]['weighted_contribution'] / total_weighted_contrib
        
        if top_contrib_ratio > 0.5:
            insights.append(f"Cluster {best_contributor} dominates performance ({top_contrib_ratio:.1%} of total contribution)")
        
        # 오분류 패턴 인사이트
        total_fp = sum(data['false_positive_count'] for data in misclassification_analysis.values())
        total_fn = sum(data['false_negative_count'] for data in misclassification_analysis.values())
        
        if total_fp > total_fn * 1.5:
            insights.append("Model shows bias toward over-prediction (excessive false positives)")
        elif total_fn > total_fp * 1.5:
            insights.append("Model shows bias toward under-prediction (excessive false negatives)")
        
        # 임계 클러스터 인사이트
        if critical_clusters:
            insights.append(f"Critical clusters identified: {critical_clusters} (removal causes >5% accuracy drop)")
        
        return {
            'best_performing_clusters': {
                'highest_accuracy': best_accuracy,
                'best_contributor': best_contributor,
                'most_efficient': most_efficient
            },
            'problematic_clusters': {
                'highest_error_rate': highest_error,
                'worst_calibrated': worst_calibrated,
                'critical_for_performance': critical_clusters
            },
            'key_insights': insights,
            'overall_metrics': {
                'total_clusters': len(cluster_ids),
                'avg_cluster_accuracy': np.mean([data['cluster_accuracy'] for data in contribution_analysis.values()]),
                'performance_variance': np.var([data['cluster_accuracy'] for data in contribution_analysis.values()]),
                'total_error_rate': (total_fp + total_fn) / sum(data['sample_count'] for data in contribution_analysis.values())
            }
        }
    
    def _create_performance_visualizations(self, results, output_dir):
        """성능 분석 시각화"""
        layer_idx = results['layer_idx']
        
        # 1. 기여도 분석 시각화
        self._plot_contribution_analysis(results['contribution_analysis'], layer_idx, output_dir)
        
        # 2. 오분류 패턴 시각화
        self._plot_misclassification_analysis(results['misclassification_analysis'], layer_idx, output_dir)
        
        # 3. 신뢰도 분석 시각화
        self._plot_confidence_analysis(results['confidence_analysis'], layer_idx, output_dir)
        
        # 4. 제거 영향 시각화
        self._plot_removal_impact(results['removal_impact'], layer_idx, output_dir)
        
        # 5. 종합 대시보드
        self._create_performance_dashboard(results, output_dir)
    
    def _plot_contribution_analysis(self, contribution_analysis, layer_idx, output_dir):
        """기여도 분석 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_ids = list(contribution_analysis.keys())
        
        # 1. 절대 기여도
        contributions = [contribution_analysis[cid]['contribution_to_overall'] for cid in cluster_ids]
        bars1 = ax1.bar([f'C{cid}' for cid in cluster_ids], contributions, color='skyblue', alpha=0.8)
        ax1.set_title('Contribution to Overall Accuracy', fontsize=14, weight='bold')
        ax1.set_ylabel('Contribution Ratio')
        ax1.grid(True, alpha=0.3)
        
        for bar, contrib in zip(bars1, contributions):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{contrib:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 2. 가중 기여도
        weighted_contribs = [contribution_analysis[cid]['weighted_contribution'] for cid in cluster_ids]
        bars2 = ax2.bar([f'C{cid}' for cid in cluster_ids], weighted_contribs, color='lightgreen', alpha=0.8)
        ax2.set_title('Weighted Contribution (Size × Accuracy)', fontsize=14, weight='bold')
        ax2.set_ylabel('Weighted Contribution')
        ax2.grid(True, alpha=0.3)
        
        for bar, contrib in zip(bars2, weighted_contribs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{contrib:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. 효율성 vs 크기
        sizes = [contribution_analysis[cid]['size_ratio'] for cid in cluster_ids]
        efficiencies = [contribution_analysis[cid]['efficiency'] for cid in cluster_ids]
        
        scatter = ax3.scatter(sizes, efficiencies, s=120, alpha=0.7, c=range(len(cluster_ids)), cmap='viridis')
        for i, cid in enumerate(cluster_ids):
            ax3.annotate(f'C{cid}', (sizes[i], efficiencies[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Cluster Size Ratio')
        ax3.set_ylabel('Efficiency Index')
        ax3.set_title('Size vs Efficiency Trade-off', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 클러스터 정확도
        accuracies = [contribution_analysis[cid]['cluster_accuracy'] for cid in cluster_ids]
        bars4 = ax4.bar([f'C{cid}' for cid in cluster_ids], accuracies, color='coral', alpha=0.8)
        ax4.set_title('Cluster Accuracy', fontsize=14, weight='bold')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        
        for bar, acc in zip(bars4, accuracies):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.suptitle(f'Layer {layer_idx} - Prediction Contribution Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Contribution analysis plot saved for layer {layer_idx}")
    
    def _plot_misclassification_analysis(self, misclassification_analysis, layer_idx, output_dir):
        """오분류 패턴 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_ids = list(misclassification_analysis.keys())
        
        # 1. 오분류 유형별 개수
        fp_counts = [misclassification_analysis[cid]['false_positive_count'] for cid in cluster_ids]
        fn_counts = [misclassification_analysis[cid]['false_negative_count'] for cid in cluster_ids]
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, fp_counts, width, label='False Positives', color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, fn_counts, width, label='False Negatives', color='blue', alpha=0.7)
        
        ax1.set_title('Misclassification Counts by Type', fontsize=14, weight='bold')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'C{cid}' for cid in cluster_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 에러율
        error_rates = [misclassification_analysis[cid]['error_rate'] for cid in cluster_ids]
        bars3 = ax2.bar([f'C{cid}' for cid in cluster_ids], error_rates, color='orange', alpha=0.8)
        ax2.set_title('Error Rate by Cluster', fontsize=14, weight='bold')
        ax2.set_ylabel('Error Rate')
        ax2.grid(True, alpha=0.3)
        
        for bar, rate in zip(bars3, error_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Precision vs Recall
        precisions = [misclassification_analysis[cid]['precision'] for cid in cluster_ids]
        recalls = [misclassification_analysis[cid]['recall'] for cid in cluster_ids]
        
        ax3.scatter(recalls, precisions, s=120, alpha=0.7, color='purple')
        for i, cid in enumerate(cluster_ids):
            ax3.annotate(f'C{cid}', (recalls[i], precisions[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        
        # 4. 오분류 신뢰도
        fp_confidences = [misclassification_analysis[cid]['fp_avg_confidence'] for cid in cluster_ids]
        fn_confidences = [misclassification_analysis[cid]['fn_avg_confidence'] for cid in cluster_ids]
        
        ax4.scatter(fp_confidences, fn_confidences, s=120, alpha=0.7, color='brown')
        for i, cid in enumerate(cluster_ids):
            if fp_confidences[i] > 0 or fn_confidences[i] > 0:  # 값이 있는 경우만 표시
                ax4.annotate(f'C{cid}', (fp_confidences[i], fn_confidences[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('FP Average Confidence')
        ax4.set_ylabel('FN Average Confidence')
        ax4.set_title('Misclassification Confidence Patterns', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Layer {layer_idx} - Misclassification Pattern Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_misclassification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Misclassification analysis plot saved for layer {layer_idx}")
    
    def _plot_confidence_analysis(self, confidence_analysis, layer_idx, output_dir):
        """신뢰도 분석 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_ids = list(confidence_analysis.keys())
        
        # 1. ECE vs Brier Score
        eces = [confidence_analysis[cid]['ece'] for cid in cluster_ids]
        brier_scores = [confidence_analysis[cid]['brier_score'] for cid in cluster_ids]
        
        ax1.scatter(eces, brier_scores, s=120, alpha=0.7, color='purple')
        for i, cid in enumerate(cluster_ids):
            ax1.annotate(f'C{cid}', (eces[i], brier_scores[i]), xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Expected Calibration Error (ECE)')
        ax1.set_ylabel('Brier Score')
        ax1.set_title('Calibration Quality (Lower is Better)', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 신뢰도 분포
        overconfident = [confidence_analysis[cid]['overconfident_ratio'] for cid in cluster_ids]
        underconfident = [confidence_analysis[cid]['underconfident_ratio'] for cid in cluster_ids]
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, overconfident, width, label='Overconfident (>0.8)', color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, underconfident, width, label='Underconfident (<0.2)', color='blue', alpha=0.7)
        
        ax2.set_title('Confidence Distribution', fontsize=14, weight='bold')
        ax2.set_ylabel('Ratio')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{cid}' for cid in cluster_ids])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 신뢰도별 정확도
        high_conf_acc = [confidence_analysis[cid]['high_confidence_accuracy'] for cid in cluster_ids]
        low_conf_acc = [confidence_analysis[cid]['low_confidence_accuracy'] for cid in cluster_ids]
        
        bars3 = ax3.bar(x - width/2, high_conf_acc, width, label='High Confidence', color='green', alpha=0.7)
        bars4 = ax3.bar(x + width/2, low_conf_acc, width, label='Low Confidence', color='orange', alpha=0.7)
        
        ax3.set_title('Accuracy by Confidence Level', fontsize=14, weight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'C{cid}' for cid in cluster_ids])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ECE 순위
        sorted_clusters = sorted(cluster_ids, key=lambda x: confidence_analysis[x]['ece'])
        sorted_eces = [confidence_analysis[cid]['ece'] for cid in sorted_clusters]
        
        colors = ['green' if ece < 0.1 else 'orange' if ece < 0.2 else 'red' for ece in sorted_eces]
        bars5 = ax4.bar([f'C{cid}' for cid in sorted_clusters], sorted_eces, color=colors, alpha=0.8)
        
        ax4.set_title('Calibration Error Ranking', fontsize=14, weight='bold')
        ax4.set_ylabel('ECE')
        ax4.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, label='Good (<0.1)')
        ax4.axhline(y=0.2, color='black', linestyle=':', alpha=0.5, label='Acceptable (<0.2)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for bar, ece in zip(bars5, sorted_eces):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{ece:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Layer {layer_idx} - Model Confidence Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Confidence analysis plot saved for layer {layer_idx}")
    
    def _plot_removal_impact(self, removal_impact, layer_idx, output_dir):
        """클러스터 제거 영향 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_ids = list(removal_impact.keys())
        
        # 1. 절대 영향
        impacts = [removal_impact[cid]['impact_on_accuracy'] for cid in cluster_ids]
        colors = ['red' if i < 0 else 'green' if i > 0 else 'gray' for i in impacts]
        
        bars1 = ax1.bar([f'C{cid}' for cid in cluster_ids], impacts, color=colors, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=-0.05, color='red', linestyle='--', alpha=0.7, label='Critical (-5%)')
        ax1.set_title('Impact of Cluster Removal on Accuracy', fontsize=14, weight='bold')
        ax1.set_ylabel('Accuracy Change')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for bar, impact in zip(bars1, impacts):
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (0.002 if impact >= 0 else -0.008),
                    f'{impact:+.3f}', ha='center', 
                    va='bottom' if impact >= 0 else 'top', fontsize=10, weight='bold')
        
        # 2. 상대 영향
        relative_impacts = [removal_impact[cid]['relative_impact'] for cid in cluster_ids]
        bars2 = ax2.bar([f'C{cid}' for cid in cluster_ids], relative_impacts, color=colors, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Relative Impact (% of Original Performance)', fontsize=14, weight='bold')
        ax2.set_ylabel('Relative Change')
        ax2.grid(True, alpha=0.3)
        
        # 3. 제거된 샘플 수 vs 영향
        samples_removed = [removal_impact[cid]['samples_removed'] for cid in cluster_ids]
        ax3.scatter(samples_removed, impacts, s=120, alpha=0.7, c=impacts, cmap='RdYlGn')
        for i, cid in enumerate(cluster_ids):
            ax3.annotate(f'C{cid}', (samples_removed[i], impacts[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Samples Removed')
        ax3.set_ylabel('Accuracy Impact')
        ax3.set_title('Sample Count vs Impact', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 임계 클러스터 식별
        critical_clusters = [cid for cid in cluster_ids if removal_impact[cid]['is_critical']]
        beneficial_clusters = [cid for cid in cluster_ids if removal_impact[cid]['impact_on_accuracy'] > 0.01]
        neutral_clusters = [cid for cid in cluster_ids if not removal_impact[cid]['is_critical'] and removal_impact[cid]['impact_on_accuracy'] <= 0.01]
        
        categories = ['Critical\n(< -5%)', 'Beneficial\n(> +1%)', 'Neutral\n(-5% to +1%)']
        counts = [len(critical_clusters), len(beneficial_clusters), len(neutral_clusters)]
        colors_cat = ['red', 'green', 'gray']
        
        bars4 = ax4.bar(categories, counts, color=colors_cat, alpha=0.8)
        ax4.set_title('Cluster Impact Categories', fontsize=14, weight='bold')
        ax4.set_ylabel('Number of Clusters')
        
        for bar, count in zip(bars4, counts):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        plt.suptitle(f'Layer {layer_idx} - Cluster Removal Impact Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_removal_impact.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Removal impact plot saved for layer {layer_idx}")
    
    def _create_performance_dashboard(self, results, output_dir):
        """종합 성능 대시보드 생성"""
        layer_idx = results['layer_idx']
        contribution_analysis = results['contribution_analysis']
        misclassification_analysis = results['misclassification_analysis']
        confidence_analysis = results['confidence_analysis']
        removal_impact = results['removal_impact']
        summary = results['summary']
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        cluster_ids = list(contribution_analysis.keys())
        
        # 1. 종합 성능 점수 (상단 전체)
        ax1 = fig.add_subplot(gs[0, :])
        
        # 종합 점수 계산 (정규화된 여러 지표의 가중합)
        performance_scores = []
        for cid in cluster_ids:
            accuracy = contribution_analysis[cid]['cluster_accuracy']
            contribution = contribution_analysis[cid]['weighted_contribution']
            efficiency = contribution_analysis[cid]['efficiency']
            error_rate = misclassification_analysis[cid]['error_rate']
            ece = confidence_analysis[cid]['ece']
            
            # 점수 계산 (0-1 스케일)
            score = (0.3 * accuracy + 
                    0.2 * (contribution * 10) + 
                    0.2 * min(efficiency / 5, 1) + 
                    0.15 * (1 - error_rate) + 
                    0.15 * (1 - min(ece * 10, 1)))
            performance_scores.append(score)
        
        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in performance_scores]
        bars = ax1.bar([f'C{cid}' for cid in cluster_ids], performance_scores, color=colors, alpha=0.8)
        ax1.set_title('Overall Performance Score by Cluster', fontsize=16, weight='bold')
        ax1.set_ylabel('Performance Score (0-1)')
        ax1.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, performance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # 2. 기여도 vs 효율성 (2행 1열)
        ax2 = fig.add_subplot(gs[1, 0])
        contributions = [contribution_analysis[cid]['weighted_contribution'] for cid in cluster_ids]
        efficiencies = [contribution_analysis[cid]['efficiency'] for cid in cluster_ids]
        
        ax2.scatter(contributions, efficiencies, s=100, alpha=0.7, c=performance_scores, cmap='RdYlGn')
        for i, cid in enumerate(cluster_ids):
            ax2.annotate(f'C{cid}', (contributions[i], efficiencies[i]), xytext=(3, 3), textcoords='offset points')
        ax2.set_xlabel('Weighted Contribution')
        ax2.set_ylabel('Efficiency')
        ax2.set_title('Contribution vs Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # 3. 오분류 패턴 (2행 2열)
        ax3 = fig.add_subplot(gs[1, 1])
        error_rates = [misclassification_analysis[cid]['error_rate'] for cid in cluster_ids]
        bars3 = ax3.bar([f'C{cid}' for cid in cluster_ids], error_rates, 
                       color=['red' if e > 0.2 else 'orange' if e > 0.1 else 'green' for e in error_rates],
                       alpha=0.8)
        ax3.set_title('Error Rate by Cluster')
        ax3.set_ylabel('Error Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. 신뢰도 품질 (2행 3열)
        ax4 = fig.add_subplot(gs[1, 2])
        eces = [confidence_analysis[cid]['ece'] for cid in cluster_ids]
        bars4 = ax4.bar([f'C{cid}' for cid in cluster_ids], eces,
                       color=['green' if e < 0.1 else 'orange' if e < 0.2 else 'red' for e in eces],
                       alpha=0.8)
        ax4.set_title('Calibration Error (ECE)')
        ax4.set_ylabel('ECE')
        ax4.grid(True, alpha=0.3)
        
        # 5. 제거 영향 (2행 4열)
        ax5 = fig.add_subplot(gs[1, 3])
        impacts = [removal_impact[cid]['impact_on_accuracy'] for cid in cluster_ids]
        colors_impact = ['red' if i < -0.05 else 'orange' if i < 0 else 'green' for i in impacts]
        bars5 = ax5.bar([f'C{cid}' for cid in cluster_ids], impacts, color=colors_impact, alpha=0.8)
        ax5.axhline(y=-0.05, color='red', linestyle='--', alpha=0.7)
        ax5.set_title('Removal Impact')
        ax5.set_ylabel('Accuracy Change')
        ax5.grid(True, alpha=0.3)
        
        # 6. 주요 인사이트 텍스트 (3행 전체)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        insights_text = "KEY INSIGHTS:\n"
        for i, insight in enumerate(summary['key_insights'], 1):
            insights_text += f"{i}. {insight}\n"
        
        best_clusters = summary['best_performing_clusters']
        insights_text += f"\nBEST PERFORMERS:\n"
        insights_text += f"• Highest Accuracy: Cluster {best_clusters['highest_accuracy']}\n"
        insights_text += f"• Best Contributor: Cluster {best_clusters['best_contributor']}\n"
        insights_text += f"• Most Efficient: Cluster {best_clusters['most_efficient']}\n"
        
        problematic = summary['problematic_clusters']
        if problematic['critical_for_performance']:
            insights_text += f"\nCRITICAL CLUSTERS: {problematic['critical_for_performance']}\n"
        
        ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 7. 메트릭 요약 표 (4행 전체)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # 표 데이터 준비
        table_data = []
        headers = ['Cluster', 'Accuracy', 'Contribution', 'Efficiency', 'Error Rate', 'ECE', 'Impact']
        
        for cid in cluster_ids:
            row = [
                f'C{cid}',
                f"{contribution_analysis[cid]['cluster_accuracy']:.3f}",
                f"{contribution_analysis[cid]['weighted_contribution']:.3f}",
                f"{contribution_analysis[cid]['efficiency']:.2f}",
                f"{misclassification_analysis[cid]['error_rate']:.3f}",
                f"{confidence_analysis[cid]['ece']:.3f}",
                f"{removal_impact[cid]['impact_on_accuracy']:+.3f}"
            ]
            table_data.append(row)
        
        table = ax7.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 헤더 스타일링
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Layer {layer_idx} - Comprehensive Performance Dashboard', fontsize=20, weight='bold')
        fig.savefig(output_dir / f'layer_{layer_idx}_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Performance dashboard saved for layer {layer_idx}")
    
    def _save_results(self, results, output_dir):
        """결과 저장"""
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # 딕셔너리의 키와 값 모두 변환
                new_dict = {}
                for k, v in obj.items():
                    # 키 변환
                    if isinstance(k, np.integer):
                        new_key = int(k)
                    elif isinstance(k, np.floating):
                        new_key = float(k)
                    else:
                        new_key = k
                    # 값 변환 (재귀)
                    new_dict[str(new_key)] = convert_numpy_types(v)  # 키를 문자열로 변환
                return new_dict
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        clean_results = convert_numpy_types(results)
        
        # JSON 저장
        results_file = output_dir / f'layer_{results["layer_idx"]}_performance_impact_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        # 요약 보고서 저장
        summary_file = output_dir / f'layer_{results["layer_idx"]}_performance_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"CLUSTER PERFORMANCE IMPACT ANALYSIS - LAYER {results['layer_idx']}\n")
            f.write("=" * 60 + "\n\n")
            
            summary = results['summary']
            
            f.write("BEST PERFORMING CLUSTERS:\n")
            f.write(f"• Highest Accuracy: Cluster {summary['best_performing_clusters']['highest_accuracy']}\n")
            f.write(f"• Best Contributor: Cluster {summary['best_performing_clusters']['best_contributor']}\n")
            f.write(f"• Most Efficient: Cluster {summary['best_performing_clusters']['most_efficient']}\n\n")
            
            f.write("PROBLEMATIC CLUSTERS:\n")
            f.write(f"• Highest Error Rate: Cluster {summary['problematic_clusters']['highest_error_rate']}\n")
            f.write(f"• Worst Calibrated: Cluster {summary['problematic_clusters']['worst_calibrated']}\n")
            if summary['problematic_clusters']['critical_for_performance']:
                f.write(f"• Critical for Performance: {summary['problematic_clusters']['critical_for_performance']}\n")
            f.write("\n")
            
            f.write("KEY INSIGHTS:\n")
            for i, insight in enumerate(summary['key_insights'], 1):
                f.write(f"{i}. {insight}\n")
            f.write("\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"• Total Clusters: {summary['overall_metrics']['total_clusters']}\n")
            f.write(f"• Average Cluster Accuracy: {summary['overall_metrics']['avg_cluster_accuracy']:.3f}\n")
            f.write(f"• Performance Variance: {summary['overall_metrics']['performance_variance']:.4f}\n")
            f.write(f"• Total Error Rate: {summary['overall_metrics']['total_error_rate']:.3f}\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    def analyze_all_layers(self, output_base_dir):
        """모든 레이어에 대해 성능 영향 분석 수행"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for layer_idx in self.layer_results.keys():
            logger.info(f"Starting performance impact analysis for layer {layer_idx}...")
            
            layer_output_dir = output_base_dir / f'layer_{layer_idx}'
            results = self.analyze_cluster_performance_impact(layer_idx, layer_output_dir)
            all_results[layer_idx] = results
        
        # 레이어 간 비교 분석
        self._create_cross_layer_analysis(all_results, output_base_dir)
        
        logger.info("Performance impact analysis completed for all layers!")
        return all_results
    
    def _create_cross_layer_analysis(self, all_results, output_dir):
        """레이어 간 성능 영향 비교 분석"""
        logger.info("Creating cross-layer performance analysis...")
        
        layers = sorted(all_results.keys())
        
        # 레이어별 최고 성능 추적
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 레이어별 최고 성능 진화
        best_accuracies = []
        worst_accuracies = []
        avg_accuracies = []
        
        for layer_idx in layers:
            contribution_analysis = all_results[layer_idx]['contribution_analysis']
            accuracies = [data['cluster_accuracy'] for data in contribution_analysis.values()]
            
            best_accuracies.append(max(accuracies))
            worst_accuracies.append(min(accuracies))
            avg_accuracies.append(np.mean(accuracies))
        
        ax1.plot(layers, best_accuracies, 'o-', label='Best Cluster', color='green', linewidth=2, markersize=8)
        ax1.plot(layers, worst_accuracies, 'o-', label='Worst Cluster', color='red', linewidth=2, markersize=8)
        ax1.plot(layers, avg_accuracies, 'o-', label='Average', color='blue', linewidth=2, markersize=8)
        ax1.fill_between(layers, worst_accuracies, best_accuracies, alpha=0.3, color='gray')
        
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Evolution Across Layers', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 레이어별 임계 클러스터 수
        critical_counts = []
        total_clusters = []
        
        for layer_idx in layers:
            removal_impact = all_results[layer_idx]['removal_impact']
            critical_count = sum(1 for data in removal_impact.values() if data['is_critical'])
            critical_counts.append(critical_count)
            total_clusters.append(len(removal_impact))
        
        ax2.bar(layers, critical_counts, color='red', alpha=0.7, label='Critical Clusters')
        ax2.bar(layers, [total - critical for total, critical in zip(total_clusters, critical_counts)], 
               bottom=critical_counts, color='lightgray', alpha=0.7, label='Non-Critical')
        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Critical vs Non-Critical Clusters by Layer', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 레이어별 평균 ECE
        avg_eces = []
        
        for layer_idx in layers:
            confidence_analysis = all_results[layer_idx]['confidence_analysis']
            eces = [data['ece'] for data in confidence_analysis.values()]
            avg_eces.append(np.mean(eces))
        
        colors = ['green' if ece < 0.1 else 'orange' if ece < 0.2 else 'red' for ece in avg_eces]
        bars3 = ax3.bar(layers, avg_eces, color=colors, alpha=0.8)
        ax3.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, label='Good (<0.1)')
        ax3.axhline(y=0.2, color='black', linestyle=':', alpha=0.7, label='Acceptable (<0.2)')
        
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Average ECE')
        ax3.set_title('Calibration Quality Across Layers', fontsize=14, weight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        for bar, ece in zip(bars3, avg_eces):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{ece:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 레이어별 성능 분산
        performance_variances = []
        
        for layer_idx in layers:
            contribution_analysis = all_results[layer_idx]['contribution_analysis']
            accuracies = [data['cluster_accuracy'] for data in contribution_analysis.values()]
            performance_variances.append(np.var(accuracies))
        
        ax4.plot(layers, performance_variances, 'o-', color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Performance Variance')
        ax4.set_title('Performance Consistency Across Layers', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 분산이 낮을수록 일관성이 높음을 표시
        min_var_layer = layers[np.argmin(performance_variances)]
        ax4.annotate(f'Most Consistent\n(Layer {min_var_layer})', 
                    xy=(min_var_layer, min(performance_variances)),
                    xytext=(min_var_layer, min(performance_variances) + 0.001),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, ha='center')
        
        plt.suptitle('Cross-Layer Performance Impact Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / 'cross_layer_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 레이어 추천 보고서 생성
        self._generate_layer_recommendations(all_results, output_dir)
        
        logger.info("Cross-layer analysis completed")
    
    def _generate_layer_recommendations(self, all_results, output_dir):
        """레이어 추천 보고서 생성"""
        layers = sorted(all_results.keys())

        recommendations = {}
        
        for layer_idx in layers:
            contribution_analysis = all_results[layer_idx]['contribution_analysis']
            confidence_analysis = all_results[layer_idx]['confidence_analysis']
            removal_impact = all_results[layer_idx]['removal_impact']
            summary = all_results[layer_idx]['summary']
            
            # 성능 메트릭 계산
            accuracies = [data['cluster_accuracy'] for data in contribution_analysis.values()]
            avg_accuracy = np.mean(accuracies)
            performance_variance = np.var(accuracies)
            
            eces = [data['ece'] for data in confidence_analysis.values()]
            avg_ece = np.mean(eces)
            
            critical_count = len(summary['problematic_clusters']['critical_for_performance'])
            total_clusters = len(contribution_analysis)
            
            # 등급 계산
            score = 0
            if avg_accuracy > 0.8: score += 25
            elif avg_accuracy > 0.7: score += 20
            elif avg_accuracy > 0.6: score += 15
            elif avg_accuracy > 0.5: score += 10
            
            if performance_variance < 0.01: score += 25
            elif performance_variance < 0.02: score += 20
            elif performance_variance < 0.05: score += 15
            elif performance_variance < 0.1: score += 10
            
            if avg_ece < 0.1: score += 25
            elif avg_ece < 0.15: score += 20
            elif avg_ece < 0.2: score += 15
            elif avg_ece < 0.3: score += 10
            
            if critical_count == 0: score += 25
            elif critical_count <= total_clusters * 0.2: score += 15
            elif critical_count <= total_clusters * 0.5: score += 10
            
            # 등급 부여
            if score >= 80: grade = 'A'
            elif score >= 65: grade = 'B'
            elif score >= 50: grade = 'C'
            else: grade = 'D'
            
            # 추천사항 생성
            if grade == 'A':
                recommendation = 'Excellent layer for in-depth analysis - high performance, consistency, and reliability'
            elif grade == 'B':
                recommendation = 'Good layer for analysis - stable performance with minor concerns'
            elif grade == 'C':
                recommendation = 'Moderate layer - useful but with notable limitations'
            else:
                recommendation = 'Poor layer for analysis - significant performance or reliability issues'
            
            recommendations[layer_idx] = {
                'grade': grade,
                'score': score,
                'recommendation': recommendation,
                'metrics': {
                    'avg_accuracy': float(avg_accuracy),
                    'performance_variance': float(performance_variance),
                    'avg_ece': float(avg_ece),
                    'critical_clusters': critical_count,
                    'total_clusters': total_clusters
                }
            }
        
        # 추천 보고서 저장
        report = {
            'analysis_summary': {
                'total_layers_analyzed': len(layers),
                'best_layer': max(recommendations.keys(), key=lambda x: recommendations[x]['score']),
                'most_reliable_layer': min(recommendations.keys(), key=lambda x: recommendations[x]['metrics']['avg_ece']),
                'recommendations_by_grade': {
                    grade: [layer for layer, data in recommendations.items() if data['grade'] == grade]
                    for grade in ['A', 'B', 'C', 'D']
                }
            },
            'layer_recommendations': recommendations
        }
        
        # JSON 저장
        with open(output_dir / 'layer_performance_recommendations.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 시각적 보고서
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 레이어별 점수
        scores = [recommendations[layer]['score'] for layer in layers]
        grades = [recommendations[layer]['grade'] for layer in layers]
        grade_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red'}
        colors = [grade_colors[grade] for grade in grades]
        
        bars1 = ax1.bar([f'L{layer}' for layer in layers], scores, color=colors, alpha=0.8)
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Layer Performance Scores', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 점수 표시
        for bar, score, grade in zip(bars1, scores, grades):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{score}\n({grade})', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 등급별 범례
        for grade, color in grade_colors.items():
            ax1.bar([], [], color=color, alpha=0.8, label=f'Grade {grade}')
        ax1.legend()
        
        # 등급 분포
        grade_counts = [len(report['analysis_summary']['recommendations_by_grade'][grade]) for grade in ['A', 'B', 'C', 'D']]
        ax2.pie(grade_counts, labels=['Grade A', 'Grade B', 'Grade C', 'Grade D'], 
               colors=[grade_colors[g] for g in ['A', 'B', 'C', 'D']], 
               autopct='%1.0f%%', startangle=90)
        ax2.set_title('Grade Distribution', fontsize=14, weight='bold')
        
        plt.suptitle('Layer Performance Recommendation Report', fontsize=16, weight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / 'layer_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 텍스트 보고서
        with open(output_dir / 'layer_recommendations.txt', 'w') as f:
            f.write("LAYER PERFORMANCE RECOMMENDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"ANALYSIS SUMMARY:\n")
            f.write(f"• Total Layers Analyzed: {report['analysis_summary']['total_layers_analyzed']}\n")
            f.write(f"• Best Overall Layer: {report['analysis_summary']['best_layer']}\n")
            f.write(f"• Most Reliable Layer: {report['analysis_summary']['most_reliable_layer']}\n\n")
            
            f.write("GRADE DISTRIBUTION:\n")
            for grade in ['A', 'B', 'C', 'D']:
                layers_in_grade = report['analysis_summary']['recommendations_by_grade'][grade]
                f.write(f"• Grade {grade}: {len(layers_in_grade)} layers {layers_in_grade}\n")
            f.write("\n")
            
            f.write("DETAILED RECOMMENDATIONS:\n")
            for layer_idx in layers:
                rec = recommendations[layer_idx]
                f.write(f"\nLayer {layer_idx} (Grade {rec['grade']}, Score: {rec['score']}):\n")
                f.write(f"  {rec['recommendation']}\n")
                f.write(f"  • Avg Accuracy: {rec['metrics']['avg_accuracy']:.3f}\n")
                f.write(f"  • Performance Variance: {rec['metrics']['performance_variance']:.4f}\n")
                f.write(f"  • Avg ECE: {rec['metrics']['avg_ece']:.3f}\n")
                f.write(f"  • Critical Clusters: {rec['metrics']['critical_clusters']}/{rec['metrics']['total_clusters']}\n")
        
        logger.info("Layer recommendations generated")


def main():
    parser = argparse.ArgumentParser(description='Cluster Performance Impact Analysis')
    parser.add_argument('--clustering_dir', type=str, required=True,
                       help='Directory containing clustering results from inference.py')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--layer_idx', type=int, default=None,
                       help='Specific layer to analyze (default: analyze all layers)')
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    # 출력 디렉토리 설정
    if args.output_dir is None:
        clustering_dir = Path(args.clustering_dir)
        # 현재 시간 정보 추가
        args.output_dir = clustering_dir / f'cluster_analysis2'

    # 분석기 초기화
    analyzer = ClusterPerformanceAnalyzer(args.clustering_dir, args.checkpoint_dir)

    if args.layer_idx is not None:
        logger.info(f"Analyzing performance impact for layer {args.layer_idx}...")
        results = analyzer.analyze_cluster_performance_impact(args.layer_idx, args.output_dir)
        
        # 주요 결과 출력
        summary = results['summary']
        logger.info(f"Analysis completed for layer {args.layer_idx}")
        logger.info(f"Best performing cluster: {summary['best_performing_clusters']['highest_accuracy']}")
        logger.info(f"Most efficient cluster: {summary['best_performing_clusters']['most_efficient']}")
        if summary['problematic_clusters']['critical_for_performance']:
            logger.info(f"Critical clusters: {summary['problematic_clusters']['critical_for_performance']}")
    else:
        logger.info("Analyzing performance impact for all layers...")
        all_results = analyzer.analyze_all_layers(args.output_dir)
        
        # 전체 요약
        best_layers = []
        for layer_idx, results in all_results.items():
            score = len([c for c in results['removal_impact'].values() if not c['is_critical']])
            best_layers.append((layer_idx, score))
        
        best_layers.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Recommended layers for analysis: {[layer for layer, _ in best_layers[:3]]}")

    logger.info(f"Performance impact analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()