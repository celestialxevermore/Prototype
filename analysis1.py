"""
클러스터 분석 스크립트

inference.py에서 생성된 클러스터링 결과를 분석하여
클러스터별 차이점을 통계적으로 검증합니다.

Usage:
    python cluster_analysis.py --clustering_dir /path/to/clustering/results --checkpoint_dir /path/to/checkpoint.pt
"""

import os
# CUDA deterministic 설정을 가장 먼저 설정
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
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# inference.py에서 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
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
        
        logger.info(f"ClusterAnalyzer initialized for {len(self.layer_results)} layers")
    
    def _load_model_and_data(self):
        """모델과 데이터 로드 (inference.py와 동일)"""
        # 체크포인트 로드
        self.checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        # 모델 초기화
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
        
        # 데이터로더 준비
        fix_seed(self.args.random_seed)
        results = prepare_embedding_dataloaders(self.args, self.args.source_dataset_name)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']

        logger.info("Model and data loaded successfully")
    
    def _load_clustering_results(self):
        """클러스터링 결과 디렉토리에서 데이터 로드"""
        self.layer_results = {}
        
        # 각 레이어별 결과 로드
        for layer_dir in self.clustering_dir.glob('layer_*'):
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
    
    def extract_predictions_and_features(self, data_loader):
        """모든 샘플의 예측값과 피처값 추출"""
        predictions = {}
        feature_values = {}
        sample_ids = {}
        labels = {}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 배치를 디바이스로 이동
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 예측값 계산
                pred, _ = self._extract_prediction_from_model(batch_on_device)
                
                # 피처값 추출
                features = self._extract_feature_values(batch_on_device)
                
                batch_size = pred.shape[0]
                
                for sample_idx in range(batch_size):
                    # 예측값 저장 (sigmoid 적용)
                    if pred.shape[1] == 1:  # binary classification
                        pred_prob = torch.sigmoid(pred[sample_idx, 0]).cpu().item()
                    else:  # multi-class
                        pred_prob = torch.softmax(pred[sample_idx], dim=0).cpu().numpy()
                    
                    predictions[sample_count] = pred_prob
                    
                    # 피처값 저장
                    feature_values[sample_count] = features[sample_idx]
                    
                    # 라벨 저장
                    if 'y' in batch:
                        labels[sample_count] = batch['y'][sample_idx].item()
                    else:
                        labels[sample_count] = -1
                    
                    # 샘플 ID 저장
                    if 'sample_ids' in batch:
                        sample_ids[sample_count] = batch['sample_ids'][sample_idx]
                    else:
                        sample_ids[sample_count] = sample_count
                    
                    sample_count += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Processed {sample_count} samples for predictions...")
        
        logger.info(f"Extracted predictions and features for {sample_count} samples")
        return predictions, feature_values, sample_ids, labels
    
    def _extract_prediction_from_model(self, batch):
        """모델에서 예측값 추출 (inference.py와 동일한 로직)"""
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
        cls_token = self.model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        # Graph Attention Layers
        for i, layer in enumerate(self.model.layers):
            norm_x = self.model.layer_norms[i](x)
            attn_output, _ = layer(desc_embeddings, norm_x)
            x = x + attn_output
        
        # 예측값 계산
        pred = x[:, 0, :]
        pred = self.model.predictor(pred)

        return pred, None
    
    def _extract_feature_values(self, batch):
        """배치에서 실제 피처값들 추출"""
        # 이 부분은 데이터셋에 따라 다를 수 있음
        # 일반적으로는 원본 피처값들이 batch에 포함되어 있어야 함
        
        feature_values = []
        batch_size = None


        # categorical features
        if 'cat_features' in batch:
            cat_features = batch['cat_features'].cpu().numpy()
            batch_size = cat_features.shape[0]
            feature_values = [cat_features[i] for i in range(batch_size)]
        
        # numerical features
        if 'num_features' in batch:
            num_features = batch['num_features'].cpu().numpy()
            if batch_size is None:
                batch_size = num_features.shape[0]
                feature_values = [num_features[i] for i in range(batch_size)]
            else:
                for i in range(batch_size):
                    feature_values[i] = np.concatenate([feature_values[i], num_features[i]])
        
        # 피처값이 없는 경우 더미 데이터 생성
        if not feature_values and batch_size is None:
            # 배치 크기를 임의로 추정
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    batch_size = value.shape[0]
                    break
            
            if batch_size:
                feature_values = [np.array([]) for _ in range(batch_size)]
        
        return feature_values
    


    def analyze_cluster_performance(self, clustering_results, attention_data, output_dir, layer_idx):
        """클러스터별 성능 및 기여도 분석"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing cluster performance for layer {layer_idx}...")
        
        # 1. 예측값과 라벨 추출
        predictions, feature_values, sample_ids, labels = self.extract_predictions_and_features(self.train_loader)
        
        # 2. 클러스터별 데이터 정리
        cluster_data = self._organize_cluster_data_for_performance(clustering_results, predictions, labels, attention_data, layer_idx)
        
        # 3. 클러스터별 실제 정확도 분석
        accuracy_results = self._analyze_cluster_accuracy(cluster_data, layer_idx)
        
        # 4. 클러스터별 예측 품질 분석
        quality_results = self._analyze_prediction_quality(cluster_data, layer_idx)
        
        # 5. 클러스터별 샘플 특성 분석
        sample_characteristics = self._analyze_sample_characteristics(cluster_data, layer_idx)
        
        # 6. 시각화
        self._visualize_cluster_performance(accuracy_results, quality_results, sample_characteristics, layer_idx, output_dir)
        
        # 7. 결과 저장
        self._save_performance_results(accuracy_results, quality_results, sample_characteristics, layer_idx, output_dir)
        
        return {
            'accuracy_results': accuracy_results,
            'quality_results': quality_results,
            'sample_characteristics': sample_characteristics
        }

    def _organize_cluster_data_for_performance(self, clustering_results, predictions, labels, attention_data, layer_idx):
        """성능 분석을 위한 클러스터별 데이터 정리"""
        cluster_assignments = clustering_results['cluster_assignments']
        cluster_labels = clustering_results['labels']
        cluster_sample_ids = clustering_results['sample_ids']
        
        cluster_data = {}
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_predictions = []
            cluster_true_labels = []
            cluster_sample_list = []
            
            for idx in cluster_indices:
                if idx < len(cluster_sample_ids):
                    sample_id = cluster_sample_ids[idx]
                    
                    if sample_id in predictions and sample_id in labels:
                        cluster_predictions.append(predictions[sample_id])
                        cluster_true_labels.append(labels[sample_id])
                        cluster_sample_list.append(sample_id)
            
            cluster_data[cluster_id] = {
                'predictions': np.array(cluster_predictions),
                'true_labels': np.array(cluster_true_labels),
                'sample_ids': cluster_sample_list,
                'n_samples': len(cluster_predictions)
            }
        
        return cluster_data

    def _analyze_cluster_accuracy(self, cluster_data, layer_idx):
        """클러스터별 실제 정확도 분석"""
        accuracy_results = {}
        
        logger.info(f"Analyzing cluster accuracy for layer {layer_idx}...")
        
        for cluster_id, data in cluster_data.items():
            if len(data['predictions']) == 0:
                continue
                
            predictions = data['predictions']
            true_labels = data['true_labels']
            
            # 이진 분류 예측 (threshold=0.5)
            pred_labels = (predictions > 0.5).astype(int)
            
            # 정확도 계산
            accuracy = (pred_labels == true_labels).mean()
            
            # 추가 메트릭
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            try:
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
                
                if len(np.unique(true_labels)) > 1:
                    auc = roc_auc_score(true_labels, predictions)
                else:
                    auc = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculating metrics for cluster {cluster_id}: {e}")
                precision = recall = f1 = auc = 0.0
            
            # 확신도별 정확도 분석
            high_confidence_mask = (predictions > 0.8) | (predictions < 0.2)
            low_confidence_mask = (predictions >= 0.2) & (predictions <= 0.8)
            
            high_conf_accuracy = 0.0
            low_conf_accuracy = 0.0
            
            if np.any(high_confidence_mask):
                high_conf_pred = pred_labels[high_confidence_mask]
                high_conf_true = true_labels[high_confidence_mask]
                high_conf_accuracy = (high_conf_pred == high_conf_true).mean()
            
            if np.any(low_confidence_mask):
                low_conf_pred = pred_labels[low_confidence_mask]
                low_conf_true = true_labels[low_confidence_mask]
                low_conf_accuracy = (low_conf_pred == low_conf_true).mean()
            
            accuracy_results[cluster_id] = {
                'n_samples': len(predictions),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'high_confidence_samples': np.sum(high_confidence_mask),
                'low_confidence_samples': np.sum(low_confidence_mask),
                'high_confidence_accuracy': high_conf_accuracy,
                'low_confidence_accuracy': low_conf_accuracy,
                'mean_prediction': np.mean(predictions),
                'std_prediction': np.std(predictions)
            }
            
            logger.info(f"Cluster {cluster_id}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        return accuracy_results

    def _analyze_prediction_quality(self, cluster_data, layer_idx):
        """클러스터별 예측 품질 분석"""
        quality_results = {}
        
        logger.info(f"Analyzing prediction quality for layer {layer_idx}...")
        
        for cluster_id, data in cluster_data.items():
            if len(data['predictions']) == 0:
                continue
                
            predictions = data['predictions']
            true_labels = data['true_labels']
            
            # Calibration 분석
            calibration_bins = np.linspace(0, 1, 11)
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(len(calibration_bins) - 1):
                bin_lower, bin_upper = calibration_bins[i], calibration_bins[i + 1]
                bin_mask = (predictions >= bin_lower) & (predictions < bin_upper)
                
                if np.any(bin_mask):
                    bin_preds = predictions[bin_mask]
                    bin_true = true_labels[bin_mask]
                    bin_pred_labels = (bin_preds > 0.5).astype(int)
                    
                    bin_accuracy = (bin_pred_labels == bin_true).mean()
                    bin_confidence = np.mean(bin_preds)
                    bin_count = len(bin_preds)
                    
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
                    bin_counts.append(bin_count)
            
            # 오분류 분석
            pred_labels = (predictions > 0.5).astype(int)
            
            fp_mask = (pred_labels == 1) & (true_labels == 0)
            fn_mask = (pred_labels == 0) & (true_labels == 1)
            tp_mask = (pred_labels == 1) & (true_labels == 1)
            tn_mask = (pred_labels == 0) & (true_labels == 0)
            
            fp_predictions = predictions[fp_mask] if np.any(fp_mask) else np.array([])
            fn_predictions = predictions[fn_mask] if np.any(fn_mask) else np.array([])
            tp_predictions = predictions[tp_mask] if np.any(tp_mask) else np.array([])
            tn_predictions = predictions[tn_mask] if np.any(tn_mask) else np.array([])
            
            quality_results[cluster_id] = {
                'calibration_bins': calibration_bins[:-1],
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts,
                'fp_count': len(fp_predictions),
                'fn_count': len(fn_predictions),
                'tp_count': len(tp_predictions),
                'tn_count': len(tn_predictions),
                'fp_mean_confidence': np.mean(fp_predictions) if len(fp_predictions) > 0 else 0,
                'fn_mean_confidence': np.mean(fn_predictions) if len(fn_predictions) > 0 else 0,
                'tp_mean_confidence': np.mean(tp_predictions) if len(tp_predictions) > 0 else 0,
                'tn_mean_confidence': np.mean(tn_predictions) if len(tn_predictions) > 0 else 0
            }
        
        return quality_results

    def _analyze_sample_characteristics(self, cluster_data, layer_idx):
        """클러스터별 샘플 특성 분석"""
        characteristics = {}
        
        logger.info(f"Analyzing sample characteristics for layer {layer_idx}...")
        
        all_predictions = []
        all_labels = []
        
        for cluster_id, data in cluster_data.items():
            all_predictions.extend(data['predictions'])
            all_labels.extend(data['true_labels'])
        
        overall_accuracy = (np.array(all_predictions) > 0.5) == np.array(all_labels)
        overall_accuracy = overall_accuracy.mean()
        
        for cluster_id, data in cluster_data.items():
            if len(data['predictions']) == 0:
                continue
                
            predictions = data['predictions']
            true_labels = data['true_labels']
            
            label_distribution = np.bincount(true_labels.astype(int))
            pred_distribution = np.histogram(predictions, bins=10, range=(0, 1))[0]
            
            characteristics[cluster_id] = {
                'cluster_size_ratio': len(predictions) / len(all_predictions),
                'label_0_count': label_distribution[0] if len(label_distribution) > 0 else 0,
                'label_1_count': label_distribution[1] if len(label_distribution) > 1 else 0,
                'label_balance': label_distribution[1] / len(predictions) if len(predictions) > 0 else 0,
                'prediction_histogram': pred_distribution,
                'prediction_bins': np.linspace(0, 1, 11),
                'entropy': self._calculate_prediction_entropy(predictions),
                'accuracy_vs_overall': (predictions > 0.5) == true_labels,
                'contributes_to_accuracy': np.sum((predictions > 0.5) == true_labels) / len(all_predictions)
            }
        
        return characteristics

    def _calculate_prediction_entropy(self, predictions):
        """예측값의 엔트로피 계산"""
        eps = 1e-8
        p1 = np.clip(predictions, eps, 1-eps)
        p0 = 1 - p1
        entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))
        return np.mean(entropy)


    def analyze_cluster_differences(self, layer_idx, output_dir):
        """특정 레이어의 클러스터별 차이 분석"""
        if layer_idx not in self.layer_results:
            logger.error(f"Layer {layer_idx} not found in clustering results")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        layer_data = self.layer_results[layer_idx]
        clusters = layer_data['clusters']
        
        logger.info(f"Analyzing cluster differences for layer {layer_idx}...")
        
        # 1. 예측값과 피처값 추출
        logger.info("Extracting predictions and feature values...")
        predictions, feature_values, sample_ids, labels = self.extract_predictions_and_features(self.train_loader)
        
        # 2. 클러스터별 데이터 정리
        cluster_analysis_data = self._organize_cluster_data(clusters, predictions, feature_values, labels)
        
        # 3. 통계적 분석 수행
        stats_results = self._perform_statistical_tests(cluster_analysis_data, layer_idx)
        
        # 4. 시각화
        self._create_cluster_comparison_plots(cluster_analysis_data, stats_results, layer_idx, output_dir)
        
        # 5. 결과 저장
        self._save_analysis_results(cluster_analysis_data, stats_results, layer_idx, output_dir)
        
        logger.info(f"Cluster analysis completed for layer {layer_idx}")
        return cluster_analysis_data, stats_results
    
    def _organize_cluster_data(self, clusters, predictions, feature_values, labels):
        """클러스터별 데이터 정리"""
        cluster_data = {}
        
        for cluster_id, samples in clusters.items():
            cluster_predictions = []
            cluster_features = []
            cluster_labels = []
            cluster_sample_ids = []
            
            for sample in samples:
                sample_id = sample['sample_id']
                
                if sample_id in predictions:
                    cluster_predictions.append(predictions[sample_id])
                    cluster_features.append(feature_values.get(sample_id, np.array([])))
                    cluster_labels.append(labels.get(sample_id, sample['label']))
                    cluster_sample_ids.append(sample_id)
            
            cluster_data[cluster_id] = {
                'predictions': np.array(cluster_predictions),
                'features': cluster_features,
                'labels': np.array(cluster_labels),
                'sample_ids': cluster_sample_ids,
                'n_samples': len(cluster_predictions)
            }
        
        return cluster_data
    
    def _perform_statistical_tests(self, cluster_data, layer_idx):
        """통계적 검정 수행"""
        stats_results = {
            'layer_idx': layer_idx,
            'label_chi2': None,
            'prediction_anova': None,
            'pairwise_prediction_tests': {},
            'feature_tests': {}
        }
        
        cluster_ids = list(cluster_data.keys())
        n_clusters = len(cluster_ids)
        
        # 1. 라벨 분포 chi-square 검정
        logger.info("Performing chi-square test for label distribution...")
        label_contingency = []
        for cluster_id in cluster_ids:
            labels = cluster_data[cluster_id]['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_row = np.zeros(len(np.unique(np.concatenate([cluster_data[cid]['labels'] for cid in cluster_ids]))))
            for i, label in enumerate(unique_labels):
                label_row[int(label)] = counts[i]
            label_contingency.append(label_row)
        
        if len(label_contingency) > 1 and all(sum(row) > 0 for row in label_contingency):
            chi2_stat, chi2_p, _, _ = chi2_contingency(label_contingency)
            stats_results['label_chi2'] = {
                'statistic': chi2_stat,
                'p_value': chi2_p,
                'significant': chi2_p < 0.05
            }
        
        # 2. 예측값 ANOVA 검정
        logger.info("Performing ANOVA test for prediction values...")
        prediction_groups = [cluster_data[cid]['predictions'] for cid in cluster_ids]
        prediction_groups = [group for group in prediction_groups if len(group) > 0]
        
        if len(prediction_groups) > 1:
            try:
                f_stat, anova_p = f_oneway(*prediction_groups)
                stats_results['prediction_anova'] = {
                    'statistic': f_stat,
                    'p_value': anova_p,
                    'significant': anova_p < 0.05
                }
            except:
                logger.warning("ANOVA test failed for predictions")
        
        # 3. 클러스터 간 예측값 t-test (pairwise)
        logger.info("Performing pairwise t-tests for predictions...")
        for i, cluster_id1 in enumerate(cluster_ids):
            for j, cluster_id2 in enumerate(cluster_ids):
                if i < j:
                    pred1 = cluster_data[cluster_id1]['predictions']
                    pred2 = cluster_data[cluster_id2]['predictions']
                    
                    if len(pred1) > 1 and len(pred2) > 1:
                        try:
                            t_stat, t_p = ttest_ind(pred1, pred2)
                            stats_results['pairwise_prediction_tests'][f'{cluster_id1}_vs_{cluster_id2}'] = {
                                'statistic': t_stat,
                                'p_value': t_p,
                                'significant': t_p < 0.05
                            }
                        except:
                            logger.warning(f"T-test failed for clusters {cluster_id1} vs {cluster_id2}")
        
        return stats_results
    
    def _create_cluster_comparison_plots(self, cluster_data, stats_results, layer_idx, output_dir):
        """클러스터 비교 시각화"""
        
        # 1. 라벨 분포 비교
        self._plot_label_distribution(cluster_data, stats_results, layer_idx, output_dir)
        
        # 2. 예측값 분포 비교
        self._plot_prediction_distribution(cluster_data, stats_results, layer_idx, output_dir)
        
        # 3. 클러스터별 요약 통계
        self._plot_cluster_summary(cluster_data, layer_idx, output_dir)
    
    def _plot_label_distribution(self, cluster_data, stats_results, layer_idx, output_dir):
        """라벨 분포 시각화 (개선된 버전)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 카운트 바 플롯
        cluster_ids = list(cluster_data.keys())
        labels_data = []
        
        for cluster_id in cluster_ids:
            labels = cluster_data[cluster_id]['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                labels_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Label': f'Label {int(label)}',
                    'Count': count
                })
        
        df_labels = pd.DataFrame(labels_data)
        
        # 스택 바 차트 (숫자 표시 추가)
        pivot_df = df_labels.pivot(index='Cluster', columns='Label', values='Count').fillna(0)
        bars = pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title(f'Layer {layer_idx} - Label Distribution by Cluster')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Count')
        ax1.legend(title='Label')
        ax1.tick_params(axis='x', rotation=45)
        
        # 각 바 위에 숫자 표시
        for container in ax1.containers:
            ax1.bar_label(container, label_type='center', fontsize=10, color='white', weight='bold')
        
        # 비율 바 차트 (숫자 표시 추가)
        pivot_df_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        bars2 = pivot_df_norm.plot(kind='bar', stacked=True, ax=ax2, color=['skyblue', 'lightcoral'])
        ax2.set_title(f'Layer {layer_idx} - Label Proportion by Cluster')
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
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Label distribution plot saved for layer {layer_idx}")
    
    def _plot_prediction_distribution(self, cluster_data, stats_results, layer_idx, output_dir):
        """예측값 분포 시각화 (개선된 버전)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        cluster_ids = list(cluster_data.keys())
        
        # 히스토그램
        for cluster_id in cluster_ids:
            predictions = cluster_data[cluster_id]['predictions']
            if len(predictions) > 0:
                ax1.hist(predictions, alpha=0.6, label=f'Cluster {cluster_id}', bins=20)
        
        ax1.set_title(f'Layer {layer_idx} - Prediction Distribution by Cluster')
        ax1.set_xlabel('Prediction Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 박스플롯
        prediction_data = []
        for cluster_id in cluster_ids:
            predictions = cluster_data[cluster_id]['predictions']
            for pred in predictions:
                prediction_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Prediction': pred
                })
        
        df_pred = pd.DataFrame(prediction_data)
        sns.boxplot(data=df_pred, x='Cluster', y='Prediction', ax=ax2)
        ax2.set_title(f'Layer {layer_idx} - Prediction Boxplot by Cluster')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # ANOVA 결과 표시
        if stats_results['prediction_anova']:
            anova_result = stats_results['prediction_anova']
            fig.suptitle(f'ANOVA test: p={anova_result["p_value"]:.4f} ' + 
                        ('(Significant)' if anova_result['significant'] else '(Not Significant)'), 
                        fontsize=14)
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Prediction distribution plot saved for layer {layer_idx}")
    
    def _plot_cluster_summary(self, cluster_data, layer_idx, output_dir):
        """클러스터별 요약 통계 시각화 (개선된 버전)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        cluster_ids = list(cluster_data.keys())
        
        # 1. 클러스터별 샘플 수 (숫자 표시)
        sample_counts = [cluster_data[cid]['n_samples'] for cid in cluster_ids]
        bars1 = ax1.bar([f'Cluster {cid}' for cid in cluster_ids], sample_counts, color='lightblue')
        ax1.set_title('Sample Count by Cluster')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 샘플 수 표시
        for bar, count in zip(bars1, sample_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # 2. 클러스터별 평균 예측값 (에러바 및 숫자 표시)
        mean_predictions = []
        std_predictions = []
        for cid in cluster_ids:
            preds = cluster_data[cid]['predictions']
            mean_predictions.append(np.mean(preds) if len(preds) > 0 else 0)
            std_predictions.append(np.std(preds) if len(preds) > 0 else 0)
        
        bars2 = ax2.bar([f'Cluster {cid}' for cid in cluster_ids], mean_predictions, 
                       yerr=std_predictions, capsize=5, color='lightgreen')
        ax2.set_title('Mean Prediction by Cluster')
        ax2.set_ylabel('Mean Prediction')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 평균값 표시
        for bar, mean_val, std_val in zip(bars2, mean_predictions, std_predictions):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.02,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')
        
        plt.suptitle(f'Layer {layer_idx} Cluster Analysis Summary', fontsize=16)
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Cluster summary plot saved for layer {layer_idx}")
    
    def _save_analysis_results(self, cluster_data, stats_results, layer_idx, output_dir):
        """분석 결과를 JSON 파일로 저장"""
        results = {
            'layer_idx': layer_idx,
            'cluster_summary': {},
            'statistical_tests': stats_results
        }
        
        # 클러스터별 요약 정보
        for cluster_id, data in cluster_data.items():
            results['cluster_summary'][f'cluster_{cluster_id}'] = {
                'n_samples': int(data['n_samples']),
                'mean_prediction': float(np.mean(data['predictions'])) if len(data['predictions']) > 0 else 0.0,
                'std_prediction': float(np.std(data['predictions'])) if len(data['predictions']) > 0 else 0.0,
                'label_distribution': {}
            }
            
            # 라벨 분포
            if len(data['labels']) > 0:
                unique_labels, counts = np.unique(data['labels'], return_counts=True)
                for label, count in zip(unique_labels, counts):
                    results['cluster_summary'][f'cluster_{cluster_id}']['label_distribution'][f'label_{int(label)}'] = int(count)
        
        # JSON 저장 (numpy 타입 처리)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # stats_results의 numpy 타입 변환 (재귀적으로 처리)
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            else:
                return convert_numpy(d)
        
        results['statistical_tests'] = clean_dict(stats_results)
        
        # 저장
        results_file = output_dir / f'layer_{layer_idx}_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {results_file}")
    
    def analyze_all_layers(self, output_base_dir):
        """모든 레이어에 대해 클러스터 분석 수행"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for layer_idx in self.layer_results.keys():
            logger.info(f"Starting analysis for layer {layer_idx}...")
            
            # 레이어별 출력 디렉토리
            layer_output_dir = output_base_dir / f'layer_{layer_idx}'
            
            # 분석 수행
            cluster_data, stats_results = self.analyze_cluster_differences(layer_idx, layer_output_dir)
            all_results[layer_idx] = {
                'cluster_data': cluster_data,
                'stats_results': stats_results
            }
        
        # 전체 레이어 비교 시각화
        self._create_cross_layer_comparison(all_results, output_base_dir)
        
        logger.info("All layer analysis completed!")
        return all_results
    
    def _create_cross_layer_comparison(self, all_results, output_dir):
        """레이어 간 비교 시각화"""
        
        # 1. 레이어별 통계적 유의성 비교
        self._plot_significance_across_layers(all_results, output_dir)
        
        # 2. 레이어별 클러스터 품질 비교  
        self._plot_cluster_quality_across_layers(all_results, output_dir)
    
    def _plot_significance_across_layers(self, all_results, output_dir):
        """레이어별 통계적 유의성 비교"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        layers = sorted(all_results.keys())
        
        # Chi-square p-values
        chi2_pvalues = []
        anova_pvalues = []
        
        for layer_idx in layers:
            stats = all_results[layer_idx]['stats_results']
            
            # Chi-square p-value
            if stats['label_chi2']:
                chi2_pvalues.append(stats['label_chi2']['p_value'])
            else:
                chi2_pvalues.append(1.0)  # 테스트 실패시 1.0
            
            # ANOVA p-value
            if stats['prediction_anova']:
                anova_pvalues.append(stats['prediction_anova']['p_value'])
            else:
                anova_pvalues.append(1.0)
        
        # Chi-square 결과
        bars1 = ax1.bar([f'Layer {l}' for l in layers], chi2_pvalues, 
                       color=['red' if p < 0.05 else 'lightcoral' for p in chi2_pvalues])
        ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α=0.05')
        ax1.set_title('Chi-square Test p-values\n(Label Distribution)')
        ax1.set_ylabel('p-value')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 유의한 결과에 별표 추가
        for i, p in enumerate(chi2_pvalues):
            if p < 0.05:
                ax1.text(i, p, '*', ha='center', va='bottom', fontsize=16, color='white')
        
        # ANOVA 결과
        bars2 = ax2.bar([f'Layer {l}' for l in layers], anova_pvalues,
                       color=['blue' if p < 0.05 else 'lightblue' for p in anova_pvalues])
        ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α=0.05')
        ax2.set_title('ANOVA Test p-values\n(Prediction Distribution)')
        ax2.set_ylabel('p-value')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 유의한 결과에 별표 추가
        for i, p in enumerate(anova_pvalues):
            if p < 0.05:
                ax2.text(i, p, '*', ha='center', va='bottom', fontsize=16, color='white')
        
        plt.suptitle('Statistical Significance Across Layers', fontsize=16)
        plt.tight_layout()
        fig.savefig(output_dir / 'cross_layer_significance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Cross-layer significance plot saved")
    
    def _plot_cluster_quality_across_layers(self, all_results, output_dir):
        """레이어별 클러스터 품질 비교 (간소화된 버전)"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        layers = sorted(all_results.keys())
        
        # 예측값 분산 비교 (클러스터 내 vs 클러스터 간)
        within_cluster_vars = []
        between_cluster_vars = []
        
        for layer_idx in layers:
            cluster_data = all_results[layer_idx]['cluster_data']
            
            # 클러스터 내 분산 (평균)
            within_vars = []
            cluster_means = []
            
            for cluster_id, data in cluster_data.items():
                if len(data['predictions']) > 1:
                    within_vars.append(np.var(data['predictions']))
                    cluster_means.append(np.mean(data['predictions']))
            
            within_cluster_vars.append(np.mean(within_vars) if within_vars else 0)
            
            # 클러스터 간 분산
            if len(cluster_means) > 1:
                between_cluster_vars.append(np.var(cluster_means))
            else:
                between_cluster_vars.append(0)
        
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, within_cluster_vars, width, label='Within Cluster', color='orange', alpha=0.7)
        bars2 = ax.bar(x + width/2, between_cluster_vars, width, label='Between Cluster', color='purple', alpha=0.7)
        
        # 값 표시
        for bar, val in zip(bars1, within_cluster_vars):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        for bar, val in zip(bars2, between_cluster_vars):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        ax.set_title('Prediction Variance: Within vs Between Clusters')
        ax.set_ylabel('Variance')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {l}' for l in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'cross_layer_quality.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Cross-layer quality plot saved")

def main():
    parser = argparse.ArgumentParser(description='Cluster Analysis')
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
    if args.output_dir is None:
        clustering_dir = Path(args.clustering_dir)
        import time 
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = clustering_dir.parent / f'cluster_analysis_{timestamp}'
    
    # 분석기 초기화
    analyzer = ClusterAnalyzer(args.clustering_dir, args.checkpoint_dir)
    
    if args.layer_idx is not None:
        # 특정 레이어만 분석
        logger.info(f"Analyzing layer {args.layer_idx}...")
        output_dir = Path(args.output_dir) / f'layer_{args.layer_idx}'
        analyzer.analyze_cluster_differences(args.layer_idx, output_dir)
    else:
        # 모든 레이어 분석
        logger.info("Analyzing all layers...")
        analyzer.analyze_all_layers(args.output_dir)

    logger.info(f"Analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()