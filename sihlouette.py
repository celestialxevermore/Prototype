"""
클러스터 품질 메트릭 평가 스크립트

inference.py에서 생성된 클러스터링 결과의 품질을 다양한 메트릭으로 평가합니다.
- Inertia (WCSS)
- Silhouette Score
- Calinski-Harabasz Index  
- Davies-Bouldin Index
- Gap Statistic
- Within/Between cluster variance analysis

Usage:
    python analysis3.py --clustering_dir /path/to/clustering/results --checkpoint_dir /path/to/checkpoint.pt
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# inference.py에서 모듈들 import
from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterQualityEvaluator:
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
        
        logger.info(f"ClusterQualityEvaluator initialized for {len(self.layer_results)} layers")
    
    def _load_model_and_data(self):
        """모델과 데이터 로드 (기존 코드와 동일)"""
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
        """클러스터링 결과 로드"""
        self.layer_results = {}
        
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

    def extract_attention_maps_and_labels(self):
        """모든 샘플의 attention maps와 라벨 추출"""
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
            for batch_idx, batch in enumerate(self.train_loader):
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
                
                if batch_idx % 50 == 0:
                    logger.info(f"Processed {sample_count} samples for attention maps...")
        
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

    def _extract_layer_embeddings(self, batch):
        """각 레이어에서 임베딩 추출"""
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

        desc_embeddings = torch.cat(desc_embeddings, dim=1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim=1)
        
        # [CLS] Token 추가
        cls_token = self.model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        layer_embeddings = {}
        
        # 각 Graph Attention Layer에서 임베딩 추출
        for i, layer in enumerate(self.model.layers):
            norm_x = self.model.layer_norms[i](x)
            attn_output, _ = layer(desc_embeddings, norm_x)
            x = x + attn_output
            
            # CLS 토큰 임베딩 저장 (첫 번째 토큰)
            layer_embeddings[i] = x[:, 0, :].clone()
        
        return layer_embeddings

    def calculate_clustering_metrics(self, layer_idx, attention_maps, labels, cluster_assignments):
        """클러스터링 품질 메트릭 계산 (attention maps 사용)"""
        metrics = {}
        
        logger.info(f"Calculating clustering metrics for layer {layer_idx}...")
        
        # 1. Inertia (WCSS) - 수동으로 계산
        metrics['inertia'] = self._calculate_inertia(attention_maps, cluster_assignments)
        
        # 2. Silhouette Score
        if len(np.unique(cluster_assignments)) > 1 and len(cluster_assignments) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(attention_maps, cluster_assignments)
            except Exception as e:
                logger.warning(f"Silhouette score calculation failed: {e}")
                metrics['silhouette_score'] = -1
        else:
            metrics['silhouette_score'] = -1
        
        # 3. Calinski-Harabasz Index
        if len(np.unique(cluster_assignments)) > 1:
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(attention_maps, cluster_assignments)
            except Exception as e:
                logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
                metrics['calinski_harabasz_score'] = 0
        else:
            metrics['calinski_harabasz_score'] = 0
        
        # 4. Davies-Bouldin Index
        if len(np.unique(cluster_assignments)) > 1:
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(attention_maps, cluster_assignments)
            except Exception as e:
                logger.warning(f"Davies-Bouldin score calculation failed: {e}")
                metrics['davies_bouldin_score'] = float('inf')
        else:
            metrics['davies_bouldin_score'] = float('inf')
        
        # 5. Within/Between Cluster Variance
        within_var, between_var = self._calculate_within_between_variance(attention_maps, cluster_assignments)
        metrics['within_cluster_variance'] = within_var
        metrics['between_cluster_variance'] = between_var
        metrics['variance_ratio'] = between_var / within_var if within_var > 0 else 0
        
        # 6. 클러스터별 세부 정보
        metrics['cluster_details'] = self._calculate_cluster_details(attention_maps, cluster_assignments)
        
        # 7. 클러스터 품질 종합 점수
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        logger.info(f"Layer {layer_idx} metrics calculated:")
        logger.info(f"  Silhouette: {metrics['silhouette_score']:.4f}")
        logger.info(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.4f}")
        logger.info(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        logger.info(f"  Quality Score: {metrics['quality_score']:.4f}")
        
        return metrics

    def _calculate_inertia(self, embeddings, cluster_assignments):
        """Inertia (WCSS) 계산"""
        inertia = 0
        unique_clusters = np.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                distances_squared = np.sum((cluster_points - cluster_center) ** 2, axis=1)
                inertia += np.sum(distances_squared)
        
        return inertia

    def _calculate_within_between_variance(self, embeddings, cluster_assignments):
        """Within-cluster와 Between-cluster variance 계산"""
        unique_clusters = np.unique(cluster_assignments)
        overall_mean = np.mean(embeddings, axis=0)
        
        within_variance = 0
        between_variance = 0
        total_points = len(embeddings)
        
        cluster_means = []
        cluster_sizes = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = embeddings[cluster_mask]
            cluster_size = len(cluster_points)
            
            if cluster_size > 0:
                cluster_mean = np.mean(cluster_points, axis=0)
                cluster_means.append(cluster_mean)
                cluster_sizes.append(cluster_size)
                
                # Within-cluster variance
                within_variance += np.sum((cluster_points - cluster_mean) ** 2)
        
        # Between-cluster variance
        for i, (cluster_mean, cluster_size) in enumerate(zip(cluster_means, cluster_sizes)):
            between_variance += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
        
        within_variance /= total_points
        between_variance /= total_points
        
        return within_variance, between_variance

    def _calculate_cluster_details(self, embeddings, cluster_assignments):
        """클러스터별 세부 정보 계산"""
        unique_clusters = np.unique(cluster_assignments)
        cluster_details = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                
                cluster_details[int(cluster_id)] = {
                    'size': len(cluster_points),
                    'mean_distance_to_center': np.mean(distances),
                    'std_distance_to_center': np.std(distances),
                    'max_distance_to_center': np.max(distances),
                    'diameter': self._calculate_cluster_diameter(cluster_points),
                    'density': self._calculate_cluster_density(cluster_points)
                }
        
        return cluster_details

    def _calculate_cluster_diameter(self, cluster_points):
        """클러스터 직경 계산 (최대 거리)"""
        if len(cluster_points) < 2:
            return 0
        
        distances = euclidean_distances(cluster_points)
        return np.max(distances)

    def _calculate_cluster_density(self, cluster_points):
        """클러스터 밀도 계산 (단순화된 버전)"""
        if len(cluster_points) < 2:
            return 0
        
        n_points = len(cluster_points)
        volume = self._estimate_cluster_volume(cluster_points)
        
        return n_points / volume if volume > 0 else 0

    def _estimate_cluster_volume(self, cluster_points):
        """클러스터 볼륨 추정 (각 차원의 범위 곱)"""
        if len(cluster_points) < 2:
            return 1e-10
        
        ranges = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
        volume = np.prod(ranges + 1e-10)  # 0으로 나누기 방지
        
        return volume

    def _calculate_quality_score(self, metrics):
        """클러스터링 품질 종합 점수 계산 (0-1 스케일)"""
        # 각 메트릭을 0-1로 정규화하고 가중평균
        
        # Silhouette Score: -1~1 -> 0~1
        silhouette_norm = (metrics['silhouette_score'] + 1) / 2
        
        # Calinski-Harabasz: 높을수록 좋음 (상대적 스케일링)
        ch_norm = min(metrics['calinski_harabasz_score'] / 1000, 1.0)
        
        # Davies-Bouldin: 낮을수록 좋음 (역수 사용)
        db_score = metrics['davies_bouldin_score']
        if db_score == float('inf') or db_score <= 0:
            db_norm = 0
        else:
            db_norm = 1 / (1 + db_score)
        
        # Variance Ratio: 높을수록 좋음
        var_ratio_norm = min(metrics['variance_ratio'] / 10, 1.0)
        
        # 가중평균 (Silhouette와 CH Index에 더 큰 가중치)
        quality_score = (0.4 * silhouette_norm + 
                        0.3 * ch_norm + 
                        0.2 * db_norm + 
                        0.1 * var_ratio_norm)
        
        return quality_score

    def calculate_gap_statistic(self, embeddings, max_k=10, n_refs=10):
        """Gap Statistic 계산"""
        logger.info("Calculating Gap Statistic...")
        
        n_samples, n_features = embeddings.shape
        
        # 실제 데이터의 클러스터링 inertia
        actual_inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            if k == 1:
                # k=1일 때는 전체 평균까지의 거리의 제곱합
                overall_mean = np.mean(embeddings, axis=0)
                inertia = np.sum(np.sum((embeddings - overall_mean) ** 2, axis=1))
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                kmeans.fit(embeddings)
                inertia = kmeans.inertia_
            actual_inertias.append(inertia)
        
        # 참조 데이터의 클러스터링 inertia
        reference_inertias = []
        
        for k in k_range:
            ref_inertias_k = []
            
            for _ in range(n_refs):
                # 각 특성의 범위 내에서 균등하게 분포된 참조 데이터 생성
                mins = np.min(embeddings, axis=0)
                maxs = np.max(embeddings, axis=0)
                reference_data = np.random.uniform(mins, maxs, (n_samples, n_features))
                
                if k == 1:
                    ref_mean = np.mean(reference_data, axis=0)
                    ref_inertia = np.sum(np.sum((reference_data - ref_mean) ** 2, axis=1))
                else:
                    ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                    ref_kmeans.fit(reference_data)
                    ref_inertia = ref_kmeans.inertia_
                
                ref_inertias_k.append(ref_inertia)
            
            reference_inertias.append(ref_inertias_k)
        
        # Gap statistic 계산
        gaps = []
        gap_stds = []
        
        for k_idx, k in enumerate(k_range):
            ref_log_inertias = np.log(reference_inertias[k_idx])
            expected_log_inertia = np.mean(ref_log_inertias)
            actual_log_inertia = np.log(actual_inertias[k_idx])
            
            gap = expected_log_inertia - actual_log_inertia
            gaps.append(gap)
            
            # 표준편차 계산
            std_ref = np.std(ref_log_inertias)
            gap_std = std_ref * np.sqrt(1 + 1/n_refs)
            gap_stds.append(gap_std)
        
        # 최적 k 찾기 (gap(k) >= gap(k+1) - se(k+1))
        optimal_k = 1
        for k_idx in range(len(gaps) - 1):
            if gaps[k_idx] >= gaps[k_idx + 1] - gap_stds[k_idx + 1]:
                optimal_k = k_range[k_idx]
                break
        
        return {
            'k_range': list(k_range),
            'gaps': gaps,
            'gap_stds': gap_stds,
            'optimal_k': optimal_k,
            'actual_inertias': actual_inertias
        }

    def analyze_layer_quality(self, layer_idx, output_dir):
        """특정 레이어의 클러스터 품질 분석"""
        if layer_idx not in self.layer_results:
            logger.error(f"Layer {layer_idx} not found in clustering results")
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing cluster quality for layer {layer_idx}...")
        
        # 1. attention maps와 클러스터 할당 추출
        attention_data = self.extract_attention_maps_and_labels()
        
        if f'layer_{layer_idx}' not in attention_data:
            logger.error(f"No attention maps found for layer {layer_idx}")
            return None
        
        # 특정 레이어의 attention maps 추출
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        sample_ids = attention_data['sample_ids']
        
        # 평탄화 (벡터화) - silhouette_analysis.py와 동일
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        # 2. 클러스터 할당 정보 추출
        cluster_assignments = self._extract_cluster_assignments(layer_idx, sample_ids)
        
        # 3. 클러스터링 메트릭 계산 (flattened attention maps 사용)
        metrics = self.calculate_clustering_metrics(layer_idx, flattened_maps, labels, cluster_assignments)
        
        # 4. Gap Statistic 계산 (시간이 오래 걸리므로 선택적)
        gap_results = self.calculate_gap_statistic(flattened_maps, max_k=min(10, len(np.unique(cluster_assignments)) + 3))
        metrics['gap_statistic'] = gap_results
        
        # 5. 시각화
        self._create_quality_visualizations(flattened_maps, cluster_assignments, metrics, layer_idx, output_dir)
        
        # 6. 결과 저장
        self._save_quality_results(metrics, layer_idx, output_dir)
        
        return metrics

    def _extract_cluster_assignments(self, layer_idx, sample_ids):
        """해당 레이어의 클러스터 할당 정보 추출"""
        layer_data = self.layer_results[layer_idx]
        clusters = layer_data['clusters']
        
        # 샘플 ID to 클러스터 ID 매핑
        sample_to_cluster = {}
        
        for cluster_id, samples in clusters.items():
            for sample in samples:
                sample_id = sample['sample_id']
                sample_to_cluster[sample_id] = cluster_id
        
        # 순서대로 클러스터 할당 배열 생성
        cluster_assignments = []
        for sample_id in sample_ids:
            if sample_id in sample_to_cluster:
                cluster_assignments.append(sample_to_cluster[sample_id])
            else:
                # 찾지 못한 샘플은 -1로 할당
                cluster_assignments.append(-1)
        
        cluster_assignments = np.array(cluster_assignments)
        
        # -1 (찾지 못한 샘플) 제거
        valid_mask = cluster_assignments != -1
        if not np.all(valid_mask):
            logger.warning(f"Found {np.sum(~valid_mask)} samples not in cluster assignments")
        
        return cluster_assignments

    def _create_quality_visualizations(self, embeddings, cluster_assignments, metrics, layer_idx, output_dir):
        """클러스터 품질 시각화"""
        
        # 1. 메트릭 요약 바 플롯
        self._plot_metrics_summary(metrics, layer_idx, output_dir)
        
        # 2. Gap Statistic 플롯
        self._plot_gap_statistic(metrics['gap_statistic'], layer_idx, output_dir)
        
        # 3. 클러스터 분포 (PCA 시각화)
        self._plot_cluster_scatter(embeddings, cluster_assignments, metrics, layer_idx, output_dir)
        
        # 4. 클러스터별 세부 정보
        self._plot_cluster_details(metrics['cluster_details'], layer_idx, output_dir)

    def _plot_metrics_summary(self, metrics, layer_idx, output_dir):
        """메트릭 요약 바 플롯"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 주요 메트릭들
        metric_names = ['Silhouette\nScore', 'Calinski-Harabasz\nIndex', 'Davies-Bouldin\nScore', 'Quality\nScore']
        metric_values = [
            metrics['silhouette_score'],
            metrics['calinski_harabasz_score'] / 1000,  # 스케일링
            1 / (1 + metrics['davies_bouldin_score']) if metrics['davies_bouldin_score'] != float('inf') else 0,
            metrics['quality_score']
        ]
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        bars1 = ax1.bar(metric_names, metric_values, color=colors)
        ax1.set_title(f'Layer {layer_idx} - Clustering Quality Metrics')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, val in zip(bars1, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 2. Variance 분석
        variance_names = ['Within Cluster', 'Between Cluster']
        variance_values = [metrics['within_cluster_variance'], metrics['between_cluster_variance']]
        
        bars2 = ax2.bar(variance_names, variance_values, color=['orange', 'purple'])
        ax2.set_title('Within vs Between Cluster Variance')
        ax2.set_ylabel('Variance')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, val in zip(bars2, variance_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Inertia vs Gap Statistic
        gap_data = metrics['gap_statistic']
        k_range = gap_data['k_range']
        gaps = gap_data['gaps']
        
        ax3.plot(k_range, gaps, 'o-', color='blue', label='Gap Statistic')
        ax3.axvline(x=gap_data['optimal_k'], color='red', linestyle='--', label=f'Optimal k={gap_data["optimal_k"]}')
        ax3.set_title('Gap Statistic')
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Gap(k)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 클러스터 크기 분포
        cluster_details = metrics['cluster_details']
        cluster_ids = list(cluster_details.keys())
        cluster_sizes = [cluster_details[cid]['size'] for cid in cluster_ids]
        
        bars4 = ax4.bar([f'C{cid}' for cid in cluster_ids], cluster_sizes, color='lightblue')
        ax4.set_title('Cluster Size Distribution')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Samples')
        ax4.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, size in zip(bars4, cluster_sizes):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{size}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Metrics summary plot saved for layer {layer_idx}")

    def _plot_gap_statistic(self, gap_data, layer_idx, output_dir):
        """Gap Statistic 상세 플롯"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        k_range = gap_data['k_range']
        gaps = gap_data['gaps']
        gap_stds = gap_data['gap_stds']
        actual_inertias = gap_data['actual_inertias']
        
        # 1. Gap Statistic with error bars
        ax1.errorbar(k_range, gaps, yerr=gap_stds, fmt='o-', capsize=5, color='blue', label='Gap ± SE')
        ax1.axvline(x=gap_data['optimal_k'], color='red', linestyle='--', alpha=0.8, 
                   label=f'Optimal k={gap_data["optimal_k"]}')
        ax1.set_title(f'Layer {layer_idx} - Gap Statistic')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Gap(k)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Actual Inertias (Elbow method)
        ax2.plot(k_range, actual_inertias, 'o-', color='green', label='Inertia')
        ax2.set_title('Elbow Method - Inertia vs k')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Inertia (WCSS)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Elbow 찾기 (간단한 버전)
        if len(k_range) > 2:
            # 곡률 계산으로 elbow 추정
            diffs = np.diff(actual_inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2  # +2는 인덱스 보정
                if elbow_idx < len(k_range):
                    ax2.axvline(x=k_range[elbow_idx], color='orange', linestyle='--', 
                               alpha=0.8, label=f'Elbow k≈{k_range[elbow_idx]}')
                    ax2.legend()
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_gap_statistic.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Gap statistic plot saved for layer {layer_idx}")

    def _plot_cluster_scatter(self, embeddings, cluster_assignments, metrics, layer_idx, output_dir):
        """PCA를 이용한 클러스터 시각화"""
        # PCA로 2D 축소
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 클러스터별 색상으로 구분
        unique_clusters = np.unique(cluster_assignments)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = cluster_assignments == cluster_id
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[color], label=f'Cluster {cluster_id}', alpha=0.7, s=30)
        
        ax1.set_title(f'Layer {layer_idx} - Cluster Visualization (PCA)')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 클러스터 중심점과 반경 표시
        for cluster_id in unique_clusters:
            mask = cluster_assignments == cluster_id
            cluster_points = embeddings_2d[mask]
            
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.std(distances) * 2  # 2 표준편차
                
                ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           alpha=0.5, s=20, label=f'Cluster {cluster_id}')
                ax2.scatter(center[0], center[1], c='red', s=100, marker='x')
                
                circle = plt.Circle(center, radius, fill=False, color='red', alpha=0.5)
                ax2.add_patch(circle)
        
        ax2.set_title('Cluster Centers and Spread')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Cluster scatter plot saved for layer {layer_idx}")

    def _plot_cluster_details(self, cluster_details, layer_idx, output_dir):
        """클러스터별 세부 정보 시각화"""
        if not cluster_details:
            logger.warning(f"No cluster details available for layer {layer_idx}")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        cluster_ids = list(cluster_details.keys())
        
        # 1. 클러스터별 평균 거리
        mean_distances = [cluster_details[cid]['mean_distance_to_center'] for cid in cluster_ids]
        bars1 = ax1.bar([f'C{cid}' for cid in cluster_ids], mean_distances, color='lightblue')
        ax1.set_title('Mean Distance to Cluster Center')
        ax1.set_ylabel('Distance')
        ax1.grid(True, alpha=0.3)
        
        for bar, dist in zip(bars1, mean_distances):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{dist:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # 2. 클러스터별 직경
        diameters = [cluster_details[cid]['diameter'] for cid in cluster_ids]
        bars2 = ax2.bar([f'C{cid}' for cid in cluster_ids], diameters, color='lightgreen')
        ax2.set_title('Cluster Diameter')
        ax2.set_ylabel('Diameter')
        ax2.grid(True, alpha=0.3)
        
        for bar, diam in zip(bars2, diameters):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{diam:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # 3. 클러스터별 밀도
        densities = [cluster_details[cid]['density'] for cid in cluster_ids]
        bars3 = ax3.bar([f'C{cid}' for cid in cluster_ids], densities, color='lightcoral')
        ax3.set_title('Cluster Density')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        
        for bar, dens in zip(bars3, densities):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{dens:.4f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # 4. 클러스터별 크기 vs 품질 산점도
        sizes = [cluster_details[cid]['size'] for cid in cluster_ids]
        compactness = [1 / (1 + cluster_details[cid]['mean_distance_to_center']) for cid in cluster_ids]
        
        scatter = ax4.scatter(sizes, compactness, c=cluster_ids, cmap='viridis', s=100, alpha=0.7)
        ax4.set_title('Cluster Size vs Compactness')
        ax4.set_xlabel('Cluster Size')
        ax4.set_ylabel('Compactness (1/(1+mean_dist))')
        ax4.grid(True, alpha=0.3)
        
        # 클러스터 ID 표시
        for i, (size, comp, cid) in enumerate(zip(sizes, compactness, cluster_ids)):
            ax4.annotate(f'C{cid}', (size, comp), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.colorbar(scatter, ax=ax4, label='Cluster ID')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'layer_{layer_idx}_cluster_details.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Cluster details plot saved for layer {layer_idx}")

    def _save_quality_results(self, metrics, layer_idx, output_dir):
        """품질 분석 결과 저장"""
        # numpy 타입을 Python 기본 타입으로 변환
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        clean_metrics = convert_numpy(metrics)
        
        # JSON 파일로 저장
        results_file = output_dir / f'layer_{layer_idx}_quality_metrics.json'
        with open(results_file, 'w') as f:
            json.dump(clean_metrics, f, indent=2)
        
        # 요약 리포트 생성
        report_file = output_dir / f'layer_{layer_idx}_quality_report.txt'
        with open(report_file, 'w') as f:
            f.write(f"Cluster Quality Analysis Report - Layer {layer_idx}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MAIN METRICS:\n")
            f.write(f"- Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, range: -1 to 1)\n")
            f.write(f"- Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)\n")
            f.write(f"- Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)\n")
            f.write(f"- Quality Score: {metrics['quality_score']:.4f} (0-1 scale, higher is better)\n\n")
            
            f.write("VARIANCE ANALYSIS:\n")
            f.write(f"- Within-cluster Variance: {metrics['within_cluster_variance']:.6f}\n")
            f.write(f"- Between-cluster Variance: {metrics['between_cluster_variance']:.6f}\n")
            f.write(f"- Variance Ratio (Between/Within): {metrics['variance_ratio']:.4f}\n\n")
            
            f.write("GAP STATISTIC:\n")
            gap_data = metrics['gap_statistic']
            f.write(f"- Optimal k: {gap_data['optimal_k']}\n")
            f.write(f"- Max Gap: {max(gap_data['gaps']):.4f}\n\n")
            
            f.write("CLUSTER SUMMARY:\n")
            cluster_details = metrics['cluster_details']
            for cluster_id, details in cluster_details.items():
                f.write(f"- Cluster {cluster_id}: {details['size']} samples, ")
                f.write(f"diameter={details['diameter']:.3f}, density={details['density']:.6f}\n")
            
            f.write("\nINTERPRETATION:\n")
            sil_score = metrics['silhouette_score']
            if sil_score > 0.5:
                f.write("- Excellent cluster separation (Silhouette > 0.5)\n")
            elif sil_score > 0.25:
                f.write("- Good cluster separation (Silhouette > 0.25)\n")
            elif sil_score > 0:
                f.write("- Weak cluster separation (Silhouette > 0)\n")
            else:
                f.write("- Poor clustering (Silhouette ≤ 0)\n")
            
            ch_score = metrics['calinski_harabasz_score']
            if ch_score > 100:
                f.write("- Strong cluster definition (CH Index > 100)\n")
            elif ch_score > 50:
                f.write("- Moderate cluster definition (CH Index > 50)\n")
            else:
                f.write("- Weak cluster definition (CH Index ≤ 50)\n")
        
        logger.info(f"Quality results saved to {output_dir}")

    def analyze_all_layers_quality(self, output_base_dir):
        """모든 레이어의 클러스터 품질 분석"""
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_quality_results = {}
        
        for layer_idx in sorted(self.layer_results.keys()):
            logger.info(f"Starting quality analysis for layer {layer_idx}...")
            
            # 레이어별 출력 디렉토리
            layer_output_dir = output_base_dir / f'layer_{layer_idx}'
            
            # 품질 분석 수행
            quality_metrics = self.analyze_layer_quality(layer_idx, layer_output_dir)
            if quality_metrics:
                all_quality_results[layer_idx] = quality_metrics
        
        # 전체 레이어 비교 분석
        self._create_cross_layer_quality_comparison(all_quality_results, output_base_dir)
        
        # 전체 요약 리포트
        self._create_overall_quality_report(all_quality_results, output_base_dir)
        
        logger.info("All layer quality analysis completed!")
        return all_quality_results

    def _create_cross_layer_quality_comparison(self, all_results, output_dir):
        """레이어 간 품질 비교 시각화"""
        if not all_results:
            logger.warning("No results available for cross-layer comparison")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        layers = sorted(all_results.keys())
        
        # 1. 주요 메트릭 비교
        silhouette_scores = [all_results[layer]['silhouette_score'] for layer in layers]
        ch_scores = [all_results[layer]['calinski_harabasz_score'] for layer in layers]
        db_scores = [all_results[layer]['davies_bouldin_score'] for layer in layers]
        quality_scores = [all_results[layer]['quality_score'] for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.2
        
        ax1.bar(x - 1.5*width, silhouette_scores, width, label='Silhouette', color='skyblue')
        ax1.bar(x - 0.5*width, [score/1000 for score in ch_scores], width, label='CH/1000', color='lightgreen')
        ax1.bar(x + 0.5*width, [1/(1+score) if score != float('inf') else 0 for score in db_scores], 
                width, label='1/(1+DB)', color='lightcoral')
        ax1.bar(x + 1.5*width, quality_scores, width, label='Quality', color='gold')
        
        ax1.set_title('Clustering Quality Metrics Across Layers')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{layer}' for layer in layers])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance Ratio 비교
        variance_ratios = [all_results[layer]['variance_ratio'] for layer in layers]
        bars2 = ax2.bar([f'L{layer}' for layer in layers], variance_ratios, color='purple', alpha=0.7)
        ax2.set_title('Between/Within Variance Ratio')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3)
        
        for bar, ratio in zip(bars2, variance_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Gap Statistic Optimal k
        optimal_ks = [all_results[layer]['gap_statistic']['optimal_k'] for layer in layers]
        bars3 = ax3.bar([f'L{layer}' for layer in layers], optimal_ks, color='orange', alpha=0.7)
        ax3.set_title('Gap Statistic Optimal k')
        ax3.set_ylabel('Optimal k')
        ax3.grid(True, alpha=0.3)
        
        for bar, k in zip(bars3, optimal_ks):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{k}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 4. 클러스터 수 vs 품질 점수
        n_clusters = [len(all_results[layer]['cluster_details']) for layer in layers]
        
        scatter = ax4.scatter(n_clusters, quality_scores, c=layers, cmap='viridis', s=100, alpha=0.7)
        ax4.set_title('Number of Clusters vs Quality Score')
        ax4.set_xlabel('Number of Clusters')
        ax4.set_ylabel('Quality Score')
        ax4.grid(True, alpha=0.3)
        
        # 레이어 라벨 표시
        for i, (n_clust, qual, layer) in enumerate(zip(n_clusters, quality_scores, layers)):
            ax4.annotate(f'L{layer}', (n_clust, qual), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.colorbar(scatter, ax=ax4, label='Layer Index')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'cross_layer_quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Cross-layer quality comparison plot saved")

    def _create_overall_quality_report(self, all_results, output_dir):
        """전체 품질 분석 요약 리포트"""
        report_file = output_dir / 'overall_quality_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("OVERALL CLUSTER QUALITY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            if not all_results:
                f.write("No results available.\n")
                return
            
            layers = sorted(all_results.keys())
            
            # 1. 최고 품질 레이어 찾기
            quality_scores = [(layer, all_results[layer]['quality_score']) for layer in layers]
            best_layer, best_score = max(quality_scores, key=lambda x: x[1])
            
            f.write("SUMMARY:\n")
            f.write(f"- Analyzed {len(layers)} layers\n")
            f.write(f"- Best performing layer: Layer {best_layer} (Quality Score: {best_score:.4f})\n")
            
            # 2. 메트릭별 최고/최저
            silhouette_scores = [(layer, all_results[layer]['silhouette_score']) for layer in layers]
            best_sil_layer, best_sil = max(silhouette_scores, key=lambda x: x[1])
            worst_sil_layer, worst_sil = min(silhouette_scores, key=lambda x: x[1])
            
            f.write(f"- Best Silhouette Score: Layer {best_sil_layer} ({best_sil:.4f})\n")
            f.write(f"- Worst Silhouette Score: Layer {worst_sil_layer} ({worst_sil:.4f})\n\n")
            
            # 3. 레이어별 상세 요약
            f.write("LAYER-BY-LAYER SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            for layer in layers:
                result = all_results[layer]
                f.write(f"Layer {layer}:\n")
                f.write(f"  Quality Score: {result['quality_score']:.4f}\n")
                f.write(f"  Silhouette: {result['silhouette_score']:.4f}\n")
                f.write(f"  CH Index: {result['calinski_harabasz_score']:.2f}\n")
                f.write(f"  DB Score: {result['davies_bouldin_score']:.4f}\n")
                f.write(f"  Clusters: {len(result['cluster_details'])}\n")
                f.write(f"  Gap Optimal k: {result['gap_statistic']['optimal_k']}\n\n")
            
            # 4. 권장사항
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            if best_score > 0.6:
                f.write(f"- Layer {best_layer} shows excellent clustering quality\n")
            elif best_score > 0.4:
                f.write(f"- Layer {best_layer} shows good clustering quality\n")
            else:
                f.write("- All layers show relatively poor clustering quality\n")
                f.write("- Consider different clustering parameters or preprocessing\n")
            
            # 클러스터 수 일관성 체크
            cluster_counts = [len(all_results[layer]['cluster_details']) for layer in layers]
            if len(set(cluster_counts)) == 1:
                f.write("- Cluster counts are consistent across layers\n")
            else:
                f.write("- Cluster counts vary across layers - consider the impact on model behavior\n")
        
        logger.info(f"Overall quality report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Cluster Quality Metrics Evaluation')
    parser.add_argument('--clustering_dir', type=str, required=True,
                       help='Directory containing clustering results from inference.py')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for quality analysis results')
    parser.add_argument('--layer_idx', type=int, default=None,
                       help='Specific layer to analyze (default: analyze all layers)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        clustering_dir = Path(args.clustering_dir)
        args.output_dir = clustering_dir / 'quality_analysis'
    
    # 분석기 초기화
    evaluator = ClusterQualityEvaluator(args.clustering_dir, args.checkpoint_dir)
    
    if args.layer_idx is not None:
        # 특정 레이어만 분석
        logger.info(f"Analyzing quality for layer {args.layer_idx}...")
        output_dir = Path(args.output_dir) / f'layer_{args.layer_idx}'
        evaluator.analyze_layer_quality(args.layer_idx, output_dir)
    else:
        # 모든 레이어 분석
        logger.info("Analyzing quality for all layers...")
        evaluator.analyze_all_layers_quality(args.output_dir)

    logger.info(f"Quality analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()