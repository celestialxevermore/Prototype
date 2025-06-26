"""
Complete Clustering Pipeline: Train clustering + Valid/Test mapping
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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pickle

current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringInference:
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
        
        # 원본 tabular 데이터 로드
        self._load_original_tabular_data()
        
    def _load_model(self):
        """체크포인트에서 모델 로드"""
        experiment_id = "inference"
        mode = "inference"
        
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
        results = prepare_embedding_dataloaders(self.args, self.args.source_data)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']
        self.num_classes = results['num_classes']
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_data}")
        logger.info(f"Training data size: {len(self.train_loader.dataset)}")
        logger.info(f"Validation data size: {len(self.val_loader.dataset)}")
        logger.info(f"Test data size: {len(self.test_loader.dataset)}")
    
    def _load_original_tabular_data(self):
        """원본 tabular 데이터 로드 - make_embed_dataset.py의 로직 활용"""
        
        # TabularToEmbeddingDataset 클래스 import 및 초기화
        from make_embed_dataset import TabularToEmbeddingDataset
        
        dataset_loader = TabularToEmbeddingDataset(self.args)
        
        # 원본 CSV 파일 로드 (TabularToEmbeddingDataset와 동일한 경로)
        base_path = "/storage/personal/eungyeop/dataset/table/"
        data_source =  "origin_table"
        csv_path = os.path.join(base_path, data_source, f"{self.args.source_data}.csv")
        
        if os.path.exists(csv_path):
            raw_data = pd.read_csv(csv_path)
            
            # 전처리 적용 (임베딩 생성 시와 동일하게)
            X, y = dataset_loader.preprocessing(raw_data, self.args.source_data)
            
            # X와 y를 합쳐서 전체 데이터 생성
            self.original_data = X.copy()
            self.original_data['target_binary'] = y
            
            logger.info(f"Loaded original tabular data from: {csv_path}")
            logger.info(f"Original data shape: {self.original_data.shape}")
            logger.info(f"Columns: {list(self.original_data.columns)}")
            
        else:
            raise FileNotFoundError(f"Original data not found: {csv_path}")
    
    def extract_attention_maps(self, data_loader, split_name):
        """
        데이터로더에서 attention maps 추출
        
        Args:
            data_loader: 데이터로더
            split_name: 'train', 'valid', 'test'
            
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
                    feature_names = self._extract_feature_names(batch_on_device)
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
                        label = -1
                    attention_data['labels'].append(label)
                    
                    # 샘플 ID (s_idx 사용)
                    if 's_idx' in batch:
                        sample_id = batch['s_idx'][sample_idx].item()
                    else:
                        sample_id = sample_count
                    attention_data['sample_ids'].append(sample_id)
                    
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} {split_name} samples...")
        
        logger.info(f"Extracted attention maps for {sample_count} {split_name} samples")
        return attention_data
    
    def _extract_attention_from_model(self, batch):
        """모델에서 attention weights와 예측값을 추출"""
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
    
    def _extract_feature_names(self, batch):
        """Feature names 추출 (임시 구현)"""
        feature_count = 0
        
        if 'cat_name_value_embeddings' in batch:
            feature_count += batch['cat_name_value_embeddings'].shape[1]
        if 'num_prompt_embeddings' in batch:
            feature_count += batch['num_prompt_embeddings'].shape[1]
        
        return [f"feature_{i}" for i in range(feature_count)]
    
    def perform_train_clustering(self, layer_idx=2, n_clusters=8, output_dir=None):
        """
        Train set의 attention maps에 대해 K-means 클러스터링 수행
        
        Args:
            layer_idx: 클러스터링할 레이어 인덱스
            n_clusters: 클러스터 수
            output_dir: 결과 저장 디렉토리
            
        Returns:
            dict: 클러스터링 결과
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train attention maps 추출
        attention_data = self.extract_attention_maps(self.train_loader, 'train')
        
        # 특정 레이어의 attention maps 사용
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        sample_ids = np.array(attention_data['sample_ids'])
        
        logger.info(f"Performing clustering on layer {layer_idx} with {len(attention_maps)} train samples")
        
        # 평탄화 (벡터화)
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        # K-means 클러스터링
        #kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        #cluster_assignments = kmeans.fit_predict(flattened_maps)
        checkpoint_file = Path(self.checkpoint_dir)
        config_folder = extract_checkpoint_config_for_folder(self.checkpoint_dir)

        # clustering1.py에서 저장된 KMeans 모델 경로 구성
        checkpoint_parent_str = str(checkpoint_file.parent)
        if '/checkpoints/' in checkpoint_parent_str:
            clustering_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/clustering/')
        else:
            clustering_parent_str = '/storage/personal/eungyeop/experiments/clustering/gpt2_mean/heart/Full'

        kmeans_model_path = (Path(clustering_parent_str) / config_folder / 
                            f'clustering1_{n_clusters}/clustering_results/layer_{layer_idx}' / 
                            f'layer_{layer_idx}_kmeans_model.pkl')

        if kmeans_model_path.exists():

            # 기존 KMeans 모델 로드
            with open(kmeans_model_path, 'rb') as f:
                kmeans = pickle.load(f)
            logger.info(f"🔥 Loaded existing KMeans model from: {kmeans_model_path}")
            
            # Train 데이터를 기존 모델로 클러스터 할당 (fit하지 않고 predict만)
            cluster_assignments = kmeans.predict(flattened_maps)
            
        else:
            pdb.set_trace()
            # 모델이 없으면 새로 생성 (fallback)
            logger.warning(f"KMeans model not found at: {kmeans_model_path}")
            logger.warning("Creating new KMeans model...")
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
        
        clustering_results = {
            'cluster_assignments': cluster_assignments,
            'cluster_centers': kmeans.cluster_centers_,
            'labels': labels,
            'sample_ids': sample_ids,
            'layer_idx': layer_idx,
            'kmeans_model': kmeans
        }
        
        # CSV 생성
        if output_dir:
            self.save_cluster_csvs(clustering_results, output_dir, n_clusters, 'train')
        
        return clustering_results
    
    def map_to_clusters_by_distance(self, attention_data, train_centroids, layer_idx):
        """
        Train centroid와의 거리 기반으로 클러스터 할당
        
        Args:
            attention_data: extract_attention_maps 결과
            train_centroids: Train으로 학습된 클러스터 centroids
            layer_idx: 사용할 레이어 인덱스
            
        Returns:
            dict: 클러스터 할당 결과
        """
        attention_maps = np.stack(attention_data[f'layer_{layer_idx}'])
        labels = np.array(attention_data['labels'])
        sample_ids = np.array(attention_data['sample_ids'])
        
        # 평탄화
        flattened_maps = attention_maps.reshape(len(attention_maps), -1)
        
        # 각 샘플과 train centroids 간의 거리 계산
        distances = pairwise_distances(flattened_maps, train_centroids, metric='euclidean')
        
        # 가장 가까운 클러스터 할당
        cluster_assignments = np.argmin(distances, axis=1)
        
        return {
            'cluster_assignments': cluster_assignments,
            'labels': labels,
            'sample_ids': sample_ids,
            'distances': distances
        }
    
    def save_cluster_csvs(self, clustering_results, output_dir, n_clusters, split_name):
        """
        클러스터별로 원본 tabular 데이터를 새로운 구조로 CSV 저장
        
        Args:
            clustering_results: 클러스터링 결과
            output_dir: 저장할 디렉토리
            n_clusters: 클러스터 수
            split_name: 'train', 'valid', 'test'
        """
        cluster_assignments = clustering_results['cluster_assignments']
        sample_ids = clustering_results['sample_ids']
        labels = clustering_results['labels']
        
        # {split}_clustering_{n_clusters} 폴더 생성
        split_dir = output_dir / f'{split_name}_clustering_{n_clusters}'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for cluster_id in range(n_clusters):
            # 각 클러스터별 폴더 생성
            cluster_dir = split_dir / f'cluster_{cluster_id}'
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            # 해당 클러스터에 속한 샘플들의 인덱스
            cluster_mask = cluster_assignments == cluster_id
            cluster_sample_ids = sample_ids[cluster_mask]
            cluster_labels = labels[cluster_mask]
            
            if len(cluster_sample_ids) > 0:
                # 원본 데이터에서 해당 인덱스의 샘플들 추출
                cluster_data = self.original_data.iloc[cluster_sample_ids].copy()
                
                # 클러스터 정보 추가
                cluster_data['cluster_id'] = cluster_id
                cluster_data['original_index'] = cluster_sample_ids
                
                # CSV 저장 (클러스터 폴더 안에)
                csv_filename = f'cluster_{cluster_id}_{split_name}.csv'
                csv_path = cluster_dir / csv_filename
                cluster_data.to_csv(csv_path, index=False)
                
                logger.info(f"Saved {split_name} cluster {cluster_id}: {len(cluster_data)} samples to {csv_path}")
                
            else:
                logger.warning(f"{split_name} cluster {cluster_id} has no samples")
        
        # 전체 요약 정보 저장
        summary = {
            'total_samples': len(sample_ids),
            'n_clusters': n_clusters,
            'split': split_name,
            'cluster_summary': {}
        }
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_count = np.sum(cluster_mask)
            cluster_labels = labels[cluster_mask]
            
            summary['cluster_summary'][f'cluster_{cluster_id}'] = {
                'sample_count': int(cluster_count),
                'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(cluster_labels, return_counts=True))}
            }
        
        summary_path = split_dir / 'clustering_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"{split_name} clustering summary saved to: {summary_path}")
        logger.info(f"✅ All {split_name} cluster CSVs saved in: {split_dir}")

    def save_whole_test_set(self, test_attention_data, output_dir):
        """
        Test set을 전체로 저장 (클러스터 분할 없음)
        
        Args:
            test_attention_data: extract_attention_maps 결과
            output_dir: 저장할 디렉토리
        """
        test_sample_ids = np.array(test_attention_data['sample_ids'])
        test_labels = np.array(test_attention_data['labels'])
        
        # Test 데이터 추출
        test_data = self.original_data.iloc[test_sample_ids].copy()
        test_data['original_index'] = test_sample_ids
        
        # Test 폴더 생성 및 저장
        test_dir = output_dir / 'test_full'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_csv_path = test_dir / 'test_full.csv'
        test_data.to_csv(test_csv_path, index=False)
        
        logger.info(f"Saved full test set: {len(test_data)} samples to {test_csv_path}")
        
        # Test set 요약 정보 저장
        test_summary = {
            'total_samples': len(test_sample_ids),
            'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))}
        }
        
        test_summary_path = test_dir / 'test_summary.json'
        with open(test_summary_path, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        logger.info(f"Test summary saved to: {test_summary_path}")

    def save_full_population_sets(self, train_results, valid_results, output_dir):
        """
        전체 population용 train/valid set 저장 (클러스터 구분 없음)
        
        Args:
            train_results: Train 클러스터링 결과
            valid_results: Valid 클러스터 할당 결과
            output_dir: 저장할 디렉토리
        """
        # Full population 폴더 생성
        full_pop_dir = output_dir / 'full_population'
        full_pop_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Full train set 저장
        train_sample_ids = train_results['sample_ids']
        train_data = self.original_data.iloc[train_sample_ids].copy()
        train_data['original_index'] = train_sample_ids
        
        train_csv_path = full_pop_dir / 'train_full.csv'
        train_data.to_csv(train_csv_path, index=False)
        logger.info(f"Saved full train set: {len(train_data)} samples to {train_csv_path}")
        
        # 2. Full valid set 저장
        valid_sample_ids = valid_results['sample_ids']
        valid_data = self.original_data.iloc[valid_sample_ids].copy()
        valid_data['original_index'] = valid_sample_ids
        
        valid_csv_path = full_pop_dir / 'valid_full.csv'
        valid_data.to_csv(valid_csv_path, index=False)
        logger.info(f"Saved full valid set: {len(valid_data)} samples to {valid_csv_path}")
        
        # 3. 요약 정보 저장
        full_pop_summary = {
            'train_samples': len(train_sample_ids),
            'valid_samples': len(valid_sample_ids),
            'train_label_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_results['labels'], return_counts=True))},
            'valid_label_distribution': {int(k): int(v) for k, v in zip(*np.unique(valid_results['labels'], return_counts=True))}
        }
        
        summary_path = full_pop_dir / 'full_population_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(full_pop_summary, f, indent=2)
        
        logger.info(f"Full population summary saved to: {summary_path}")
        logger.info(f"✅ Full population train/valid sets saved in: {full_pop_dir}")

    def run_complete_pipeline(self, layer_idx=2, n_clusters=8, output_dir=None):
        """
        전체 파이프라인 실행: Train clustering + Valid/Test mapping
        
        Args:
            layer_idx: 클러스터링할 레이어 인덱스
            n_clusters: 클러스터 수
            output_dir: 결과 저장 디렉토리
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Train 클러스터링
        logger.info("=== Step 1: Train Clustering ===")
        train_results = self.perform_train_clustering(layer_idx, n_clusters, output_dir)
        
        # Train centroids 저장
        centroids_path = output_dir / 'cluster_centroids.npy'
        np.save(centroids_path, train_results['cluster_centers'])
        logger.info(f"Train centroids saved to: {centroids_path}")
        
        # 2. Valid attention maps 추출 및 클러스터 매핑
        logger.info("=== Step 2: Valid Cluster Mapping ===")
        valid_attention_data = self.extract_attention_maps(self.val_loader, 'valid')
        valid_results = self.map_to_clusters_by_distance(
            valid_attention_data, 
            train_results['cluster_centers'], 
            layer_idx
        )
        
        # Valid CSV 저장
        self.save_cluster_csvs(valid_results, output_dir, n_clusters, 'valid')
        
        # 3. Test attention maps 추출 및 클러스터 매핑
        logger.info("=== Step 3: Test Cluster Mapping ===")
        test_attention_data = self.extract_attention_maps(self.test_loader, 'test')
        self.save_whole_test_set(test_attention_data, output_dir)
        logger.info("=== Step 4: Saving Full Population Train/Valid Sets ===")
        self.save_full_population_sets(train_results, valid_results, output_dir)
        
        logger.info("✅ Complete clustering pipeline finished!")
        
        # 결과 요약
        logger.info("\n=== CLUSTERING SUMMARY ===")
        logger.info(f"Train samples: {len(train_results['sample_ids'])}")
        logger.info(f"Valid samples: {len(valid_results['sample_ids'])}")
        logger.info(f"Test samples: {len(test_attention_data['sample_ids'])}")

def extract_checkpoint_config_for_folder(checkpoint_path):
    """체크포인트 파일명에서 설정 정보를 추출해서 폴더명으로 변환"""
    filename = Path(checkpoint_path).stem
    
    # 날짜/시간 패턴 제거 (20250617_173832 형태)
    import re
    filename_clean = re.sub(r'_\d{8}_\d{6}', '', filename)
    
    # "Embed:carte_desc_Edge:mlp_A:att" 형태를 파싱
    pattern = r'Embed:([^:_]+(?:_[^:_]+)*?)_Edge:([^:_]+)_A:([^:_]+)'
    match = re.match(pattern, filename_clean)
    
    if match:
        embed_type = match.group(1)  # carte, carte_desc, ours, ours2
        edge_attr = match.group(2)   # mlp, no_use, normal
        attn_type = match.group(3)   # att, gat
        
        # 폴더명 생성: Embed-carte_desc_Edge-mlp_A-att
        folder_name = f"Embed-{embed_type}_Edge-{edge_attr}_A-{attn_type}"
        return folder_name
    else:
        # 패턴 매칭 실패시 원본 사용하되 콜론을 대시로 변경
        logger.warning(f"Could not parse config from filename: {filename_clean}")
        return filename_clean.replace(':', '-')


def main():
    parser = argparse.ArgumentParser(description='Complete Clustering Pipeline')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--layer_idx', type=int, default=2,
                       help='Layer index for clustering (default: 2)')
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='Number of clusters for K-means')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--train_only', action='store_true',
                       help='Only perform train clustering (skip valid/test)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정: clustering 폴더로 변경
    if args.output_dir is None:
        checkpoint_file = Path(args.checkpoint_dir)
        config_folder = extract_checkpoint_config_for_folder(args.checkpoint_dir)
        
        # 체크포인트 파일의 부모 디렉토리 경로를 가져와서 변환
        checkpoint_parent_str = str(checkpoint_file.parent)
        
        # checkpoints를 clustering으로 변경
        if '/checkpoints/' in checkpoint_parent_str:
            clustering_parent_str = checkpoint_parent_str.replace('/checkpoints/', '/clustering/')
        else:
            # fallback: 직접 경로 구성
            clustering_parent_str = '/storage/personal/eungyeop/experiments/clustering/gpt2_mean/heart/Full'
        
        # 최종 출력 경로: .../clustering/.../config_folder/clustering_{n_clusters}/
        args.output_dir = Path(clustering_parent_str) / config_folder / f'clustering_{args.n_clusters}'
    
    logger.info(f"📁 Results will be saved to: {args.output_dir}")

    # Pipeline 실행
    pipeline = ClusteringInference(args.checkpoint_dir)
    
    if args.train_only:
        # Train만 수행
        logger.info("Performing train-only clustering...")
        train_results = pipeline.perform_train_clustering(
            layer_idx=args.layer_idx,
            n_clusters=args.n_clusters,
            output_dir=args.output_dir
        )
        
        # Train centroids 저장
        centroids_path = args.output_dir / 'cluster_centroids.npy'
        np.save(centroids_path, train_results['cluster_centers'])
        logger.info(f"Train centroids saved to: {centroids_path}")
        
        logger.info(f"Train clustering completed! Results saved to {args.output_dir}")
    else:
        # 전체 파이프라인 실행
        pipeline.run_complete_pipeline(
            layer_idx=args.layer_idx,
            n_clusters=args.n_clusters,
            output_dir=args.output_dir
        )
        
        logger.info(f"Complete clustering pipeline completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()