"""
Cluster-based Training Pipeline: MoE vs Full Population comparison
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import torch
from datetime import datetime
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, pairwise_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil 
p = psutil.Process()

p.cpu_affinity(range(1, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
current_dir = Path(__file__).resolve().parent
import sys
# clustering/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가

from utils.util import setup_logger, format_time, fix_seed
from models.LogReg import logistic_regression_benchmark
from utils.metrics import compute_overall_accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterTrainingExperiment:
    def __init__(self, clustering_dir, device='cpu'):
        """
        Args:
            clustering_dir (str): 클러스터링 결과가 저장된 디렉토리
            device (str): 'cuda' 또는 'cpu'
        """
        self.clustering_dir = Path(clustering_dir)
        self.device = device
        
        # 클러스터 수 추출 (clustering_8 -> 8)
        self.n_clusters = int(self.clustering_dir.name.split('_')[-1])
        
        logger.info(f"Initialized cluster training experiment")
        logger.info(f"Clustering directory: {self.clustering_dir}")
        logger.info(f"Number of clusters: {self.n_clusters}")
        
        # 클러스터별 모델 저장용
        self.cluster_models = {}
        self.cluster_results = {}
        
    def load_cluster_data(self, cluster_id, split):
        """
        특정 클러스터의 특정 split 데이터 로드
        
        Args:
            cluster_id (int): 클러스터 ID
            split (str): 'train', 'valid', 'test'
            
        Returns:
            tuple: (X, y) 또는 None if not exists
        """
        if split == 'test':
            csv_path = self.clustering_dir / f'{split}_clustering_{self.n_clusters}' / f'cluster_{cluster_id}' / f'cluster_{cluster_id}_{split}.csv'
        else:
            csv_path = self.clustering_dir / f'{split}_clustering_{self.n_clusters}' / f'cluster_{cluster_id}' / f'cluster_{cluster_id}_{split}.csv'
        
        if not csv_path.exists():
            logger.warning(f"Data not found: {csv_path}")
            return None, None
            
        df = pd.read_csv(csv_path)
        
        # target_binary 컬럼을 y로, 나머지를 X로
        if 'target_binary' not in df.columns:
            logger.error(f"target_binary column not found in {csv_path}")
            return None, None
            
        # 메타데이터 컬럼 제거
        meta_columns = ['cluster_id', 'original_index']
        feature_columns = [col for col in df.columns if col not in ['target_binary'] + meta_columns]
        
        X = df[feature_columns]
        y = df['target_binary']
        
        logger.info(f"Loaded cluster {cluster_id} {split}: {len(df)} samples, {len(feature_columns)} features")
        return X, y
    
    def load_full_population_data(self):
        """
        전체 population 데이터 로드
        
        Returns:
            tuple: (X_train, X_valid, y_train, y_valid)
        """
        train_path = self.clustering_dir / 'full_population' / 'train_full.csv'
        valid_path = self.clustering_dir / 'full_population' / 'valid_full.csv'
        
        # Train 데이터
        if not train_path.exists():
            logger.error(f"Full population train data not found: {train_path}")
            return None, None, None, None
            
        train_df = pd.read_csv(train_path)
        meta_columns = ['original_index']
        feature_columns = [col for col in train_df.columns if col not in ['target_binary'] + meta_columns]
        
        X_train_full = train_df[feature_columns]
        y_train_full = train_df['target_binary']
        
        # Valid 데이터
        if not valid_path.exists():
            logger.error(f"Full population valid data not found: {valid_path}")
            return None, None, None, None
            
        valid_df = pd.read_csv(valid_path)
        X_valid_full = valid_df[feature_columns]
        y_valid_full = valid_df['target_binary']
        
        logger.info(f"Loaded full population - Train: {len(X_train_full)}, Valid: {len(X_valid_full)}")
        return X_train_full, X_valid_full, y_train_full, y_valid_full

    def load_test_data(self):
        """
        전체 Test 데이터 로드
        
        Returns:
            tuple: (X_test, y_test)
        """
        test_path = self.clustering_dir / 'test_full' / 'test_full.csv'
        
        if not test_path.exists():
            logger.error(f"Test data not found: {test_path}")
            return None, None
            
        test_df = pd.read_csv(test_path)
        meta_columns = ['original_index']
        feature_columns = [col for col in test_df.columns if col not in ['target_binary'] + meta_columns]
        
        X_test = test_df[feature_columns]
        y_test = test_df['target_binary']
        
        logger.info(f"Loaded test data: {len(X_test)} samples")
        return X_test, y_test
    
    def train_cluster_models(self, args):
        """
        각 클러스터별로 LogReg 모델 학습 및 저장
        
        Args:
            args: 실험 설정
            
        Returns:
            dict: 클러스터별 모델 결과 및 저장된 모델 정보
        """
        logger.info("=== Training cluster-wise models ===")
        
        cluster_results = {}
        models_dir = self.clustering_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        for cluster_id in range(self.n_clusters):
            logger.info(f"Training cluster {cluster_id}...")
            
            # 클러스터 데이터 로드
            X_train, y_train = self.load_cluster_data(cluster_id, 'train')
            X_valid, y_valid = self.load_cluster_data(cluster_id, 'valid')
            
            if X_train is None or X_valid is None:
                logger.warning(f"Skipping cluster {cluster_id} due to missing data")
                continue
                
            if len(X_train) == 0 or len(X_valid) == 0:
                logger.warning(f"Skipping cluster {cluster_id} due to empty data")
                continue
            
            # 클래스가 1개만 있는 경우 스킵
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Skipping cluster {cluster_id} - only one class in training data")
                continue
            
            # 이진 분류 여부 확인
            is_binary = (len(np.unique(y_train)) == 2)
            
            try:
                # LogReg 학습하고 실제 모델 저장
                result = self.train_and_save_cluster_model(
                    cluster_id, X_train, X_valid, y_train, y_valid, 
                    args, is_binary, models_dir
                )
                
                if result is not None:
                    cluster_results[f'cluster_{cluster_id}'] = result
                    logger.info(f"Cluster {cluster_id} training completed - Valid AUC: {result['valid_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training cluster {cluster_id}: {str(e)}")
                continue
        
        logger.info(f"Completed training {len(cluster_results)} cluster models")
        return cluster_results
    
    def train_and_save_cluster_model(self, cluster_id, X_train, X_valid, y_train, y_valid, args, is_binary, models_dir):
        """
        개별 클러스터 모델 학습 및 저장 - 기존 LogReg benchmark 사용
        """
        from models.LogReg import logistic_regression_benchmark
        from models.MLP import mlp_benchmark
        import pickle
        
        try:
            # 기존 LogReg benchmark 함수 사용 (공정한 비교)
            result, trained_model = logistic_regression_benchmark(
                args,
                X_train, X_valid, X_valid,  # test는 valid로 임시 사용 (validation만 필요)
                y_train, y_valid, y_valid,
                is_binary=is_binary
            )
            
            # 결과에서 validation 성능 추출
            valid_auc = result.get('test_lr_auc', 0.0)  # test 자리에 valid를 넣었으므로
            valid_auprc = result.get('test_lr_auprc', 0.0)
            best_c = result.get('best_lr_c', 1.0)
            best_penalty = result.get('best_lr_penalty', 'l2')
            
            # benchmark에서 받은 모델 바로 저장
            model_path = models_dir / f'cluster_{cluster_id}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model, f)
            
            cluster_result = {
                'cluster_id': cluster_id,
                'valid_auc': valid_auc,
                'valid_auprc': valid_auprc,
                'best_params': {'C': best_c, 'penalty': best_penalty},
                'model_path': str(model_path),
                'train_samples': len(X_train),
                'valid_samples': len(X_valid),
                'train_label_dist': dict(zip(*np.unique(y_train, return_counts=True))),
                'valid_label_dist': dict(zip(*np.unique(y_valid, return_counts=True)))
            }
            
            logger.info(f"Cluster {cluster_id}: AUC={valid_auc:.4f}, AUPRC={valid_auprc:.4f}, "
                    f"C={best_c}, penalty={best_penalty}, Samples={len(X_train)}")
            
            return cluster_result
            
        except Exception as e:
            logger.error(f"Error training cluster {cluster_id}: {e}")
            return None
    
    def train_full_population_model(self, args):
        """
        전체 population으로 단일 모델 학습 및 저장
        
        Args:
            args: 실험 설정
            
        Returns:
            dict: 전체 모델 결과
        """
        logger.info("=== Training full population model ===")
        
        # 전체 데이터 로드
        X_train_full, X_valid_full, y_train_full, y_valid_full = self.load_full_population_data()
        
        if X_train_full is None:
            logger.error("Failed to load full population data")
            return None
        
        # 이진 분류 여부 확인
        is_binary = (len(np.unique(y_train_full)) == 2)
        
        try:
            # 모델 학습 및 저장
            models_dir = self.clustering_dir / 'trained_models'
            models_dir.mkdir(exist_ok=True)
            
            result = self.train_and_save_full_model(
                X_train_full, X_valid_full, y_train_full, y_valid_full,
                args, is_binary, models_dir
            )
            
            if result is not None:
                logger.info(f"Full population training completed - Valid AUC: {result['valid_auc']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training full population model: {str(e)}")
            return None
    
    def train_and_save_full_model(self, X_train, X_valid, y_train, y_valid, args, is_binary, models_dir):
        """
        전체 population 모델 학습 및 저장 - 기존 LogReg benchmark 사용
        """
        from models.LogReg import logistic_regression_benchmark
        import pickle
        
        try:
            # 기존 LogReg benchmark 함수 사용 (클러스터와 동일한 방식)
            result, trained_model = logistic_regression_benchmark(
                args,
                X_train, X_valid, X_valid,  # test는 valid로 임시 사용
                y_train, y_valid, y_valid,
                is_binary=is_binary
            )
            
            # 결과에서 validation 성능 추출
            valid_auc = result.get('test_lr_auc', 0.0)
            valid_auprc = result.get('test_lr_auprc', 0.0)
            best_c = result.get('best_lr_c', 1.0)
            best_penalty = result.get('best_lr_penalty', 'l2')
            
            # benchmark에서 받은 모델 바로 저장
            model_path = models_dir / 'full_population_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model, f)
            
            full_result = {
                'valid_auc': valid_auc,
                'valid_auprc': valid_auprc,
                'best_params': {'C': best_c, 'penalty': best_penalty},
                'model_path': str(model_path),
                'train_samples': len(X_train),
                'valid_samples': len(X_valid),
                'train_label_dist': dict(zip(*np.unique(y_train, return_counts=True))),
                'valid_label_dist': dict(zip(*np.unique(y_valid, return_counts=True)))
            }
            
            logger.info(f"Full population: AUC={valid_auc:.4f}, AUPRC={valid_auprc:.4f}, "
                    f"C={best_c}, penalty={best_penalty}, Samples={len(X_train)}")
            
            return full_result
            
        except Exception as e:
            logger.error(f"Error training full population model: {e}")
            return None
    
    def extract_test_attention_maps(self, checkpoint_dir):
        """
        체크포인트를 로드해서 test 데이터에서 attention maps 직접 추출
        
        Args:
            checkpoint_dir (str): TabularFLM 체크포인트 경로
            
        Returns:
            dict: layer별 attention maps
        """
        logger.info("Extracting attention maps from test data using checkpoint...")
        
        try:
            import torch
            from models.TabularFLM import Model
            from dataset.data_dataloaders import prepare_embedding_dataloaders
            from utils.util import fix_seed
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_dir, map_location=self.device)
            tabular_args = checkpoint['args']
            
            logger.info(f"Loaded checkpoint from {checkpoint_dir}")
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}, Val AUC: {checkpoint['val_auc']:.4f}")
            
            # TabularFLM 모델 초기화
            model = Model(
                tabular_args,
                tabular_args.input_dim,
                tabular_args.hidden_dim,
                tabular_args.output_dim,
                tabular_args.num_layers,
                tabular_args.dropout_rate,
                tabular_args.llm_model,
                "attention_extraction",
                "inference"
            ).to(self.device)
            
            # 모델 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info("TabularFLM model loaded successfully")
            
            # 데이터로더 준비
            results = prepare_embedding_dataloaders(tabular_args, tabular_args.source_data)
            _, _, test_loader = results['loaders']
            
            logger.info(f"Test dataloader prepared: {len(test_loader.dataset)} samples")
            
            # Attention maps 추출
            attention_data = {
                'layer_0': [],
                'layer_1': [], 
                'layer_2': [],
                'labels': [],
                'sample_ids': []
            }
            
            sample_count = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # 배치를 디바이스로 이동
                    batch_on_device = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    
                    # 모델에서 attention weights 추출
                    pred, attention_weights = self._extract_attention_from_tabular_model(model, batch_on_device)
                    
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
                        logger.info(f"Processed {sample_count} test samples...")
            
            logger.info(f"Extracted attention maps for {sample_count} test samples")
            return attention_data
            
        except Exception as e:
            logger.error(f"Error in extract_test_attention_maps: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_attention_from_tabular_model(self, model, batch):
        """TabularFLM 모델에서 attention weights와 예측값을 추출"""
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
        cls_token = model.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        # Graph Attention Layers
        for i, layer in enumerate(model.layers):
            norm_x = model.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_weights)
            x = x + attn_output
        
        # 예측값 계산
        pred = x[:, 0, :]
        pred = model.predictor(pred)

        return pred, attention_weights
    
    def predict_moe_on_test(self, cluster_results, checkpoint_dir):
        """
        MoE 방식으로 test 데이터 예측 (체크포인트로부터 실시간 attention 추출)
        
        Args:
            cluster_results: 클러스터별 모델 결과
            checkpoint_dir: TabularFLM 체크포인트 경로
            
        Returns:
            dict: MoE 예측 결과
        """
        logger.info("=== MoE prediction on test data ===")
        import pickle
        
        # 1. 전체 test 데이터 로드
        X_test_full, y_test_full = self.load_test_data()
        if X_test_full is None:
            logger.error("Failed to load test data")
            return None
        
        # 2. Test 데이터의 attention maps 추출
        test_attention_data = self.extract_test_attention_maps(checkpoint_dir)
        if test_attention_data is None:
            logger.error("Failed to extract test attention maps")
            return None
        
        # 3. 저장된 centroids 로드
        centroids_path = self.clustering_dir / 'cluster_centroids.npy'
        if not centroids_path.exists():
            logger.error(f"Centroids not found: {centroids_path}")
            return None
        
        centroids = np.load(centroids_path)
        logger.info(f"Loaded centroids with shape: {centroids.shape}")
        
        # 4. Test 샘플들의 attention maps를 centroid와 거리 계산으로 클러스터 할당
        test_attention_maps = np.stack(test_attention_data['layer_2'])  # layer_2 사용
        test_flattened = test_attention_maps.reshape(len(test_attention_maps), -1)
        
        # 거리 계산 및 클러스터 할당
        distances = pairwise_distances(test_flattened, centroids, metric='euclidean')
        test_cluster_assignments = np.argmin(distances, axis=1)
        
        logger.info(f"Test cluster assignments: {np.bincount(test_cluster_assignments)}")
        
        # 5. 각 test 샘플을 할당된 클러스터의 모델로 예측 (순서로 매칭)
        all_predictions_proba = []  # AUC/AUPRC용
        all_true_labels = []
        successful_predictions = 0
        
        for i in range(len(test_attention_maps)):
            assigned_cluster = test_cluster_assignments[i]
            cluster_key = f'cluster_{assigned_cluster}'
            
            if cluster_key not in cluster_results:
                logger.warning(f"No model for cluster {assigned_cluster}, skipping sample {i}")
                continue
            
            # 해당 클러스터 모델 로드
            model_path = cluster_results[cluster_key]['model_path']
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # i번째 샘플 사용 (순서 매칭)
                test_sample_df = X_test_full.iloc[[i]]
                true_label = y_test_full.iloc[i]
                
                # 확률값만 수집 (threshold 적용 안함)
                prediction_proba = model.predict_proba(test_sample_df)[0, 1]  # 이진분류 positive class
                
                all_predictions_proba.append(prediction_proba)
                all_true_labels.append(true_label)
                successful_predictions += 1
                
                if successful_predictions % 50 == 0:
                    logger.info(f"Processed {successful_predictions} predictions...")
                
            except Exception as e:
                logger.error(f"Error predicting sample {i} with cluster {assigned_cluster}: {e}")
                continue
        
        if len(all_predictions_proba) == 0:
            logger.error("No predictions generated for MoE")
            return None
        
        # 6. AUROC/AUPRC만 계산
        test_auc = roc_auc_score(all_true_labels, all_predictions_proba)
        test_auprc = average_precision_score(all_true_labels, all_predictions_proba)
        
        # 클러스터별 예측 분포 로깅
        cluster_pred_counts = {}
        for i in range(successful_predictions):
            cluster_id = test_cluster_assignments[i]
            cluster_pred_counts[cluster_id] = cluster_pred_counts.get(cluster_id, 0) + 1
        
        logger.info(f"Predictions per cluster: {cluster_pred_counts}")
        
        moe_result = {
            'test_auc': test_auc,
            'test_auprc': test_auprc,
            'total_test_samples': len(all_predictions_proba),
            'method': 'MoE',
            'cluster_assignments': test_cluster_assignments[:successful_predictions].tolist()
        }
        
        logger.info(f"MoE prediction completed - Test AUC: {test_auc:.4f}, Test AUPRC: {test_auprc:.4f}")
        logger.info(f"Total test samples predicted: {len(all_predictions_proba)}")
        
        return moe_result
    
    def predict_full_on_test(self, full_result):
        """
        Full population 모델로 test 데이터 예측 (실제 모델 로드 및 예측)
        
        Args:
            full_result: 전체 모델 결과
            
        Returns:
            dict: Full population 예측 결과
        """
        logger.info("=== Full population prediction on test data ===")
        import pickle
        
        # Test 데이터 로드
        X_test, y_test = self.load_test_data()
        if X_test is None:
            logger.error("Failed to load test data")
            return None
        
        # 전체 모델 로드
        model_path = full_result['model_path']
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 확률값만 계산
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # 이진분류 가정
            
            # AUROC/AUPRC만 계산
            test_auc = roc_auc_score(y_test, y_pred_proba)
            test_auprc = average_precision_score(y_test, y_pred_proba)
            
            full_test_result = {
                'test_auc': test_auc,
                'test_auprc': test_auprc,
                'total_test_samples': len(X_test),
                'method': 'Full_Population'
            }
            
            logger.info(f"Full population test prediction completed - Test AUC: {test_auc:.4f}, Test AUPRC: {test_auprc:.4f}")
            logger.info(f"Total test samples: {len(X_test)}")
            
            return full_test_result
            
        except Exception as e:
            logger.error(f"Error in full population test prediction: {e}")
            return None
    
    def compare_performance(self, moe_result, full_result):
        """
        MoE vs Full Population 성능 비교
        
        Args:
            moe_result: MoE 예측 결과
            full_result: Full population 예측 결과
            
        Returns:
            dict: 비교 결과
        """
        logger.info("=== Performance Comparison ===")
        
        comparison = {
            'MoE': {
                'test_auc': moe_result['test_auc'],
                'test_auprc': moe_result['test_auprc'],
                'test_samples': moe_result['total_test_samples'],
            },
            'Full_Population': {
                'test_auc': full_result['test_auc'],
                'test_auprc': full_result['test_auprc'],
                'test_samples': full_result['total_test_samples'],
            }
        }
        
        # 성능 차이 계산
        auc_diff = moe_result['test_auc'] - full_result['test_auc']
        auprc_diff = moe_result['test_auprc'] - full_result['test_auprc']
        
        logger.info(f"MoE - AUC: {moe_result['test_auc']:.4f}, AUPRC: {moe_result['test_auprc']:.4f}")
        logger.info(f"Full - AUC: {full_result['test_auc']:.4f}, AUPRC: {full_result['test_auprc']:.4f}")
        logger.info(f"Differences (MoE - Full): AUC: {auc_diff:+.4f}, AUPRC: {auprc_diff:+.4f}")
        
        comparison['Performance_Difference'] = {
            'auc_diff': auc_diff,
            'auprc_diff': auprc_diff,
            'better_method_auc': 'MoE' if auc_diff > 0 else 'Full_Population',
            'better_method_auprc': 'MoE' if auprc_diff > 0 else 'Full_Population'
        }
        
        return comparison

def prepare_cluster_results(moe_result, full_result):
    """
    결과를 기존 format에 맞게 변환
    """
    results = {
        'Best_results': {
            "MoE": {
                "MoE_best_test_auc": moe_result['test_auc'],
                "MoE_best_test_auprc": moe_result['test_auprc'],
                "MoE_test_samples": moe_result['total_test_samples'],
            },
            "Full_Population": {
                "Full_best_test_auc": full_result['test_auc'],
                "Full_best_test_auprc": full_result['test_auprc'],
                "Full_test_samples": full_result['total_test_samples'],
            },
            "Performance_Comparison": {
                "auc_difference": moe_result['test_auc'] - full_result['test_auc'],
                "auprc_difference": moe_result['test_auprc'] - full_result['test_auprc'],
                "better_method_auc": 'MoE' if (moe_result['test_auc'] - full_result['test_auc']) > 0 else 'Full_Population',
                "better_method_auprc": 'MoE' if (moe_result['test_auprc'] - full_result['test_auprc']) > 0 else 'Full_Population',
            }
        }
    }
    return results

def save_cluster_results(clustering_dir, results, n_clusters):
    """
    결과를 clustering 디렉토리에 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cluster_training_results_{n_clusters}_{timestamp}.json"
    filepath = clustering_dir / filename
    
    # 디렉토리에서 dataset 정보 추출
    # /storage/.../clustering/gpt2_mean/heart/Full/Embed-carte_Edge-mlp_A-gat/clustering_8
    path_parts = str(clustering_dir).split('/')
    
    dataset_name = "unknown"
    config_name = "unknown"
    
    # 경로에서 dataset과 config 추출
    for i, part in enumerate(path_parts):
        if part in ['heart', 'adult', 'cleveland', 'bank', 'diabetes', 'blood', 'car', 'communities', 'credit-g', 'myocardial', 'heart_statlog', 'hungarian', 'switzerland']:
            dataset_name = part
        if part.startswith('Embed-') and '_Edge-' in part and '_A-' in part:
            config_name = part
    
    data = {
        "Experimental Memo": "Cluster-based MoE vs Full Population LogReg comparison (AUROC/AUPRC only)",
        "dataset": dataset_name,
        "configuration": config_name,
        "clustering_directory": str(clustering_dir),
        "timestamp": timestamp,
        "experimental_settings": {
            "n_clusters": n_clusters,
            "model_type": "LogisticRegression",
            "comparison_type": "MoE_vs_Full_Population",
            "evaluation_metrics": ["AUROC", "AUPRC"],
            "threshold_optimization": "None (threshold-independent evaluation)"
        },
        "results": results['Best_results']
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Results saved to {filepath}")
    return filepath

def extract_checkpoint_config_for_folder(clustering_dir):
    """클러스터링 디렉토리에서 설정 정보 추출"""
    path_parts = str(clustering_dir).split('/')
    
    # Embed-carte_Edge-mlp_A-gat 형태 찾기
    config_folder = None
    for part in path_parts:
        if part.startswith('Embed-') and '_Edge-' in part and '_A-' in part:
            config_folder = part
            break
    
    return config_folder if config_folder else "unknown_config"

def get_args():
    parser = argparse.ArgumentParser(description='Cluster-based Training Experiment')
    
    parser.add_argument('--clustering_dir', type=str, required=True,
                       help='Path to clustering results directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to TabularFLM checkpoint for attention extraction')
    parser.add_argument('--random_seed', type=int, default=2021, 
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    return args

def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    
    logger.info(f"Starting cluster training experiment")
    logger.info(f"Clustering directory: {args.clustering_dir}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # 실험 실행
    experiment = ClusterTrainingExperiment(args.clustering_dir, args.device)
    
    # 1. 클러스터별 모델 학습
    logger.info("Step 1: Training cluster-wise models...")
    cluster_results = experiment.train_cluster_models(args)
    
    # 2. 전체 population 모델 학습
    logger.info("Step 2: Training full population model...")
    full_result = experiment.train_full_population_model(args)
    
    if not cluster_results or full_result is None:
        logger.error("Failed to train models")
        return
    
    # 3. Test 데이터에서 성능 비교 (checkpoint_dir 전달)
    logger.info("Step 3: Evaluating on test data...")
    moe_result = experiment.predict_moe_on_test(cluster_results, args.checkpoint_dir)
    full_test_result = experiment.predict_full_on_test(full_result)
    
    if moe_result is None or full_test_result is None:
        logger.error("Failed to evaluate on test data")
        return
    
    # 4. 성능 비교
    logger.info("Step 4: Comparing performance...")
    comparison = experiment.compare_performance(moe_result, full_test_result)
    
    # 5. 결과 저장
    logger.info("Step 5: Saving results...")
    results = prepare_cluster_results(moe_result, full_test_result)
    save_cluster_results(Path(args.clustering_dir), results, experiment.n_clusters)
    
    # 6. 상세 결과 출력
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Number of clusters: {experiment.n_clusters}")
    logger.info(f"MoE Test - AUC: {moe_result['test_auc']:.4f}, AUPRC: {moe_result['test_auprc']:.4f}")
    logger.info(f"Full Test - AUC: {full_test_result['test_auc']:.4f}, AUPRC: {full_test_result['test_auprc']:.4f}")
    
    auc_improvement = moe_result['test_auc'] - full_test_result['test_auc']
    auprc_improvement = moe_result['test_auprc'] - full_test_result['test_auprc']
    
    logger.info(f"Improvements (MoE vs Full):")
    logger.info(f"  AUC: {auc_improvement:+.4f}")
    logger.info(f"  AUPRC: {auprc_improvement:+.4f}")
    
    if auc_improvement > 0:
        logger.info("🎉 MoE approach shows better AUC performance!")
    else:
        logger.info("📊 Full population approach shows better AUC performance.")
        
    if auprc_improvement > 0:
        logger.info("🎉 MoE approach shows better AUPRC performance!")
    else:
        logger.info("📊 Full population approach shows better AUPRC performance.")
    
    logger.info("="*50)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")
    logger.info("✅ Cluster training experiment completed!")

if __name__ == "__main__":
    main()