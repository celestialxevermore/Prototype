"""
Complete Cosine Similarity Matrix Analyzer: 모든 클러스터 쌍의 cosine similarity 매트릭스 분석
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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

current_dir = Path(__file__).resolve().parent
import sys
# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteCosineSimilarityAnalyzer:
    def __init__(self, checkpoint_dir, clustering_dir, device='cuda'):
        """
        Args:
            checkpoint_dir (str): 체크포인트 파일 경로
            clustering_dir (str): 클러스터링 결과 디렉토리
            device (str): 'cuda' 또는 'cpu'
        """
        self.checkpoint_dir = checkpoint_dir
        self.clustering_dir = Path(clustering_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.args = self.checkpoint['args']
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"Clustering directory: {clustering_dir}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
        # 클러스터링 결과 로드
        self._load_clustering_results()
        
        # 모든 클러스터의 embedding 추출
        self._extract_all_cluster_embeddings()
        
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
    
    def _load_clustering_results(self):
        """클러스터링 결과 로드"""
        # train_label_clustering_{n_clusters} 폴더에서 클러스터링 정보 로드
        train_dirs = list(self.clustering_dir.glob('train_label_clustering_*'))
        if not train_dirs:
            raise FileNotFoundError(f"No train clustering directories found in {self.clustering_dir}")
        
        train_dir = train_dirs[0]  # 첫 번째 발견된 디렉토리 사용
        
        # 요약 정보 로드
        summary_path = train_dir / 'label_clustering_summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(f"Clustering summary not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            self.clustering_summary = json.load(f)
        
        logger.info(f"Loaded clustering summary from: {summary_path}")
        
        # 라벨별 클러스터 정보 파싱
        self.cluster_info = {}
        for label_key, label_data in self.clustering_summary['label_cluster_summary'].items():
            label_num = int(label_key.split('_')[1])
            self.cluster_info[label_num] = {
                'n_clusters': label_data['n_clusters'],
                'total_samples': label_data['total_samples'],
                'clusters': {}
            }
            
            # 각 클러스터별 샘플 ID 로드
            for cluster_key, cluster_data in label_data['cluster_summary'].items():
                cluster_num = int(cluster_key.split('_')[1])
                
                # CSV 파일에서 실제 샘플 ID들 로드
                csv_path = train_dir / f'label_{label_num}' / f'cluster_{cluster_num}' / f'label_{label_num}_cluster_{cluster_num}_train.csv'
                if csv_path.exists():
                    cluster_df = pd.read_csv(csv_path)
                    sample_ids = cluster_df['original_index'].values
                    self.cluster_info[label_num]['clusters'][cluster_num] = {
                        'sample_ids': sample_ids,
                        'sample_count': len(sample_ids)
                    }
                    logger.info(f"Label {label_num} Cluster {cluster_num}: {len(sample_ids)} samples")
        
        logger.info("✅ Clustering results loaded successfully")
    
    def _extract_all_cluster_embeddings(self):
        """모든 클러스터의 embedding들을 미리 추출해서 저장"""
        logger.info("Extracting embeddings for all clusters...")
        
        self.cluster_embeddings = {}
        
        for label_num, label_data in self.cluster_info.items():
            self.cluster_embeddings[label_num] = {}
            
            for cluster_num, cluster_data in label_data['clusters'].items():
                sample_ids = cluster_data['sample_ids']
                
                logger.info(f"Extracting embeddings for Label {label_num} Cluster {cluster_num} ({len(sample_ids)} samples)...")
                embedding_data = self.extract_embeddings_for_samples(sample_ids, split='train')
                
                if embedding_data['embeddings'].size > 0:
                    self.cluster_embeddings[label_num][cluster_num] = embedding_data['embeddings']
                    logger.info(f"✅ Label {label_num} Cluster {cluster_num}: {embedding_data['embeddings'].shape}")
                else:
                    logger.warning(f"⚠️ No embeddings found for Label {label_num} Cluster {cluster_num}")
        
        logger.info("✅ All cluster embeddings extracted")
    
    def extract_embeddings_for_samples(self, sample_ids, split='train'):
        """
        특정 샘플 ID들에 대한 embedding 추출
        
        Args:
            sample_ids (list): 추출할 샘플 ID 리스트
            split (str): 'train', 'valid', 'test'
            
        Returns:
            dict: {'embeddings': embeddings, 'sample_ids': actual_sample_ids}
        """
        if split == 'train':
            data_loader = self.train_loader
        elif split == 'valid':
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        
        # 목표 샘플 ID를 set으로 변환 (빠른 검색을 위해)
        target_sample_ids = set(sample_ids)
        
        extracted_embeddings = []
        extracted_sample_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 배치를 디바이스로 이동
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 배치 내 샘플 확인
                batch_sample_ids = batch.get('s_idx', None)
                if batch_sample_ids is None:
                    continue
                
                # 목표 샘플이 있는지 확인
                batch_mask = []
                for i, sample_id in enumerate(batch_sample_ids):
                    if sample_id.item() in target_sample_ids:
                        batch_mask.append(i)
                        extracted_sample_ids.append(sample_id.item())
                
                if not batch_mask:
                    continue  # 이 배치에는 목표 샘플이 없음
                
                # 모델에서 embedding 추출 (CLS token embedding 사용)
                embeddings = self._extract_cls_embeddings(batch_on_device)
                
                # 목표 샘플들의 embedding만 추출
                for idx in batch_mask:
                    extracted_embeddings.append(embeddings[idx].detach().cpu().numpy())
        
        if not extracted_embeddings:
            return {'embeddings': np.array([]), 'sample_ids': []}
        
        embeddings_array = np.stack(extracted_embeddings)
        
        return {
            'embeddings': embeddings_array,
            'sample_ids': extracted_sample_ids
        }
    
    def _extract_cls_embeddings(self, batch):
        """모델에서 CLS token embedding 추출"""
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        desc_embeddings = [] 
        name_value_embeddings = [] 
        
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            cat_name_value_embeddings = batch['cat_name_value_embeddings'].to(self.device)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device)
            
            name_value_embeddings.append(cat_name_value_embeddings)
            desc_embeddings.append(cat_desc_embeddings)
            
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device)
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
        
        # Graph Attention Layers 통과
        for i, layer in enumerate(self.model.layers):
            norm_x = self.model.layer_norms[i](x)
            attn_output, _ = layer(desc_embeddings, norm_x)
            x = x + attn_output
        
        # CLS token embedding 반환 (첫 번째 토큰)
        cls_embeddings = x[:, 0, :]  # [batch_size, hidden_dim]
        
        return cls_embeddings
    
    def compute_complete_cosine_similarity_matrix(self, output_dir=None):
        """
        모든 클러스터 쌍 간의 cosine similarity matrix 계산 및 시각화
        
        Returns:
            dict: 완전한 분석 결과
        """
        logger.info("Computing complete cosine similarity matrix...")
        
        # 라벨별 클러스터 개수 확인
        label_0_clusters = list(self.cluster_embeddings[0].keys())
        label_1_clusters = list(self.cluster_embeddings[1].keys())
        
        n_clusters_0 = len(label_0_clusters)
        n_clusters_1 = len(label_1_clusters)
        
        logger.info(f"Label 0: {n_clusters_0} clusters, Label 1: {n_clusters_1} clusters")
        logger.info(f"Total comparisons: {n_clusters_0} × {n_clusters_1} = {n_clusters_0 * n_clusters_1}")
        
        # 전체 결과 구성
        complete_results = {
            'label_0_clusters': label_0_clusters,
            'label_1_clusters': label_1_clusters,
            'n_clusters_0': n_clusters_0,
            'n_clusters_1': n_clusters_1
        }
        
        # 시각화
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._create_cosine_similarity_grid(complete_results, output_dir)
        
        return complete_results
    
    def _create_cosine_similarity_grid(self, results, output_dir):
        """모든 클러스터 쌍의 cosine similarity matrix를 subplot으로 시각화"""
        
        n_clusters_0 = results['n_clusters_0']
        n_clusters_1 = results['n_clusters_1']
        
        # 클러스터 쌍별로 개별 cosine similarity matrix 생성
        fig, axes = plt.subplots(n_clusters_0, n_clusters_1, 
                                figsize=(4*n_clusters_1, 4*n_clusters_0))
        
        # axes를 2D 배열로 만들기
        if n_clusters_0 == 1 and n_clusters_1 == 1:
            axes = np.array([[axes]])
        elif n_clusters_0 == 1:
            axes = axes.reshape(1, -1)
        elif n_clusters_1 == 1:
            axes = axes.reshape(-1, 1)
        
        # 각 클러스터 쌍에 대해 cosine similarity matrix 그리기
        for i, cluster_0 in enumerate(results['label_0_clusters']):
            for j, cluster_1 in enumerate(results['label_1_clusters']):
                
                ax = axes[i, j]
                
                # 해당 클러스터 쌍의 embedding 가져오기
                embeddings_0 = self.cluster_embeddings[0][cluster_0]
                embeddings_1 = self.cluster_embeddings[1][cluster_1]
                
                # Cosine similarity matrix 계산
                cosine_sim_matrix = cosine_similarity(embeddings_0, embeddings_1)
                
                # Heatmap 그리기
                im = ax.imshow(cosine_sim_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # 제목
                ax.set_title(f'L0 C{cluster_0} vs L1 C{cluster_1}', 
                           fontsize=12, fontweight='bold')
                
                # 축 라벨 (작게)
                if i == n_clusters_0 - 1:  # 마지막 행에만 x축 라벨
                    ax.set_xlabel(f'L1 C{cluster_1}', fontsize=10)
                if j == 0:  # 첫 번째 열에만 y축 라벨
                    ax.set_ylabel(f'L0 C{cluster_0}', fontsize=10)
                
                # 틱 제거 (깔끔하게)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Cosine Similarity Matrices for All Cluster Pairs', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 공통 컬러바
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.90, 0.93])
        
        # 저장
        plt.savefig(output_dir / 'all_cosine_similarity_matrices.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ Cosine similarity grid saved to: {output_dir}")
        
        # 개별 매트릭스들도 따로 저장
        self._save_individual_matrices(results, output_dir)
    
    def _save_individual_matrices(self, results, output_dir):
        """각 클러스터 쌍의 cosine similarity matrix를 개별 파일로 저장"""
        individual_dir = output_dir / 'individual_matrices'
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for cluster_0 in results['label_0_clusters']:
            for cluster_1 in results['label_1_clusters']:
                
                # 해당 클러스터 쌍의 embedding 가져오기
                embeddings_0 = self.cluster_embeddings[0][cluster_0]
                embeddings_1 = self.cluster_embeddings[1][cluster_1]
                
                # Cosine similarity matrix 계산
                cosine_sim_matrix = cosine_similarity(embeddings_0, embeddings_1)
                
                # 개별 플롯 생성
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                im = ax.imshow(cosine_sim_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                ax.set_title(f'Cosine Similarity: L0 C{cluster_0} vs L1 C{cluster_1}\n'
                           f'({len(embeddings_0)} × {len(embeddings_1)} samples)', 
                           fontsize=14, fontweight='bold', pad=20)
                
                ax.set_xlabel(f'Label 1 Cluster {cluster_1} Samples', fontsize=12)
                ax.set_ylabel(f'Label 0 Cluster {cluster_0} Samples', fontsize=12)
                
                # 컬러바
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Cosine Similarity', fontsize=12)
                
                plt.tight_layout()
                
                # 저장
                filename = f'L0_C{cluster_0}_vs_L1_C{cluster_1}.png'
                plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                # numpy 배열도 저장
                np.save(individual_dir / f'L0_C{cluster_0}_vs_L1_C{cluster_1}.npy', 
                       cosine_sim_matrix)
        
        logger.info(f"✅ Individual matrices saved to: {individual_dir}")
    



def main():
    parser = argparse.ArgumentParser(description='Complete Cosine Similarity Matrix Visualization')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--clustering_dir', type=str, required=True,
                       help='Path to clustering results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        clustering_dir = Path(args.clustering_dir)
        args.output_dir = clustering_dir / 'cosine_matrices'
    
    logger.info(f"🔍 Creating cosine similarity matrices for all cluster pairs")
    logger.info(f"📁 Results will be saved to: {args.output_dir}")
    
    # 분석 실행
    analyzer = CompleteCosineSimilarityAnalyzer(args.checkpoint_dir, args.clustering_dir, args.device)
    results = analyzer.compute_complete_cosine_similarity_matrix(args.output_dir)
    
    if results:
        logger.info("✅ Cosine similarity matrix visualization completed successfully!")
        logger.info(f"   • Generated {results['n_clusters_0']} × {results['n_clusters_1']} matrices")
    else:
        logger.error("❌ Analysis failed!")
    
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        clustering_dir = Path(args.clustering_dir)
        args.output_dir = clustering_dir / 'complete_cosine_analysis'
    
    logger.info(f"🔍 Computing complete cosine similarity matrix for all cluster pairs")
    logger.info(f"📁 Results will be saved to: {args.output_dir}")
    
    # 분석 실행
    analyzer = CompleteCosineSimilarityAnalyzer(args.checkpoint_dir, args.clustering_dir, args.device)
    results = analyzer.compute_complete_cosine_similarity_matrix(args.output_dir)
    
    if results:
        logger.info("✅ Complete cosine similarity matrix analysis completed successfully!")
        
        # 주요 결과 요약 출력
        #mean_matrix = results['mean_similarity_matrix']
        # logger.info(f"\n🎯 Key Findings:")
        # logger.info(f"   • Matrix Size: {mean_matrix.shape[0]} × {mean_matrix.shape[1]}")
        # logger.info(f"   • Global Mean Similarity: {np.mean(mean_matrix):.4f}")
        # logger.info(f"   • Most Similar Pair: {np.max(mean_matrix):.4f}")
        # logger.info(f"   • Most Different Pair: {np.min(mean_matrix):.4f}")
        
    else:
        logger.error("❌ Analysis failed!")

if __name__ == "__main__":
    main()