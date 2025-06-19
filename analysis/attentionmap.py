"""
Checkpoint별 Attention Maps 저장 스크립트

각 checkpoint에서 attention maps을 추출하여 저장합니다.
저장 경로: /storage/personal/eungyeop/experiments/attention_map/{llm_model}/{source_data}/{mode}/{config}/

Usage:
    python save_attention_maps.py --checkpoint_dir /path/to/checkpoint/dir  # 디렉토리 내 모든 .pt 파일
    python save_attention_maps.py --checkpoint_dir /path/to/checkpoint/dir --pattern "*Edge-True*"  # 특정 패턴 파일만
"""

import os
import sys
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
import re
from torch.utils.data import ConcatDataset, DataLoader
import fnmatch

# 현재 스크립트 파일 위치 (analysis/attentionmap.py)
current_dir = Path(__file__).resolve().parent

# analysis/의 부모 디렉토리 (즉, ProtoLLM/)
root_dir = current_dir.parent
sys.path.append(str(root_dir))  # models, dataset, utils 등이 위치한 루트 디렉토리를 추가

from models.TabularFLM import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders, get_few_shot_embedding_samples
from utils.util import setup_logger, fix_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_deterministic():
    """완전한 deterministic 설정"""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("✅ Deterministic mode enabled")

def extract_config_from_checkpoint(checkpoint_path):
    """체크포인트 파일명에서 설정 정보를 추출"""
    filename = Path(checkpoint_path).stem
    
    # 날짜/시간 패턴 제거 (20250617_173832 형태)
    filename_clean = re.sub(r'_\d{8}_\d{6}$', '', filename)
    
    # "Embed:carte_desc_Edge:mlp_A:att" 형태를 파싱
    pattern = r'Embed:([^:_]+(?:_[^:_]+)*?)_Edge:([^:_]+)_A:([^:_]+)'
    match = re.match(pattern, filename_clean)
    
    if match:
        embed_type = match.group(1)  # carte, carte_desc, ours, ours2
        edge_attr = match.group(2)   # mlp, True, False
        attn_type = match.group(3)   # att, gat
        
        return {
            'embed_type': embed_type,
            'edge_attr': edge_attr,
            'attn_type': attn_type,
            'config_str': f"Embed-{embed_type}_Edge-{edge_attr}_A-{attn_type}"
        }
    else:
        # 패턴 매칭 실패시 기본값
        logger.warning(f"Could not parse config from filename: {filename}")
        logger.warning(f"Cleaned filename: {filename_clean}")
        return {
            'embed_type': 'unknown',
            'edge_attr': 'unknown',
            'attn_type': 'unknown',
            'config_str': filename_clean.replace(':', '-')
        }

class AttentionMapExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path (str): 체크포인트 파일 경로
            device (str): 'cuda' 또는 'cpu'
        """
        ensure_deterministic()
        
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.args = self.checkpoint['args']
        
        # 설정 정보 추출
        self.config = extract_config_from_checkpoint(checkpoint_path)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Config: {self.config['config_str']}")
        logger.info(f"Checkpoint epoch: {self.checkpoint['epoch']}, Val AUC: {self.checkpoint.get('val_auc', 'N/A')}")
        
        # 모델 초기화 및 로드
        self._load_model()
        
        # 데이터로더 준비
        self._prepare_dataloaders()
        
    def _load_model(self):
        """체크포인트에서 모델 로드"""
        experiment_id = "attention_extraction"
        mode = "extraction"
        
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
        """데이터로더 준비 (deterministic)"""
        fix_seed(self.args.random_seed)
        
        # 전체 데이터셋 로더 준비
        results = prepare_embedding_dataloaders(self.args, self.args.source_data)
        self.train_loader, self.val_loader, self.test_loader = results['loaders']
        self.num_classes = results['num_classes']
        
        # 전체 데이터셋 결합 (동일한 순서 보장)
        combined_dataset = ConcatDataset([
            self.train_loader.dataset,
            self.val_loader.dataset, 
            self.test_loader.dataset
        ])
        
        self.combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # 순서 고정
            num_workers=0   # 재현성 위해 단일 프로세스
        )
        
        logger.info(f"Dataloaders prepared for dataset: {self.args.source_data}")
        logger.info(f"Total samples: {len(combined_dataset)}")
    
    def extract_and_save_attention_maps(self, output_base_dir="/storage/personal/eungyeop/experiments/attention_map"):
        """
        Attention maps을 추출하여 샘플별 개별 파일로 저장
        
        Args:
            output_base_dir (str): 기본 저장 디렉토리
        """
        # 저장 경로 생성
        output_dir = Path(output_base_dir) / self.args.llm_model / self.args.source_data / "Full" / self.config['config_str']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting attention maps to: {output_dir}")
        
        # 메타데이터 준비
        metadata = {
            'checkpoint_path': str(self.checkpoint_path),
            'config': self.config,
            'num_layers': len(self.model.layers),
            'total_samples': 0,
            'args_dict': vars(self.args),
            'feature_names': None
        }
        
        sample_count = 0
        saved_files = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.combined_loader):
                # 배치를 디바이스로 이동
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # 모델 forward (attention weights 추출)
                pred, attention_weights = self._extract_attention_from_model(batch_on_device)
                
                # Feature names 추출 (첫 번째 배치에서만)
                if metadata['feature_names'] is None:
                    feature_names = self.model.extract_feature_names(batch_on_device)
                    metadata['feature_names'] = ["CLS"] + feature_names
                
                # 배치 크기 확인
                batch_size = attention_weights[0].shape[0]
                
                for sample_idx in range(batch_size):
                    # 라벨 가져오기
                    if 'y' in batch:
                        label = batch['y'][sample_idx].item()
                    else:
                        label = -1  # 라벨이 없는 경우
                    
                    # 각 레이어의 attention maps 수집
                    sample_attention_maps = {}
                    for layer_idx, layer_attention in enumerate(attention_weights):
                        # Multi-head attention을 평균내어 단일 attention map으로 변환
                        attention_map = layer_attention[sample_idx].mean(dim=0)  # [seq_len, seq_len]
                        attention_numpy = attention_map.detach().cpu().numpy()
                        sample_attention_maps[f'layer_{layer_idx}'] = attention_numpy
                    
                    # 샘플별 파일로 저장
                    sample_filename = f"sample_{sample_count}_label_{label}.npz"
                    sample_filepath = output_dir / sample_filename
                    
                    # 샘플 메타데이터 포함
                    sample_data = {
                        'sample_id': sample_count,
                        'label': label,
                        'batch_idx': batch_idx,
                        'batch_sample_idx': sample_idx,
                        **sample_attention_maps
                    }
                    
                    np.savez_compressed(sample_filepath, **sample_data)
                    saved_files.append(sample_filename)
                    sample_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        
        # 전체 메타데이터 업데이트 및 저장
        metadata['total_samples'] = sample_count
        metadata['saved_files'] = saved_files
        
        import json
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Attention maps saved successfully!")
        logger.info(f"   Total samples: {sample_count}")
        logger.info(f"   Layers per sample: {len(self.model.layers)}")
        logger.info(f"   Attention map shape: {sample_attention_maps['layer_0'].shape}")
        logger.info(f"   Sample files: {sample_count} .npz files")
        logger.info(f"   Metadata file: {metadata_path}")
        logger.info(f"   Directory: {output_dir}")
        
        return output_dir, metadata_path
    
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

def find_checkpoint_files(checkpoint_path, pattern="*.pt"):
    """디렉토리 또는 파일에서 패턴에 맞는 .pt 파일 찾기"""
    checkpoint_path = Path(checkpoint_path)
    
    # 파일인 경우 해당 파일만 반환
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pt':
        if re.search(r'\d{8}_\d{6}', checkpoint_path.stem):  # 날짜 패턴 확인
            return [checkpoint_path]
        else:
            return []
    
    # 디렉토리인 경우 패턴에 맞는 파일들 찾기
    if checkpoint_path.is_dir():
        # 모든 .pt 파일 찾기
        pt_files = list(checkpoint_path.rglob("*.pt"))
        
        # 파일명에 날짜가 포함된 것들만 필터링 (임시 파일 제외)
        filtered_files = []
        for pt_file in pt_files:
            if re.search(r'\d{8}_\d{6}', pt_file.stem):  # 날짜 패턴이 있는 파일
                # 패턴 매칭 확인
                if fnmatch.fnmatch(pt_file.name, pattern):
                    filtered_files.append(pt_file)
        
        return filtered_files
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Extract and save attention maps from checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files OR path to specific checkpoint file')
    parser.add_argument('--pattern', type=str, default="*.pt",
                       help='Pattern to match checkpoint files (e.g., "*Edge-True*", "*gat*")')
    parser.add_argument('--output_dir', type=str, 
                       default="/storage/personal/eungyeop/experiments/attention_map",
                       help='Base output directory for attention maps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 처리할 checkpoint 파일들 수집
    checkpoint_files = find_checkpoint_files(args.checkpoint_dir, args.pattern)
    
    if not checkpoint_files:
        checkpoint_path = Path(args.checkpoint_dir)
        if checkpoint_path.is_file():
            logger.error(f"Checkpoint file {args.checkpoint_dir} does not match the expected pattern (must contain date like 20250617_163404)!")
        else:
            logger.error(f"No checkpoint files found in {args.checkpoint_dir} matching pattern '{args.pattern}'!")
        return
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files matching pattern '{args.pattern}':")
    for i, file in enumerate(checkpoint_files):
        logger.info(f"  {i+1}. {file.name}")
    
    # 각 checkpoint에 대해 attention maps 추출
    success_count = 0
    for i, checkpoint_path in enumerate(checkpoint_files):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_files)}: {checkpoint_path}")
        logger.info(f"{'='*80}")
        
        try:
            # Attention map 추출기 초기화
            extractor = AttentionMapExtractor(str(checkpoint_path), device=args.device)
            
            # Attention maps 추출 및 저장
            save_path, metadata_path = extractor.extract_and_save_attention_maps(args.output_dir)
            
            success_count += 1
            logger.info(f"✅ Successfully processed {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {checkpoint_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n🎯 SUMMARY:")
    logger.info(f"   Total files: {len(checkpoint_files)}")
    logger.info(f"   Successfully processed: {success_count}")
    logger.info(f"   Failed: {len(checkpoint_files) - success_count}")
    logger.info(f"   Pattern used: {args.pattern}")
    logger.info(f"   Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()