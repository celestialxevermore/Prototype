import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import PowerTransformer, StandardScaler
import json
import os
from torch import Tensor
import math
import torch.nn.init as nn_init
import logging

logger = logging.getLogger(__name__)

class AdaptiveGraphAttention(nn.Module):
    def __init__(
        self,
        args,
        input_dim : int,
        hidden_dim : int,
        n_heads : int, 
        dropout : float = 0.1,
        threshold : float = 0.5
    ):
        super().__init__()
        assert input_dim % n_heads == 0 
        self.args = args
        self.n_heads = n_heads 
        self.head_dim = input_dim // n_heads 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.threshold = threshold
        self.attn_dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        if self.args.attn_type in ['gat_v1', 'gat_v2']:
            if self.args.edge_type == 'normal':
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
            elif self.args.edge_type == 'mlp':
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
                self.edge_update = nn.Sequential(
                    nn.Linear(input_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
            elif self.args.edge_type == 'no_use':
                self.attn_proj = nn.Linear(self.head_dim * 2, 1)
        elif self.args.attn_type == 'gate':
            if self.args.edge_type == 'normal':
                self.gate_proj = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
            elif self.args.edge_type == 'mlp':
                self.gate_proj = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
                self.edge_update = nn.Sequential(
                    nn.Linear(input_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
            elif self.args.edge_type == 'no_use':
                self.gate_proj = nn.Linear(self.head_dim * 2, 1)
                self.content_proj = nn.Linear(self.head_dim * 2, 1)
        
        self.out_proj = nn.Linear(input_dim, input_dim)
        nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
        
        if hasattr(self, 'attn_proj'):
            nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'gate_proj'):
            nn_init.xavier_uniform_(self.gate_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'content_proj'):
            nn_init.xavier_uniform_(self.content_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'edge_update'):
            for module in self.edge_update:
                if isinstance(module, nn.Linear):
                    nn_init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))

    def _no_self_interaction(self, adjacency_matrix):
        batch_size, seq_len, _ = adjacency_matrix.shape
        diag_mask = 1.0 - torch.eye(seq_len, device=adjacency_matrix.device).unsqueeze(0)
        return adjacency_matrix * diag_mask

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        self.adjacency = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        self.adjacency = self._no_self_interaction(self.adjacency)
        
        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency 
        new_adjacency[:, 0, 1:] = 1.0 # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0 # Var -> CLS
        
        self.new_adjacency = new_adjacency
        
        '''
            Attention
        '''
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention 계산 방식 선택
        if self.args.attn_type == 'gat_v1':
            # GAT-style attention (concat + MLP)
            if self.args.edge_type == "normal":
                # Case 1: GAT with edge attributes - MLP([q | k | edge_attr])
                target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                cls_edge_attr = desc_embeddings

                edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = target_desc  # 변수 노드 간
                edge_attr[:, 0, 1:] = cls_edge_attr  # CLS->변수
                
                edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
                edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
                
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qke_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1),
                    edge_attr
                ], dim = -1)
                
                attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
            elif self.args.edge_type == "mlp":
                node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
                node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
                cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
                edge_dim = var_edge_attr.size(-1)
                edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = var_edge_attr  # 변수 노드 간
                edge_attr[:, 0, 1:] = cls_edge_attr   # CLS->변수
                edge_attr[:, 1:, 0] = cls_edge_attr   # 변수->CLS
                
                # 4. 차원 맞추기 (edge_update 필요)
                edge_attr = self.edge_update(edge_attr)  # [batch, new_seq, new_seq, n_heads * head_dim]
                edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
                edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
                
                # 5. Attention 계산 (기존과 동일)
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qke_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1),
                    edge_attr
                ], dim = -1)
                
                attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
            elif self.args.edge_type == 'no_use':
                # Case 2: GAT without edge attributes - MLP([q | k])
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                
                attn_weights = self.attn_proj(qk_expanded).squeeze(-1)
        elif self.args.attn_type == 'gat_v2':
           # GAT-v2 style attention (LeakyReLU before attention)
           if self.args.edge_type == "normal":
               target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
               cls_edge_attr = desc_embeddings

               edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
               edge_attr[:, 1:, 1:] = target_desc
               edge_attr[:, 0, 1:] = cls_edge_attr
               
               edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
               edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
               edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
               
               q_expanded = q.unsqueeze(3)
               k_expanded = k.unsqueeze(2)
               qke_expanded = torch.cat([
                   q_expanded.expand(-1,-1,-1, new_seq, -1),
                   k_expanded.expand(-1,-1, new_seq, -1, -1),
                   edge_attr
               ], dim = -1)
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qke_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
           elif self.args.edge_type == "mlp":
               node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
               node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
               var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
               cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
               edge_dim = var_edge_attr.size(-1)
               edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
               edge_attr[:, 1:, 1:] = var_edge_attr
               edge_attr[:, 0, 1:] = cls_edge_attr
               edge_attr[:, 1:, 0] = cls_edge_attr
               
               edge_attr = self.edge_update(edge_attr)
               edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
               edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
               edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
               
               q_expanded = q.unsqueeze(3)
               k_expanded = k.unsqueeze(2)
               qke_expanded = torch.cat([
                   q_expanded.expand(-1,-1,-1, new_seq, -1),
                   k_expanded.expand(-1,-1, new_seq, -1, -1),
                   edge_attr
               ], dim = -1)
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qke_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
           elif self.args.edge_type == 'no_use':
               q_expanded = q.unsqueeze(3)
               k_expanded = k.unsqueeze(2)
               qk_expanded = torch.cat([
                   q_expanded.expand(-1,-1,-1, new_seq, -1),
                   k_expanded.expand(-1,-1, new_seq, -1, -1)
               ], dim = -1)
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qk_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
        elif self.args.attn_type == "gate":
            # Gate attention mechanism
            if self.args.edge_type == "normal":
                # Edge attributes 준비
                target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                cls_edge_attr = desc_embeddings

                edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = target_desc
                edge_attr[:, 0, 1:] = cls_edge_attr
                
                edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
                edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
                
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qke_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1),
                    edge_attr
                ], dim = -1)
                
                # Gate mechanism: σ(W_g * [q|k|e]) * tanh(W_c * [q|k|e])
                gate_values = torch.sigmoid(self.gate_proj(qke_expanded))  # [batch, n_heads, seq, seq, 1]
                content_values = torch.tanh(self.content_proj(qke_expanded))  # [batch, n_heads, seq, seq, 1]
                attn_weights = (gate_values * content_values).squeeze(-1)  # [batch, n_heads, seq, seq]
                
            elif self.args.edge_type == "mlp":
                node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
                node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
                cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
                edge_dim = var_edge_attr.size(-1)
                edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = var_edge_attr
                edge_attr[:, 0, 1:] = cls_edge_attr
                edge_attr[:, 1:, 0] = cls_edge_attr
                
                edge_attr = self.edge_update(edge_attr)
                edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
                edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
                
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qke_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1),
                    edge_attr
                ], dim = -1)
                
                # Gate mechanism: σ(W_g * [q|k|e]) * tanh(W_c * [q|k|e])
                gate_values = torch.sigmoid(self.gate_proj(qke_expanded))
                content_values = torch.tanh(self.content_proj(qke_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)
                
            elif self.args.edge_type == 'no_use':
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                
                # Gate mechanism: σ(W_g * [q|k]) * tanh(W_c * [q|k])
                gate_values = torch.sigmoid(self.gate_proj(qk_expanded))
                content_values = torch.tanh(self.content_proj(qk_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)

        elif self.args.attn_type == "att":
            # Case 3: Standard dot-product attention (use_edge_attr는 무시)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            # Default: standard dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Graph structure masking
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim = -1)
        self.attn_weights = self.attn_dropout(attn_weights)

        # Context 계산
        context = torch.matmul(self.attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)
        
        return output, attn_weights

class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model, experiment_id, mode):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.threshold = args.threshold  # 엣지 프루닝 임계값 (명령줄 인자에서 가져옴)
        self.frozen = args.frozen  # 그래프 구조 고정 여부 (초기에는 False로 설정)
        self.args = args
        self.llm_model = llm_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.num_layers = num_layers
        self.source_data = args.source_data
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.meta_type = args.meta_type
        self.enc_type = args.enc_type
        self.experiment_id = experiment_id
        self.mode = mode
        self.num_classes = args.num_classes
        
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))

        '''
            MLP(CONCAT[Name embedding, Value embedding])
            - In order to infuse the information of name and value simultaneously. 
        '''
        self.layers = nn.ModuleList([
            AdaptiveGraphAttention(
                args = args,
                input_dim = self.input_dim, 
                hidden_dim = self.hidden_dim,
                n_heads = args.n_heads,
                dropout = args.dropout_rate,
                threshold = self.threshold
            ) for _ in range(args.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(args.num_layers)
        ])
        self.dropout = nn.Dropout(args.dropout_rate)
        self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(hidden_dim, output_dim)
            ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes ==2 else nn.CrossEntropyLoss()
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def set_attention_save_dir(self, experiment_id, mode):
        """
        Attention map 저장 디렉토리를 기존 clustering과 동일한 구조로 설정
        
        Args:
            experiment_id (str): 실험 ID (예: "20250604_224705")
            mode (str): "Full" 또는 "Few"
        """
        base_viz_dir = f"/storage/personal/eungyeop/experiments/visualization/{self.args.llm_model}/{self.args.source_data}/{mode}/{experiment_id}"
        self.attention_save_dir = os.path.join(base_viz_dir, 'attention_maps')
        os.makedirs(self.attention_save_dir, exist_ok=True)
        logger.info(f"Attention maps will be saved to: {self.attention_save_dir}")

    def extract_feature_names(self, batch):
        """
        배치에서 feature names를 추출하는 함수 (시각화 코드에서 가져옴)
        
        Args:
            batch: 입력 배치 데이터
            
        Returns:
            list: feature names 리스트
        """
        feature_names = []
        
        # Categorical features
        if 'cat_desc_texts' in batch:
            for feature in batch['cat_desc_texts']:
                if isinstance(feature, tuple):
                    clean_name = str(feature[0])
                else:
                    try:
                        clean_name = feature.split("'")[1] if "'" in feature else feature
                        clean_name = clean_name.split(',')[0]
                    except:
                        clean_name = str(feature)
                feature_names.append(clean_name)

        # Numerical features
        if 'num_desc_texts' in batch:
            for feature in batch['num_desc_texts']:
                if isinstance(feature, tuple):
                    clean_name = str(feature[0])
                else:
                    try:
                        clean_name = feature.split("'")[1] if "'" in feature else feature
                        clean_name = clean_name.split(',')[0]
                    except:
                        clean_name = str(feature)
                feature_names.append(clean_name)
        
        # 중복 제거 (순서 유지)
        seen = set()
        unique_features = []
        for feat in feature_names:
            if feat not in seen:
                seen.add(feat)
                unique_features.append(feat)
        
        return unique_features

    def save_attention_maps_to_file(self, attention_weights, batch, labels=None, sample_ids=None):
        """
        Attention maps을 feature names와 함께 numpy 파일로 저장합니다.
        
        Args:
            attention_weights (list): 각 레이어의 attention weights
            batch: 입력 배치 (feature names 추출용)
            labels (torch.Tensor, optional): 샘플의 라벨
            sample_ids (list, optional): 샘플 ID들
        """
        if not hasattr(self, 'attention_save_dir') or self.attention_save_dir is None:
            logger.warning("Attention save directory not set. Skipping attention map saving.")
            return
        
        # Feature names 추출
        feature_names = self.extract_feature_names(batch)
        all_node_names = ["CLS"] + feature_names
        
        # 각 레이어별로 처리
        for layer_idx, layer_attention in enumerate(attention_weights):
            # [batch_size, n_heads, seq_len, seq_len]
            batch_size = layer_attention.shape[0]
            
            for batch_idx in range(batch_size):
                # Multi-head attention을 평균내어 단일 attention map으로 변환
                attention_map = layer_attention[batch_idx].mean(dim=0)  # [seq_len, seq_len]
                attention_numpy = attention_map.detach().cpu().numpy()
                
                # 메타데이터 준비
                sample_id = sample_ids[batch_idx] if sample_ids is not None else self.attention_counter
                label = labels[batch_idx].item() if labels is not None else "unknown"
                
                # numpy.savez로 attention map과 feature names를 함께 저장
                filename = f"layer_{layer_idx}_sample_{sample_id}_label_{label}.npz"
                filepath = os.path.join(self.attention_save_dir, filename)
                
                np.savez(filepath,
                        attention_map=attention_numpy,
                        feature_names=np.array(all_node_names),
                        layer_idx=layer_idx,
                        sample_id=sample_id,
                        label=label)
                
                self.attention_counter += 1
        
        logger.info(f"Attention maps saved for {batch_size} samples across {len(attention_weights)} layers to {self.attention_save_dir}")

    def remove_feature(self, batch, desc_embeddings, name_value_embeddings):
        """
        batch에서 args.del_feat에 지정된 변수들을 완전히 제거하고
        desc_embeddings, name_value_embeddings도 함께 업데이트
        """
        removed = getattr(self.args, 'del_feat', [])
        
        if not removed:
            return desc_embeddings, name_value_embeddings
        
        removed_set = set(removed)
        filtered_desc_embeddings = []
        filtered_name_value_embeddings = []
        
        # Categorical features 처리
        if 'cat_desc_texts' in batch:
            cat_feature_names = [feature_tuple[0] if isinstance(feature_tuple, tuple) else str(feature_tuple) 
                                for feature_tuple in batch['cat_desc_texts']]
            
            # 유지할 인덱스 찾기
            keep_indices = [i for i, name in enumerate(cat_feature_names) if name not in removed_set]
            removed_cat = [name for name in cat_feature_names if name in removed_set]
            
            if removed_cat:
                # batch에서 직접 제거
                batch['cat_desc_texts'] = [batch['cat_desc_texts'][i] for i in keep_indices]
                batch['cat_desc_embeddings'] = batch['cat_desc_embeddings'][:, keep_indices, :]
                batch['cat_name_value_embeddings'] = batch['cat_name_value_embeddings'][:, keep_indices, :]
            
            # desc_embeddings, name_value_embeddings에 추가
            if keep_indices:  # 남은 categorical features가 있으면
                filtered_desc_embeddings.append(batch['cat_desc_embeddings'].to(self.device))
                filtered_name_value_embeddings.append(batch['cat_name_value_embeddings'].to(self.device))
        
        # Numerical features 처리
        if 'num_desc_texts' in batch:
            num_feature_names = [feature_tuple[0] if isinstance(feature_tuple, tuple) else str(feature_tuple) 
                                for feature_tuple in batch['num_desc_texts']]
            
            # 유지할 인덱스 찾기
            keep_indices = [i for i, name in enumerate(num_feature_names) if name not in removed_set]
            removed_num = [name for name in num_feature_names if name in removed_set]
            
            if removed_num:
                # batch에서 직접 제거
                batch['num_desc_texts'] = [batch['num_desc_texts'][i] for i in keep_indices]
                batch['num_desc_embeddings'] = batch['num_desc_embeddings'][:, keep_indices, :]
                batch['num_prompt_embeddings'] = batch['num_prompt_embeddings'][:, keep_indices, :]
            
            # desc_embeddings, name_value_embeddings에 추가
            if keep_indices:  # 남은 numerical features가 있으면
                filtered_desc_embeddings.append(batch['num_desc_embeddings'].to(self.device))
                filtered_name_value_embeddings.append(batch['num_prompt_embeddings'].to(self.device))
        
        return filtered_desc_embeddings, filtered_name_value_embeddings

    def extract_cls_attention_weights(self, data_loader, device, top_k=3):
        """
        완전히 학습된 모델에서 CLS 토큰의 attention weight를 추출하여 중요한 변수들을 찾음
        
        Args:
            data_loader: 데이터 로더
            device: 디바이스
            top_k: 상위 몇 개의 변수를 추출할지
        
        Returns:
            important_features: 중요한 변수들의 이름 리스트 (attention 높은 순)
            all_features: 전체 변수들의 이름 리스트
            unimportant_features: 안중요한 변수들의 이름 리스트 (attention 낮은 순)
        """
        self.eval()
        all_feature_names = None
        all_attentions = []  # 모든 배치의 attention 저장
        
        with torch.no_grad():
            for batch in data_loader:
                # Feature names 추출 (첫 번째 배치에서만)
                if all_feature_names is None:
                    all_feature_names = self.extract_feature_names(batch)
                
                # 임베딩 준비 (전체 변수 사용 - remove_feature 호출하지 않음)
                desc_embeddings, name_value_embeddings = self._prepare_embeddings_for_attention(batch)
                
                # CLS token 추가
                cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
                name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
                x = name_value_embeddings
                
                # 마지막 레이어까지 forward pass
                for i, layer in enumerate(self.layers):
                    norm_x = self.layer_norms[i](x)
                    attn_output, attn_weights = layer(desc_embeddings, norm_x)
                    
                    if i == len(self.layers) - 1:  # 마지막 레이어의 attention 저장
                        # CLS -> 다른 노드들로의 attention weights 추출
                        # attn_weights: [batch_size, n_heads, seq_len, seq_len]
                        cls_to_vars_attention = attn_weights[:, :, 0, 1:]  # [batch, heads, num_vars]
                        all_attentions.append(cls_to_vars_attention)
                    
                    x = x + attn_output
        
        # 모든 배치의 attention을 연결하고 평균화
        final_attention = torch.cat(all_attentions, dim=0).mean(dim=(0, 1))  # [num_vars]
        
        # 🔥 전체 변수의 attention 순위 계산 (높은 순으로 정렬)
        sorted_indices = torch.argsort(final_attention, descending=True)
        
        # Top-k 중요 변수 (attention 높은 순)
        if len(final_attention) < top_k:
            top_k = len(final_attention)
        
        top_indices = sorted_indices[:top_k]
        important_features = [all_feature_names[i] for i in top_indices]
        
        # 🔥 Bottom-k 안중요 변수 (attention 낮은 순)
        bottom_indices = sorted_indices[-top_k:] if len(sorted_indices) >= top_k else []
        unimportant_features = [all_feature_names[i] for i in bottom_indices]
        
        logger.info(f"CLS Attention weights for top {top_k} features:")
        for i, (idx, feat) in enumerate(zip(top_indices, important_features)):
            weight = final_attention[idx].item()
            logger.info(f"  {i+1}. {feat}: {weight:.6f}")
        
        logger.info(f"CLS Attention weights for bottom {len(unimportant_features)} features:")
        for i, (idx, feat) in enumerate(zip(bottom_indices, unimportant_features)):
            weight = final_attention[idx].item()
            logger.info(f"  {i+1}. {feat}: {weight:.6f}")
        
        return important_features, all_feature_names, unimportant_features

    def _prepare_embeddings_for_attention(self, batch):
        """
        Attention 분석용 임베딩 준비 함수 (remove_feature 호출하지 않음)
        전체 변수로 attention 분석할 때 사용
        """
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
        
        return desc_embeddings, name_value_embeddings

    def _prepare_embeddings(self, batch):
        """
        배치에서 임베딩을 준비하는 헬퍼 함수 (기존 predict 함수에서 추출)
        일반 학습/예측용 - remove_feature 호출함
        """
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
        
        desc_embeddings, name_value_embeddings = self.remove_feature(
            batch, desc_embeddings, name_value_embeddings
        )
        
        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim=1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim=1)
        
        return desc_embeddings, name_value_embeddings

    def forward(self, batch, y):        
        target = y.to(self.device)
        
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            # Multi-class classification  
            target = target.squeeze()
            target = target.long()
        pred = self.predict(batch)
        loss = self.criterion(pred, target)
        return loss

    def predict(self, batch):
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
        
        desc_embeddings, name_value_embeddings = self.remove_feature(
            batch, desc_embeddings, name_value_embeddings
        )
        
        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim = 1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim = 1)
        
        '''
            1. [CLS] Token
        '''
        attention_weights = [] 
        cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        
        '''
            2. Graph Attention Layers
        '''
        for i, layer in enumerate(self.layers):
            norm_x = self.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_weights)
            x = x + attn_output
    
        pred = x[:, 0, :]
        pred = self.predictor(pred)

        return pred