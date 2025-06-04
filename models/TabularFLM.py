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
        input_dim : int,
        hidden_dim : int,
        n_heads : int, 
        dropout : float = 0.1,
        threshold : float = 0.5
    ):
        super().__init__()
        assert input_dim % n_heads == 0 
        self.n_heads = n_heads 
        self.head_dim = input_dim // n_heads 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.frozen = False  # 그래프 구조 고정 여부
        self.prelu = nn.PReLU()
        self.edge_update = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        self.threshold = threshold
        self.topology_bias = nn.Parameter(torch.zeros(1))
        self.attn_dropout = nn.Dropout(dropout)

        self.alpha_param = nn.Parameter(torch.tensor(0.1))

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.attn_proj = nn.Linear(self.head_dim * 3, 1)   # [q | k | edge_attr] -> attention score
        
        
        self.out_proj = nn.Linear(input_dim, input_dim)
        nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
    
    def _no_self_interaction(self, adjacency_matrix):
        batch_size, seq_len, _ = adjacency_matrix.shape
        diag_mask = 1.0 - torch.eye(seq_len, device=adjacency_matrix.device).unsqueeze(0)
        return adjacency_matrix * diag_mask

    

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        self.adjacency = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        self.adjacency = self._no_self_interaction(self.adjacency)
        # 1. 변수 노드 간의 edge_attr
        node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)  # [batch, seq_len, seq_len, dim*2]

        # 2. CLS-변수 노드 간의 edge_attr
        cls_edge_attr = torch.cat([
            desc_embeddings.unsqueeze(1),  # CLS->변수: 변수의 description
            desc_embeddings.unsqueeze(1)   # 변수->CLS: 동일한 description
        ], dim=-1)

        # 3. Edge attribute 합치기
        edge_dim = var_edge_attr.size(-1)
        edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=var_edge_attr.device)
        edge_attr[:, 1:, 1:] = var_edge_attr  # 변수 노드 간
        edge_attr[:, 0, 1:] = cls_edge_attr.squeeze(1)  # CLS->변수
        edge_attr[:, 1:, 0] = cls_edge_attr.squeeze(1)  # 변수->CLS

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device= self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency 
        new_adjacency[:, 0, 1:] = 1.0 # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0 # Var -> CLS
        
        self.new_adjacency = new_adjacency
        '''
            5. Attention
        '''
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

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

        attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim = -1)
        self.attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)
        return output, attn_weights

class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.threshold = args.threshold  # 엣지 프루닝 임계값 (명령줄 인자에서 가져옴)
        self.frozen = args.frozen  # 그래프 구조 고정 여부 (초기에는 False로 설정)
        self.args = args
        self.llm_model = llm_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.source_dataset_name = args.source_dataset_name
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.meta_type = args.meta_type
        self.enc_type = args.enc_type
                # GMM 관련 속성 추가
        self.use_gmm = args.use_gmm 
        self.use_gmm2 = args.use_gmm2
        self.num_prototypes = args.num_prototypes
        self.stage_num = args.gmm_stage_num
        self.momentum = args.gmm_momentum 
        self.beta = args.gmm_beta 
        self.lambd = args.gmm_lambda
        self.eps = args.gmm_eps
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))
        max_seq_len = 100
        self.global_table = nn.Parameter(torch.Tensor(1, max_seq_len, self.input_dim))
        nn.init.kaiming_uniform_(self.global_table, a = math.sqrt(5))

        # PyTorch Attention Map Clustering 관련 변수들
        self.attention_maps = []  # List of attention map tensors (현재 에포크만)
        self.cluster_centroids = None  # [num_clusters, seq_len, seq_len]
        self.cluster_assignments = []  # 각 attention map의 클러스터 할당
        self.num_clusters = getattr(args, 'num_attention_clusters', 5)
        self.clustering_update_freq = getattr(args, 'clustering_update_freq', 10)
        self.attention_count = 0
        self.current_epoch = -1  # 현재 에포크 추적용
        self.collect_attention = False  # 🆕 attention 수집 모드 플래그

        '''
            MLP(CONCAT[Name embedding, Value embedding])
            - In order to infuse the information of name and value simultaneously. 
        '''
        self.sample_fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
       
        self.layers = nn.ModuleList([
            AdaptiveGraphAttention(
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
        # self.post_norms = nn.ModuleList([
        #     nn.LayerNorm(self.input_dim) for _ in range(args.num_layers)
        # ])
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

                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        self._init_weights()

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def _frobenius_distance(self, A, B):
        """
        두 attention map 간의 Frobenius norm 거리 계산 (PyTorch)
        A, B: [seq_len, seq_len]
        """
        diff = A - B
        return torch.norm(diff, p='fro')

    def _compute_attention_distances(self, attention_maps, centroids):
        """
        모든 attention map과 centroid 간의 거리 행렬 계산
        attention_maps: [n_maps, seq_len, seq_len]
        centroids: [n_clusters, seq_len, seq_len]
        return: [n_maps, n_clusters]
        """
        n_maps = len(attention_maps)
        n_clusters = centroids.shape[0]
        distances = torch.zeros(n_maps, n_clusters, device=centroids.device)
        
        for i, attn_map in enumerate(attention_maps):
            for j in range(n_clusters):
                distances[i, j] = self._frobenius_distance(attn_map, centroids[j])
        
        return distances

    def _update_clustering(self):
        """
        PyTorch 기반 K-means 스타일 클러스터링
        """
        if len(self.attention_maps) < self.num_clusters:
            return
        
        # attention maps를 tensor stack으로 변환
        attention_tensor = torch.stack(self.attention_maps)  # [n_maps, seq_len, seq_len]
        n_maps, seq_len, _ = attention_tensor.shape
        
        # 초기 centroid 설정 (첫 번째 클러스터링이면 랜덤 선택)
        if self.cluster_centroids is None:
            indices = torch.randperm(n_maps)[:self.num_clusters]
            self.cluster_centroids = attention_tensor[indices].clone()
        
        max_iterations = 10
        for iteration in range(max_iterations):
            # 1. 각 attention map을 가장 가까운 centroid에 할당
            distances = self._compute_attention_distances(self.attention_maps, self.cluster_centroids)
            old_assignments = self.cluster_assignments.copy() if self.cluster_assignments else None
            self.cluster_assignments = torch.argmin(distances, dim=1).tolist()
            
            # 수렴 체크
            if old_assignments is not None and self.cluster_assignments == old_assignments:
                break
            
            # 2. Centroid 업데이트
            new_centroids = torch.zeros_like(self.cluster_centroids)
            for cluster_id in range(self.num_clusters):
                cluster_mask = [i for i, assign in enumerate(self.cluster_assignments) if assign == cluster_id]
                
                if cluster_mask:  # 빈 클러스터가 아니면
                    cluster_maps = torch.stack([self.attention_maps[i] for i in cluster_mask])
                    new_centroids[cluster_id] = torch.mean(cluster_maps, dim=0)
                else:  # 빈 클러스터면 랜덤 재할당
                    random_idx = torch.randint(0, n_maps, (1,)).item()
                    new_centroids[cluster_id] = self.attention_maps[random_idx].clone()
            
            self.cluster_centroids = new_centroids

        logger.info(f"PyTorch clustering updated: {len(self.attention_maps)} maps, {self.num_clusters} clusters")

    def reset_epoch_clustering(self):
        """
        시각화 에포크에서만 attention maps 리셋 및 수집 시작
        """
        self.attention_maps = []  # 새로운 수집을 위해 리셋
        self.cluster_assignments = []  # 클러스터 할당도 리셋
        self.cluster_centroids = None  # 클러스터 중심도 리셋
        self.attention_count = 0
        self.collect_attention = True  # 🆕 수집 모드 활성화
        logger.info(f"Attention clustering reset and collection started for visualization epoch")
    
    def stop_attention_collection(self):
        """
        attention 수집 중단
        """
        self.collect_attention = False
        logger.info(f"Attention collection stopped")

    def update_attention_clustering(self):
        """
        외부에서 호출 가능한 클러스터링 업데이트 메소드 (시각화용)
        """
        if len(self.attention_maps) < self.num_clusters:
            logger.info(f"Not enough attention maps for clustering: {len(self.attention_maps)}/{self.num_clusters}")
            return False
        
        try:
            self._update_clustering()
            return True
        except Exception as e:
            logger.error(f"Clustering update failed: {e}")
            return False

    def get_clustering_info(self):
        """
        클러스터링 정보 반환 (시각화용)
        """
        return {
            'attention_maps': self.attention_maps,
            'cluster_centroids': self.cluster_centroids,
            'cluster_assignments': self.cluster_assignments,
            'num_clusters': self.num_clusters,
            'attention_count': self.attention_count
        }


    def save_cluster_centroids(self, save_dir, epoch):
        if self.cluster_centroids is None:
            return 
        
        # centroids 폴더 생성
        centroid_dir = os.path.join(save_dir, 'centroids')
        os.makedirs(centroid_dir, exist_ok=True)

        for cluster_id, centroid in enumerate(self.cluster_centroids):
            # 클러스터별 폴더 생성 (centroids 하위에)
            cluster_folder = os.path.join(centroid_dir, f'cluster_{cluster_id}')
            os.makedirs(cluster_folder, exist_ok=True)
            
            centroid_np = centroid.detach().cpu().numpy() 

            # 클러스터별 폴더에 저장
            filename = f"epoch_{epoch}.npy"
            filepath = os.path.join(cluster_folder, filename)
            np.save(filepath, centroid_np)
            logger.info(f"Saved centroid for cluster {cluster_id}: {filepath}")


    def forward(self, batch, y):
        target = y.to(self.device).view(-1,1).float()
        pred = self.predict(batch)
        loss = self.criterion(pred, target)
        return loss
    
    def predict(self, batch):
        
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
        #name_value_embeddings = self.sample_fusion(name_value_embeddings)
        '''
            1. [CLS] Token
        '''
        attention_weights = [] 
        cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        batch_size , seq_len, input_dim = x.size()
        '''
            2. Global Table Token
        '''
        for i, layer in enumerate(self.layers):
            norm_x = self.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_weights)
            x = x + attn_output
            
        # 모든 레이어 처리 완료 후, 마지막 레이어의 attention만 클러스터링에 사용
        if self.training and self.collect_attention:  # 🆕 수집 모드일 때만
            final_layer_attention = attention_weights[-1]  # 마지막 레이어 (Layer 2)
            batch_size = final_layer_attention.shape[0]
            
            for batch_idx in range(batch_size):
                sample_attention = final_layer_attention[batch_idx].mean(dim=0)  # [seq_len, seq_len]
                
                # 최종 attention map만 저장 (현재 에포크만)
                self.attention_maps.append(sample_attention.detach().clone())
                self.attention_count += 1        
        pred = x[:, 0, :]
        pred = self.predictor(pred)

        return pred

    def froze_topology(self):
        self.frozen = True
        logger.info("Graph topology frozen. Continuing with fixed structure.")