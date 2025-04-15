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
        self.global_topology_proj_head = self.desc_head_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
                )
        self.global_topology_proj_tail = self.desc_tail_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
                )
        for m in self.global_topology_proj_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.global_topology_proj_tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.adaptive_weight_proj_head = nn.Linear(input_dim, hidden_dim)
        self.adaptive_weight_proj_tail = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.adaptive_weight_proj_head.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.adaptive_weight_proj_tail.weight, gain=1 / math.sqrt(2))
        

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

        self.G = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        adjacency = torch.ones_like(self.G) / seq_len 
        

        """
            Description embedding Self attention
        """

        desc_embeddings_head = self.global_topology_proj_head(desc_embeddings)
        desc_embeddings_tail = self.global_topology_proj_tail(desc_embeddings)
        desc_embeddings_head = desc_embeddings_head / desc_embeddings_head.norm(dim=-1, keepdim=True)
        desc_embeddings_tail = desc_embeddings_tail / desc_embeddings_tail.norm(dim=-1, keepdim=True)
        self.global_sim = torch.matmul(desc_embeddings_head, desc_embeddings_tail.transpose(-1, -2))
        self.global_topology_A = torch.sigmoid(self.global_sim + self.topology_bias)

        var_embeddings_head = self.adaptive_weight_proj_head(name_value_embeddings[:, 1:, :])
        var_embeddings_tail = self.adaptive_weight_proj_tail(name_value_embeddings[:, 1:, :])

        self.sample_sim = torch.matmul(var_embeddings_head, var_embeddings_tail.transpose(-1, -2))
        self.global_topology_A = self._no_self_interaction(self.global_topology_A)
        
        self.G = self.global_topology_A * self.sample_sim

        #self.G = torch.clamp(self.G, min = 0.0)
        diag_indices = torch.arange(seq_len, device=self.G.device)
        self.G[:, diag_indices, diag_indices] = -1e9
        threshold = self.G.max(dim=-1, keepdim=True)[0] * self.alpha_param.clamp(min=1e-5,max=1.0)
        mask = (self.G < threshold)

        ## 이 위치에 대각 제거 
        adjacency = torch.softmax(self.G, dim=-1)
        # softmax 후 마스킹된 위치를 정확히 0으로 설정
        adjacency = torch.where(mask, torch.zeros_like(adjacency), adjacency)
        pdb.set_trace()
        # 남은 값들을 다시 정규화 (각 행의 합이 1이 되도록)
        row_sums = adjacency.sum(dim=-1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        adjacency = adjacency / row_sums

        self.adjacency = adjacency

        '''
            4. Edge Attributes & Adjacency 
        '''
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

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=adjacency.device)
        new_adjacency[:, 1:, 1:] = adjacency 
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
        name_value_embeddings = self.sample_fusion(name_value_embeddings)
        '''
            1. [CLS] Token
        '''
        attention_weights = [] 
        cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        x = name_value_embeddings
        for i, layer in enumerate(self.layers):
            norm_x = self.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_weights)
            x = x + attn_output
        pred = x[:, 0, :]
        pred = self.predictor(pred)

        
        return pred

    def froze_topology(self):

        self.frozen = True
        logger.info("Graph topology frozen. Continuing with fixed structure.")