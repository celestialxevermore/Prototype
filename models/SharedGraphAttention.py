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


class SharedGraphAttention(nn.Module):
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
        if self.args.no_self_loop:
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