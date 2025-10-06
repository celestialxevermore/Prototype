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
import torch.nn.init as init
from utils.affinity import BasisSlotAffinityGAT

logger = logging.getLogger(__name__)
def _chk(name: str, t: torch.Tensor):
    with torch.no_grad():
        shape = list(t.shape)
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        is_finite = torch.isfinite(t)

        if not is_finite.any():
            print(f"[NaN@]{name} shape={shape} | ALL non-finite (nan/inf)!")
            return

        t_f = t[is_finite]
        t_min = t_f.min().item()
        t_max = t_f.max().item()
        t_mean = t_f.mean().item()
        t_std = t_f.std().item()

        print(
            f"[NaN@]{name} shape={shape} "
            f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} std={t_std:.6g} "
            f"| has_nan={has_nan} has_inf={has_inf}"
        )

class CoordinatorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_basis: int, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.k_basis = k_basis
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = float(temperature)

        # 경로/인덱스 유지: 0:Linear, 1:LayerNorm, 2:ReLU, 3:Dropout, 4:Linear
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.k_basis),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        logits = self.coordinate_mlp(cls_emb)                # [B, K]
        return F.softmax(logits / max(self.temperature, 1e-6), dim=-1)


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





class BasisGATLayer(nn.Module):
    """
    - gat_v1 / gat_v2 / gate 분기만 지원
    - edge_type: normal / mlp / no_use 지원 (mlp면 edge_update MLP 사용)
    - CLS->Var on, Var->CLS off, Var-Var는 (self_loop 옵션 반영) 구조 마스크 적용
    - 선택적 mask_M를 pre-softmax logit bias로 주입(A안):
        mask_M: [B, n_heads, seq_len, seq_len]  (seq_len = new_seq - 1, Var-Var 블록)
        logits[:, :, 1:, 1:] += logit(mask_M) * gamma
    반환:
      basis_outputs: [B, new_seq, n_heads, head_dim]
      attn_weights : [B, n_heads, new_seq, new_seq]
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        assert args.attn_type in ['gat_v1', 'gat_v2', 'gate'], "attn_type must be one of {gat_v1,gat_v2,gate}"

        self.args = args
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = input_dim // n_heads
        self.attn_dropout = nn.Dropout(dropout)

        self.gw_eps = self.args.gw_eps 
        self.gw_sigma = self.args.gw_sigma 
        self.gw_outer_iters = self.args.gw_outer_iters 
        self.gw_sink_iters = self.args.gw_sinkhorn_iters 
        self.gw_tol = self.args.gw_sinkhorn_eps 

        # V
        self.v_proj = nn.Linear(input_dim, input_dim)

        # branch-specific projections
        if self.args.attn_type in ['gat_v1', 'gat_v2']:
            if self.args.edge_type in ['normal', 'mlp']:
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
                if self.args.edge_type == 'mlp':
                    self.edge_update = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    )
            elif self.args.edge_type == 'no_use':
                self.attn_proj = nn.Linear(self.head_dim * 2, 1)

        elif self.args.attn_type == 'gate':
            if self.args.edge_type in ['normal', 'mlp']:
                self.gate_proj    = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
                if self.args.edge_type == 'mlp':
                    self.edge_update = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    )
            elif self.args.edge_type == 'no_use':
                self.gate_proj    = nn.Linear(self.head_dim * 2, 1)
                self.content_proj = nn.Linear(self.head_dim * 2, 1)

        # === initialization: xavier_uniform with gain=1/sqrt(2) (bias는 기본값 유지/0으로 초기화) ===
        nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))

        if hasattr(self, 'attn_proj'):
            nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'gate_proj'):
            nn_init.xavier_uniform_(self.gate_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'content_proj'):
            nn_init.xavier_uniform_(self.content_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'edge_update'):
            for m in self.edge_update:
                if isinstance(m, nn.Linear):
                    nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))

    @staticmethod
    def _no_self_interaction(adj: torch.Tensor) -> torch.Tensor:
        # adj: [B, S, S]
        B, S, _ = adj.shape
        diag = torch.eye(S, device=adj.device).unsqueeze(0)
        return adj * (1.0 - diag)
    @staticmethod 
    def _attn_from_Qhat(Q_hat: torch.tensor, new_adjacency: torch.Tensor) -> torch.Tensor:
        """
        Q_hat: [B,H,N,N] (Var-Var 확률행렬)
        new_adjacency: [B, N+1, N+1] (CLS 포함 구조마스크; CLS->Var=1, Var->CLS=0, Var-Var=1)
        return attn [B,H,N,N] (각 행 확률합=1)
        """
        B, H, S, _ = Q_hat.shape 
        S_one = S + 1 
        slot_adjacency = Q_hat.new_zeros(B, H, S_one, S_one)
        slot_adjacency[:, :, 1:, 1:] = Q_hat 
        slot_adjacency[:, :, 0, 1:] = 1.0 
        slot_adjacency[:, :, 1:, 0] = 0.0
        slot_adjacency = slot_adjacency * new_adjacency.unsqueeze(1) 
        row_sum = slot_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_adjacency = slot_adjacency / row_sum 
        return slot_adjacency     
    @staticmethod
    def _logit_bias_from_prob(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = prob.clamp(min=eps, max=1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def forward(self, 
        desc_embeddings:torch.Tensor, 
        name_value_embeddings:torch.Tensor,
        shared_attn:torch.Tensor,
        prior_Q:torch.Tensor = None, # [B,M,S,S] Global node space affinity
        DG:torch.Tensor = None, # [B,M,K,K] distance on slot space 
        b:torch.Tensor = None):
        """ 
            desc_embeddings : [B,S,D] 
            name_value_embeddings : [B,T,D] 
            prior_Q (optional) : [B,M,S,S] (Var-Var prior from slots) Global node space affinity
            DG (optional) : [B,M,K,K] Global slot space affinity
            b (optional) : [B,H,K]
        """
        B, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1
         
        var_adj = torch.ones(B, seq_len, seq_len, device=name_value_embeddings.device)
        if self.args.no_self_loop:
            var_adj = self._no_self_interaction(var_adj)
        new_adjacency = torch.zeros(B, new_seq, new_seq, device = name_value_embeddings.device)
        new_adjacency[:,1:,1:] = var_adj
        new_adjacency[:,0,1:] = 1.0 
        new_adjacency[:,1:,0] = 0.0
        self.new_adjacency = new_adjacency 
        # SharedGAT에서 받은 attention에서 Var–Var 블록만 추출
        if shared_attn.dim() == 4 and shared_attn.size(2) == new_seq:
            P_var = shared_attn[:, :, 1:, 1:]         # [B,H,S,S]
        else:
            # 이미 Var–Var로만 넘어온 경우라고 보고 그대로 사용
            P_var = shared_attn                       # [B,H,S,S]
        P_var = P_var.detach()
        P_norm = BasisSlotAffinityGAT.normalize_affinity(P_var, sym=True)
        DP = BasisSlotAffinityGAT.affinity_to_distance(P_norm)

        eps = 1e-8 
        deg_P = 0.5 * (P_norm.sum(dim=-1) + P_norm.sum(dim=-2))
        a = deg_P / deg_P.sum(dim=-1, keepdim=True).clamp_min(eps)

        _, gw_val = BasisSlotAffinityGAT._entropic_gw(DP,DG,a,b,eps=self.gw_eps,outer_iters=self.gw_outer_iters,sinkhorn_iters=self.gw_sink_iters,tol=self.gw_tol)
        alpha = BasisSlotAffinityGAT.alpha_from_gw(gw_val, sigma=self.gw_sigma)
        Q_hat = BasisSlotAffinityGAT.sharpen_Q(prior_Q, alpha)
        attn_weights = self._attn_from_Qhat(Q_hat, new_adjacency)
        attn_weights = self.attn_dropout(attn_weights) 

        # final attention
        v = self.v_proj(name_value_embeddings) 
        v = v.view(B, new_seq, self.n_heads, self.head_dim).transpose(1,2)
        context = torch.matmul(attn_weights, v)
        basis_outputs = context.transpose(1,2).contiguous()
        return basis_outputs, attn_weights
    
class Model(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, dropout_rate, llm_model, experiment_id, mode):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args         = args
        self.llm_model    = llm_model
        self.input_dim    = input_dim      # LLM dim (e.g., 768)
        self.hidden_dim   = hidden_dim
        self.output_dim   = output_dim
        self.dropout_rate = dropout_rate
        self.source_data  = args.source_data
        self.mode         = mode
        self.num_classes  = args.num_classes
        self.n_slots = self.args.n_slots
        self.slot_dim = self.args.slot_dim
        self.num_shared_layers = self.args.num_shared_layers
        self.num_basis_layers = self.args.num_basis_layers
        # CLS
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.uniform_(self.cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))

        self.shared_layers = nn.ModuleList([ 
            SharedGraphAttention(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim,
            n_heads = args.n_heads, dropout = self.dropout_rate, threshold = getattr(args, 'threshold', 0.5)
            ) for _ in range(self.num_shared_layers)
        ])
        self.shared_layer_norms = nn.ModuleList([ 
            nn.LayerNorm(self.input_dim) for _ in range(self.num_shared_layers)
        ])

        self.basis_affinity = BasisSlotAffinityGAT(
            args, input_dim = self.input_dim, n_slots = self.n_slots, slot_dim = self.slot_dim
        )

        self.basis_layers = nn.ModuleList([ 
            BasisGATLayer(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, n_heads = args.n_heads, dropout = self.dropout_rate)
            for _ in range(self.num_basis_layers)
        ])
        self.basis_layer_norms = nn.ModuleList([ 
            nn.LayerNorm(self.input_dim) for _ in range(self.num_basis_layers)
        ])

        self.expert_predictors = nn.ModuleList([
            nn.Linear(self.input_dim // args.n_heads, output_dim) for _ in range(args.n_heads)
        ])

        # Coordinator (weights over heads/bases)
        self.coordinator = CoordinatorMLP(
            self.input_dim, hidden_dim, args.n_heads, self.dropout_rate,
            getattr(args, 'coord_softmax_temp', 1.0)
        )

        # Source/Target residual heads (on CLS)
        self.n_src = len(args.source_data) if isinstance(args.source_data, (list, tuple)) else 1
        hid = min(128, self.input_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hid),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hid, self.output_dim),
            ) for _ in range(self.n_src)
        ])
        self.thead = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hid, self.output_dim),
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    # Few-shot freeze policy
    def set_freeze_target(self):
        for p in self.parameters():
            p.requires_grad = False
        # coordinator / layer norms / target head만 오픈
        for p in self.coordinator.parameters():
            p.requires_grad = True
        for ln in self.basis_layer_norms:
            for p in ln.parameters():
                p.requires_grad = False
        for p in self.thead.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def get_coordinates(self, batch):
        self.eval()
        # gather
        desc_list, nv_list = [], []
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            desc_list.append(batch['cat_desc_embeddings'].to(self.device))
            nv_list.append(batch['cat_name_value_embeddings'].to(self.device))
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            desc_list.append(batch['num_desc_embeddings'].to(self.device))
            nv_list.append(batch['num_prompt_embeddings'].to(self.device))
        if not desc_list or not nv_list:
            raise ValueError("No categorical or numerical features found in batch")

        desc = torch.cat(desc_list, dim=1)  # [B,S,D]
        nv   = torch.cat(nv_list ,dim=1)    # [B,S,D]

        x_shared = torch.cat([self.cls.expand(nv.size(0), 1, self.input_dim), nv], dim = 1) 
        for l in range(self.num_shared_layers):
            nx = self.shared_layer_norms[l](x_shared)
            out, _ = self.shared_layers[l](desc, nx)
            x_shared = x_shared + out 
        cls_coord = x_shared[:, 0, :]
        c = self.coordinator(cls_coord)
        return c

    def set_kmeans_centroids(self, centroids: torch.Tensor):
        self.register_buffer("centroids", centroids.detach(), persistent=False)
        self.best_k = int(centroids.size(0))

    def set_coord_temperature(self, t: float):
        self.coordinator.temperature = float(t)

    # ---- training ----
    def forward(self, batch, y):
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()

        pred = self.predict(batch)
        loss = self.criterion(pred, target)

        # (1) 슬롯 규제(겹침/사용량) 그냥 더하기
        if self.training and hasattr(self, "_last_slot_loss") and (self._last_slot_loss is not None):
            loss = loss + self._last_slot_loss
        z_student = pred 
        z_teacher = self._last_logits_shared.detach() 
        if self.num_classes == 2:
            z_student = torch.stack([torch.zeros_like(z_student), z_student], dim=-1)
            z_teacher = torch.stack([torch.zeros_like(z_teacher), z_teacher], dim=-1)
        logQ = F.log_softmax(z_student,dim=-1)
        P = F.softmax(z_teacher, dim=-1)
        kd_loss = F.kl_div(logQ, P, reduction='batchmean')
        loss = loss + kd_loss 

        # (2) 기존 Few-shot coord KL 유지 (타깃 에피소드에서 좌표 분포 정렬)
        lam = float(getattr(self.args, "coord_reg_lambda", 0.0))
        if (self.mode == 'Few') and (lam > 0.0) and hasattr(self, "centroids"):
            c = getattr(self, "_last_coordinates", None)
            if c is not None:
                from utils.coord_Kmeans import build_centroid_target
                q = build_centroid_target(
                    c, self.centroids,
                    tau=float(getattr(self.args, "coord_tau", 0.3)),
                    mode=str(getattr(self.args, "coord_target_mode", "soft"))
                ).to(c.device)
                eps = 1e-8
                c_safe = c.clamp_min(eps)
                q_safe = q.clamp_min(eps)
                kl = torch.sum(q_safe * (q_safe.log() - c_safe.log()), dim=1).mean()
                loss = loss + lam * kl

        return loss

    # ---- inference ----
    def predict(self, batch):
        # gather
        desc_embeddings, name_value_embeddings = [], []
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            name_value_embeddings.append(batch['cat_name_value_embeddings'].to(self.device))
            desc_embeddings.append(batch['cat_desc_embeddings'].to(self.device))
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            name_value_embeddings.append(batch['num_prompt_embeddings'].to(self.device))
            desc_embeddings.append(batch['num_desc_embeddings'].to(self.device))
        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc = torch.cat(desc_embeddings, dim=1)  # [B,S,D]
        nv   = torch.cat(name_value_embeddings, dim=1)

        # ---- shared blocks -> coordinator ----
        x_shared = torch.cat([self.cls.expand(nv.size(0), 1, self.input_dim), nv], dim=1)
        last_attn_shared = None 

        for l in range(self.num_shared_layers):
            nx = self.shared_layer_norms[l](x_shared)
            out, attn_shared = self.shared_layers[l](desc, nx)
            x_shared = x_shared + out
            last_attn_shared = attn_shared 

        shared_cls = x_shared[:, 0, :]
        self._last_coordinates = self.coordinator(shared_cls.detach())

        if 'src_idx' in batch:
            logits_shared = self.heads[int(batch['src_idx'])](shared_cls)
        elif getattr(self.args, 'use_target_head', False):
            logits_shared = self.thead(shared_cls)
        else:
            logits_shared = self.thead(shared_cls)
        self._last_logits_shared = logits_shared


        # ---- global/slot prior Q and regularizers ----
        bias_log, Q_slot, slot_loss, DG, b = self.basis_affinity(desc, nv)
        self._last_bias_log  = bias_log          # log Q  (시각화/기록용)
        self._last_Q_slot    = Q_slot            # Q      (확률) [B,M,S,S]
        self._last_slot_loss = (slot_loss if self.training else None)

        # === 핵심: GAT의 pre-softmax bias로 넣을 프라이어 확률 ===
        mask_M = torch.clamp(self._last_Q_slot, min=1e-6, max=1.0 - 1e-6)  # [B,M,S,S]

        # ---- basis GAT stack (Q를 pre-softmax logit bias로 사용) ----
        x_basis  = torch.cat([self.cls.expand(nv.size(0), 1, self.input_dim), nv], dim=1)
        last_att = None
        for l in range(self.num_basis_layers):
            norm_x = self.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.basis_layers[l](desc, norm_x,shared_attn = last_attn_shared, prior_Q=mask_M, DG=DG,b=b)
            x_basis = x_basis + basis_outputs.reshape(x_basis.size(0), x_basis.size(1), self.input_dim)
            last_att = att
        if last_att is not None:
            # Var-Var 블록만 저장: [B,M,S,S]
            self._last_P_basis = last_att[:, :, 1:, 1:]

        # ---- experts & mixture ----
        expert_outputs = basis_outputs[:, 0, :, :]  # [B,H,head_dim]
        preds = [self.expert_predictors[i](expert_outputs[:, i, :]) for i in range(self.args.n_heads)]
        expert_predictions = torch.stack(preds, dim=1)  # [B,H,C]
        logits_basis_mix  = torch.sum(self._last_coordinates.unsqueeze(-1) * expert_predictions, dim=1)  # [B,C]
        return logits_basis_mix