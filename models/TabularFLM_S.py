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

logger = logging.getLogger(__name__)

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


class SubgraphAttention(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1, sim_threshold: float = 0.5, no_self_loop: bool = True):
        super().__init__() 
        self.input_dim = input_dim 
        self.dropout = dropout 
        self.sim_threshold = sim_threshold 
        self.no_self_loop = no_self_loop 

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.uniform_(self.cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))

        self._init_weights() 
    def _init_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn_init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))
            if m.bias is not None:
                nn_init.zeros_(m.bias)

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape 
        seq_len = new_seq - 1 
        q = self.q_proj(name_value_embeddings)
        k = self.k_proj(name_value_embeddings)
        v = self.v_proj(name_value_embeddings)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_dim)
        if self.no_self_loop:
            mask = torch.ones_like(attn_weights)
            mask[:, 1:, 1:] = 1.0 - torch.eye(seq_len, device=name_value_embeddings.device)
            attn_weights = attn_weights * mask 
        attn_weights = F.softmax(attn_weights, dim=-1)
        subgraph_mask = (attn_weights > self.sim_threshold).float() 
        subgraph_embeddings = [] 
        self.debug_info = {
            'attn_weights' : attn_weights, 
            'subgraph_mask' : subgraph_mask,
            'threshold' : self.sim_threshold,
        }
        for b in range(batch_size):
            batch_embeddings = [] 
            used = set() 
            for i in range(1, new_seq):
                if (i-1) in used: 
                    continue 
                connected_indices = torch.where(subgraph_mask[b, i, 1:] > 0)[0] 
                # 문법 오류 수정
                if len(connected_indices) == 0:
                    subgraph_nodes = [i]
                else:
                    subgraph_nodes = [i] + (connected_indices + 1).tolist()
                subgraph_nodes = sorted(list(set(subgraph_nodes)))
                for node in subgraph_nodes:
                    used.add(node - 1)
                subgraph_emb = name_value_embeddings[b, subgraph_nodes, :]
                cls_token = self.cls.squeeze(0)
                subgraph_with_cls = torch.cat([cls_token, subgraph_emb], dim=0)
                batch_embeddings.append(subgraph_with_cls)
            subgraph_embeddings.append(batch_embeddings)  # 누락된 부분 추가
        

        return subgraph_embeddings
        

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
        if self.args.no_self_loop:
            self.adjacency = self._no_self_interaction(self.adjacency)

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency
        new_adjacency[:, 0, 1:] = 1.0  # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0  # Var -> CLS

        self.new_adjacency = new_adjacency

        # Attention
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

        if self.args.attn_type == 'gat_v1':
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
                attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
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
                attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
            elif self.args.edge_type == 'no_use':
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                attn_weights = self.attn_proj(qk_expanded).squeeze(-1)

        elif self.args.attn_type == 'gat_v2':
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
                activated_features = F.leaky_relu(qke_expanded)
                attn_weights = self.attn_proj(activated_features).squeeze(-1)
            elif self.args.edge_type == 'no_use':
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                activated_features = F.leaky_relu(qk_expanded)
                attn_weights = self.attn_proj(activated_features).squeeze(-1)

        elif self.args.attn_type == "gate":
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
                gate_values = torch.sigmoid(self.gate_proj(qke_expanded))
                content_values = torch.tanh(self.content_proj(qke_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)
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
                gate_values = torch.sigmoid(self.gate_proj(qk_expanded))
                content_values = torch.tanh(self.content_proj(qk_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)

        elif self.args.attn_type == "att":
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Graph structure masking
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim = -1)
        self.attn_weights = self.attn_dropout(attn_weights)

        # Context
        context = torch.matmul(self.attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)

        return output, attn_weights


class BasisGATLayer(AdaptiveGraphAttention):
    def __init__(self, args, input_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__(args, input_dim, hidden_dim, n_heads, dropout)
        if hasattr(self, 'out_proj'):
            del self.out_proj

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        self.adjacency = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        if self.args.no_self_loop:
            self.adjacency = self._no_self_interaction(self.adjacency)

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency
        new_adjacency[:, 0, 1:] = 1.0
        new_adjacency[:, 1:, 0] = 0.0

        self.new_adjacency = new_adjacency

        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

        q_expanded = q.unsqueeze(3).expand(-1, -1, -1, new_seq, -1)
        k_expanded = k.unsqueeze(2).expand(-1, -1, new_seq, -1, -1)

        node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)
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

        qke_expanded = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1)

        if self.args.attn_type == 'gat_v2':
            activated_features = F.leaky_relu(qke_expanded)
            attn_weights = self.attn_proj(activated_features).squeeze(-1)
        else:
            attn_weights = self.attn_proj(qke_expanded).squeeze(-1)

        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        basis_outputs = context.transpose(1, 2)  # [B, new_seq, n_heads, head_dim]
        return basis_outputs, attn_weights


class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model, experiment_id, mode):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.threshold = args.threshold
        self.frozen = args.frozen
        self.args = args
        self.llm_model = llm_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.source_data = args.source_data
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.llm_model = args.llm_model
        self.meta_type = args.meta_type
        self.experiment_id = experiment_id
        self.sim_threshold = args.sim_threshold
        self.mode = mode
        self.num_classes = args.num_classes
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.uniform_(self.cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))
        self.dropout = nn.Dropout(args.dropout_rate)


        self.subgraph_attention = SubgraphAttention(input_dim, dropout_rate, self.sim_threshold, no_self_loop=False)


        # Shared GAT blocks
        self.shared_layers = nn.ModuleList([
            AdaptiveGraphAttention(
                args = args,
                input_dim = self.input_dim,
                hidden_dim = self.hidden_dim,
                n_heads = args.n_heads,
                dropout = args.dropout_rate,
                threshold = self.threshold
            ) for _ in range(args.num_layers - 1)
        ])
        self.shared_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(args.num_layers - 1)
        ])

        # Basis and experts
        self.coordinator = CoordinatorMLP(input_dim, hidden_dim, args.k_basis, args.dropout_rate, args.coord_softmax_temp)
        self.basis_layer = BasisGATLayer(args, input_dim, hidden_dim, args.k_basis, args.dropout_rate)
        self.basis_layer_norm = nn.LayerNorm(input_dim)
        self.expert_predictors = nn.ModuleList([
            nn.Linear(input_dim // args.k_basis, output_dim) for _ in range(args.k_basis)
        ])

        # Source heads (fixed-count, prebuilt) + target head
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

        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    # === Target few-shot freeze policy ===
    def set_freeze_target(self):
        # 0) freeze all
        for p in self.parameters():
            p.requires_grad = False
        # 1) coordinator open
        for p in self.coordinator.parameters():
            p.requires_grad = True
        # 2) LayerNorms open (shared LNs + basis LN)
        for ln in self.shared_layer_norms:
            for p in ln.parameters():
                p.requires_grad = True
        for p in self.basis_layer_norm.parameters():
            p.requires_grad = True
        # 3) shared edge_update open
        for blk in self.shared_layers:
            if hasattr(blk, 'edge_update'):
                for p in blk.edge_update.parameters():
                    p.requires_grad = True
        # 4) BasisGAT keep frozen (do nothing)
        # 5) target head open
        for p in self.thead.parameters():
            p.requires_grad = True
        # (expert_predictors, cls param 등은 동결)

    def set_attention_save_dir(self, experiment_id, mode):
        base_viz_dir = f"/storage/personal/eungyeop/experiments/visualization/{self.args.llm_model}/{self.args.source_data}/{mode}/{experiment_id}"
        self.attention_save_dir = os.path.join(base_viz_dir, 'attention_maps')
        os.makedirs(self.attention_save_dir, exist_ok=True)
        logger.info(f"Attention maps will be saved to: {self.attention_save_dir}")

    def extract_feature_names(self, batch):
        feature_names = []
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
        seen = set()
        unique_features = []
        for feat in feature_names:
            if feat not in seen:
                seen.add(feat)
                unique_features.append(feat)
        return unique_features

    def save_attention_maps_to_file(self, attention_weights, batch, labels=None, sample_ids=None):
        if not hasattr(self, 'attention_save_dir') or self.attention_save_dir is None:
            logger.warning("Attention save directory not set. Skipping attention map saving.")
            return
        feature_names = self.extract_feature_names(batch)
        all_node_names = ["CLS"] + feature_names
        for layer_idx, layer_attention in enumerate(attention_weights):
            batch_size = layer_attention.shape[0]
            for batch_idx in range(batch_size):
                attention_map = layer_attention[batch_idx].mean(dim=0)
                attention_numpy = attention_map.detach().cpu().numpy()
                sample_id = sample_ids[batch_idx] if sample_ids is not None else self.attention_counter
                label = labels[batch_idx].item() if labels is not None else "unknown"
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
        removed = getattr(self.args, 'del_feat', [])
        if not removed:
            return desc_embeddings, name_value_embeddings
        removed_set = set(removed)
        filtered_desc_embeddings = []
        filtered_name_value_embeddings = []

        if 'cat_desc_texts' in batch:
            cat_feature_names = [feature_tuple[0] if isinstance(feature_tuple, tuple) else str(feature_tuple)
                                 for feature_tuple in batch['cat_desc_texts']]
            keep_indices = [i for i, name in enumerate(cat_feature_names) if name not in removed_set]
            if len(keep_indices) != len(cat_feature_names):
                batch['cat_desc_texts'] = [batch['cat_desc_texts'][i] for i in keep_indices]
                batch['cat_desc_embeddings'] = batch['cat_desc_embeddings'][:, keep_indices, :]
                batch['cat_name_value_embeddings'] = batch['cat_name_value_embeddings'][:, keep_indices, :]
            if keep_indices:
                filtered_desc_embeddings.append(batch['cat_desc_embeddings'].to(self.device))
                filtered_name_value_embeddings.append(batch['cat_name_value_embeddings'].to(self.device))

        if 'num_desc_texts' in batch:
            num_feature_names = [feature_tuple[0] if isinstance(feature_tuple, tuple) else str(feature_tuple)
                                 for feature_tuple in batch['num_desc_texts']]
            keep_indices = [i for i, name in enumerate(num_feature_names) if name not in removed_set]
            if len(keep_indices) != len(num_feature_names):
                batch['num_desc_texts'] = [batch['num_desc_texts'][i] for i in keep_indices]
                batch['num_desc_embeddings'] = batch['num_desc_embeddings'][:, keep_indices, :]
                batch['num_prompt_embeddings'] = batch['num_prompt_embeddings'][:, keep_indices, :]
            if keep_indices:
                filtered_desc_embeddings.append(batch['num_desc_embeddings'].to(self.device))
                filtered_name_value_embeddings.append(batch['num_prompt_embeddings'].to(self.device))

        return filtered_desc_embeddings, filtered_name_value_embeddings

    @torch.no_grad()
    def get_coordinates(self, batch):
        self.eval()
        desc_list, nv_list = [], []
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            desc_list.append(batch['cat_desc_embeddings'].to(self.device))
            nv_list.append(batch['cat_name_value_embeddings'].to(self.device))
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            desc_list.append(batch['num_desc_embeddings'].to(self.device))
            nv_list.append(batch['num_prompt_embeddings'].to(self.device))
        if not desc_list or not nv_list:
            raise ValueError("No categorical or numerical features found in batch")

        desc = torch.cat(desc_list, dim=1)
        nv   = torch.cat(nv_list ,dim=1)

        cls_token = self.cls.expand(nv.size(0), -1, -1)   # FIX
        x = torch.cat([cls_token, nv], dim=1)

        for i, layer in enumerate(self.shared_layers):
            x = x + layer(desc, self.shared_layer_norms[i](x))[0]

        shared_cls = x[:, 0, :]
        c = self.coordinator(shared_cls)                  # [B, K]
        return c

    def set_kmeans_centroids(self, centroids: torch.Tensor):
        # [J, K] 고정 버퍼로 보관
        self.register_buffer("centroids", centroids.detach(), persistent=False)
        self.best_k = int(centroids.size(0))
    def set_coord_temperature(self, t: float):
        self.coordinator.temperature = float(t)

    def forward(self, batch, y):
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()

        pred = self.predict(batch)
        loss = self.criterion(pred, target)

        # === 좌표-KL 정규화 (Few 단계에서만, 버퍼/하이퍼파라미터 존재 시) ===
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
                # KL(q || c)
                kl = torch.sum(q_safe * (q_safe.log() - c_safe.log()), dim=1).mean()
                loss = loss + lam * kl

        return loss

    def predict(self, batch):
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)  # (사용처 유지)
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

        desc_embeddings, name_value_embeddings = self.remove_feature(batch, desc_embeddings, name_value_embeddings)

        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim = 1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim = 1)

        # 1) CLS
        attention_weights = []
        cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
        x = torch.cat([cls_token, name_value_embeddings], dim=1)

        ####

            # Subgraph Attention Debug
        ####
        subgraph_embeddings = self.subgraph_attention(desc_embeddings, x)


        # 2) Shared blocks
        for i, layer in enumerate(self.shared_layers):
            norm_x = self.shared_layer_norms[i](x)
            attn_output, attn_w = layer(desc_embeddings, norm_x)
            attention_weights.append(attn_w)
            x = x + attn_output

        shared_cls = x[:, 0, :]
        coordinates = self.coordinator(shared_cls)
        self._last_coordinates = coordinates 

        # 3) Basis + experts
        norm_x = self.basis_layer_norm(x)
        basis_outputs, _ = self.basis_layer(desc_embeddings, norm_x)
        expert_outputs = basis_outputs[:, 0, :, :]  # [B, k, head_dim]
        expert_predictions = []
        for i in range(self.args.k_basis):
            pred_i = self.expert_predictors[i](expert_outputs[:, i, :])
            expert_predictions.append(pred_i)
        expert_predictions = torch.stack(expert_predictions, dim = 1)  # [B, k, out_dim]
        pred = torch.sum(coordinates.unsqueeze(-1) * expert_predictions, dim=1)  # [B, out_dim]

        # 4) Residual source/target head
        if 'src_idx' in batch:
            si = int(batch['src_idx'])
            pred = pred + self.heads[si](shared_cls)
        elif getattr(self.args, 'use_target_head', False):
            pred = pred + self.thead(shared_cls)

        return pred
