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


class BasisGATLayer_MUL(nn.Module):
    """
    entropic_gw, prior_Q, DG, b 모두 제거한 바닐라 버전.
    - gat_v1 / gat_v2 / gate 지원
    - edge_type: normal / mlp / no_use 지원
    - CLS->Var on, Var->CLS off, Var-Var 구조 마스크 적용
    반환:
        basis_outputs: [B, new_seq, n_heads, head_dim]
        attn_weights : [B, n_heads, new_seq, new_seq]
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, num_basis_heads: int, dropout: float = 0.1):
        super().__init__()
        assert input_dim % num_basis_heads == 0, "input_dim must be divisible by n_heads"
        assert args.attn_type in ['gat_v1', 'gat_v2', 'gate'], "attn_type must be one of {gat_v1,gat_v2,gate}"

        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basis_heads = num_basis_heads
        self.head_dim = input_dim // num_basis_heads
        self.attn_dropout = nn.Dropout(dropout)

        # === Q/K/V projection ===
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        # === Attention branch ===
        if args.attn_type in ['gat_v1', 'gat_v2']:
            if args.edge_type in ['normal', 'mlp']:
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
                if args.edge_type == 'mlp':
                    self.edge_update = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    )
            elif args.edge_type == 'no_use':
                self.attn_proj = nn.Linear(self.head_dim * 2, 1)

        elif args.attn_type == 'gate':
            if args.edge_type in ['normal', 'mlp']:
                self.gate_proj = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
                if args.edge_type == 'mlp':
                    self.edge_update = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim, input_dim)
                    )
            elif args.edge_type == 'no_use':
                self.gate_proj = nn.Linear(self.head_dim * 2, 1)
                self.content_proj = nn.Linear(self.head_dim * 2, 1)

        # === Initialization ===
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    @staticmethod
    def _no_self_interaction(adj: torch.Tensor) -> torch.Tensor:
        B, S, _ = adj.shape
        diag = torch.eye(S, device=adj.device).unsqueeze(0)
        return adj * (1.0 - diag)

    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor):
        """
        desc_embeddings: [B, S, D]
        name_value_embeddings: [B, new_seq, D], new_seq = S + 1 (CLS 포함)
        """
        B, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        # === Adjacency (CLS->Var on / Var->CLS off) ===
        var_adj = torch.ones(B, seq_len, seq_len, device=name_value_embeddings.device)
        if getattr(self.args, "no_self_loop", False):
            var_adj = self._no_self_interaction(var_adj)

        new_adj = torch.zeros(B, new_seq, new_seq, device=name_value_embeddings.device)
        new_adj[:, 1:, 1:] = var_adj
        new_adj[:, 0, 1:] = 1.0
        new_adj[:, 1:, 0] = 0.0
        self.new_adjacency = new_adj

        # === Q/K/V ===
        q = self.q_proj(name_value_embeddings).view(B, new_seq, self.num_basis_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(B, new_seq, self.num_basis_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(B, new_seq, self.num_basis_heads, self.head_dim).transpose(1, 2)

        q_exp = q.unsqueeze(3)
        k_exp = k.unsqueeze(2)

        # === Edge attributes (optional) ===
        if self.args.edge_type in ['normal', 'mlp']:
            node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
            node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
            var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)
            cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
            edge_dim = var_edge_attr.size(-1)

            edge_attr = torch.zeros(B, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
            edge_attr[:, 1:, 1:] = var_edge_attr
            edge_attr[:, 0, 1:] = cls_edge_attr
            edge_attr[:, 1:, 0] = cls_edge_attr

            if self.args.edge_type == 'mlp' and hasattr(self, 'edge_update'):
                edge_attr = self.edge_update(edge_attr)

            edge_attr = edge_attr.view(B, new_seq, new_seq, self.num_basis_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            edge_attr = edge_attr * new_adj.unsqueeze(1).unsqueeze(-1)
        else:
            edge_attr = None

        # === Attention logits ===
        q_expanded = q_exp.expand(-1, -1, -1, new_seq, -1)
        k_expanded = k_exp.expand(-1, -1, new_seq, -1, -1)

        if self.args.attn_type in ['gat_v1', 'gat_v2']:
            qke = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1) if edge_attr is not None else torch.cat([q_expanded, k_expanded], dim=-1)
            if self.args.attn_type == 'gat_v2':
                qke = F.leaky_relu(qke)
            logits = self.attn_proj(qke).squeeze(-1)

        elif self.args.attn_type == 'gate':
            qke = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1) if edge_attr is not None else torch.cat([q_expanded, k_expanded], dim=-1)
            gate_values = torch.sigmoid(self.gate_proj(qke))
            content_values = torch.tanh(self.content_proj(qke))
            logits = (gate_values * content_values).squeeze(-1)

        else:
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # === Structure mask ===
        mask = (new_adj.unsqueeze(1) == 0).float() * -1e9
        logits = logits + mask

        # === Attention & Output ===
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        basis_outputs = context.transpose(1, 2).contiguous()  # [B, new_seq, H, head_dim]
        return basis_outputs, attn_weights


class BasisGATLayer_IND(nn.Module):
    """
    entropic_gw, prior_Q, DG, b 모두 제거한 바닐라 버전.
    - gat_v1 / gat_v2 / gate 지원
    - edge_type: normal / mlp / no_use 지원
    - CLS->Var on, Var->CLS off, Var-Var 구조 마스크 적용
    반환:
        basis_outputs: [B, new_seq, n_heads, head_dim]
        attn_weights : [B, n_heads, new_seq, new_seq]
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, num_basis_heads: int, dropout: float = 0.1):
        super().__init__()
        assert input_dim % num_basis_heads == 0, "input_dim must be divisible by n_heads"
        assert args.attn_type in ['gat_v1', 'gat_v2', 'gate'], "attn_type must be one of {gat_v1,gat_v2,gate}"

        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basis_heads = num_basis_heads
        self.head_dim = self.input_dim // self.num_basis_heads
        self.attn_dropout = nn.Dropout(dropout)

        # === 각 head별 독립적 Q/K/V/attn_proj 생성 ===
        self.q_proj_list = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_basis_heads)])
        self.k_proj_list = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_basis_heads)])
        self.v_proj_list = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_basis_heads)])

        self.ovr_M, self.ovr_G = None, None 


        if args.attn_type in ['gat_v1', 'gat_v2']:
            self.attn_proj_list = nn.ModuleList([
                nn.Linear(self.head_dim * 3 if args.edge_type in ['normal', 'mlp'] else self.head_dim * 2, 1)
                for _ in range(num_basis_heads)
            ])
        elif args.attn_type == 'gate':
            self.gate_proj_list = nn.ModuleList([
                nn.Linear(self.head_dim * 3 if args.edge_type in ['normal', 'mlp'] else self.head_dim * 2, 1)
                for _ in range(num_basis_heads)
            ])
            self.content_proj_list = nn.ModuleList([
                nn.Linear(self.head_dim * 3 if args.edge_type in ['normal', 'mlp'] else self.head_dim * 2, 1)
                for _ in range(num_basis_heads)
            ])

        if args.edge_type == 'mlp':
            self.edge_update = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )

        # === Initialization (레거시 유지) ===
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    @staticmethod
    def _no_self_interaction(adj: torch.Tensor) -> torch.Tensor:
        B, S, _ = adj.shape
        diag = torch.eye(S, device=adj.device).unsqueeze(0)
        return adj * (1.0 - diag)

    def forward(self, name_embeddings: torch.Tensor, value_embeddings: torch.Tensor):
        B, new_seq, _ = value_embeddings.shape
        seq_len = new_seq - 1

        # === Adjacency (CLS->Var on / Var->CLS off) ===
        var_adj = torch.ones(B, seq_len, seq_len, device=value_embeddings.device)
        if getattr(self.args, "no_self_loop", False):
            var_adj = self._no_self_interaction(var_adj)

        new_adj = torch.zeros(B, new_seq, new_seq, device=value_embeddings.device)
        new_adj[:, 1:, 1:] = var_adj
        new_adj[:, 0, 1:] = 1.0
        new_adj[:, 1:, 0] = 0.0
        self.new_adjacency = new_adj

        # === Edge attributes (optional) ===
        if self.args.edge_type in ['normal', 'mlp']:
            node_i_desc = name_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
            node_j_desc = name_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
            var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)
            cls_edge_attr = torch.cat([name_embeddings, name_embeddings], dim=-1)
            edge_dim = var_edge_attr.size(-1)

            edge_attr = torch.zeros(B, new_seq, new_seq, edge_dim, device=name_embeddings.device)
            edge_attr[:, 1:, 1:] = var_edge_attr
            edge_attr[:, 0, 1:] = cls_edge_attr
            edge_attr[:, 1:, 0] = cls_edge_attr

            if self.args.edge_type == 'mlp' and hasattr(self, 'edge_update'):
                edge_attr = self.edge_update(edge_attr)

            # edge_dim(=input_dim*2)을 head_dim에 맞게 투사
            edge_attr = edge_attr.view(B, new_seq, new_seq, self.num_basis_heads, -1)
            edge_attr = edge_attr[..., :self.head_dim]  # 간단한 head_dim matching (trim or project)
            edge_attr = edge_attr * new_adj.unsqueeze(-1).unsqueeze(-1)
            edge_attr = edge_attr.permute(0, 3, 1, 2, 4)  # [B, H, new_seq, new_seq, head_dim]
        else:
            edge_attr = None

        # === Head별 독립 계산 ===
        head_outputs, head_attns = [], []
        for h in range(self.num_basis_heads):
            q = self.q_proj_list[h](value_embeddings)  # [B, new_seq, head_dim]
            k = self.k_proj_list[h](value_embeddings)
            v = self.v_proj_list[h](value_embeddings)

            q_exp = q.unsqueeze(2)
            k_exp = k.unsqueeze(1)
            q_expanded = q_exp.expand(-1, new_seq, new_seq, -1)
            k_expanded = k_exp.expand(-1, new_seq, new_seq, -1)

            # head별 edge_attr 선택
            edge_attr_h = edge_attr[:, h] if edge_attr is not None else None

            if edge_attr_h is not None:
                qke = torch.cat([q_expanded, k_expanded, edge_attr_h], dim=-1)
            else:
                qke = torch.cat([q_expanded, k_expanded], dim=-1)

            # === Attention logits ===
            if self.args.attn_type in ['gat_v1', 'gat_v2']:
                if self.args.attn_type == 'gat_v2':
                    qke = F.leaky_relu(qke)
                logits = self.attn_proj_list[h](qke).squeeze(-1)
            elif self.args.attn_type == 'gate':
                gate_values = torch.sigmoid(self.gate_proj_list[h](qke))
                content_values = torch.tanh(self.content_proj_list[h](qke))
                logits = (gate_values * content_values).squeeze(-1)
            else:
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            mask = (new_adj == 0).float() * -1e9
            logits = logits + mask

            G = self.ovr_G 
            if G is None and self.ovr_M is not None:
                G = torch.sigmoid(self.ovr_M)
            if G is not None:
                G = G * new_adj 
                logits = logits + (1.0 - G) * (-1e9)
            pdb.set_trace()

            attn_weights = F.softmax(logits, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, v)

            head_outputs.append(context.unsqueeze(2))
            head_attns.append(attn_weights.unsqueeze(1))

        basis_outputs = torch.cat(head_outputs, dim=2)
        attn_weights = torch.cat(head_attns, dim=1)

        return basis_outputs, attn_weights