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
from utils.affinity import PairwiseMLPScorer, RelationQueryScorer, FSTHeadwiseEntmaxScorer, FSTHeadwiseEntmaxScorerNodeFirst

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

        # Q/K/V
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
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
        nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
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

        # affinity gate strength γ
        self.gate_strength = float(getattr(self.args, "affinity_gate_gamma", 1.0))

    @staticmethod
    def _no_self_interaction(adj: torch.Tensor) -> torch.Tensor:
        # adj: [B, S, S]
        B, S, _ = adj.shape
        diag = torch.eye(S, device=adj.device).unsqueeze(0)
        return adj * (1.0 - diag)

    @staticmethod
    def _logit_bias_from_prob(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = prob.clamp(min=eps, max=1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def forward(self, desc_embeddings: torch.Tensor,
                      name_value_embeddings: torch.Tensor,
                      mask_M: torch.Tensor = None):
        """
        desc_embeddings        : [B, seq_len, D]        (변수 설명 임베딩)
        name_value_embeddings  : [B, new_seq, D]        (CLS + 변수)
        mask_M (optional)      : [B, n_heads, seq_len, seq_len] (Var-Var 확률)
        """
        B, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1  # exclude CLS

        # === structure adjacency ===
        var_adj = torch.ones(B, seq_len, seq_len, device=name_value_embeddings.device)
        if self.args.no_self_loop:
            var_adj = self._no_self_interaction(var_adj)

        new_adjacency = torch.zeros(B, new_seq, new_seq, device=name_value_embeddings.device)
        new_adjacency[:, 1:, 1:] = var_adj
        new_adjacency[:, 0, 1:]  = 1.0  # CLS -> Var
        new_adjacency[:, 1:, 0]  = 0.0  # Var -> CLS
        self.new_adjacency = new_adjacency  # for debugging/visualization

        # === Q/K/V ===
        q = self.q_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,d]
        k = self.k_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

        # === edge attributes if needed ===
        if self.args.attn_type in ['gat_v1', 'gat_v2', 'gate']:
            if self.args.edge_type in ['normal', 'mlp']:
                # build var-var & cls-var attributes in the SAME way you already used
                node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)        # [B,S,S,D]
                node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)        # [B,S,S,D]
                var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)                 # [B,S,S,2D]
                cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)         # [B,S,2D]
                edge_dim = var_edge_attr.size(-1)

                edge_attr = torch.zeros(B, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = var_edge_attr
                edge_attr[:, 0, 1:]  = cls_edge_attr
                edge_attr[:, 1:, 0]  = cls_edge_attr

                if self.args.edge_type == 'mlp' and hasattr(self, 'edge_update'):
                    edge_attr = self.edge_update(edge_attr)  # [B,T,T,D]

                # [B,T,T,D] -> [B,H,T,T,head_dim]
                edge_attr = edge_attr.view(B, new_seq, new_seq, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)

        # === build logits by branch ===
        q_exp = q.unsqueeze(3)  # [B,H,1,T,d]→[B,H,T,T,d] via expand
        k_exp = k.unsqueeze(2)  # [B,H,T,1,d]→[B,H,T,T,d]
        q_expanded = q_exp.expand(-1, -1, -1, new_seq, -1)
        k_expanded = k_exp.expand(-1, -1, new_seq, -1, -1)

        if self.args.attn_type in ['gat_v1', 'gat_v2']:
            if self.args.edge_type in ['normal', 'mlp']:
                qke = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1)  # [..., 3*head_dim]
            elif self.args.edge_type == 'no_use':
                qke = torch.cat([q_expanded, k_expanded], dim=-1)
            if self.args.attn_type == 'gat_v2':
                qke = F.leaky_relu(qke)
            logits = self.attn_proj(qke).squeeze(-1)  # [B,H,T,T]

        elif self.args.attn_type == 'gate':
            if self.args.edge_type in ['normal', 'mlp']:
                qke = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1)
            elif self.args.edge_type == 'no_use':
                qke = torch.cat([q_expanded, k_expanded], dim=-1)
            gate_values    = torch.sigmoid(self.gate_proj(qke))
            content_values = torch.tanh(self.content_proj(qke))
            logits = (gate_values * content_values).squeeze(-1)  # [B,H,T,T]

        # === pre-softmax bias with mask_M (A안) ===
        if mask_M is not None:
            if mask_M.dim() != 4 or mask_M.size(0) != B or mask_M.size(1) != self.n_heads \
               or mask_M.size(2) != seq_len or mask_M.size(3) != seq_len:
                raise ValueError(f"mask_M must be [B, {self.n_heads}, {seq_len}, {seq_len}], got {list(mask_M.shape)}")
            bias_full = torch.zeros_like(logits)  # [B,H,T,T]
            bias_var  = self._logit_bias_from_prob(mask_M) * self.gate_strength
            bias_full[:, :, 1:, 1:] = bias_var
            logits = logits + bias_full

        # === structural masking + softmax ===
        NEG_INF = -1e9
        struct_mask = (new_adjacency.unsqueeze(1) == 0).to(logits.dtype) * NEG_INF  # [B,1,T,T]
        logits = logits + struct_mask
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # [B,H,T,T]

        # === context ===
        context = torch.matmul(attn_weights, v)               # [B,H,T,d]
        basis_outputs = context.transpose(1, 2).contiguous()  # [B,T,H,d]
        return basis_outputs, attn_weights


class Model(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim,
                 num_layers, dropout_rate, llm_model, experiment_id, mode):
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

        # CLS
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.uniform_(self.cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))

        # ----- Relation scorer (FFN1 + scorer) -----
        self.mask_share_across_layers = bool(getattr(args, 'mask_share_across_layers', True))
        rel_proj_dim   = int(getattr(args, 'rel_proj_dim', 768))
        rel_hidden_dim = int(getattr(args, 'rel_hidden_dim', 512))
        rel_symmetric  = bool(getattr(args, 'rel_symmetric', False))
        no_self_loop   = bool(getattr(args, 'no_self_loop', True))
        scorer_type    = str(getattr(args, 'relation_scorer_type', 'pair_mlp'))
        self.num_basis_layers = int(getattr(args, 'num_basis_layers', 3))

        # FFN1: project LLM embeddings to relation-space
        self.rel_proj = nn.Sequential(
            nn.Linear(self.input_dim, rel_proj_dim),
            nn.LayerNorm(rel_proj_dim),
            nn.ReLU()
        )

        # scorer: make M [B,H,S,S]
        if scorer_type == 'pair_mlp':
            self.relation_scorer = PairwiseMLPScorer(
                rel_input_dim=rel_proj_dim, rel_hidden_dim=rel_hidden_dim, k_basis=args.k_basis,
                dropout_rate=self.dropout_rate, mask_symmetric=rel_symmetric, no_self_loop=no_self_loop
            )
        elif scorer_type == 'query':
            self.relation_scorer = RelationQueryScorer(
                rel_input_dim=rel_proj_dim, k_basis=args.k_basis, rel_hidden_dim=rel_hidden_dim,
                dropout_rate=self.dropout_rate, mask_symmetric=rel_symmetric, no_self_loop=no_self_loop
            )
        elif scorer_type == 'fst_headwise_entmax':
            self.relation_scorer = FSTHeadwiseEntmaxScorer(
                rel_proj_dim=rel_proj_dim, rel_hidden_dim=rel_hidden_dim, k_basis=args.k_basis,
                dropout_rate=self.dropout_rate, mask_symmetric=rel_symmetric, no_self_loop=no_self_loop
            )
        elif scorer_type == 'fst_pair_first_headwise_entmax':
            self.relation_scorer = FSTHeadwiseEntmaxScorerNodeFirst(
                rel_input_dim=rel_proj_dim, rel_hidden_dim=rel_hidden_dim, k_basis=args.k_basis,
                dropout_rate=self.dropout_rate, mask_symmetric=rel_symmetric, no_self_loop=no_self_loop
            )
        else:
            raise ValueError(f"Unknown relation_scorer_type: {scorer_type}")

        # ----- BasisGAT stack -----
        self.basis_layers = nn.ModuleList([
            BasisGATLayer(args, input_dim, hidden_dim, args.k_basis, self.dropout_rate)
            for _ in range(self.num_basis_layers)
        ])
        self.basis_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(self.num_basis_layers)
        ])

        # Experts (one per basis head)
        self.expert_predictors = nn.ModuleList([
            nn.Linear(self.input_dim // args.k_basis, output_dim) for _ in range(args.k_basis)
        ])

        # Coordinator (weights over heads/bases)
        self.coordinator = CoordinatorMLP(
            self.input_dim, hidden_dim, args.k_basis, self.dropout_rate,
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

        # init (Linear only)
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
                p.requires_grad = True
        for p in self.thead.parameters():
            p.requires_grad = True
        # (선택) relation 모듈도 열고 싶으면 아래 주석 해제
        # for p in self.rel_proj.parameters(): p.requires_grad = True
        # for p in self.relation_scorer.parameters(): p.requires_grad = True

        # (원하면 relation_scorer/edge_update 일부만 열 수도 있음)
        # 5) (선택) relation_scorer는 프리징 유지 (pretrain에서 학습되었다고 가정)
        # for p in self.relation_scorer.parameters(): p.requires_grad = False

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

        cls_token = self.cls.expand(nv.size(0), 1, self.input_dim)
        x = torch.cat([cls_token, nv], dim=1)  # [B,T,D]

        # 첫 레이어 기준으로 마스크 생성 → 통과
        norm_x = self.basis_layer_norms[0](x)
        E_vars = norm_x[:, 1:, :]
        E_rel  = self.rel_proj(E_vars)
        M = self.relation_scorer(E_rel)       # [B,H,S,S]
        basis_outputs, _ = self.basis_layers[0](desc, norm_x, mask_M=M)

        cls_for_coord = basis_outputs[:, 0, :, :].contiguous().view(nv.size(0), self.input_dim)
        c = self.coordinator(cls_for_coord)    # [B,K]
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

        # 좌표-KL (Few일 때)
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

        cls_token = self.cls.expand(nv.size(0), 1, self.input_dim)
        x = torch.cat([cls_token, nv], dim=1)     # [B,T,D]

        M0 = None
        for l in range(self.num_basis_layers):
            norm_x = self.basis_layer_norms[l](x)
            E_vars = norm_x[:, 1:, :]            # [B,S,D]

            if self.mask_share_across_layers and (l > 0):
                M = M0
            else:
                E_rel = self.rel_proj(E_vars)    # [B,S,rel_proj_dim]
                M = self.relation_scorer(E_rel)  # [B,H,S,S]
                if self.mask_share_across_layers and l == 0:
                    M0 = M

            basis_outputs, _ = self.basis_layers[l](desc, norm_x, mask_M=M)
            # residual
            x = x + basis_outputs.reshape(x.size(0), x.size(1), self.input_dim)

        # 마지막 층 CLS(H,d) → [B,input_dim]
        cls_for_coord = basis_outputs[:, 0, :, :].contiguous().view(nv.size(0), self.input_dim)
        coordinates = self.coordinator(cls_for_coord)  # [B,H]
        self._last_coordinates = coordinates

        # experts
        expert_outputs = basis_outputs[:, 0, :, :]  # [B,H,d]
        preds = [self.expert_predictors[i](expert_outputs[:, i, :]) for i in range(self.args.k_basis)]
        expert_predictions = torch.stack(preds, dim=1)  # [B,H,out]
        pred = torch.sum(coordinates.unsqueeze(-1) * expert_predictions, dim=1)

        # residual head
        if 'src_idx' in batch:
            pred = pred + self.heads[int(batch['src_idx'])](cls_for_coord)
        elif getattr(self.args, 'use_target_head', False):
            pred = pred + self.thead(cls_for_coord)

        return pred