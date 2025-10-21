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
from utils.affinity import BasisSlotAffinityGAT
from models.coordinate import CoordinatorMLP
from models.SharedGraphAttention import SharedGraphAttention
logger = logging.getLogger(__name__)


class BasisGATLayer(nn.Module):
    """
    entropic_gw, prior_Q, DG, b 모두 제거한 바닐라 버전.
    - gat_v1 / gat_v2 / gate 지원
    - edge_type: normal / mlp / no_use 지원
    - CLS->Var on, Var->CLS off, Var-Var 구조 마스크 적용
    반환:
        basis_outputs: [B, new_seq, n_heads, head_dim]
        attn_weights : [B, n_heads, new_seq, new_seq]
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        assert args.attn_type in ['gat_v1', 'gat_v2', 'gate'], "attn_type must be one of {gat_v1,gat_v2,gate}"

        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
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
        q = self.q_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(B, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

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

            edge_attr = edge_attr.view(B, new_seq, new_seq, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
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
        # CLS
        self.shared_cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        self.basis_cls = nn.Parameter(Tensor(1,1,self.input_dim))
        nn.init.uniform_(self.shared_cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))
        nn.init.uniform_(self.basis_cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))
        self.num_basis_layers = int(getattr(args, 'num_basis_layers', 3))
        self.num_shared_layers = int(getattr(args, 'num_shared_layers', 3))


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

        # Experts (one per basis head)
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

        x_shared = torch.cat([self.shared_cls.expand(nv.size(0), 1, self.input_dim), nv], dim = 1) 
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

        # (2) 기존 Few-shot coord KL 유지 (타깃 에피소드에서 좌표 분포 정렬)
        lam = float(getattr(self.args, "coord_reg_lambda", 0.0))
        if (self.mode == 'Few') and (lam > 0.0) and hasattr(self, "centroids"):
            c = getattr(self, "_last_coordinates", None)
            if c is not None:
                from utils.coord_Kmeans import build_centroid_target
                recon_coords, assign_q = build_centroid_target(
                    c, self.centroids,
                    tau=float(getattr(self.args, "coord_tau", 0.3)),
                    mode=str(getattr(self.args, "coord_target_mode", "soft"))
                )
                eps = 1e-8
                c_safe = c.clamp_min(eps)
                temp = float(getattr(self.args, "coord_softmax_temp",1.0))
                recon_logprob = F.log_softmax(recon_coords / max(temp,1e-6),dim=1)
                coord_reg = F.kl_div(recon_logprob, c_safe, reduction='batchmean')
                loss = loss + lam * coord_reg
                self._last_assign_q = assign_q.detach()
        # ---- Disentanglement Loss (λ=0.1, margin=2 고정) ----
        if hasattr(self, "_last_P_basis"):
            A = self._last_P_basis  # [B, H, S, S]
            B, H, S, _ = A.shape
            A_cols = A.permute(0, 2, 1, 3)  # [B, S, H, S]
            dists = torch.cdist(A_cols, A_cols, p=1)  # [B, S, H, H]
            avg_dists = torch.mean(dists, 1)          # [B, H, H]
            mean_dist = (2 * torch.triu(avg_dists, diagonal=1).sum(dim=(1, 2)) / (H * (H - 1))).mean()
            dis_loss = F.relu(2 - mean_dist)
            loss = loss + 0.3 * dis_loss
            self._last_dis_loss = dis_loss.detach()

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
        x_shared = torch.cat([self.shared_cls.expand(nv.size(0), 1, self.input_dim), nv], dim=1)
        for l in range(self.num_shared_layers):
            nx = self.shared_layer_norms[l](x_shared)
            out, _ = self.shared_layers[l](desc, nx)
            x_shared = x_shared + out
        cls_for_coord = x_shared[:, 0, :]
        coordinates = self.coordinator(cls_for_coord)
        self._last_coordinates = coordinates

        # ---- basis GAT stack: prior_Q=Q_hat만 사용 ----
        x_basis  = torch.cat([self.basis_cls.expand(nv.size(0), 1, self.input_dim), nv], dim=1)
        last_att = None
        for l in range(self.num_basis_layers):
            norm_x = self.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.basis_layers[l](desc, norm_x)
            x_basis = x_basis + basis_outputs.reshape(x_basis.size(0), x_basis.size(1), self.input_dim)
            last_att = att

        if last_att is not None:
            # Var-Var 블록만 저장: [B,H,S,S]
            self._last_P_basis = last_att[:, :, 1:, 1:]

        # ---- experts & mixture ----
        expert_outputs = basis_outputs[:, 0, :, :]  # [B,H,head_dim]
        preds = [self.expert_predictors[i](expert_outputs[:, i, :]) for i in range(self.args.n_heads)]
        expert_predictions = torch.stack(preds, dim=1)  # [B,H,C]
        pred = torch.sum(coordinates.unsqueeze(-1) * expert_predictions, dim=1)  # [B,C]

        # residual heads
        if 'src_idx' in batch:
            pred = pred + self.heads[int(batch['src_idx'])](cls_for_coord)
        elif getattr(self.args, 'use_target_head', False):
            pred = pred + self.thead(cls_for_coord)

        return pred

    
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