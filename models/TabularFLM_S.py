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
from models.BasisGraphAttention import BasisGATLayer_MUL, BasisGATLayer_IND
logger = logging.getLogger(__name__)


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


        if args.basis_type == 'mul':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_MUL(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, n_heads = args.n_heads, dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        elif args.basis_type == 'ind':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_IND(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, n_heads = args.n_heads, dropout = self.dropout_rate)
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
        # ---- Disentanglement Loss (λ=0.1, margin=2 고정, Disentangled Attention Graph Neural Network for Alzheimer’s Disease Diagnosis code) ----
        if hasattr(self, "_last_P_basis"):
            A = self._last_P_basis  # [B, H, S, S]
            B, H, S, _ = A.shape
            A_cols = A.permute(0, 2, 1, 3)  # [B, S, H, S]
            dists = torch.cdist(A_cols, A_cols, p=1)  # [B, S, H, H]
            avg_dists = torch.mean(dists, 1)          # [B, H, H]
            mean_dist = (2 * torch.triu(avg_dists, diagonal=1).sum(dim=(1, 2)) / (H * (H - 1))).mean()
            dis_loss = F.relu(2 - mean_dist)
            loss = loss + 0.3 * dis_loss
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
        self.expert_outputs = basis_outputs[:, 0, :, :]  # [B,H,head_dim]
        preds = [self.expert_predictors[i](self.expert_outputs[:, i, :]) for i in range(self.args.n_heads)]
        expert_predictions = torch.stack(preds, dim=1)  # [B,H,C]
        pred = torch.sum(coordinates.unsqueeze(-1) * expert_predictions, dim=1)  # [B,C]

        # residual heads
        if 'src_idx' in batch:
            pred = pred + self.heads[int(batch['src_idx'])](cls_for_coord)
        elif getattr(self.args, 'use_target_head', False):
            pred = pred + self.thead(cls_for_coord)

        return pred