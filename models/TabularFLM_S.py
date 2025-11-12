import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from torch import Tensor
import math
import torch.nn.init as nn_init
import logging
from models.coordinate import CoordinatorMLP 
from models.LCG import LatentCompositeGraph, GraphQuantizer 
from models.LCGGNN import LatentCompositeGNN
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
        self.n_graphs = self.args.n_graphs
        self.graph_dim = self.args.graph_dim
        
        # CLS
        self.basis_cls = nn.Parameter(Tensor(1,1,self.input_dim))
        nn.init.uniform_(self.basis_cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))
        self.num_basis_layers = args.num_basis_layers

        # ---- Latent Composite Graph Components ---- 
        # (1) LatentCompositeGraph : learnable latent composite graphs (codebook)
        self.latent_graph = LatentCompositeGraph(args, input_dim = self.graph_dim, n_graphs = args.n_graphs, n_nodes = args.n_nodes, node_dim = self.input_dim)
        # (2) GraphQuantizer : FGW-based quantization module 
        self.graph_quantizer = GraphQuantizer(args, alpha = args.fgw_alpha)

        # (3) LatentCompositeGNN : Head-wise message passing + readout 
        self.gnn_experts = LatentCompositeGNN(
            args,input_dim = self.input_dim // args.num_basis_heads, hidden_dim = self.hidden_dim, num_basis_heads = args.num_basis_heads, dropout = self.dropout_rate
        ) 

        if args.basis_type == 'mul':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_MUL(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = args.num_basis_heads, dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        elif args.basis_type == 'ind':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_IND(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = args.num_basis_heads, dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        self.basis_layer_norms = nn.ModuleList([ 
            nn.LayerNorm(self.input_dim) for _ in range(self.num_basis_layers)
        ])

        # Coordinator (weights over heads/bases)
        self.coordinator = CoordinatorMLP(args, self.input_dim, hidden_dim, args.num_basis_heads, self.dropout_rate)

        # Source/Target residual heads (on CLS)
        self.n_src = len(args.source_data) if isinstance(args.source_data, (list, tuple)) else 1
        hid = min(128, self.input_dim)
        self.sheads = nn.ModuleList([
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

        self.ghead = nn.Sequential(
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
        for p in self.coordinator.parameters():
            p.requires_grad = True
        for ln in self.basis_layer_norms:
            for p in ln.parameters():
                p.requires_grad = True
        for p in self.thead.parameters():
            p.requires_grad = True
        for p in self.latent_graph.parameters():
            p.requires_grad = True 
        for p in self.graph_quantizer.parameters():
            p.requires_grad = True 
        for p in self.gnn_experts.parameters():
            p.requires_grad = True
        for p in self.ghead.parameters(): # ghead unfreeze 시켜주는게 성능에는 더 좋고 Computational Cost도 적음. 
            p.requires_grad = True

    @torch.no_grad()
    def get_coordinates(self, batch):
        self.eval()
        # gather
        desc_list, n_list, v_list = [], [], []
        if all(k in batch for k in ['cat_name_embeddings', 'cat_value_embeddings', 'cat_desc_embeddings']):
            n_list.append(batch['cat_name_embeddings'].to(self.device))
            v_list.append(batch['cat_value_embeddings'].to(self.device))
            desc_list.append(batch['cat_desc_embeddings'].to(self.device))
            
        if all(k in batch for k in ['num_name_embeddings', 'num_prompt_embeddings','num_desc_embeddings']):
            n_list.append(batch['num_name_embeddings'].to(self.device))
            v_list.append(batch['num_prompt_embeddings'].to(self.device))
            desc_list.append(batch['num_desc_embeddings'].to(self.device))
        if not desc_list or not n_list or not v_list:
            raise ValueError("No categorical or numerical features found in batch")

        desc = torch.cat(desc_list, dim=1)  # [B,S,D]
        name = torch.cat(n_list , dim=1)
        value   = torch.cat(v_list , dim=1)    # [B,S,D]

        coordinates = self.coordinator(desc, name, value).mean(dim=1)
        return coordinates 

    def set_kmeans_centroids(self, centroids: torch.Tensor):
        self.register_buffer("centroids", centroids.detach(), persistent=False)
        self.best_k = int(centroids.size(0))
    def set_coord_temperature(self, t: float):
        self.coordinator.temperature = float(t)
    # ---- training ----
    def forward(self, batch, y):
        total_loss = 0.0 
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()


        # === Predict ===
        global_pred = self.predict(batch)

        # === Local output ===
        local_output = self.x_basis[:, 0, :]

        if 'src_idx' in batch:
            local_pred = self.sheads[int(batch['src_idx'])](local_output)
        elif self.args.use_target_head:
            local_pred = self.thead(local_output)
        else:
            local_pred = self.thead(local_output)
        
        # === 3. Base Classification losses === 
        global_loss = self.criterion(global_pred, target)
        local_loss = self.criterion(local_pred, target)
        task_loss = 0.5 * (global_loss + local_loss)
        total_loss += task_loss 
        # === 4. KL consistency loss === 
        if self.args.kl_gamma > 0.0:
            p_local = F.log_softmax(local_pred.detach(), dim=-1)
            p_global = F.softmax(global_pred, dim=-1)
            kl_loss = F.kl_div(p_local, p_global, reduction='batchmean')
            total_loss += self.args.kl_gamma * kl_loss 

        # === 5. FGW loss (distribution alignment between Multiple source <-> Latent Composite Graph) ===
        total_loss += self.args.fgw_alpha * self.fgw_loss 

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
                total_loss += lam * coord_reg
                self._last_assign_q = assign_q.detach()
        
        # ---- Diversifying Loss (P-Space Coordinate Constraint) ----
        
        if hasattr(self, "_last_coordinates") and self.args.diversifying_loss is True:
            coordinates = self._last_coordinates
            labels = y.to(self.device)
            distance = (coordinates.unsqueeze(1) - coordinates.unsqueeze(0)).abs().sum(dim=2)
            label_similarity = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            positive_mask = label_similarity
            div_loss = torch.sum(distance * positive_mask) / (torch.sum(distance) + 1e-8)
            total_loss += 0.3 * div_loss
        
        return total_loss


    # ---- inference ----
    def predict(self, batch):
        # gather
        desc_embeddings, name_embeddings, value_embeddings = [], [], []
        if all(k in batch for k in ['cat_name_embeddings', 'cat_value_embeddings', 'cat_desc_embeddings']):
            desc_embeddings.append(batch['cat_desc_embeddings'].to(self.device))
            name_embeddings.append(batch['cat_name_embeddings'].to(self.device))
            value_embeddings.append(batch['cat_value_embeddings'].to(self.device))
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            desc_embeddings.append(batch['num_desc_embeddings'].to(self.device))
            name_embeddings.append(batch['num_name_embeddings'].to(self.device))
            value_embeddings.append(batch['num_prompt_embeddings'].to(self.device))

        if not desc_embeddings or not name_embeddings or not value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")
        desc = torch.cat(desc_embeddings, dim = 1)  # [B,S,D]
        name   = torch.cat(name_embeddings, dim = 1)
        value = torch.cat(value_embeddings, dim = 1)

        # (2) coordinator weights (desc + nv -> coord) ----
        coordinates = self.coordinator(desc, name, value).mean(dim=1)
        self._last_coordinates = coordinates

        # (3) basis GAT stack ---- 
        
        x_basis  = torch.cat([self.basis_cls.expand(value.size(0), 1, self.input_dim), value], dim=1)
        last_att = None
        for l in range(self.num_basis_layers):
            norm_x = self.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.basis_layers[l](name, norm_x)
            x_basis = x_basis + basis_outputs.reshape(x_basis.size(0), x_basis.size(1), self.input_dim)
            last_att = att
        self.x_basis = x_basis
        if last_att is not None:
            self._last_P_basis = last_att[:, :, 1:, 1:] # P_affinity 
        # (4) FGW-based quantization ---- 
        self.basis_outputs_for_viz = basis_outputs

        Fy_res, Ay_res, fgw_loss = self.graph_quantizer(
            self._last_P_basis, 
            basis_outputs, self.latent_graph
        )
        self.fgw_loss = fgw_loss
        # (5) Head-wise GNN message passing & readout ---- 
        expert_outputs = self.gnn_experts(Fy_res, Ay_res) # [B, H, D]
        # (6) Coordinator-weighted combination ----
        # # (7) Global prediction
        expert_outputs = (coordinates.unsqueeze(1).unsqueeze(-1) * expert_outputs).sum(dim = 2)
        global_output = expert_outputs.reshape(expert_outputs.size(0), -1)
        global_pred = self.ghead(global_output)

        # (8) Classificaion heads ---- 
        local_output = x_basis[:, 0, :]
        if 'src_idx' in batch:
            local_pred = self.sheads[int(batch['src_idx'])](local_output)
        elif getattr(self.args, 'use_target_head', False):
            local_pred = self.thead(local_output)
        else:
            local_pred = self.thead(local_output)
        self._last_local_pred = local_pred

        return global_pred