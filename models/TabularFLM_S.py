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
        self.num_basis_layers = args.num_basis_layers
        self.alpha = self.args.alpha 
        self.eps = self.args.eps
        self.basis_cls = nn.Parameter(Tensor(1,1,self.input_dim))
        nn.init.uniform_(self.basis_cls, a=-1/math.sqrt(self.input_dim), b=1/math.sqrt(self.input_dim))
        self.switch_epoch = 40
        self.current_epoch = 0

        #---- Latent Composite Graph Components ---- 
        # (1) LatentCompositeGraph : learnable latent composite graphs (codebook)
        self.latent_graph = LatentCompositeGraph(args, input_dim = self.graph_dim, n_graphs = args.n_graphs, n_nodes = args.n_nodes, node_dim = self.input_dim)
        
        # (2) GraphQuantizer : FGW-based quantization module 
        self.graph_quantizer = GraphQuantizer(args, alpha = self.alpha, eps = self.eps)

        # (3) LatentCompositeGNN : Head-wise message passing + readout 
        self.gnn_experts = LatentCompositeGNN(
            args,input_dim = self.input_dim, hidden_dim = self.hidden_dim, dropout = self.dropout_rate
        ) 

        if args.basis_type == 'mul':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_MUL(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = 1 , dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        elif args.basis_type == 'ind':
            self.basis_layers = nn.ModuleList([ 
                BasisGATLayer_IND(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = 1, dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        self.basis_layer_norms = nn.ModuleList([ 
            nn.LayerNorm(self.input_dim) for _ in range(self.num_basis_layers)
        ])


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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def extract_description(self, batch):
        desc_embeddings = [] 
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        if 'cat_desc_embeddings' in batch: desc_embeddings.append(batch['cat_desc_embeddings'])
        if 'num_desc_embeddings' in batch: desc_embeddings.append(batch['num_desc_embeddings'])
        if not desc_embeddings: return None 
        return torch.cat(desc_embeddings, dim = 1)

    # Few-shot freeze policy
    def set_freeze_target(self):
        for p in self.parameters():
            p.requires_grad = False
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
    # ---- training ----
    def forward(self, batch, y):
        total_loss = 0.0 
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()

        global_pred, local_pred = self.predict(batch, return_all = True)
        # === 3. Base Classification losses === 
        global_loss = self.criterion(global_pred, target)
        local_loss = self.criterion(local_pred, target)
        kl_loss = 0.0
        if self.args.kl_gamma > 0.0:
            # Teacher: Local (잘하는 놈 -> Detach 필수!)
            # 확률 분포(Softmax)로 변환
            p_teacher = F.softmax(local_pred.detach(), dim=-1)
            
            # Student: Global (배우는 놈 -> Log Softmax)
            p_student = F.log_softmax(global_pred, dim=-1)
            
            # KL(Teacher || Student) 계산
            kl_loss = F.kl_div(p_student, p_teacher, reduction='batchmean')

        # 4. Total Loss 합산
        # Local Loss + Global Loss + FGW Loss + KL Loss
        total_loss = local_loss + global_loss + (self.args.fgw_alpha * self.fgw_loss) + (self.args.kl_gamma * kl_loss)
        
        return total_loss

    # ---- inference ----
    def predict(self, batch, return_all = False):
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
        name   = torch.cat(name_embeddings, dim = 1)
        value = torch.cat(value_embeddings, dim = 1)

        # (2) latent composite graph, latent composite affinity 
        lcg_feat, lcg_struct = self.latent_graph()
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
            self._last_P_basis = 1.0 - last_att[:, 0, 1:, 1:] # P_affinity 
        self.basis_outputs_for_viz = basis_outputs
        # [수정] Dynamic Detach 로직 추가 (필수)
        if self.current_epoch < self.switch_epoch:
            # Phase 1~3: GAT 보호 (Detach ON)
            source_feat_in = self.x_basis[:, 1:, :].detach()
            source_struct_in = self._last_P_basis.detach()
        else:
            # Phase 4: 완전체 (Detach OFF)
            source_feat_in = self.x_basis[:, 1:, :]
            source_struct_in = self._last_P_basis



        q_lcg_feat, q_lcg_struct, coordinates, fgw_loss = self.graph_quantizer( 
            source_struct = source_struct_in, 
            source_feat = source_feat_in, 
            lcg_struct = lcg_struct, 
            lcg_feat = lcg_feat, 
            batch = batch
        )

        self.fgw_loss = fgw_loss
        # # (5) LCG-wise GNN message passing & readout ---- 
        expert_outputs = self.gnn_experts(q_lcg_feat, q_lcg_struct) # [B, H, D]
        # (6) Coordinator-weighted combination ----
        # # (7) Global prediction
        expert_outputs = (coordinates.unsqueeze(-1) * expert_outputs).sum(dim = 1)
        global_pred = self.ghead(expert_outputs)

        # (8) Classificaion heads ---- 
        local_output = x_basis[:, 0, :]
        if 'src_idx' in batch:
            local_pred = self.sheads[int(batch['src_idx'])](local_output)
        elif getattr(self.args, 'use_target_head', False):
            local_pred = self.thead(local_output)
        else:
            local_pred = self.thead(local_output)
        self._last_local_pred = local_pred
        if self.training or return_all:
            return global_pred, local_pred
        else:
            if self.current_epoch < self.switch_epoch:
                return local_pred 
            else:
                return global_pred
