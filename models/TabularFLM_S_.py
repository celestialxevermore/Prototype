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
from models.BasisGraphAttention import BasisGATLayer_MUL, BasisGATLayer_IND, BasisScaledDotLayer, BasisGraphormerLayer
import copy
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
        self.use_ema = getattr(args, 'ema', False)
        self.ema_decay = 0.999
        if self.use_ema:
            self.latent_graph_ema = copy.deepcopy(self.latent_graph)
            for p in self.latent_graph_ema.parameters():
                p.requires_grad = False 
            print(">>> [Model] EMA Enabled for LCG.")
        else: 
            self.latent_graph_ema = None 
        #---- Latent Composite Graph Components ---- 
        # (1) LatentCompositeGraph : learnable latent composite graphs (codebook)
        self.latent_graph = LatentCompositeGraph(args, input_dim = self.graph_dim, n_graphs = args.n_graphs, n_nodes = args.n_nodes, node_dim = self.input_dim)
        
        # (2) GraphQuantizer : FGW-based quantization module 
        self.graph_quantizer = GraphQuantizer(args, alpha = args.alpha, tau = args.tau, eps = self.eps)

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
        elif args.basis_type == 'sdpa':
            self.basis_layers = nn.ModuleList([
                BasisScaledDotLayer(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = 1, dropout = self.dropout_rate)
                for _ in range(self.num_basis_layers)
            ])
        elif args.basis_type == 'graphormer':
            self.basis_layers = nn.ModuleList([
                BasisGraphormerLayer(args, input_dim = self.input_dim, hidden_dim = self.hidden_dim, num_basis_heads = 1, dropout = self.dropout_rate)
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
        self.ghead2 = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hid, self.output_dim),
        )
        # Loss (num_classes==1: regression, ==2: binary, >2: multiclass)
        if self.num_classes == 1:
            self.criterion = nn.MSELoss()
        elif self.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)
    def set_freeze_target(self):
            for p in self.parameters(): p.requires_grad = False
            for p in self.basis_layers.parameters(): p.requires_grad = True
            for p in self.basis_layer_norms.parameters(): p.requires_grad = True
            for p in self.thead.parameters(): p.requires_grad = True
            for p in self.ghead.parameters(): p.requires_grad = True
    def forward(self, batch, y):
        total_loss = 0.0
        target = y.to(self.device)
        if self.num_classes <= 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()

        if not getattr(self.args, 'use_lcg', False):
            local_pred = self.predict(batch)
            loss = self.criterion(local_pred, target)
            return loss
        else:
            current_mode = getattr(self, 'mode', 'Full')

            if current_mode == 'Few':
                global_pred, local_pred = self.predict(batch, return_all=True)
                local_loss = self.criterion(local_pred, target)
                global_loss = self.criterion(global_pred, target)
                total_loss = local_loss + global_loss
                #total_loss = global_loss
                self._last_local_loss = float(local_loss.detach().item())
                self._last_global_loss = 0.0
                self._last_reg_loss = 0.0
                self._last_total_loss = float(local_loss.detach().item())
            else:
                global_pred, local_pred = self.predict(batch, return_all=True)
                local_loss = self.criterion(local_pred, target)
                global_loss = self.criterion(global_pred, target)
                reg_loss = self.reg_loss

                _ablate_reg = getattr(self.args, 'ablate_reg_loss', False)
                _reg_term = 0.0 if _ablate_reg else (self.args.fgw_alpha * reg_loss)
                total_loss = local_loss + global_loss + _reg_term

                self._last_local_loss = float(local_loss.detach().item())
                self._last_global_loss = float(global_loss.detach().item())
                self._last_reg_loss = float(reg_loss.detach().item() if torch.is_tensor(reg_loss) else float(reg_loss))
                self._last_total_loss = float(total_loss.detach().item())

            return total_loss

    def predict(self, batch, return_all=False):
        # gather embeddings
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
        name = torch.cat(name_embeddings, dim=1)
        value = torch.cat(value_embeddings, dim=1)

        # === GAT encoder ===
        x_basis = torch.cat([self.basis_cls.expand(value.size(0), 1, self.input_dim), value], dim=1)
        last_att = None
        for l in range(self.num_basis_layers):
            norm_x = self.basis_layer_norms[l](x_basis)
            basis_outputs, att = self.basis_layers[l](name, norm_x)
            x_basis = x_basis + basis_outputs.reshape(x_basis.size(0), x_basis.size(1), self.input_dim)
            last_att = att
        self.x_basis = x_basis
        self.basis_outputs_for_viz = basis_outputs

        # === Local prediction (CLS → head) ===
        local_output = x_basis[:, 0, :]
        if 'src_idx' in batch:
            local_pred = self.sheads[int(batch['src_idx'])](local_output)
        else:
            local_pred = self.thead(local_output)
        self._last_local_pred = local_pred

        # === Phase 1: no LCG ===
        if not getattr(self.args, 'use_lcg', False):
            return local_pred

        # === Few-shot training: local만 쓰므로 global path 스킵 ===
        current_mode = getattr(self, 'mode', 'Full')
        # if current_mode == 'Few' and self.training:
        #     return local_pred, global_pred

        # === Global path: FGW → coordinates → experts → ghead ===
        if last_att is not None:
            attn = last_att[:, 0, 1:, 1:]  # (B, N, N)
            log_attn = -torch.log(attn.clamp_min(1e-8))
            with torch.no_grad():
                B = log_attn.shape[0]
                flat = log_attn.view(B, -1)
                lo = torch.quantile(flat, 0.05, dim=1).view(B, 1, 1)
                hi = torch.quantile(flat, 0.95, dim=1).view(B, 1, 1)
                span = (hi - lo).clamp_min(1e-8)
            self._last_P_basis = ((log_attn - lo) / span).clamp(0.0, 1.0)

        lcg_feat, lcg_struct = self.latent_graph()
        q_lcg_feat, q_lcg_struct, coordinates, reg_loss = self.graph_quantizer(
            source_struct=self._last_P_basis,
            source_feat=self.x_basis[:, 1:, :],
            lcg_struct=lcg_struct,
            lcg_feat=lcg_feat,
            batch=batch,
        )
        self.reg_loss = reg_loss
        self._last_coordinates = coordinates
        
        expert_outputs_per = self.gnn_experts(q_lcg_feat, q_lcg_struct)
        self._last_expert_outputs_per = expert_outputs_per

        expert_outputs = (coordinates.unsqueeze(-1) * expert_outputs_per).sum(dim=1)
        self._last_expert_combined = expert_outputs

        global_pred = self.ghead(expert_outputs)

        if return_all:
            return global_pred, local_pred
        else:
            return global_pred
    