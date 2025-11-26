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
        # self.latent_graph = LatentCompositeGraph(args, input_dim = self.graph_dim, n_graphs = args.n_graphs, n_nodes = args.n_nodes, node_dim = self.input_dim)
        # # (2) GraphQuantizer : FGW-based quantization module 
        # self.graph_quantizer = GraphQuantizer(args, alpha = 0.9, eps = 0.1)

        # # (3) LatentCompositeGNN : Head-wise message passing + readout 
        # self.gnn_experts = LatentCompositeGNN(
        #     args,input_dim = self.input_dim // args.num_basis_heads, hidden_dim = self.hidden_dim, num_basis_heads = args.num_basis_heads, dropout = self.dropout_rate
        # ) 

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
        for ln in self.basis_layer_norms:
            for p in ln.parameters():
                p.requires_grad = True
        for p in self.thead.parameters():
            p.requires_grad = True
        
    # ---- training ----
    def forward(self, batch, y):
        total_loss = 0.0 
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()

        # === Predict ===
        local_pred = self.predict(batch)

        # === Local output ===        
        # === 3. Base Classification losses === 
        local_loss = self.criterion(local_pred, target)
        total_loss = local_loss         
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

        # (8) Classificaion heads ---- 
        local_output = x_basis[:, 0, :]
        if 'src_idx' in batch:
            local_pred = self.sheads[int(batch['src_idx'])](local_output)
        elif getattr(self.args, 'use_target_head', False):
            local_pred = self.thead(local_output)
        else:
            local_pred = self.thead(local_output)
        self._last_local_pred = local_pred

        return local_pred