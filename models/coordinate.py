import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from torch import Tensor 
import math 
import torch.nn.init as nn_init
import logging 
import os 

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
