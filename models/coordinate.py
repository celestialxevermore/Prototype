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
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = float(temperature)

        # desc + name + value 결합 후 projection
        concat_dim = self.input_dim * 3  # [desc, name value]
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(concat_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.n_heads),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def forward(self, desc: torch.Tensor, name: torch.Tensor, value : torch.Tensor) -> torch.Tensor:
        """
        cls_emb: [B, D]
        desc:    [B, S, D]
        name:      [B, S, D]
        value:      [B, S, D]
        """
        # desc와 nv의 평균을 사용하여 요약 (feature-wise mean pooling)
        # CLS, desc_mean, nv_mean 결합
        fused = torch.cat([desc, name, value], dim=-1)  # [B, 3D]

        logits = self.coordinate_mlp(fused)  # [B, K]
        return F.softmax(logits / max(self.temperature, 1e-6), dim=-1)