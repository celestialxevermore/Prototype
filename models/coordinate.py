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
    """
        GAT encoder의 CLS embedding을 입력 받아,
        K개의 basis function(Attention head)에 대한 좌표(가중치)를 생성한다.
    """
    def __init__(self, input_dim: int, hidden_dim : int, k_basis : int, dropout : float = 0.1):
        """
            Args:
                input_dim (int) CLS dimension 
                hidden_dim (int) MLP
                k_basis (int) : 생성할 좌표의 개수 (Basis function / Expert head의 개수)
                dropout (float) : dropout rate
        """
        super().__init__()
        self.k_basis = k_basis 
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.dropout = dropout 
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.k_basis),
            nn.Softmax(dim = -1)
        )
        self._init_weights() 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)
    def forward(self, cls_emb : torch.Tensor) -> torch.Tensor:
        """
            Args:
                cls (torch.Tensor) : [batch_size, input_dim]
            Returns:
                torch.Tensor : [batch_size, k_basis]
        """    
        coordinates = self.coordinate_mlp(cls_emb)
        
        return coordinates