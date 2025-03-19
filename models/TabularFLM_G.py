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


class AdaptiveGraphAttention(nn.Module):
    def __init__(
        self,
        input_dim : int,
        n_heads : int, 
        dropout : float = 0.1
    ):
        super().__init__()
        assert input_dim % n_heads == 0 
        self.n_heads = n_heads 
        self.head_dim = input_dim // n_heads 
        self.input_dim = input_dim

        self.edge_update = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        self.sample_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        self.topology_bias = nn.Parameter(torch.zeros(1))
        self.attn_dropout = nn.Dropout(dropout)

        self.global_topology_proj = nn.Linear(input_dim, input_dim)
        self.adaptive_weight_proj = nn.Linear(input_dim, input_dim)

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.attn_proj = nn.Linear(self.head_dim * 3, 1)  # [q | k | edge_attr] -> attention score
        
        
        self.out_proj = nn.Linear(input_dim, input_dim)
        # nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        # nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        # nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        #nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        #nn_init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
    def forward(self, desc_embeddings, name_embeddings, value_embeddings, cls_token, edge_attr = None):
        batch_size, seq_len, _ = value_embeddings.shape 

        '''
            1. Glboal Topology
        '''
        desc_embeddings = self.global_topology_proj(desc_embeddings)
        global_sim = torch.matmul(desc_embeddings, desc_embeddings.transpose(-1, -2))
        global_topology = torch.sigmoid(global_sim + self.topology_bias)


        '''
            2. Sample-wise Weight
        '''
        
        name_value_embeddings = torch.cat([name_embeddings, value_embeddings], dim = -1)
        name_value_embeddings = self.sample_fusion(name_value_embeddings)
        sample_sim = torch.matmul(name_value_embeddings, name_value_embeddings.transpose(-1, -2))
        adaptive_weight = torch.sigmoid(sample_sim)
        adjacency_matrix = global_topology * adaptive_weight 

        '''
            3. Self Connection Delete
        '''

        diag_mask = 1.0 - torch.eye(seq_len, device = value_embeddings.device).unsqueeze(0)
        adjacency = adjacency_matrix * diag_mask 

        '''
            4. [CLS] Token
        '''
        cls_token = cls_token.expand(batch_size, -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)
        new_seq = seq_len + 1

        # Edge attributes 생성
        # 1. 변수 노드 간의 edge_attr
        node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)  # [batch, seq_len, seq_len, dim*2]

        # 2. CLS-변수 노드 간의 edge_attr
        cls_edge_attr = torch.cat([
            desc_embeddings.unsqueeze(1),  # CLS->변수: 변수의 description
            desc_embeddings.unsqueeze(1)   # 변수->CLS: 동일한 description
        ], dim=-1)  # [batch, 1, seq_len, dim*2]

        # 3. Edge attribute 합치기
        edge_attr = torch.zeros(batch_size, new_seq, new_seq, var_edge_attr.size(-1), device=var_edge_attr.device)
        edge_attr[:, 1:, 1:] = var_edge_attr  # 변수 노드 간
        edge_attr[:, 0, 1:] = cls_edge_attr.squeeze(1)  # CLS->변수
        edge_attr[:, 1:, 0] = cls_edge_attr.squeeze(1)  # 변수->CLS

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=adjacency.device)
        new_adjacency[:, 1:, 1:] = adjacency 
        new_adjacency[:, 0, 1:] = 1.0 
        new_adjacency[:, 1:, 0] = 0.0 

        # q, k, v projection
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

        if edge_attr is not None:
            edge_attr = self.edge_update(edge_attr)
            edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
            edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
            edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)

        return output, attn_weights
            
class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.args = args
        self.llm_model = llm_model
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.source_dataset_name = args.source_dataset_name
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.meta_type = args.meta_type
        self.enc_type = args.enc_type
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))
        def init_mlp(module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # ReLU에 적합한 He 초기화
                    if m.bias is not None:
                        nn_init.zeros_(m.bias)

        self.layers = nn.ModuleList([
            AdaptiveGraphAttention(
                input_dim = self.input_dim, 
                n_heads = args.n_heads,
                dropout = args.dropout_rate
            ) for _ in range(args.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(args.num_layers)
        ])
        self.dropout = nn.Dropout(args.dropout_rate)
        self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)


    def forward(self, batch, y):
        pred = self.predict(batch)
        target = y.to(self.device).view(-1,1).float()
        loss = self.criterion(pred, target)
        return loss
    
    def predict(self, batch):
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        
        desc_embeddings = [] 
        name_embeddings = [] 
        value_embeddings = [] 

        sample_embeddings = []

        if all(k in batch for k in ['cat_name_embeddings', 'cat_desc_embeddings', 'cat_value_embeddings']):
            cat_name_embeddings = batch['cat_name_embeddings'].to(self.device).squeeze(-2)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
            cat_value_embeddings = batch['cat_value_embeddings'].to(self.device).squeeze(-2)

            desc_embeddings.append(cat_desc_embeddings)
            name_embeddings.append(cat_name_embeddings)
            value_embeddings.append(cat_value_embeddings)    
            

        if all(k in batch for k in ['num_name_embeddings', 'num_desc_embeddings', 'num_prompt_embeddings']):
            num_name_embeddings = batch['num_name_embeddings'].to(self.device).squeeze(-2)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
            
            
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)
            
            
            desc_embeddings.append(num_desc_embeddings)
            name_embeddings.append(num_name_embeddings)
            value_embeddings.append(num_prompt_embeddings)
            
        if not desc_embeddings or not name_embeddings or not value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim = 1)
        name_embeddings = torch.cat(name_embeddings, dim = 1)
        value_embeddings = torch.cat(value_embeddings, dim = 1)



        # if self.args.label == 'add':
        #     sample_embeddings = torch.concat([label_description_embeddings, sample_embeddings], dim=1)

        x = value_embeddings 
        for i, layer in enumerate(self.layers):
            norm_x = self.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, name_embeddings, norm_x, self.cls)
            x = x + attn_output[:, 1:, :]
        cls_repr = x[:, 0, :]
        pred = self.predictor(cls_repr)
        
        # x = value_embeddings
        # cls_hidden = self.cls.expand(value_embeddings.size(0), -1, -1)  # 초기 CLS
        
        # for i, layer in enumerate(self.layers):
        #     norm_x = self.layer_norms[i](x)
        #     attn_output, attn_weights = layer(desc_embeddings, name_embeddings, norm_x, cls_hidden)
        #     x = attn_output[:, 1:, :]  # 변수 노드 업데이트
        #     cls_hidden = attn_output[:, 0, :].unsqueeze(1)  # 업데이트된 CLS를 다음 layer로
            
        # pred = self.predictor(cls_hidden.squeeze(1))

        #pdb.set_trace()
        return pred

