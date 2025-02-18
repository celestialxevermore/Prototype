import pdb
import torch
import torch.nn as nn 

import numpy as np 
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, DynamicEdgeConv, EdgeConv,RGCNConv, TransformerConv, GINConv, global_mean_pool

from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, Set2Set
from torch_scatter import scatter_mean
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.use_deterministic_algorithms(False)
import json 

class NORM_GNN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3):
        super(NORM_GNN, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Return graph representation after global mean pooling
        return global_mean_pool(x, data.batch)  

class GAT_edge(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
        super(GAT_edge, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = num_layers
        self.heads = heads

        self.edge_dim = args.input_dim

        # GAT_edge_3와 동일하게 edge_mlp 추가 (FD=='ND' 시 사용)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(
                self.input_dim,
                self.hidden_dim // self.heads,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    self.hidden_dim,
                    self.hidden_dim // self.heads,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        # Output layer
        self.convs.append(
            GATConv(
                self.hidden_dim,
                self.hidden_dim // self.heads,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(self.hidden_dim // self.heads))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # FD=='ND'인 경우, desc_attr와 합쳐서 edge_mlp에 통과시킴
        if self.args.FD == 'ND':
            desc_attr = data.desc_attr  # GAT_edge_3와 동일
            new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
        else:
            new_edge_attr = edge_attr
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        
        return global_mean_pool(x, data.batch)

class GAT_edge_2(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
        super(GAT_edge_2, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        
        # FD=='ND' 인 경우 desc_attr + edge_attr 합쳐서 MLP 통과
        self.use_desc = (args.FD == 'ND')
        if self.use_desc:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.input_dim * 2, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),

                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                
                nn.Linear(self.hidden_dim, self.input_dim)
            )
        # graph_type=='full'인 경우 heads 등 자동 조정
        if 'full' in args.graph_type:
            self.heads = min(4, heads)
            self.num_layers = min(3, num_layers)
            self.dropout_rate = dropout_rate * 1.5
        else:
            self.heads = heads
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate

        self.edge_dim = self.input_dim
        self.output_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # 첫 번째 레이어
        first_hidden = self.hidden_dim // self.heads
        self.convs.append(
            GATConv(
                self.input_dim,
                first_hidden,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
        self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 중간 레이어
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    first_hidden * self.heads,
                    first_hidden,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
            self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 마지막 레이어
        self.convs.append(
            GATConv(
                first_hidden * self.heads,
                hidden_dim,
                heads=1,
                edge_dim=self.edge_dim,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Dropout, Skip proj 세팅
        self.dropout = nn.Dropout(self.dropout_rate)
        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
            elif i == self.num_layers - 1:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
            else:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # ND이면 desc_attr 활용
        if self.use_desc:
            desc_attr = data.desc_attr
            new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
        else:
            new_edge_attr = edge_attr

        layer_outputs = [x]

        # 공통 레이어 순회
        for i in range(self.num_layers):
            current = x

            x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
            x = self.bns[i](x)
            x = self.layer_norms[i](x)

            # Skip
            if i > 0:
                skip = self.skip_projs[i](current)
                x = x + skip

            x = F.elu(x)
            x = self.dropout(x)

            layer_outputs.append(x)

        return global_mean_pool(x, data.batch)

# class GAT_edge_3(nn.Module):
#     def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
#         super(GAT_edge_3, self).__init__()
#         self.args = args
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.hidden_dim
        
#         # FD=='ND' 인 경우 desc_attr + edge_attr 합쳐서 MLP 통과
#         self.use_desc = (args.FD == 'ND')
#         if self.use_desc:
#             self.edge_mlp = nn.Sequential(
#                 nn.Linear(self.input_dim * 2, self.input_dim),
#                 nn.LayerNorm(self.input_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_rate),
                
#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.LayerNorm(self.hidden_dim),
#                 nn.ReLU(),

#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.LayerNorm(self.hidden_dim),
#                 nn.ReLU(),
                
#                 nn.Linear(self.hidden_dim, self.input_dim)
#             )
#         # graph_type=='full'인 경우 heads 등 자동 조정
#         if 'full' in args.graph_type:
#             self.heads = min(4, heads)
#             self.num_layers = min(3, num_layers)
#             self.dropout_rate = dropout_rate * 1.5
#         else:
#             self.heads = heads
#             self.num_layers = num_layers
#             self.dropout_rate = dropout_rate

#         self.edge_dim = self.input_dim
#         self.output_dim = hidden_dim
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.layer_norms = nn.ModuleList()

#         # 첫 번째 레이어
#         first_hidden = self.hidden_dim // self.heads
#         self.convs.append(
#             GATv2Conv(
#                 self.input_dim,
#                 first_hidden,
#                 heads=self.heads,
#                 edge_dim=self.edge_dim,
#                 concat=True
#             )
#         )
#         self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#         self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

#         # 중간 레이어
#         for _ in range(self.num_layers - 2):
#             self.convs.append(
#                 GATv2Conv(
#                     first_hidden * self.heads,
#                     first_hidden,
#                     heads=self.heads,
#                     edge_dim=self.edge_dim,
#                     concat=True
#                 )
#             )
#             self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#             self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

#         # 마지막 레이어
#         self.convs.append(
#             GATv2Conv(
#                 first_hidden * self.heads,
#                 hidden_dim,
#                 heads=1,
#                 edge_dim=self.edge_dim,
#                 concat=False
#             )
#         )
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
#         self.layer_norms.append(nn.LayerNorm(hidden_dim))

#         # Dropout, Skip proj 세팅
#         self.dropout = nn.Dropout(self.dropout_rate)
#         self.skip_projs = nn.ModuleList()
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
#             elif i == self.num_layers - 1:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
#             else:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         # ND이면 desc_attr 활용
#         if self.use_desc:
#             desc_attr = data.desc_attr
#             new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
#         else:
#             new_edge_attr = edge_attr

#         layer_outputs = [x]

#         # 공통 레이어 순회
#         for i in range(self.num_layers):
#             current = x

#             x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
#             x = self.bns[i](x)
#             x = self.layer_norms[i](x)

#             # Skip
#             if i > 0:
#                 skip = self.skip_projs[i](current)
#                 x = x + skip

#             x = F.elu(x)
#             x = self.dropout(x)

#             layer_outputs.append(x)

#         return global_mean_pool(x, data.batch)
class GAT_edge_3(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3, heads=8):
        super(GAT_edge_3, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        # FD=='ND' 인 경우 desc_attr + edge_attr 합쳐서 MLP 통과
        self.use_desc = (args.FD == 'ND')
        if self.use_desc:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.input_dim * 2, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                
                nn.Linear(self.hidden_dim, self.input_dim)
            )

        # graph_type=='full'인 경우 heads 등 자동 조정
        if 'full' in args.graph_type:
            self.heads = min(4, heads)
            self.num_layers = min(2, num_layers)  # Star Graph에서는 2~3 layer가 적절
            self.dropout_rate = dropout_rate * 1.2
        else:
            self.heads = heads
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate

        self.edge_dim = self.input_dim
        self.output_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # 첫 번째 레이어
        first_hidden = self.hidden_dim // self.heads
        self.convs.append(
            GATv2Conv(
                self.input_dim,
                first_hidden,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=True,
                dropout=self.dropout_rate  # GAT 내부에서도 Dropout 적용
            )
        )
        self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
        self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 중간 레이어
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    first_hidden * self.heads,
                    first_hidden,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    concat=True,
                    dropout=self.dropout_rate
                )
            )
            self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
            self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 마지막 레이어
        self.convs.append(
            GATv2Conv(
                first_hidden * self.heads,
                hidden_dim,
                heads=1,  # 최종 예측에서는 single head 사용
                edge_dim=self.edge_dim,
                concat=False,
                dropout=self.dropout_rate
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Dropout, Skip proj 세팅
        self.dropout = nn.Dropout(self.dropout_rate)
        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
            elif i == self.num_layers - 1:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
            else:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # ND이면 desc_attr 활용
        if self.use_desc:
            desc_attr = data.desc_attr
            new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
        else:
            new_edge_attr = edge_attr

        layer_outputs = [x]

        # 공통 레이어 순회
        for i in range(self.num_layers):
            current = x

            x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
            x = self.bns[i](x)
            x = self.layer_norms[i](x)

            # Residual Skip Connection 추가
            if i > 0:
                skip = self.skip_projs[i](current)
                x = x + skip

            x = F.elu(x)
            x = self.dropout(x)

            layer_outputs.append(x)

        return global_mean_pool(x, data.batch)

# class GAT_edge_3(nn.Module):
#     def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
#         super(GAT_edge_3, self).__init__()
#         self.args = args
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.hidden_dim
#         self.edge_dim = args.input_dim


#         self.edge_mlp = nn.Sequential(
#                     nn.Linear(self.input_dim * 2, self.input_dim),
#                     nn.LayerNorm(self.input_dim),
#                     nn.ReLU(),
#                     nn.Dropout(dropout_rate),
                    
#                     nn.Linear(self.input_dim, self.hidden_dim),
#                     nn.LayerNorm(self.hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(self.hidden_dim, self.hidden_dim // 2),
#                     nn.LayerNorm(self.hidden_dim // 2),
#                     nn.ReLU(),
#                     nn.Dropout(dropout_rate),
                    
#                     nn.Linear(self.hidden_dim // 2, self.hidden_dim),
#                     nn.LayerNorm(self.hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(self.hidden_dim, self.input_dim)
#                 )
        
#         if 'full' in args.graph_type:
#             self.heads = min(4, heads)
#             self.num_layers = min(3, num_layers)
#             self.dropout_rate = dropout_rate * 1.5
#             self.edge_dropout_rate = 0.3
#         else:
#             self.heads = heads
#             self.num_layers = num_layers
#             self.dropout_rate = dropout_rate
#             self.edge_dropout_rate = 0.1

#         self.output_dim = hidden_dim
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.layer_norms = nn.ModuleList()

#         # Edge importance projection
#         self.edge_importance = nn.Sequential(
#             nn.Linear(self.input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#         first_hidden = self.hidden_dim // self.heads
#         self.convs.append(GATConv(
#             self.input_dim,
#             first_hidden,
#             heads = self.heads, 
#             dropout = self.dropout_rate,
#             add_self_loops = True,
#             edge_dim = self.edge_dim
#         ))
#         self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#         self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

#         for _ in range(self.num_layers - 2):
#             self.convs.append(GATConv(
#                 first_hidden * self.heads,
#                 first_hidden,
#                 heads = self.heads,
#                 dropout = self.dropout_rate,
#                 add_self_loops = True,
#                 edge_dim = self.edge_dim
#             ))
#             self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#             self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

#         # Output layer
#         self.convs.append(GATConv(
#             first_hidden * self.heads,
#             hidden_dim,
#             heads = 1,
#             concat = False,
#             dropout = self.dropout_rate,
#             add_self_loops = True,
#             edge_dim = self.edge_dim
#         ))
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
#         self.layer_norms.append(nn.LayerNorm(hidden_dim))

#         self.dropout = nn.Dropout(self.dropout_rate)

#         # Skip connection을 위한 projection
#         self.skip_projs = nn.ModuleList()
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
#             elif i == self.num_layers - 1:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
#             else:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

#     def _edge_dropout(self, edge_index, edge_attr):
#         if self.training:
#             importance_scores = self.edge_importance(edge_attr).squeeze()
#             dropout_probs = torch.sigmoid(-importance_scores) * self.edge_dropout_rate
#             mask = torch.rand_like(dropout_probs) > dropout_probs
#             return edge_index[:, mask], edge_attr[mask]
#         return edge_index, edge_attr

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         if 'full' in self.args.graph_type:
            
#             edge_index, edge_attr = self._edge_dropout(edge_index, edge_attr)

#         layer_outputs = [x] 
    
#         if self.args.FD =='ND':
#             desc_attr = data.desc_attr
#             new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim= -1))
#             for i in range(self.num_layers):
#                 current = x
#                 x = self.convs[i](x, edge_index, edge_attr = new_edge_attr)
#                 x = self.bns[i](x)
#                 x = self.layer_norms[i](x) 
#                 if i > 0:
#                     skip = self.skip_projs[i](current)
#                     x = x + skip
            
#             x = F.elu(x)
#             x = self.dropout(x)
#             layer_outputs.append(x)
#             return global_mean_pool(x, data.batch)

#         for i in range(self.num_layers):
#             current = x
#             x = self.convs[i](x, edge_index, edge_attr = edge_attr)
#             x = self.bns[i](x)
#             x = self.layer_norms[i](x) 
#         if i > 0:
#             skip = self.skip_projs[i](current)
#             x = x + skip
        
#             x = F.elu(x)
#             x = self.dropout(x)
#             layer_outputs.append(x)
#         return global_mean_pool(x, data.batch)

class CenterAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_type='CA_m'):
        super().__init__()
        self.attention_type = attention_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)
        self.scale = self.hidden_dim ** -0.5
    def forward(self, x, edge_index, Z):
        # center와 연결된 leaf node 찾기
        center_node = x[0].unsqueeze(0)
        center_edges = (edge_index[0] == 0)
        
        
        
        leaf_Z = Z[center_edges]  # [num_leaves, feature_dim]
        leaf_Z = torch.cat([center_node, leaf_Z], dim=0)
        
        attention_scores = torch.matmul(self.query(leaf_Z), self.key(leaf_Z).transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        center_repr = torch.matmul(attention_weights, self.value(leaf_Z))[0]
        return center_repr

class GAT_edge_4(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
        super(GAT_edge_4, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.set2set = Set2Set(self.hidden_dim, processing_steps=3)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        if self.args.center_type in ['CA_m', 'CA_f']:
            self.center_attention = CenterAttention(input_dim, hidden_dim, args.center_type)
        self.use_desc = (args.FD == 'ND')
        if self.use_desc:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.input_dim * 2, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),

                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                
                nn.Linear(self.hidden_dim, self.input_dim)
            )
        if 'full' in args.graph_type:
            self.heads = min(4, heads)
            self.num_layers = min(3, num_layers)
            self.dropout_rate = dropout_rate * 1.5
        else:
            self.heads = heads
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate

        self.edge_dim = self.input_dim

        # GAT 레이어들
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        first_hidden = self.hidden_dim // self.heads
        # 첫 번째
        self.convs.append(
            GATConv(
                in_channels=self.input_dim,
                out_channels=first_hidden,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
        self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 중간 레이어
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=first_hidden * self.heads,
                    out_channels=first_hidden,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
            self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 마지막 레이어
        self.convs.append(
            GATConv(
                in_channels=first_hidden * self.heads,
                out_channels=hidden_dim,
                heads=1,
                edge_dim=self.edge_dim,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Dropout, Skip proj
        self.dropout = nn.Dropout(self.dropout_rate)
        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
            elif i == self.num_layers - 1:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
            else:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if self.use_desc:
            desc_attr = data.desc_attr
            new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
        else:
            new_edge_attr = edge_attr

        # Center Attention을 먼저 수행 (x가 아직 input_dim 차원일 때)
        if hasattr(self, 'center_attention'):
            Z = torch.mul(new_edge_attr, x[edge_index[1]])
            center_repr = self.center_attention(x, edge_index, Z)
            x[0] = center_repr

        for i in range(self.num_layers):
            current = x
            x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
            x = self.bns[i](x)
            x = self.layer_norms[i](x)

            # Skip connection
            skip = self.skip_projs[i](current)
            x = x + skip

            x = F.elu(x)
            x = self.dropout(x)

        # (C) Pooling
        #pdb.set_trace()
        out = self.set2set(x, data.batch)
        out = self.proj(out)
        return out

class GAT_edge_5(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
        super(GAT_edge_5, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.set2set = Set2Set(self.hidden_dim, processing_steps=3)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        if self.args.center_type in ['CA_m', 'CA_f']:
            self.center_attention = CenterAttention(input_dim, hidden_dim, args.center_type)
        self.use_desc = (args.FD == 'ND')
        if self.use_desc:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.input_dim * 2, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),

                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                
                nn.Linear(self.hidden_dim, self.input_dim)
            )
        if 'full' in args.graph_type:
            self.heads = min(4, heads)
            self.num_layers = min(3, num_layers)
            self.dropout_rate = dropout_rate * 1.5
        else:
            self.heads = heads
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate

        self.edge_dim = self.input_dim

        # GAT 레이어들
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        first_hidden = self.hidden_dim // self.heads
        # 첫 번째
        self.convs.append(
            GATv2Conv(
                in_channels=self.input_dim,
                out_channels=first_hidden,
                heads=self.heads,
                edge_dim=self.edge_dim,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
        self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 중간 레이어
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=first_hidden * self.heads,
                    out_channels=first_hidden,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
            self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

        # 마지막 레이어
        self.convs.append(
            GATv2Conv(
                in_channels=first_hidden * self.heads,
                out_channels=hidden_dim,
                heads=1,
                edge_dim=self.edge_dim,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Dropout, Skip proj
        self.dropout = nn.Dropout(self.dropout_rate)
        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
            elif i == self.num_layers - 1:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
            else:
                self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if self.use_desc:
            desc_attr = data.desc_attr
            new_edge_attr = self.edge_mlp(torch.cat([edge_attr, desc_attr], dim=-1))
        else:
            new_edge_attr = edge_attr

        # Center Attention을 먼저 수행 (x가 아직 input_dim 차원일 때)
        if hasattr(self, 'center_attention'):
            Z = torch.mul(new_edge_attr, x[edge_index[1]])
            center_repr = self.center_attention(x, edge_index, Z)
            x[0] = center_repr

        for i in range(self.num_layers):
            current = x
            x = self.convs[i](x, edge_index, edge_attr=new_edge_attr)
            x = self.bns[i](x)
            x = self.layer_norms[i](x)

            # Skip connection
            skip = self.skip_projs[i](current)
            x = x + skip

            x = F.elu(x)
            x = self.dropout(x)

        # (C) Pooling
        out = self.set2set(x, data.batch)
        out = self.proj(out)
        return out

class Model(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4):
        super(Model, self).__init__()
        self.args = args
        heads = args.heads
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        # GNN 모델 초기화
        if args.model_type == 'GAT_edge':
            self.gnn = GAT_edge(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, heads)
            gnn_out_dim = hidden_dim //heads  # GAT_edge의 출력 차원
        elif args.model_type == 'GAT_edge_2':
            self.gnn = GAT_edge_2(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, heads)
            gnn_out_dim = hidden_dim  # GAT_edge_2의 출력 차원
        elif args.model_type == 'GAT_edge_3':
            self.gnn = GAT_edge_3(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, heads)
            gnn_out_dim = hidden_dim  # GAT_edge_3의 출력 차원
        elif args.model_type == 'GAT_edge_4':
            self.gnn = GAT_edge_4(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, heads)
            gnn_out_dim = hidden_dim  # GAT_edge_4의 출력 차원
        elif args.model_type == 'GAT_edge_5':
            self.gnn = GAT_edge_5(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, heads)
            gnn_out_dim = hidden_dim  # GAT_edge_5의 출력 차원
        else:  # NORM_GNN
            self.gnn = NORM_GNN(args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
            gnn_out_dim = hidden_dim  # NORM_GNN의 출력 차원
        
        # MLP for final prediction
        self.mlp1 = nn.Sequential(
            nn.Linear(gnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, 1)
        )
        

    def forward(self, data):
        pred = self.predict(data)
        target = data.y.view(-1,1).float()

        loss = self.criterion(pred, target)
        return loss
    def predict(self, data):
        graph_repr = self.gnn(data)
        pred = self.mlp1(graph_repr)
        return pred





'''
    Old ones
'''
# class GAT_edge_4(nn.Module):
#     def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
#         super(GAT_edge_4, self).__init__()
#         self.args = args
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.hidden_dim
#         self.edge_dim = args.input_dim

#         if 'full' in args.graph_type:
#             self.heads = min(4, heads)
#             self.num_layers = min(3, num_layers)
#             self.dropout_rate = dropout_rate * 1.5
#             self.edge_dropout_rate = 0.3
#         else:
#             self.heads = heads
#             self.num_layers = num_layers
#             self.dropout_rate = dropout_rate
#             self.edge_dropout_rate = 0.1

#         self.edge_dim = args.input_dim
#         self.output_dim = hidden_dim 

#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.layer_norms = nn.ModuleList()

#         first_hidden = self.hidden_dim // self.heads
#         self.convs.append(GATConv(
#             self.input_dim,
#             first_hidden,
#             heads = self.heads,
#             edge_dim = self.edge_dim,
#             concat = True
#         ))
#         self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#         self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))    

#         for _ in range(self.num_layers - 2):
#             self.convs.append(GATConv(
#                 first_hidden * self.heads,
#                 first_hidden,
#                 heads = self.heads,
#                 edge_dim = self.edge_dim,
#                 concat = True
#             ))
#             self.bns.append(nn.BatchNorm1d(first_hidden * self.heads))
#             self.layer_norms.append(nn.LayerNorm(first_hidden * self.heads))

#         self.convs.append(GATConv(
#             first_hidden * self.heads,
#             hidden_dim,
#             heads = 1,
#             edge_dim = self.edge_dim,
#             concat = False
#         ))
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
#         self.layer_norms.append(nn.LayerNorm(hidden_dim))

#         self.dropout = nn.Dropout(self.dropout_rate)

#         # Skip connection을 위한 projection
#         self.skip_projs = nn.ModuleList()
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.skip_projs.append(nn.Linear(self.input_dim, first_hidden * self.heads))
#             elif i == self.num_layers - 1:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, hidden_dim))
#             else:
#                 self.skip_projs.append(nn.Linear(first_hidden * self.heads, first_hidden * self.heads))

#     def forward(self, data):
#         if self.args.graph_type in ['full_one', 'full_mean']:
#             x, edge_index, edge_attr, desc_attr, center_attr, leaf_attr = data.x, data.edge_index, data.edge_attr, data.desc_attr, data.center_attr, data.leaf_attr
#         elif self.args.graph_type == 'star':
#             x, edge_index, edge_attr, desc_attr = data.x, data.edge_index, data.edge_attr, data.desc_attr
        
#         x = x.mean(dim=1)
#         edge_attr = edge_attr.mean(dim=1)
#         desc_attr = desc_attr.mean(dim=1)

#         layer_outputs = []
#         layer_outputs.append(x)
        
#         for i in range(self.num_layers):
#             current = x
#             x = self.convs[i](x, edge_index, edge_attr = edge_attr)
#             x = self.bns[i](x)
#             x = self.layer_norms[i](x)
#             if i > 0:
#                 skip = self.skip_projs[i](current)
#                 x = x + skip
            
#             x = F.elu(x)
#             x = self.dropout(x)
#             layer_outputs.append(x)

#         return global_mean_pool(x, data.batch)