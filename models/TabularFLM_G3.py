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

logger = logging.getLogger(__name__)

class AdaptiveGraphAttention(nn.Module):
    def __init__(
        self,
        input_dim : int,
        n_heads : int, 
        dropout : float = 0.1,
        threshold : float = 0.5
    ):
        super().__init__()
        assert input_dim % n_heads == 0 
        self.n_heads = n_heads 
        self.head_dim = input_dim // n_heads 
        self.input_dim = input_dim
        
        self.frozen = False  # 그래프 구조 고정 여부
        
        self.edge_update = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.threshold = threshold
        self.topology_bias = nn.Parameter(torch.zeros(1))
        self.attn_dropout = nn.Dropout(dropout)

        self.global_topology_proj = nn.Linear(input_dim, input_dim)
        self.adaptive_weight_proj = nn.Linear(input_dim, input_dim)

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.attn_proj = nn.Linear(self.head_dim * 3, 1)   # [q | k | edge_attr] -> attention score
        
        
        self.out_proj = nn.Linear(input_dim, input_dim)
        nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
    
    def _no_self_interaction(self, adjacency_matrix):
        batch_size, seq_len, _ = adjacency_matrix.shape
        diag_mask = 1.0 - torch.eye(seq_len, device=adjacency_matrix.device).unsqueeze(0)
        return adjacency_matrix * diag_mask

    def _debug_fr_graph(self, global_topology, global_topology_A, sample_sim, adjacency_matrix, adjacency):
        """
        FR-Graph 생성 과정의 디버깅 정보를 출력하는 함수
        """
        print(f"global_topology 범위: 최소={global_topology.min().item():.4f}, 최대={global_topology.max().item():.4f}")
        print(f"global_topology 평균: {global_topology.mean().item():.4f}")
        
        print(f"sample_sim 범위: 최소={sample_sim.min().item():.4f}, 최대={sample_sim.max().item():.4f}")
        print(f"sample_sim 평균: {sample_sim.mean().item():.4f}")
        
        print(f"adjacency_matrix 범위: 최소={adjacency_matrix.min().item():.4f}, 최대={adjacency_matrix.max().item():.4f}")
        print(f"adjacency_matrix 평균: {adjacency_matrix.mean().item():.4f}")
        print(f"임계값: {self.threshold}")
        
        print(f"adjacency 범위: 최소={adjacency.min().item():.4f}, 최대={adjacency.max().item():.4f}")
        print(f"adjacency 평균: {adjacency.mean().item():.4f}")
        
        # 임계값 이상 엣지 비율 계산
        threshold_ratio = (global_topology_A > 0).float().mean().item()
        print(f"임계값 이상 엣지 비율: {threshold_ratio:.4f}")
        
        # 범위별 분포 분석
        adj_flat = adjacency.flatten()
        min_val = adj_flat.min().item()
        max_val = adj_flat.max().item()
        num_bins = 10
        step = (max_val - min_val) / num_bins
        
        for i in range(num_bins):
            lower = min_val + i * step
            upper = min_val + (i + 1) * step
            count = ((adj_flat >= lower) & (adj_flat < upper)).sum().item()
            print(f"범위 [{lower:.4f}, {upper:.4f}): {int(count)} 개")

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        '''
            1. Glboal Topology
        '''
        desc_embeddings = self.global_topology_proj(desc_embeddings)
        desc_embeddings_ = desc_embeddings / desc_embeddings.norm(dim=-1, keepdim=True)
        global_sim = torch.matmul(desc_embeddings_, desc_embeddings_.transpose(-1, -2))
        global_topology = torch.sigmoid(global_sim + self.topology_bias)

        if not self.frozen:
            global_topology_A = (global_topology > self.threshold).float() - global_topology.detach() + global_topology
        else:
            global_topology_A = (global_topology > self.threshold).float()

        '''
            2. Sample-wise Weight
        '''
        var_embeddings = name_value_embeddings[:, 1:, :]
        var_embeddings_ = var_embeddings / var_embeddings.norm(dim=-1, keepdim=True)
        sample_sim = torch.matmul(var_embeddings_, var_embeddings_.transpose(-1, -2))

        global_topology_A = self._no_self_interaction(global_topology_A)
        G = global_topology_A * sample_sim 
        adjacency = torch.softmax(G, dim=-1)
        
        # 디버깅: 필요시 주석 해제
        # self._debug_fr_graph(global_topology, global_topology_A, sample_sim, adjacency_matrix, adjacency)
        
        '''
            4. Edge Attributes & Adjacency 
        '''
        # 1. 변수 노드 간의 edge_attr
        node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)  # [batch, seq_len, seq_len, dim*2]

        # 2. CLS-변수 노드 간의 edge_attr
        cls_edge_attr = torch.cat([
            desc_embeddings.unsqueeze(1),  # CLS->변수: 변수의 description
            desc_embeddings.unsqueeze(1)   # 변수->CLS: 동일한 description
        ], dim=-1)

        # 3. Edge attribute 합치기
        edge_dim = var_edge_attr.size(-1)
        edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=var_edge_attr.device)
        edge_attr[:, 1:, 1:] = var_edge_attr  # 변수 노드 간
        edge_attr[:, 0, 1:] = cls_edge_attr.squeeze(1)  # CLS->변수
        edge_attr[:, 1:, 0] = cls_edge_attr.squeeze(1)  # 변수->CLS

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=adjacency.device)
        new_adjacency[:, 1:, 1:] = adjacency 
        new_adjacency[:, 0, 1:] = 1.0 # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0 # Var -> CLS 
        '''
            5. Attention
        '''
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)


        edge_attr = self.edge_update(edge_attr)
        edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
        edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
        edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)

        q_expanded = q.unsqueeze(3)
        k_expanded = k.unsqueeze(2)
        qke_expanded = torch.cat([
            q_expanded.expand(-1,-1,-1, new_seq, -1),
            k_expanded.expand(-1,-1, new_seq, -1, -1),
            edge_attr
        ], dim = -1)

        attn_weights = self.attn_proj(qke_expanded).squeeze(-1)

        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)
        return output, attn_weights




class GMM(nn.Module):
    """
        k : The number of Prototypes 
        stage_num : EM step counter
        momentum : The momentum of the prototype update 
    """
    def __init__(
            self,
            args, 
            num_prototypes : int, 
            stage_num  : int = 5, 
            momentum : float = 0.9, 
            beta : float = 1.0,
            lambd : float = 0.1, 
            eps : float = 1e-6,
    ):
        super(GMM, self).__init__()
        self.num_prototypes = num_prototypes
        self.input_dim = args.input_dim
        self.stage_num = stage_num
        self.momentum = momentum 
        self.beta = beta
        self.lambd = lambd
        self.eps = eps

        prototypes = torch.Tensor(num_prototypes, self.input_dim)
        nn.init.kaiming_uniform_(prototypes, a = math.sqrt(5))
        prototypes = prototypes / (1e-6 + prototypes.norm(dim=1, keepdim=True))
        self.register_buffer("prototypes", prototypes)
        
    def forward(self, cls : torch.Tensor, is_train = True):
        assert cls.dim() == 2
        b, d = cls.shape
        device = cls.device
        
        local_proto = self.prototypes.to(device).expand(b, -1, -1).contiguous()
        _cls = cls 
        with torch.no_grad():
            for _ in range(self.stage_num):
                latent = torch.einsum("bd, bkd->bk", cls, local_proto)
                r = F.softmax(latent, dim = 1)
                new_proto = torch.mm(r.t(), _cls)
                new_proto = new_proto / (new_proto.norm(dim=1, keepdim=True) + self.eps)
                local_proto = new_proto.unsqueeze(0).expand(b, -1, -1).contiguous()

        dot = torch.einsum("bd,bkd->bk", cls, local_proto)
        r = F.softmax(dot, dim=1)
        
        z_recon = torch.mm(r, new_proto)
        z_out = self.beta * z_recon + cls

        if is_train:
            old_proto = self.prototypes.to(device)
            new_proto_for_update = new_proto.unsqueeze(0)
            updated = self.momentum * old_proto + (1-self.momentum)*new_proto_for_update            
            self.prototypes = updated.detach().cpu()
            
        return r, z_out


class GMM2(nn.Module):
    def __init__(
            self,
            args, 
            num_prototypes : int, 
            stage_num  : int = 5, 
            momentum : float = 0.9, 
            beta : float = 1.0,
            lambd : float = 0.1, 
            eps : float = 1e-6,
    ):
        super(GMM2, self).__init__()
        self.num_prototypes = num_prototypes
        self.input_dim = args.input_dim
        self.stage_num = stage_num
        self.momentum = momentum 
        self.beta = beta
        self.lambd = lambd
        self.eps = eps

        prototypes = torch.Tensor(num_prototypes, self.input_dim)
        nn.init.kaiming_uniform_(prototypes, a = math.sqrt(5))
        prototypes = prototypes / (1e-6 + prototypes.norm(dim=1, keepdim=True))
        
        self.prototypes = nn.Parameter(prototypes.clone())
        self.register_buffer("running_prototypes", prototypes.clone())
        
    def _normalize_prototypes(self):
        with torch.no_grad():
            norm = self.prototypes.norm(dim=1, keepdim=True) + self.eps
            self.prototypes.data = self.prototypes.data / norm
            
    def forward(self, cls : torch.Tensor, is_train = True):
        assert cls.dim() == 2
        b, d = cls.shape
        device = cls.device
        
        self._normalize_prototypes()
        local_proto = self.prototypes.unsqueeze(0).expand(b, -1, -1).contiguous()
        _cls = cls 
        
        for _ in range(self.stage_num):
            latent = torch.einsum("bd, bkd->bk", cls, local_proto)
            latent = latent / self.lambd
            r = F.softmax(latent, dim = 1)
            
            new_proto = torch.mm(r.t(), _cls)
            new_proto = new_proto / (new_proto.norm(dim=1, keepdim=True) + self.eps)
            local_proto = new_proto.unsqueeze(0).expand(b, -1, -1).contiguous()

        dot = torch.einsum("bd,bkd->bk", cls, local_proto)
        dot = dot / self.lambd
        r = F.softmax(dot, dim=1)

        z_recon = torch.mm(r, new_proto)
        z_out = self.beta * z_recon + cls

        recon_loss = F.mse_loss(z_recon, cls)
        entropy_loss = - (r * torch.log(r + self.eps)).sum(dim = 1).mean()
        proto_similarity = torch.mm(new_proto, new_proto.t())
        eye = torch.eye(self.num_prototypes, device = device)
        diversity_loss = torch.mean(torch.abs(proto_similarity * (1 - eye)))

        if is_train:
            with torch.no_grad():
                old_proto = self.running_prototypes.to(device)
                updated = self.momentum * old_proto + (1 - self.momentum) * new_proto 
                updated = updated / (updated.norm(dim = 1, keepdim = True) + self.eps)
                self.running_prototypes.copy_(updated.detach())
                
                update_ratio = 0.01 
                self.prototypes.data = (1 - update_ratio) * self.prototypes.data + update_ratio * new_proto.detach()

        return r, z_out, {
            'recon_loss': recon_loss,
            'entropy_loss': entropy_loss,
            'diversity_loss': diversity_loss
        }


class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.threshold = args.threshold  # 엣지 프루닝 임계값 (명령줄 인자에서 가져옴)
        self.frozen = args.frozen  # 그래프 구조 고정 여부 (초기에는 False로 설정)
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
                # GMM 관련 속성 추가
        self.use_gmm = args.use_gmm 
        self.use_gmm2 = args.use_gmm2
        self.num_prototypes = args.num_prototypes
        self.stage_num = args.gmm_stage_num
        self.momentum = args.gmm_momentum 
        self.beta = args.gmm_beta 
        self.lambd = args.gmm_lambda
        self.eps = args.gmm_eps
        

        if self.use_gmm:
            self.gmm = GMM(self.args, self.num_prototypes, self.stage_num, self.momentum, self.beta, self.lambd, self.eps)
        elif self.use_gmm2:
            self.gmm = GMM2(self.args, self.num_prototypes, self.stage_num, self.momentum, self.beta, self.lambd, self.eps)
        
        # GMM 모듈 초
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))

        '''
            MLP(CONCAT[Name embedding, Value embedding])
            - In order to infuse the information of name and value simultaneously. 
        '''
        self.sample_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        self.layers = nn.ModuleList([
            AdaptiveGraphAttention(
                input_dim = self.input_dim, 
                n_heads = args.n_heads,
                dropout = args.dropout_rate,
                threshold = self.threshold
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
        target = y.to(self.device).view(-1,1).float()
        
        # GMM2 사용 시 predict에서 손실도 함께 받음
        if self.use_gmm2 and self.training:
            pred, gmm_losses = self.predict(batch)
            loss = self.criterion(pred, target)
            
            # GMM 손실 계산
            gmm_loss_weight = 0.1
            gmm_loss = (
                #gmm_losses['recon_loss'] + 
                #0.1 * gmm_losses['entropy_loss'] + 
                gmm_losses['diversity_loss']
            )
            return loss + gmm_loss_weight * gmm_loss
    
        # 일반적인 경우
        else:
            pred = self.predict(batch)
            loss = self.criterion(pred, target)
            return loss
    
    def predict(self, batch):
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        desc_embeddings = [] 
        name_value_embeddings = [] 
        

        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            cat_name_value_embeddings = batch['cat_name_value_embeddings'].to(self.device).squeeze(-2)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
            
            name_value_embeddings.append(cat_name_value_embeddings)
            desc_embeddings.append(cat_desc_embeddings)
            
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
            name_value_embeddings.append(num_prompt_embeddings)
            desc_embeddings.append(num_desc_embeddings)
            
        
        if not desc_embeddings or not name_value_embeddings:
            raise ValueError("No categorical or numerical features found in batch")

        desc_embeddings = torch.cat(desc_embeddings, dim = 1)
        name_value_embeddings = torch.cat(name_value_embeddings, dim = 1)
    
        '''
            1. [CLS] Token
        '''
        cls_token = self.cls.expand(name_value_embeddings.size(0), -1, -1)
        name_value_embeddings = torch.cat([cls_token, name_value_embeddings], dim=1)    

        x = name_value_embeddings 
        for i, layer in enumerate(self.layers):
            norm_x = self.layer_norms[i](x)
            attn_output, attn_weights = layer(desc_embeddings, norm_x)
            x = x + attn_output
        pred = x[:, 0, :]
        if self.use_gmm:
            r, z_out = self.gmm(pred, is_train = True)
            pred = self.predictor(z_out)
            return pred
        elif self.use_gmm2:
            r, z_out, gmm_losses = self.gmm(pred, is_train = True)
            pred = self.predictor(z_out)
            if self.training:
                return pred, gmm_losses
            else:
                return pred
        else:
            pred = self.predictor(pred)
            z_out = pred

        
        return pred

    def froze_topology(self):

        self.frozen = True
        logger.info("Graph topology frozen. Continuing with fixed structure.")