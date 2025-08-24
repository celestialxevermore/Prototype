import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from torch import Tensor 
from models.coordinate import CoordinatorMLP
import math 
import torch.nn.init as nn_init
import logging 
import os


logger = logging.getLogger(__name__)


class AdaptiveGraphAttention(nn.Module):
    def __init__(
        self,
        args,
        input_dim : int,
        hidden_dim : int,
        n_heads : int, 
        dropout : float = 0.1,
        threshold : float = 0.5, 
        connectivity_mode : str  = 'baseline'
    ):
        super().__init__()
        assert input_dim % n_heads == 0 
        self.args = args
        self.n_heads = n_heads 
        self.head_dim = input_dim // n_heads 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.threshold = threshold
        self.attn_dropout = nn.Dropout(dropout)
        self.connectivity_mode = connectivity_mode
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        if self.args.attn_type in ['gat_v1', 'gat_v2']:
            if self.args.edge_type == 'normal':
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
            elif self.args.edge_type == 'mlp':
                self.attn_proj = nn.Linear(self.head_dim * 3, 1)
                self.edge_update = nn.Sequential(
                    nn.Linear(input_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
            elif self.args.edge_type == 'no_use':
                self.attn_proj = nn.Linear(self.head_dim * 2, 1)
        elif self.args.attn_type == 'gate':
            if self.args.edge_type == 'normal':
                self.gate_proj = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
            elif self.args.edge_type == 'mlp':
                self.gate_proj = nn.Linear(self.head_dim * 3, 1)
                self.content_proj = nn.Linear(self.head_dim * 3, 1)
                self.edge_update = nn.Sequential(
                    nn.Linear(input_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
            elif self.args.edge_type == 'no_use':
                self.gate_proj = nn.Linear(self.head_dim * 2, 1)
                self.content_proj = nn.Linear(self.head_dim * 2, 1)
        
        self.out_proj = nn.Linear(input_dim, input_dim)
        nn_init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
        
        if hasattr(self, 'attn_proj'):
            nn_init.xavier_uniform_(self.attn_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'gate_proj'):
            nn_init.xavier_uniform_(self.gate_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'content_proj'):
            nn_init.xavier_uniform_(self.content_proj.weight, gain=1 / math.sqrt(2))
        if hasattr(self, 'edge_update'):
            for module in self.edge_update:
                if isinstance(module, nn.Linear):
                    nn_init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))

    def _no_self_interaction(self, adjacency_matrix):
        batch_size, seq_len, _ = adjacency_matrix.shape
        diag_mask = 1.0 - torch.eye(seq_len, device=adjacency_matrix.device).unsqueeze(0)
        return adjacency_matrix * diag_mask

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        self.adjacency = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        if self.args.no_self_loop:
            self.adjacency = self._no_self_interaction(self.adjacency)

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency 
        new_adjacency[:, 0, 1:] = 1.0 # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0 # Var -> CLS
        
        self.new_adjacency = new_adjacency
        
        '''
            Attention
        '''
        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention 계산 방식 선택
        if self.args.attn_type == 'gat_v1':
            # GAT-style attention (concat + MLP)
            if self.args.edge_type == "normal":
                # Case 1: GAT with edge attributes - MLP([q | k | edge_attr])
                target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                cls_edge_attr = desc_embeddings

                edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = target_desc  # 변수 노드 간
                edge_attr[:, 0, 1:] = cls_edge_attr  # CLS->변수
                
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
            elif self.args.edge_type == "mlp":
                node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
                node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
                cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
                edge_dim = var_edge_attr.size(-1)
                edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = var_edge_attr  # 변수 노드 간
                edge_attr[:, 0, 1:] = cls_edge_attr   # CLS->변수
                edge_attr[:, 1:, 0] = cls_edge_attr   # 변수->CLS
                
                # 4. 차원 맞추기 (edge_update 필요)
                edge_attr = self.edge_update(edge_attr)  # [batch, new_seq, new_seq, n_heads * head_dim]
                edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
                edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
                edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)
                
                # 5. Attention 계산 (기존과 동일)
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qke_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1),
                    edge_attr
                ], dim = -1)
                
                attn_weights = self.attn_proj(qke_expanded).squeeze(-1)
            elif self.args.edge_type == 'no_use':
                # Case 2: GAT without edge attributes - MLP([q | k])
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                
                attn_weights = self.attn_proj(qk_expanded).squeeze(-1)
        elif self.args.attn_type == 'gat_v2':
           # GAT-v2 style attention (LeakyReLU before attention)
           if self.args.edge_type == "normal":
               target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
               cls_edge_attr = desc_embeddings

               edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
               edge_attr[:, 1:, 1:] = target_desc
               edge_attr[:, 0, 1:] = cls_edge_attr
               
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
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qke_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
           elif self.args.edge_type == "mlp":
               node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
               node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
               var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
               cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
               edge_dim = var_edge_attr.size(-1)
               edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
               edge_attr[:, 1:, 1:] = var_edge_attr
               edge_attr[:, 0, 1:] = cls_edge_attr
               edge_attr[:, 1:, 0] = cls_edge_attr
               
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
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qke_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
           elif self.args.edge_type == 'no_use':
               q_expanded = q.unsqueeze(3)
               k_expanded = k.unsqueeze(2)
               qk_expanded = torch.cat([
                   q_expanded.expand(-1,-1,-1, new_seq, -1),
                   k_expanded.expand(-1,-1, new_seq, -1, -1)
               ], dim = -1)
               
               # GAT-v2: LeakyReLU before attention
               activated_features = F.leaky_relu(qk_expanded)
               attn_weights = self.attn_proj(activated_features).squeeze(-1)
        elif self.args.attn_type == "gate":
            # Gate attention mechanism
            if self.args.edge_type == "normal":
                # Edge attributes 준비
                target_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                cls_edge_attr = desc_embeddings

                edge_attr = torch.zeros(batch_size, new_seq, new_seq, desc_embeddings.size(-1), device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = target_desc
                edge_attr[:, 0, 1:] = cls_edge_attr
                
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
                
                # Gate mechanism: σ(W_g * [q|k|e]) * tanh(W_c * [q|k|e])
                gate_values = torch.sigmoid(self.gate_proj(qke_expanded))  # [batch, n_heads, seq, seq, 1]
                content_values = torch.tanh(self.content_proj(qke_expanded))  # [batch, n_heads, seq, seq, 1]
                attn_weights = (gate_values * content_values).squeeze(-1)  # [batch, n_heads, seq, seq]
                
            elif self.args.edge_type == "mlp":
                node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
                node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim = -1)
                cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
                edge_dim = var_edge_attr.size(-1)
                edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
                edge_attr[:, 1:, 1:] = var_edge_attr
                edge_attr[:, 0, 1:] = cls_edge_attr
                edge_attr[:, 1:, 0] = cls_edge_attr
                
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
                
                # Gate mechanism: σ(W_g * [q|k|e]) * tanh(W_c * [q|k|e])
                gate_values = torch.sigmoid(self.gate_proj(qke_expanded))
                content_values = torch.tanh(self.content_proj(qke_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)
                
            elif self.args.edge_type == 'no_use':
                q_expanded = q.unsqueeze(3)
                k_expanded = k.unsqueeze(2)
                qk_expanded = torch.cat([
                    q_expanded.expand(-1,-1,-1, new_seq, -1),
                    k_expanded.expand(-1,-1, new_seq, -1, -1)
                ], dim = -1)
                
                # Gate mechanism: σ(W_g * [q|k]) * tanh(W_c * [q|k])
                gate_values = torch.sigmoid(self.gate_proj(qk_expanded))
                content_values = torch.tanh(self.content_proj(qk_expanded))
                attn_weights = (gate_values * content_values).squeeze(-1)

        elif self.args.attn_type == "att":
            # Case 3: Standard dot-product attention (use_edge_attr는 무시)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            # Default: standard dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Graph structure masking
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim = -1)
        self.attn_weights = self.attn_dropout(attn_weights)

        # Context 계산
        context = torch.matmul(self.attn_weights, v)
        context = context.transpose(1,2).reshape(batch_size, new_seq, self.input_dim)
        output = self.out_proj(context)
        
        return output, attn_weights

class BasisGATLayer(AdaptiveGraphAttention):

    def __init__(self, args, input_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        # 부모 클래스(AdaptiveGraphAttention)의 __init__을 그대로 호출합니다.
        super().__init__(args, input_dim, hidden_dim, n_heads, dropout)
        
        # [핵심 변경점]
        # 헤드별 출력을 독립적으로 사용하기 위해 최종 프로젝션을 제거합니다.
        if hasattr(self, 'out_proj'):
            del self.out_proj

    def forward(self, desc_embeddings, name_value_embeddings):
        batch_size, new_seq, _ = name_value_embeddings.shape
        seq_len = new_seq - 1

        # Adjacency Matrix 설정
        self.adjacency = torch.ones(batch_size, seq_len, seq_len, device=name_value_embeddings.device)
        if self.args.no_self_loop:
            self.adjacency = self._no_self_interaction(self.adjacency)

        new_adjacency = torch.zeros(batch_size, new_seq, new_seq, device=self.adjacency.device)
        new_adjacency[:, 1:, 1:] = self.adjacency
        new_adjacency[:, 0, 1:] = 1.0  # CLS -> Var
        new_adjacency[:, 1:, 0] = 0.0  # Var -> CLS (비대칭 어텐션)
        
        self.new_adjacency = new_adjacency


        q = self.q_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(name_value_embeddings).view(batch_size, new_seq, self.n_heads, self.head_dim).transpose(1, 2)

        #attn_weights = None # Initialize
        q_expanded = q.unsqueeze(3).expand(-1, -1, -1, new_seq, -1)
        k_expanded = k.unsqueeze(2).expand(-1, -1, new_seq, -1, -1)

        node_i_desc = desc_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j_desc = desc_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        var_edge_attr = torch.cat([node_i_desc, node_j_desc], dim=-1)
        cls_edge_attr = torch.cat([desc_embeddings, desc_embeddings], dim=-1)
        edge_dim = var_edge_attr.size(-1)
        edge_attr = torch.zeros(batch_size, new_seq, new_seq, edge_dim, device=desc_embeddings.device)
        edge_attr[:, 1:, 1:] = var_edge_attr
        edge_attr[:, 0, 1:] = cls_edge_attr
        edge_attr[:, 1:, 0] = cls_edge_attr
        
        edge_attr = self.edge_update(edge_attr)
        edge_attr = edge_attr.view(batch_size, new_seq, new_seq, self.n_heads, self.head_dim)
        edge_attr = edge_attr.permute(0, 3, 1, 2, 4)
        edge_attr = edge_attr * new_adjacency.unsqueeze(1).unsqueeze(-1)

        qke_expanded = torch.cat([q_expanded, k_expanded, edge_attr], dim=-1)

        if self.args.attn_type == 'gat_v2':
            activated_features = F.leaky_relu(qke_expanded)
            attn_weights = self.attn_proj(activated_features).squeeze(-1)
        else: # gat_v1
            attn_weights = self.attn_proj(qke_expanded).squeeze(-1)

        
        # 마스킹 및 Softmax
        mask = (new_adjacency.unsqueeze(1) == 0).float() * -1e9
        attn_weights = attn_weights + mask 
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # context shape: [batch, n_heads, new_seq, head_dim]

        # [핵심] 헤드별 결과물을 분리된 상태로 유지하여 반환
        basis_outputs = context.transpose(1, 2)  # shape: [batch, new_seq, n_heads, head_dim]
        return basis_outputs, attn_weights

# ===============================
# Model
# ===============================
class Model(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model, experiment_id, mode):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # config
        self.args = args
        self.threshold = args.threshold
        self.llm_model = llm_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = args.num_layers
        self.dropout_rate = dropout_rate
        self.meta_type = args.meta_type
        self.experiment_id = experiment_id
        self.mode = mode
        self.num_classes = args.num_classes
        self.source_data = args.source_data

        # loss
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()

        # CLS
        self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
        nn.init.uniform_(self.cls, a=-1 / math.sqrt(self.input_dim), b=1 / math.sqrt(self.input_dim))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.attention_counter = 0
        self.attention_save_dir = None

        # shared encoder (num_layers > 1)
        assert self.num_layers > 1, "num_layers must be > 1"
        self.shared_layers = nn.ModuleList([
            AdaptiveGraphAttention(args, self.input_dim, self.hidden_dim, args.n_heads, self.dropout_rate, self.threshold)
            for _ in range(self.num_layers - 1)
        ])
        self.shared_layer_norms = nn.ModuleList([nn.LayerNorm(self.input_dim) for _ in range(self.num_layers - 1)])

        # coordinator + basis
        self.coordinator = CoordinatorMLP(self.input_dim, self.hidden_dim, args.k_basis, self.dropout_rate)
        self.basis_layer = BasisGATLayer(args, self.input_dim, self.hidden_dim, args.k_basis, self.dropout_rate)
        self.basis_layer_norm = nn.LayerNorm(self.input_dim)
        self.expert_predictors = nn.ModuleList([
            nn.Linear(self.input_dim // args.k_basis, self.output_dim) for _ in range(args.k_basis)
        ])
        self._init_weights()

    # ---- FREEZE: target에서 명시적으로 호출 ----
    def freeze(self):
        # 모두 동결
        for p in self.parameters():
            p.requires_grad = False

        # LayerNorm 전역 활성
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        # 학습 파트: coordinator, basis_layer, basis_layer_norm, expert heads
        for p in self.coordinator.parameters():
            p.requires_grad = True
        for p in self.basis_layer.parameters():
            p.requires_grad = True
        for p in self.basis_layer_norm.parameters():
            p.requires_grad = True
        for head in self.expert_predictors:
            for p in head.parameters():
                p.requires_grad = True

        # CLS, shared_layers 등 백본은 그대로 동결
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[freeze()] trainable={trainable}/{total} ({trainable/total*100:.2f}%)")

    # ---- init helpers ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    # ---- viz utilities ----
    def set_attention_save_dir(self, experiment_id, mode):
        base_viz_dir = f"/storage/personal/eungyeop/experiments/visualization/{self.args.llm_model}/{self.args.source_data}/{mode}/{experiment_id}"
        self.attention_save_dir = os.path.join(base_viz_dir, 'attention_maps')
        os.makedirs(self.attention_save_dir, exist_ok=True)

    def extract_feature_names(self, batch):
        names = []
        if 'cat_desc_texts' in batch:
            for t in batch['cat_desc_texts']:
                names.append(str(t[0] if isinstance(t, tuple) else t).split(",")[0].split("'")[0])
        if 'num_desc_texts' in batch:
            for t in batch['num_desc_texts']:
                names.append(str(t[0] if isinstance(t, tuple) else t).split(",")[0].split("'")[0])
        uniq = []
        seen = set()
        for n in names:
            if n not in seen:
                seen.add(n); uniq.append(n)
        return uniq

    def save_attention_maps_to_file(self, attention_weights, batch, labels=None, sample_ids=None):
        if self.attention_save_dir is None:
            return
        node_names = ["CLS"] + self.extract_feature_names(batch)
        for layer_idx, layer_attention in enumerate(attention_weights):
            B = layer_attention.shape[0]
            for b in range(B):
                att = layer_attention[b].mean(dim=0).detach().cpu().numpy()
                sid = sample_ids[b] if sample_ids is not None else self.attention_counter
                lab = labels[b].item() if labels is not None else "unknown"
                np.savez(os.path.join(self.attention_save_dir, f"layer_{layer_idx}_sample_{sid}_label_{lab}.npz"),
                         attention_map=att, feature_names=np.array(node_names),
                         layer_idx=layer_idx, sample_id=sid, label=lab)
                self.attention_counter += 1

    # ---- optional feature removal ----
    def remove_feature(self, batch, desc_list, nv_list):
        removed = getattr(self.args, 'del_feat', [])
        if not removed:
            return desc_list, nv_list
        rset = set(removed)
        out_desc, out_nv = [], []
        if 'cat_desc_texts' in batch:
            names = [t[0] if isinstance(t, tuple) else str(t) for t in batch['cat_desc_texts']]
            keep = [i for i, n in enumerate(names) if n not in rset]
            if keep:
                batch['cat_desc_texts'] = [batch['cat_desc_texts'][i] for i in keep]
                batch['cat_desc_embeddings'] = batch['cat_desc_embeddings'][:, keep, :]
                batch['cat_name_value_embeddings'] = batch['cat_name_value_embeddings'][:, keep, :]
                out_desc.append(batch['cat_desc_embeddings'].to(self.device))
                out_nv.append(batch['cat_name_value_embeddings'].to(self.device))
        if 'num_desc_texts' in batch:
            names = [t[0] if isinstance(t, tuple) else str(t) for t in batch['num_desc_texts']]
            keep = [i for i, n in enumerate(names) if n not in rset]
            if keep:
                batch['num_desc_texts'] = [batch['num_desc_texts'][i] for i in keep]
                batch['num_desc_embeddings'] = batch['num_desc_embeddings'][:, keep, :]
                batch['num_prompt_embeddings'] = batch['num_prompt_embeddings'][:, keep, :]
                out_desc.append(batch['num_desc_embeddings'].to(self.device))
                out_nv.append(batch['num_prompt_embeddings'].to(self.device))
        return out_desc or desc_list, out_nv or nv_list

    # ---- forward / predict ----
    def forward(self, batch, y):
        target = y.to(self.device)
        if self.num_classes == 2:
            target = target.view(-1, 1).float()
        else:
            target = target.squeeze().long()
        pred = self.predict(batch)
        return self.criterion(pred, target)

    def predict(self, batch):
        desc_list, nv_list = [], []
        if all(k in batch for k in ['cat_name_value_embeddings', 'cat_desc_embeddings']):
            desc_list.append(batch['cat_desc_embeddings'].to(self.device))
            nv_list.append(batch['cat_name_value_embeddings'].to(self.device))
        if all(k in batch for k in ['num_prompt_embeddings', 'num_desc_embeddings']):
            desc_list.append(batch['num_desc_embeddings'].to(self.device))
            nv_list.append(batch['num_prompt_embeddings'].to(self.device))
        desc_list, nv_list = self.remove_feature(batch, desc_list, nv_list)
        if not desc_list or not nv_list:
            raise ValueError("No categorical or numerical features found in batch")

        desc = torch.cat(desc_list, dim=1)
        nv   = torch.cat(nv_list, dim=1)

        # CLS + shared encoder (항상 동결 → no_grad + detach)
        attentions = []
        cls_token = self.cls.expand(nv.size(0), -1, -1)
        x = torch.cat([cls_token, nv], dim=1)
        with torch.no_grad():
            for i, layer in enumerate(self.shared_layers):
                x = x + layer(desc, self.shared_layer_norms[i](x))[0]
        x = x.detach()

        # Coordinator + Basis (학습)
        shared_cls = x[:, 0, :]
        coordinates = self.coordinator(shared_cls)  # [B, k]
        basis_in = self.basis_layer_norm(x)
        basis_out, _ = self.basis_layer(desc, basis_in)  # [B, S_all, H, Hd]
        expert_out = basis_out[:, 0, :, :]  # CLS만 [B, H, Hd]

        preds = []
        for i in range(self.args.k_basis):
            preds.append(self.expert_predictors[i](expert_out[:, i, :]))  # [B, out]
        preds = torch.stack(preds, dim=1)  # [B, H, out]
        return torch.sum(coordinates.unsqueeze(-1) * preds, dim=1)  # [B, out]
        