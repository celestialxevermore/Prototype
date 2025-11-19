import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pdb 

class lightGraphNeuralNet(nn.Module):
    """
        Args 
        Fy_res : [B, K, D]
        Ay_sel : [B, K, K]
    """
    def __init__(self, input_dim : int, hidden_dim : int, dropout : float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

        # weight init 
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.update.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.update.bias)

    def forward(self, Fy_res : torch.Tensor, Ay_sel : torch.Tensor):
        """
            Args:
                Fy_res : [B, K, D]
                Ay_sel : [B, K, K]
            Returns:
                out : [B, K, D]
        """
        Ay_norm = Ay_sel / (Ay_sel.sum(dim=-1, keepdim=True) + 1e-8)

        # Message Passing 
        agg = torch.bmm(Ay_norm, Fy_res)
        agg = F.relu(self.linear(agg))
        update = self.update(self.dropout(agg))

        # Residual + LayerNorm
        out = self.norm(Fy_res + update)
        return out
class GraphReadout(nn.Module):
    """
        Graph-level readout : mean pooling + projection
    """
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int, dropout : float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
    def forward(self, node_embeddings : torch.Tensor):
        """
            Args:
                node_embeddings : [B, K, D]
            Returns:
                graph_embedding : [B, D]
        """
        graph_embedding = node_embeddings.mean(dim=1)
        return self.norm(self.proj(graph_embedding))

class LatentCompositeGNN(nn.Module):
    """
    Head x Latent Composite Graph (M개) GNN projection independently
    Args:
        Fy_res: [B, H, M, K, D]
        Ay_sel: [B, H, M, K, K]
    Returns:
        expert_outputs : [B, H, M, D]
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, num_basis_heads: int, dropout: float = 0.1):
        super().__init__()
        self.args = args 
        self.n_heads = num_basis_heads
        self.n_graphs = self.args.n_graphs
        self.input_dim = self.args.input_dim // self.args.num_basis_heads

        self.graph_gnns = nn.ModuleList([
            lightGraphNeuralNet(self.input_dim, hidden_dim, dropout) for _ in range(self.n_graphs)
        ])
        self.readouts = nn.ModuleList([
            GraphReadout(self.input_dim, hidden_dim, self.input_dim, dropout) for _ in range(self.n_graphs)
        ])

    def forward(self, Fy_res: torch.Tensor, Ay_sel: torch.Tensor, assign_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Fy_res: [B, H, M, K, D]
            Ay_sel: [B, H, M, K, K]
            assign_idx : [B, H, M, K] # graph idx for each node
        Returns:
            graph_outputs: [B, H, M, K, D]
            graph_outputs: [B, H, M, D]
        """# Fy_res는 GraphQuantizer에서 M=1 차원이 제거된 상태로 들어왔다고 가정합니다.
        Fy_res = Fy_res.squeeze(2)  # [B, H, K, D]
        Ay_sel = Ay_sel.squeeze(2)  # [B, H, K,
        B, H, K, D = Fy_res.shape
        H_outputs = [] 

        # ⭐️ B x H 이중 포문 시작 ⭐️
        for i in range(B): # Batch 루프
            head_outputs = []
            for h in range(H): # Head 루프
                # 1. 동적 Dispatch 인덱스 추출 (Python Integer)
                m_idx = assign_idx[i, h].item() 
                
                # 2. 입력 슬라이싱 (현재 [K, D] 형태)
                fy_input = Fy_res[i, h] 
                ay_input = Ay_sel[i, h]
                
                # 3. GNN Expert 호출 (batch size 1로 unsqueeze(0) 필수)
                node_out = self.graph_gnns[m_idx](fy_input.unsqueeze(0), ay_input.unsqueeze(0)) # [1, K, D]
                
                # 4. Readout 및 Squeeze (최종 [D] 벡터 획득)
                # readouts[m]은 [1, D]를 출력하므로 squeeze(0)로 [D] 만듦
                graph_output = self.readouts[m_idx](node_out).squeeze(0) 
                
                head_outputs.append(graph_output)
            
            # 5. H개의 결과를 [H, D] 형태로 쌓음
            H_outputs.append(torch.stack(head_outputs, dim=0)) 
            
        # 6. B개의 결과를 [B, H, D] 형태로 최종 결합
        expert_outputs = torch.stack(H_outputs, dim=0)

        return expert_outputs