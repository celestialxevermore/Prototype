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

        # Message Passing 
        agg = torch.bmm(Ay_sel, Fy_res)
        agg = F.relu(self.linear(agg))
        update = self.update(self.dropout(agg))

        # Residual + LayerNorm
        out = self.norm(Fy_res + update)
        return out
class GraphReadout(nn.Module):
    """
    Graph-level readout : Mean Pooling + Projection
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, node_embeddings: torch.Tensor):
        """
        Args:
            node_embeddings : [B, K, D]
        Returns:
            graph_embedding : [B, D]
        """
        # Mean Pooling over nodes (K)
        graph_embedding = node_embeddings.mean(dim=1)
        return self.norm(self.proj(graph_embedding))


class LatentCompositeGNN(nn.Module):
    """
    M개의 Independent GNN Experts
    각 LCG(m)은 자신의 전담 GNN(m)을 통과함.
    """
    def __init__(self, args, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.args = args
        self.n_graphs = args.n_graphs # M
        self.input_dim = input_dim    # D

        # [M]개의 독립적인 GNN 생성
        self.graph_gnns = nn.ModuleList([
            lightGraphNeuralNet(self.input_dim, hidden_dim, dropout) for _ in range(self.n_graphs)
        ])
        
        # [M]개의 독립적인 Readout 생성
        self.readouts = nn.ModuleList([
            GraphReadout(self.input_dim, hidden_dim, self.input_dim, dropout) for _ in range(self.n_graphs)
        ])

    def forward(self, Fy_res: torch.Tensor, Ay_sel: torch.Tensor):
        """
        Args:
            Fy_res: [M, K, D] (Static LCG Features) - 배치 차원 없음!
            Ay_sel: [M, K, K] (Static LCG Structure)
        Returns:
            expert_outputs : [1, M, D] (Broadcasting을 위해 1 추가)
        """
        # [수정] 3D 텐서가 들어오므로 B를 풀지 않고 M, K, D만 가져옴
        M, K, D = Fy_res.shape
        expert_outputs = []

        for m in range(self.n_graphs):
            # [Pairing] m-th LCG -> m-th GNN
            
            # Input slicing: [K, D] & [K, K]
            node_in = Fy_res[m]
            adj_in = Ay_sel[m]
            
            # [핵심 수정] lightGraphNeuralNet은 3D 입력을 원함 ([B, K, D])
            # 따라서 unsqueeze(0)으로 가상의 배치 차원(1)을 만들어줌
            # [K, D] -> [1, K, D]
            node_in = node_in.unsqueeze(0)
            adj_in = adj_in.unsqueeze(0)
            
            # 1. GNN Update -> [1, K, D]
            node_out = self.graph_gnns[m](node_in, adj_in)
            
            # 2. Readout (Pooling) -> [1, D]
            graph_vec = self.readouts[m](node_out)
            
            # 리스트에 추가
            expert_outputs.append(graph_vec.unsqueeze(1)) # [1, 1, D]

        # Concat: [1, M, D]
        # 이렇게 하면 나중에 [B, M, 1]인 coordinates와 곱할 때 
        # 자동으로 Broadcasting([B, M, D]) 되어 계산됨. (메모리/속도 이득)
        expert_outputs = torch.cat(expert_outputs, dim=1)
        
        return expert_outputs