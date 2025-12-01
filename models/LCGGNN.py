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
    
    [개선점]
    __init__ 마지막에 _init_experts_as_identity()를 호출하여,
    초기에 GNN이 입력값(K-Means Centroid)을 왜곡하지 않고 그대로 통과시키도록 설정함.
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
        self._init_experts_as_identity()

    def _init_experts_as_identity(self):
        """
        GNN Experts와 Readout의 가중치를 Identity(단위 행렬)에 가깝게 초기화하여
        학습 초기에 K-Means로 찾은 LCG Centroid 정보가 
        Random Weight에 의해 파괴되지 않고 그대로 전파되도록 함.
        """
        # 1. GNN Layer 초기화
        for gnn in self.graph_gnns: 
            # lightGraphNeuralNet 내부 구조: linear -> relu -> update -> residual
            
            # (1) Linear (Input -> Hidden)
            if hasattr(gnn, 'linear'):
                nn.init.eye_(gnn.linear.weight)
                if gnn.linear.bias is not None:
                    nn.init.zeros_(gnn.linear.bias)
            
            # (2) Update (Hidden -> Input)
            if hasattr(gnn, 'update'):
                nn.init.eye_(gnn.update.weight)
                if gnn.update.bias is not None:
                    nn.init.zeros_(gnn.update.bias)
            
        # 2. Readout Layer 초기화
        for readout in self.readouts: 
            # GraphReadout 내부 구조: proj (Linear -> ReLU -> Linear)
            for m in readout.proj:
                if isinstance(m, nn.Linear):
                    # 차원이 달라도 eye_는 가능한 만큼 대각선에 1을 채워줌
                    nn.init.eye_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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
            
            node_in = node_in.unsqueeze(0)
            adj_in = adj_in.unsqueeze(0)
            
            # 1. GNN Update -> [1, K, D]
            node_out = self.graph_gnns[m](node_in, adj_in)
            
            # 2. Readout (Pooling) -> [1, D]
            graph_vec = self.readouts[m](node_out)
            
            # 리스트에 추가
            expert_outputs.append(graph_vec.unsqueeze(1)) # [1, 1, D]

        expert_outputs = torch.cat(expert_outputs, dim=1)
        
        return expert_outputs