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

    def forward(self, Fy_res: torch.Tensor, Ay_sel: torch.Tensor):
        """
        Args:
            Fy_res: [B, H, M, K, D]
            Ay_sel: [B, H, M, K, K]
        Returns:
            graph_outputs: [B, H, M, K, D]
            graph_outputs: [B, H, M, D]
        """
        B, H, M, K, D = Fy_res.shape
        expert_outputs = []

        for m in range(self.n_graphs):
            node_out = self.graph_gnns[m](Fy_res[:, :, m].reshape(B * H, K, D), Ay_sel[:, :, m].reshape(B * H, K, K))  # [B, K, D]
            graph_output = self.readouts[m](node_out)                    # [B, D]
            graph_output = graph_output.view(B, H, -1)
            expert_outputs.append(graph_output.unsqueeze(2))
        expert_outputs = torch.cat(expert_outputs, dim=2)  # [B, H, D]

        return expert_outputs