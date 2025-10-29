import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as nn_init 
import math 

class FGWUtils:
    """ 
        Differentiable Fused Gromov-Wasserstein utilities.
        - Pure math layer (no parameters)
    """
    @staticmethod 
    def _sinkhorn_ot(a, b, C, eps = 0.05, iters= 30, tolerance = 1e-6):
        """
            a : [B, H, N] (distribution over N features of H attention heads)
            b : [B, M, K] (distribution over K features of M latent composite graph)
            C : [B, H, M, N, K] Cost matrix of Head H x LCG M 

            return \Pi : [B, H, M, N, K] 
        """
        B, H,  M, N, K = C.shape 
        Kmat = torch.exp(-C / max(eps, 1e-6)).clamp_min(1e-12) # [B,H,M,N,K]

        # Initialization 
        u = torch.ones(B, H, M, N, device=C.device) / N # Head Marginal distribution 
        v = torch.ones(B, H, M, K, device=C.device) / K # Latent Composite Graph Marginal distribution 

        for _ in range(iters):
            Kv = torch.einsum("bhmnk,bhmk->bhmn",Kmat,v) + tolerance 
            u = a.unsqueeze(2) / Kv 

            KTu = torch.einsum("bhmnk,bhmn->bhmk", Kmat, u) + tolerance 
            v = b.unsqueeze(1) / KTu 
        Pi = (u.unsqueeze(-1) * Kmat) * v.unsqueeze(-2) # Transport Plan Heed H x Latent composite M 
        return Pi
    @staticmethod 
    def _Gromov_Wasserstein_cost(Dx : torch.Tensor, Dy : torch.Tensor, Pi : torch.Tensor) -> torch.Tensor:
        Dx2, Dy2 = Dx ** 2, Dy ** 2 
        # (1) sum over K dimension -> mass_j : [B, H, M, N]
        mass_j = Pi.sum(dim=-1)
        term1 = torch.einsum("bhnj,bhmj,bhmn", Dx2, mass_j).unsqueeze(-1)

        # (2) sum over N dimension -> mass_l : [B, H, M, K]
        mass_l = Pi.sum(dim=-2)
        term2 = torch.einsum("bmkl,bhml->bhmk", Dy2, mass_l).unsqueeze(-2)

        # (3) cross term 
        cross = torch.einsum("bhij,bmkl,bhmjl->bhmik", Dx, Dy, Pi)
        C = term1 + term2 - 2.0 * cross 
        return C.clamp_min(0.0)
    @staticmethod 
    def _fused_Gromov_Wasserstein(Fx : torch.Tensor, Fy: torch.Tensor, Dx: torch.Tensor, Dy: torch.Tensor, a : torch.Tensor, b : torch.Tensor, alpha : float = 0.5, eps : float = 0.05, sinkhorn_iters : int = 20):
        """
            Fx : [B, H, N, D] Source node features 
            Fy : [M, K, D] Latent composite node embeddings 
            Dx : [B, H, N, N] Source structure distances
            Dy : [M, K, K] Target (latent composite) structure distances
            a : [B, H, S] Source marginal distribution 
            b : [B, M, K] Target marginal distribution 
        """
        # (1) Feature-level cost 
        C_feat = torch.cdist(Fx.unsqueeze(2), Fy.unsqueeze(0).unsqueeze(0), p=2) ** 2 # [B, H, M, N, K]

        # (2) Initial Transport Plan Pi 
        Pi_init = torch.ones_like(C_feat) / (C_feat.size(-2) * C_feat.size(-1))

        # (3) Gromov Wasserstein structure term 
        C_gw = LatentCompositeGraph._Gromov_Wasserstein_cost(Dx, Dy, Pi_init)
        
        # (4) Fused cost
        C_fused = (1 - alpha) * C_feat + alpha * C_gw 
        return C_fused 
    @staticmethod 
    def entropic_FGW(Fx, Fy, Dx, Dy, a, b, alpha = 0.5, eps = 0.05, outer_iters = 10, sinkhorn_iters = 30):
        """
            Iteratively refine the Fused Gromov Wasserstein Transport Plan 
            Returns:
                Pi : optimal Transport Plan
                loss_values : scalar FGW discrepancies
        """
        Pi = torch.einsum("bhn, bmk->bhmnk",a,b)
        for _ in range(outer_iters):
            C = FGWUtils._fused_Gromov_Wasserstein(Fx, Fy, Dx, Dy, a, b, alpha, eps)
            Pi = FGWUtils._sinkhorn_ot(a, b, C, eps = eps, iters= sinkhorn_iters)
        fgw_values = (C * Pi).sum(dim=[-2, -1])
        return Pi, fgw_values 




class LatentCompositeGraph(nn.Modlue):
    """
        Learnable dictionary of latent composite graphs. 
        - Stores M latent graphs, each with K nodes and D-dimensional node embeddings. 
        - Provides structural (Dy) and feature representations for FGW-based matching.
    """

    def __init__(self, args, input_dim : int, n_graphs: int , n_nodes:int, node_dim: int):
        """
        Args:
            args: argparse or config object
            input_dim (int) : feature dimension per node 
            n_graphs (int) : number of latent composite graphs (M)
            n_nodes (int) : number of nodes per latent graph (K)
            
        """
        super().__init__()
        self.args = args 
        self.D = int(input_dim)
        self.H = int(args.n_heads)
        self.M = int(n_graphs)
        self.K = int(n_nodes)
        self.node_embeddings = nn.Parameter(torch.empty(self.M, self.K, self.input_dim))
        nn_init.xavier_uniform_(self.node_embeddings)

    @staticmethod 
    def _row_softmax(x, temperature = 1.0, mask=None):
        if mask is not None:
            x = x.masked_fill(mask, float("-inf"))
        t = max(temperature, 1e-6)
        return F.softmax(x / t, dim=-1)
    @staticmethod 
    def _bistochastic_sinkhorn(M, iters=20, eps=1e-6):
        M = M.clamp_min(eps)
        for _ in range(iters):
            M = M / (M.sum(dim=-1, keepdim=True) + eps)
            M = M / (M.sum(dim=-2, keepdim=True) + eps)
        return M 
    @staticmethod 
    def normalize_affinity(P:torch.Tensor, sym:bool=True, eps:float = 1e-8) -> torch.Tensor:
        f"""
            Args:
                P : Graph Attention Affinity 
            Return:
                Symetric Graph Attention Affinity tilde P
        """
        if sym:
            P = 0.5 * (P + P.transpose(-1,-2))
        P = P.clamp_min(eps)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        return P 
    @staticmethod 
    def affinity_to_distance(P:torch.Tensor) -> torch.Tensor:
        return (1.0 - P).clamp_min(0.0) 
    @staticmethod 
    def cosine_slot_cost(G : torch.Tensor, eps : float = 1e-8) -> torch.Tensor:
        diag = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(eps)
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2)).clamp_min(eps)
        cosG = (G / denom).clamp(-1.0, 1.0)
        M = (1.0 - cosG).clamp_min(0.0)
        M = M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
        return M 
    @staticmethod 
    def cosine_slot_cost_from_U(U: torch.Tensor, eps : float = 1e-8) -> torch.Tensor:
        """ 
            Compute cosine-based intra-graph distance matrix for each latent graph.

            Args:
                U : [M, K, D] M latent node embeddings 
            Returns:
            Dy : [M, K, K] cosine distance matrix per graph 
        """
        U_norm = F.normalize(U, p=2, dim=-1, eps = eps)
        cosine_similarity = torch.einsum("mkr,mjr->mkj", U_norm, U_norm).clamp(-1.0, 1.0)
        Dy = 0.5 * (1.0 - cosine_similarity) 
        Dy = Dy - torch.diag_embed(torch.diagonal(Dy, dim1=-2, dim2=-1)).clamp(0.0,1.0)
        return Dy

    def forward(self):
        """ 
            Returns:
                node_embeddings: [M, K, D]
                Dy : [M, K, K] cosine-based structural distances for each latent graph 
        """
        Dy = self.cosine_slot_cost_from_U(self.node_embeddings)
        return self.node_embeddings, Dy
        

class GraphQuantizer(nn.Module):
    """
        perform graph-level quantization using FGW distances. 
        Selects one latent composite graph per head and applies VQ-stype stop-gradient update. 
    """
    def __init__(self, args, alpha = 0.5, eps = 0.05, outer_iters = 10 , sinkhorn_iters = 30):
        super().__init__() 
        self.args = args 
        self.alpha = alpha 
        self.vq_beta = self.vq_beta 
        self.eps = eps 
        self.outer_iters = outer_iters 
        self.sinkhorn_iters = sinkhorn_iters 
        
    def forward(self, P_affinity : torch.Tensor, basis_outputs : torch.Tensor, latent_graph : torch.Tensor):
        """
            Args:
                P_affinity : [B, H, N, N]
                basis_outputs : [B, N+1, H, D/H]
                latent_graph : LatentCompositeGraph instance 
                detach_encoder : bool (True -> encoder stopgrad, False -> dictionary stopgrad)       
            Returns:
                assign_idx : [B, H] index of selected latent composite graph per head
        """
        # (0) dimension 
        B, H, N, _ = P_affinity.shape 
        
        # (1) Source graph & affinity
        Fx = basis_outputs[:, 1:, :, :].permute(0, 2, 1, 3) #[B, H, N, D/H]
        Dx = LatentCompositeGraph.normalize_affinity(P_affinity)
        # (2) Latent composite graph & affinity 
        Fy, Dy = latent_graph()

        # (3) Uniform marginals 
        a = torch.ones(B, H, N, device = Fx.device) / N 
        b = torch.ones(B, latent_graph.M, latent_graph.K, device=Fx.device) / latent_graph.K 

        # ---- Step 1. Fused Gromov-Wasserstein distance computation for selection ---- 
        with torch. no_grad():
            C_fused = FGWUtils._fused_Gromov_Wasserstein(Fx, Fy, Dx, Dy, a, b, alpha = self.alpha, eps = self.eps)
            fgw_dist = C_fused.mean(dim=[-1,-2])
            assign_idx = torch.argmin(fgw_dist, dim=-1)
        
        # ---- Step 2. Gather selected latent graphs ---- 
        Fy_sel = Fy[assign_idx] # [B, H, K, D]
        Dy_sel = Dy[assign_idx] # [B, H, K, K]

        # ---- (3) FGW reconstruction update (stop-gradient) ---- 
        loss_dict = FGWUtils._fused_Gromov_Wasserstein(
            Fx.detach(), Fy_sel, Dx.detach(), Dy_sel,
            a, b, alpha = self.alpha, eps = self.eps
        ).mean() 
        loss_enc = FGWUtils._fused_Gromov_Wasserstein(
            Fx, Dy_sel.detach(), Dx, Dy_sel.detach(),
            a, b, alpha = self.alpha, eps = self.eps 
        ).mean() 
        loss_fgw = loss_dict + self.vq_beta * loss_enc 
        return loss_fgw         
