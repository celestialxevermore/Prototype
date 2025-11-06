import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as nn_init 
import math 
import pdb
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
        Pi = (u.unsqueeze(-1) * Kmat) * v.unsqueeze(-2) # Transport Plan Head H x Latent composite M 
        return Pi

    @staticmethod 
    def _pairwise_feature_cost(Fx, Fy):
        """
            Fx : [B, H, N, D]
            Fy : [M, K, D]
            Return : C_feat [B, H, M, N, K]
        """
        if Fy.dim() == 3 : # global dictionary (assign)
            B, H, N, D = Fx.shape
            M, K, _ = Fy.shape
            # Broadcast feature tensors 
            Fx_ = Fx.unsqueeze(2).unsqueeze(-2) #[B, H, 1, N, 1, D]
            Fy_ = Fy.unsqueeze(0).unsqueeze(0).unsqueeze(3) # [1, 1, M, 1, K, D]
            return ((Fx_ - Fy_) ** 2).sum(dim=-1) #[B, H, M, N, K]
        else: # [B, H, K ,D] (reconstruct)
            B, H, N, D = Fx.shape
            _, _, K, _ = Fy.shape 
            Fx_ = Fx.unsqueeze(-2) # [B, H, N, 1, D]
            Fy_ = Fy.unsqueeze(2) # [B, H, 1, K ,D]
            return ((Fx_ - Fy_) ** 2).sum(-1) # [B, H, N, K]

    @staticmethod 
    def _Gromov_Wasserstein_cost(Dx, Dy, Pi):
        """
        Dx : [B, H, N, N]
        Dy : [M, K, K] (assign) or [B, H, K, K]
        Pi : [B, H, M, N, K] (assign - current transport plan) or [B, H, N, K] (reconstruct - current transport plan)
        Returns:
            C_gw : [B, H, M, N, K]
        """
        if Dy.dim() == 3: # (assign)
            B, H, M, N, K = Pi.shape 
            Dx_ = Dx.unsqueeze(2).unsqueeze(-1).unsqueeze(-1) # [B, H, 1, N, N, 1, 1]
            Dy_ = Dy.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3) # [1, 1, M, 1, 1, K, K]
            diff2 = (Dx_ - Dy_) ** 2 
            return torch.einsum("bhmnNkK,bhmnK->bhmnK", diff2, Pi)
        else: # (reconstruct)
            B, H, N, K = Pi.shape 
            Dx_ = Dx.unsqueeze(-1).unsqueeze(-1) # [B, H, N, N, 1, 1]
            Dy_ = Dy.unsqueeze(2).unsqueeze(2) # [B, H, 1, 1, K, K]
            diff2 = (Dx_ - Dy_) ** 2 
            return torch.einsum("bhnNkK,bhnK->bhnK", diff2, Pi)
    
    # ---- 1. assign (no_grad) ---- 
    @staticmethod 
    def assign_FGW(Fx, Fy, Dx, Dy, a, b, alpha = 0.5, eps = 0.05, outer_iters = 10, sinkhorn_iters = 30):
        """
            assign phase : Select the nearest latent composite graph on each (B, H)
            Fx : [B, H, N, D]
            Fy : [M, K, D]
            Dx : [B, H, N, N]
            Dy : [M, K, K]
            Return : 
            fgw_values : [B, H, M]
            Pi : [B, H, M, N, K]
        """
        with torch.no_grad():
            Pi = torch.einsum("bhn,bmk->bhmnk",a,b)
            for _ in range(outer_iters):
                C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy)
                C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy, Pi)
                C = (1 - alpha) * C_feat + alpha * C_gw 
                Pi = FGWUtils._sinkhorn_ot(a, b, C, eps = eps, iters = sinkhorn_iters)
            fgw_values = (C * Pi).sum((-2, -1))
        return Pi, fgw_values
    # ---- 2. reconstruct (grad_on) ---- 
    @staticmethod 
    def reconstruct_FGW(Fx, Fy_sel, Dx, Dy_sel, a, b, alpha = 0.5, eps = 0.05, outer_iters = 10, sinkhorn_iters = 30):
        """
            reconstruction phase : update assigned latent composite graphs on node-level
            Fx : [B, H, N, D]
            Fy_sel : [B, H, K, D]
            Dx : [B, H, N, N]
            Dy_sel : [B, H, K, K]
            Return : 
                Fy_res : [B, H, K, D] : (updated latent composite graph node embeddings)
                Dy_res : [D, H, K, K] : (updated latent composite graph affinity matrices)
                fgw_loss : scalar
        """
        Pi = torch.einsum("bhn,bhk->bhnk", a, b)
        for _ in range(outer_iters):
            C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy_sel)
            C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy_sel, Pi)
            C = (1 - alpha) * C_feat + alpha * C_gw 
            Pi = FGWUtils._sinkhorn_ot(a, b, C.unsqueeze(2), eps = eps, iters = sinkhorn_iters)[:, :, 0]
        # residual update 
        Fy_res = Fy_sel #+ torch.einsum("bhnk,bhnd->bhkd",Pi,Fx)
        Dy_res = LatentCompositeGraph.cosine_slot_cost_from_U(Fy_res)
        fgw_loss = (C * Pi).sum((-2, -1)).mean() 
        return Fy_res, Dy_res, fgw_loss 


    @staticmethod
    def _fused_Gromov_Wasserstein(Fx, Fy, Dx, Dy, alpha=0.5):
        """
        Compute fused cost C_fused = (1 - α)*C_feat + α*C_gw
        Fx: [B, H, N, D]
        Fy: [M, K, D]
        Dx: [B, H, N, N]
        Dy: [M, K, K]
        Return: C_fused [B, H, M, N, K]
        """
        # 1. Feature-level cost
        C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy)
        # 2. Uniform initial Pi
        Pi_init = torch.ones_like(C_feat) / (C_feat.size(-2) * C_feat.size(-1))
        # 3. Structure-level cost
        C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy, Pi_init)
        # 4. Combine
        C_fused = (1 - alpha) * C_feat + alpha * C_gw
        return C_fused 

class LatentCompositeGraph(nn.Module):
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
        self.H = int(args.num_basis_heads)
        self.M = int(n_graphs)
        self.K = int(n_nodes)
        self.D = self.args.input_dim // self.args.num_basis_heads
        self.node_embeddings = nn.Parameter(torch.empty(self.M, self.K, self.D))
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
    def cosine_slot_cost_from_U(U):
        """
        Compute cosine-based affinity matrix from node embeddings U
        Supports both:
            U [M, K, D]  (no batch/head)
            U [B, H, K, D] (with batch & head)
        Returns:
            Dy [M, K, K] or [B, H, K, K]
        """
        U_norm = F.normalize(U, p=2, dim=-1)

        if U.dim() == 3:
            # [M, K, D]
            cosine_similarity = torch.einsum("mrd,mjd->mrj", U_norm, U_norm)
        elif U.dim() == 4:
            # [B, H, K, D]
            cosine_similarity = torch.einsum("bhkd,bhjd->bhkj", U_norm, U_norm)
        else:
            raise ValueError(f"Unexpected input shape for cosine_slot_cost_from_U: {U.shape}")

        # enforce symmetry
        cosine_similarity = 0.5 * (cosine_similarity + cosine_similarity.transpose(-1, -2))
        return cosine_similarity.clamp(-1.0, 1.0)

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
        self.vq_beta = self.args.vq_beta 
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
            Pi_all, fgw_values = FGWUtils.assign_FGW(
                Fx, Fy, Dx, Dy, a, b,
                alpha = self.alpha, eps = self.eps, 
                outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters 
            )
            assign_idx = torch.argmin(fgw_values, dim=-1)
        
        # ---- Step 2. Gather selected latent graphs ---- 
        Fy_sel = Fy[assign_idx] # [B, H, K, D]
        Dy_sel = Dy[assign_idx] # [B, H, K, K]
        b_sel = b[torch.arange(B).unsqueeze(1), assign_idx]
        #Pi_sel = Pi_all[torch.arange(B)[:, None], torch.arange(H)[None, :], assign_idx] # [B,H,N,K]

        # ---- Step 3. FGW-based reconstruction 
        # (a) dictionary update (encoder frozen)
        Fy_res_sel, Dy_res_sel, loss_dict = FGWUtils.reconstruct_FGW(
            Fx.detach(), Fy_sel, Dx.detach(), Dy_sel, a, b_sel, 
            alpha = self.alpha, eps = self.eps, 
            outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters
        )
        # (b) encoder update (dictionary frozen)
        _, _, loss_enc = FGWUtils.reconstruct_FGW(
            Fx, Fy_sel.detach(), Dx, Dy_sel.detach(), a, b_sel, 
            alpha = self.alpha, eps = self.eps, 
            outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters 
        )

        # ---- Step 4. SOM regularizaion for unselected Latent Composite Graphs 
        B, H = assign_idx.shape 
        M, K, D = Fy.shape
        mask = torch.ones((B, H, M), dtype = torch.bool, device = Fy.device)
        mask.scatter_(2, assign_idx.unsqueeze(-1), False) 
        unsel_idx = mask.nonzero(as_tuple = True)

        Fy_unsel = Fy[unsel_idx[2]].view(B, H, M-1, K, D)
        Dy_unsel = Dy[unsel_idx[2]].view(B, H, M-1, K, K)

        Fx_rep = Fx.detach().repeat_interleave(M-1, dim = 0)
        Dx_rep = Dx.detach().repeat_interleave(M-1, dim = 0)
        a_rep = a.repeat_interleave(M-1, dim = 0)
        b_rep = torch.ones_like(b_sel).repeat_interleave(M-1, dim = 0)

        
        Fy_unsel_flat = Fy_unsel.view(B * (M - 1), H, K, D)
        Dy_unsel_flat = Dy_unsel.view(B * (M - 1), H, K, K)
        Fy_res_unsel, Dy_res_unsel, loss_som = FGWUtils.reconstruct_FGW(
            Fx_rep, Fy_unsel_flat, Dx_rep, Dy_unsel_flat, 
            a_rep, b_rep, 
            alpha = self.alpha, eps = self.eps, 
            outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters
        )


        Fy_res_unsel = Fy_res_unsel.view(B, M - 1, H, K, D).transpose(1,2)
        Dy_res_unsel = Dy_res_unsel.view(B, M - 1, H, K, K).transpose(1,2)

        Fy_res_sel = Fy_res_sel.unsqueeze(2)
        Dy_res_sel = Dy_res_sel.unsqueeze(2)

        Fy_res_all = torch.cat([Fy_res_sel, Fy_res_unsel], dim = 2)
        Dy_res_all = torch.cat([Dy_res_sel, Dy_res_unsel], dim = 2)
        Ay_res_all = 1.0 - Dy_res_all.clamp(0.0, 1.0)
        
        fgw_loss = loss_dict.mean() + self.vq_beta * loss_enc.mean() + self.vq_beta * loss_som.mean() 
        if self.args.diversifying_loss:
            # Ay_res_all: [B, H, M, K, K]
            B, H, M, K, _ = Ay_res_all.shape

            # (1) batch 평균 (각 LCG의 평균 affinity 구조)
            A = Ay_res_all.mean(dim=0)  # [H, M, K, K]

            # (2) LCG 단위로 head, node를 flatten
            A_flat = A.permute(1, 0, 2, 3).reshape(M, -1)  # [M, H*K*K]

            # (3) pairwise L1 거리 계산
            dists = torch.cdist(A_flat, A_flat, p=1)  # [M, M]

            # (4) 평균 거리 계산 (upper triangle만)
            mean_dist = (2 * torch.triu(dists, diagonal=1).sum() / (M * (M - 1)))

            # (5) margin-based disentanglement loss
            dis_loss = F.relu(2.0 - mean_dist)
            # (6) 전체 손실에 추가
            fgw_loss = fgw_loss + 0.3 * dis_loss
        return Fy_res_all, Ay_res_all, fgw_loss 
        # ---- Step 4. SOM regularizaion for unselected Latent Composite Graphs 
        # B, H = assign_idx.shape 
        
        # M, K, D = Fy.shape
        # mask = torch.ones((B, H, M), dtype=torch.bool, device=Fy.device)
        # mask.scatter_(2, assign_idx.unsqueeze(-1), False)
        # unsel_idx = mask.nonzero(as_tuple=True)

        # # 선택되지 않은 LCG들만 추출
        # Fy_unsel = Fy[unsel_idx[2]].view(B, H, M-1, K, D)
        # Dy_unsel = Dy[unsel_idx[2]].view(B, H, M-1, K, K)

        # # === FGW reconstruction (unselected 부분은 생략) ===
        # # 대신 원본 Fy, Dy를 그대로 사용
        # Fy_res_unsel = Fy_unsel
        # Dy_res_unsel = Dy_unsel

        # # 선택된 LCG 차원 맞추기
        # Fy_res_sel = Fy_res_sel.unsqueeze(2)
        # Dy_res_sel = Dy_res_sel.unsqueeze(2)

        # # === 전체 LCG 병합 ===
        # Fy_res_all = torch.cat([Fy_res_sel, Fy_res_unsel], dim=2)  # [B, H, M, K, D]
        # Dy_res_all = torch.cat([Dy_res_sel, Dy_res_unsel], dim=2)  # [B, H, M, K, K]
        # Ay_res_all = 1.0 - Dy_res_all.clamp(0.0, 1.0)

        # # === FGW total loss ===
        # fgw_loss = (
        #     loss_dict.mean()
        #     + self.vq_beta * loss_enc.mean()
        # )
        # # 초기화는 한 번만
        # if not hasattr(self, "assign_counts"):
        #     self.assign_counts = torch.zeros(M, dtype=torch.long, device=Fy.device)

        # # assign 결과 누적
        # unique_idx, counts = torch.unique(assign_idx, return_counts=True)
        # self.assign_counts[unique_idx] += counts

        # # 로그 확인
        # print("assign_counts:", self.assign_counts.tolist())

        # print("Loss components:", loss_dict.mean().item(), loss_enc.mean().item())
        # return Fy_res_all, Ay_res_all, fgw_loss