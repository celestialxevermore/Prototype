import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from entmax import entmax_bisect as entmax
import math
import pdb


class BasisSlotAffinityGAT(nn.Module):
    """
    Slot-only Affinity (no EMA, no prototypes)

    Returns:
        bias_log: [B,H,N,N]  - log Q (detach) : for GAT bias
        Q       : [B,H,N,N]  - sample-conditional global prior
        total_reg: scalar    - regularization term (slot overlap/usage + optional G diversity etc.)
    """
    def __init__(self, args, input_dim: int, n_slots: int , slot_dim: int):
        super().__init__()
        self.args = args
        self.D = int(input_dim)
        self.H = int(args.n_heads)                 # number of heads (basis)
        self.M = int(n_slots)   # slots (independent from H) 
        self.K = int(slot_dim)

        # ---------------- Hyperparameters ----------------
        self.no_self_loop        = args.no_self_loop
        self.tau_aff             = args.slot_aff_temp          # (currently unused) sample affinity temperature
        self.tau_slot            = args.slot_graph_temp        # slot graph temperature
        self.slot_orth_lambda    = args.slot_orth_lambda
        self.scale               = 1.0 / math.sqrt(self.D)
        self.slot_usage_lambda   = args.slot_usage_lambda
        # G constraints
        self.g_mode              = args.slot_g_mode
        self.g_diag_zero         = args.slot_g_diag_zero
        self.g_sparse_l1         = args.slot_g_sparse_l1
        self.g_ent_lambda        = args.slot_g_ent_lambda
        self.g_temp              = args.slot_g_temp
        self.g_div_lambda        = args.g_frob_div_lambda

        # Kernel-specific
        self.kernel_rank         = args.slot_kernel_rank if args.slot_kernel_rank > 0 else self.K
        self.laplacian_lambda    = args.slot_laplacian_lambda

        self.gw_eps = args.gw_eps 
        self.gw_sigma = args.gw_sigma
        self.gw_outer_iters = args.gw_outer_iters 
        self.gw_sinkhorn_iters = args.gw_sinkhorn_iters 
        self.gw_sinkhorn_eps = args.gw_sinkhorn_eps 

        # ---------------- Layers ----------------
        # Fusion embedding -> slot assignment
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.D * 2, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D)
        )
        for m in self.fusion_mlp:
            if isinstance(m, nn.Linear):
                nn_init.xavier_uniform_(m.weight)
                nn_init.zeros_(m.bias)

        self.slot_proj = nn.Linear(self.D, self.M)
        nn_init.xavier_uniform_(self.slot_proj.weight)
        nn_init.zeros_(self.slot_proj.bias)
        # Global slot–slot matrix params
        self.U_param = nn.Parameter(torch.empty(self.M, self.K, self.kernel_rank))
        nn_init.xavier_uniform_(self.U_param)
        self.G_param = None

    # ---------------- utils ----------------
    @staticmethod
    def _row_softmax(x, temperature=1.0, mask=None):
        if mask is not None:
            x = x.masked_fill(mask, float("-inf"))
        t = max(temperature, 1e-6)
        return F.softmax(x / t, dim=-1)

    @staticmethod
    def _bistochastic_sinkhorn(M, iters=10, eps=1e-6):
        # M: [H,K,K] (assume nonnegative)
        M = M.clamp_min(eps)
        for _ in range(iters):
            M = M / (M.sum(dim=-1, keepdim=True) + eps)  # row normalize
            M = M / (M.sum(dim=-2, keepdim=True) + eps)  # col normalize
        return M
    @staticmethod
    def normalize_affinity(P:torch.Tensor,sym:bool=True, eps:float = 1e-8) -> torch.Tensor:
        if sym:
            P = 0.5 * (P + P.transpose(-1,-2))
        P = P.clamp_min(eps)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        return P 
    
    @staticmethod
    def affinity_to_distance(P:torch.Tensor) -> torch.Tensor:
        eps = 1e-6 
        P = P.clamp_min(eps)
        return -torch.log(P)
        #return (1.0 - P).clamp_min(0.0)

    @staticmethod
    def cosine_slot_cost(G: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if G.dim() == 3:
            Gt = G.unsqueeze(0)
        elif G.dim() == 4:
            Gt = G
        else:
            raise ValueError(f"cosine_slot_cost expects G with dim 3 or 4, got {G.dim()}")
        diag = torch.diagonal(Gt, dim1=-2, dim2=-1).clamp_min(eps)
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2)).clamp_min(eps)
        cosG = (Gt / denom).clamp(-1.0, 1.0)
        M = (1.0 - cosG).clamp_min(0.0)
        M = M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
        return M if G.dim() == 4 else M.squeeze(0)

    @staticmethod
    def cosine_slot_cost_from_U(U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute cosine-based slot cost directly from U (before UU^T).
        Args:
            U: [H, K, R]  - slot embeddings (nonnegative, after softplus)
            eps: small value for numerical stability
        Returns:
            DG: [H, K, K] - slot-to-slot cosine distance matrix
        """
        # normalize slot embeddings along last dim (R)
        U_norm = F.normalize(U, p=2, dim=-1, eps=eps)   # [H, K, R]

        # cosine similarity between slots
        cos_sim = torch.einsum("mkr,mjr->mkj", U_norm, U_norm)  # [H, K, K]
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        DG = 0.5 * (1.0 - cos_sim)       
        # remove self-distance (diagonal = 0)
        DG = DG - torch.diag_embed(torch.diagonal(DG, dim1=-2, dim2=-1))
        DG = DG.clamp(0.0, 1.0)
        #cosine distance
        return DG
    # @staticmethod
    # def cosine_slot_cost_from_U(U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    #     """
    #     RBF 기반 슬롯 거리 DG 생성.
    #     U: [H, K, R]
    #     return DG: [H, K, K]  with zero diagonal (no hard clamp to 0/1)
    #     """
    #     H, K, R = U.shape
    #     # pairwise squared Euclidean per head
    #     # d2[h,i,j] = ||U[h,i]-U[h,j]||^2
    #     Ui2 = (U ** 2).sum(dim=-1, keepdim=True)                      # [H,K,1]
    #     d2 = Ui2 + Ui2.transpose(-2, -1) - 2 * (U @ U.transpose(-2,-1))  # [H,K,K]
    #     d2 = d2.clamp_min(0.0)

    #     # per-head bandwidth: tau_h^2 = median(offdiag d2)  (median heuristic)
    #     eye = torch.eye(K, device=U.device, dtype=torch.bool).unsqueeze(0)  # [1,K,K]
    #     d2_off = d2.masked_select(~eye.expand(H, -1, -1)).view(H, K*(K-1))  # [H, K*(K-1)]
    #     tau2 = d2_off.median(dim=1).values.clamp_min(eps)                   # [H]

    #     # RBF kernel with per-head tau; clamp exponent to avoid underflow
    #     # DG = 1 - K,  K = exp(-d2 / (2*tau^2))
    #     scale = (2.0 * tau2).view(H, 1, 1)                                   # [H,1,1]
    #     x = (d2 / scale).clamp(max=30.0)                                     # stabilize
    #     K = torch.exp(-x)                                                     # [H,K,K]
    #     DG = 1.0 - K

    #     # zero diagonal (no hard [0,1] clamp to keep gradients smooth)
    #     DG = DG - torch.diag_embed(torch.diagonal(DG, dim1=-2, dim2=-1))
    #     return DG

    @staticmethod
    def alpha_from_gw(gw_val: torch.Tensor, sigma: float) -> torch.Tensor:
        sigma = max(float(sigma), 1e-6)
        # 행별 표준화가 더 안정적
        mean = gw_val.mean(dim=-1, keepdim=True)
        std  = gw_val.std(dim=-1, keepdim=True).clamp_min(1e-6)
        z    = (gw_val - mean) / std
        return torch.softmax(-z / sigma, dim=-1)  # sigma=temperature

    @staticmethod
    def sharpen_Q(Q_slot: torch.Tensor, alpha: torch.Tensor, eps=1e-8):
        """
        Q_slot : [B,M,N,N]   (SGS로 만든 slot affinity)
        alpha  : [B,H,M]     (GW로 얻은 head-slot 가중치)
        return : [B,H,N,N]   (head별로 재조합된 affinity)
        """
        B, M, N, _ = Q_slot.shape
        _, H, _ = alpha.shape

        # normalize alpha if needed
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True).clamp_min(eps))

        # [B,H,M] × [B,M,N,N] → [B,H,N,N]
        Q_hat = torch.einsum("bhm,bmij->bhij", alpha, Q_slot)

        return Q_hat
    @staticmethod
    def _sinkhorn_ot(a, b, C, eps=0.05, iters=30, tol=1e-6):
        """
        a : [B,H,N]    (distribution over N features per head)
        b : [B,M,K]    (distribution over K slots per global slot m)
        C : [B,H,M,N,K] cost
        return Pi: [B,H,M,N,K]
        """
        B, H, M, N, K = C.shape
        
        Kmat = torch.exp(-C / max(eps, 1e-6)).clamp_min(1e-12)  # [B,H,M,N,K]

        # 초기화
        u = torch.ones(B, H, M, N, device=C.device) / N
        v = torch.ones(B, H, M, K, device=C.device) / K

        for _ in range(iters):
            # [B,H,M,N]
            Kv = torch.einsum("bhmnk,bhmk->bhmn", Kmat, v) + tol
            u = a.unsqueeze(2) / Kv  # broadcast a: [B,H,1,N] -> [B,H,M,N]

            # [B,H,M,K]
            KTu = torch.einsum("bhmnk,bhmn->bhmk", Kmat, u) + tol
            v = b.unsqueeze(1) / KTu  # broadcast b: [B,1,M,K] -> [B,H,M,K]

        Pi = (u.unsqueeze(-1) * Kmat) * v.unsqueeze(-2)  # [B,H,M,N,K]

        return Pi

    @staticmethod
    def _gw_cost_matrix(Dx: torch.Tensor, Dy: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """
        Dx : [B,H,N,N]
        Dy : [B,M,K,K]
        Pi : [B,H,M,N,K]
        return: [B,H,M,N,K] cost tensor
        """
        Dx2, Dy2 = Dx ** 2, Dy ** 2
        # (1) sum over K dimension → mass_j: [B,H,M,N]
        mass_j = Pi.sum(dim=-1)
        term1 = torch.einsum("bhnj,bhmj->bhmn", Dx2, mass_j).unsqueeze(-1)  # [B,H,M,N,1]

        # (2) sum over N dimension → mass_l: [B,H,M,K]
        mass_l = Pi.sum(dim=-2)
        term2 = torch.einsum("bmkl,bhml->bhmk",Dy2,mass_l).unsqueeze(-2)
        # (3) cross term
        cross  = torch.einsum("bhij,bmkl,bhmjl->bhmik", Dx, Dy, Pi)
        C = term1 + term2 - 2.0 * cross

        return C.clamp_min(0.0)



    @staticmethod
    def _entropic_gw(DP: torch.tensor, DG:torch.tensor, a:torch.tensor,b:torch.tensor,eps: float = 0.05, outer_iters: int = 10, sinkhorn_iters: int = 30, tol: float = 1e-6):
        #Pi = torch.einsum("bhn,bhk->bhnk",a,b)
        Pi = torch.einsum("bhn,bmk->bhmnk", a, b)
        for _ in range(outer_iters):
            C = BasisSlotAffinityGAT._gw_cost_matrix(DP, DG, Pi)
            Pi = BasisSlotAffinityGAT._sinkhorn_ot(a,b,C,eps=eps,iters=sinkhorn_iters,tol=tol)

        gw_val = (C * Pi).sum(dim=[-2,-1])

        return Pi, gw_val
    

    # ---------------- G construction ----------------
    def _make_G_and_regs(self, S):
        """
        S: [B,N,K]
        Return:
          G: [H,K,K]  - global slot graph with constraints
          g_reg: scalar regularization
          L: [H,K,K] or None (for kernel Laplacian)
        """
        eps = 1e-8
        g_reg_terms = []
        L = None
        #U = F.softplus(self.U_param)  # [H,K,R], nonnegative for positive affinities
        U = F.normalize(self.U_param, p=2, dim=-1)   # ✅ 방향만 쓰게 정규화
        G_cos = torch.einsum("mkr,mjr->mkj", U, U).clamp(-1.0,1.0)  # [H,K,K], symmetric & PSD
        G_cos = G_cos - torch.diag_embed(torch.diagonal(G_cos, dim1=-2, dim2=-1)) 
        G = ((G_cos + 1.0) * 0.5).clamp(0.0,1.0)
        if self.g_div_lambda > 0.0:
            H = G.size(0)
            if H > 1:
                V = F.normalize(G.reshape(H, -1), p=2, dim=1, eps=1e-8)
                Gram = torch.matmul(V, V.t())
                off_mean = (Gram.sum() - torch.diag(Gram).sum()) / (H * (H - 1))
                g_reg_terms.append(self.g_div_lambda * off_mean)

        if self.g_sparse_l1 > 0.0:
            off = G * (1.0 - torch.eye(self.K, device=G.device).view(1, self.K, self.K))
            g_reg_terms.append(self.g_sparse_l1 * off.abs().mean())

        if self.laplacian_lambda > 0.0:
            D = torch.diag_embed(G.sum(dim=-1))
            L = D - G
        g_reg = torch.stack(g_reg_terms).sum() if g_reg_terms else torch.tensor(0.0, device=S.device)
        #pdb.set_trace()
        return G, g_reg, L, U

    # ---------------- Laplacian smoothness ----------------
    def _laplacian_smoothness(self, S, L):
        """
        S: [B,N,K], L: [M,K,K]
        return scalar mean_bm Tr(S_b^T L_m S_b)
        """
        if L is None:
            return torch.tensor(0.0, device=S.device)
        # broadcast L over batch: [B,M,K,K]
        B = S.size(0)
        Lb = L.unsqueeze(0).expand(B, -1, -1, -1)  # [B,M,K,K]
        y = torch.einsum("bnk,bmkj->b m n j", S, Lb)   # [B,M,N,K]
        S_exp = S.unsqueeze(1)                         # [B,1,N,K] -> [B,M,N,K] by broadcast in mul
        term_bm = (S_exp * y).sum(dim=(-1, -2))       # [B,M]
        return term_bm.mean()

    # ---------------- Debug utils ----------------
    def _current_G(self):
        # For visualization: return the same affinity used in forward (0~1, diag=0)
        U = F.normalize(self.U_param, p=2, dim=-1)
        G_cos = torch.einsum("hkr,hjr->hkj", U, U).clamp(-1.0, 1.0)
        G_cos = G_cos - torch.diag_embed(torch.diagonal(G_cos, dim1=-2, dim2=-1))
        G = ((G_cos + 1.0) * 0.5).clamp(0.0, 1.0)
        G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
        return G

    def _current_U(self):

        #U = F.softplus(self.U_param)
        #return U
        return F.normalize(self.U_param, p=2,dim=-1)

    def export_G_numpy(self):
        G = self._current_G()
        return G.detach().cpu().numpy()
    def export_U_numpy(self):
        U = self._current_U()
        return U.detach().cpu().numpy()
    def export_slot_cost(self): 
        U = self._current_U()
        M = self.cosine_slot_cost_from_U(U)
        return M.detach().cpu().numpy() 

    # ---------------- Forward ----------------
    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor):
        B, N, D = name_value_embeddings.shape
        device = name_value_embeddings.device
        eps = 1e-8

        # (1) Fusion embedding
        z = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim=-1))  # [B,N,D]
        # (2) Slot assignment S: [B,N,M]
        S_logits = self.slot_proj(z)
        S = F.softmax(S_logits, dim=-1)
        # (3) Global slot graph G & Q(x)
        G, g_reg, L, U = self._make_G_and_regs(S)      # G:[H,K,K]
        
        #A_slot = torch.einsum("bnk,mkl,bjl->bmnj", S, G, S)  # [B,H,N,N]

        A_slot = torch.einsum("bnm,mkl,bjm->bmnj", S, G, S)
        if self.no_self_loop:
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, 1, N, N)
            A_slot = A_slot.masked_fill(eye, float("-inf"))
        Q = self._row_softmax(A_slot, temperature=self.tau_slot)        # [B,H,N,N]

        reg_terms = []
        if self.slot_orth_lambda > 0.0:
            StS = torch.einsum("bnk,bnl->bkl", S, S) / max(float(N), 1.0)  # [B,K,K]
            offdiag = StS - torch.diag_embed(torch.diagonal(StS, dim1=-2, dim2=-1))
            reg_terms.append(self.slot_orth_lambda * (offdiag ** 2).mean())

        if self.slot_usage_lambda > 0.0:
            u = S.mean(dim=1)                               # [B,K], 평균 슬롯 사용률
            u = u / (u.sum(dim=-1, keepdim=True) + eps)     # 정규화
            uniform = torch.full_like(u, 1.0 / self.M)      # 균등 분포
            usage_kl = (u.clamp_min(eps) * (u.clamp_min(eps).log() - uniform.log())).sum(dim=-1).mean()
            reg_terms.append(self.slot_usage_lambda * usage_kl)
            
        if (self.g_mode in ("kernel","gw")) and (self.laplacian_lambda > 0.0):
            reg_terms.append(self.laplacian_lambda * self._laplacian_smoothness(S, L))

        total_reg = torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0, device=device)
        total_reg = total_reg + g_reg
        DG = BasisSlotAffinityGAT.cosine_slot_cost_from_U(U)
        DG = DG.unsqueeze(0).expand(B, -1, -1, -1)          # [B,M,K,K]
        deg_G = 0.5 * (G.sum(dim=-1) + G.sum(dim=-2))       # [M,K]
        b = deg_G / deg_G.sum(dim=-1, keepdim=True).clamp_min(eps)  # [M,K]
        b = b.unsqueeze(0).expand(B, -1, -1)                # [B,M,K]

        bias_log = torch.log(Q.clamp_min(eps))
        return bias_log, Q, total_reg, DG, b