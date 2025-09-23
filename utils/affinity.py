import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from entmax import entmax_bisect as entmax
import math


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
        self.align_kl_lambda     = args.slot_align_kl_lambda   # (currently unused)
        self.slot_orth_lambda    = args.slot_orth_lambda
        self.scale               = 1.0 / math.sqrt(self.D)
        self.use_l2norm          = args.affinity_l2norm
        self.slot_usage_lambda   = args.slot_usage_lambda
        # G constraints
        self.g_mode              = args.slot_g_mode            # "markov" | "kernel"
        self.g_diag_zero         = args.slot_g_diag_zero
        self.g_sparse_l1         = args.slot_g_sparse_l1
        self.g_ent_lambda        = args.slot_g_ent_lambda
        self.g_temp              = args.slot_g_temp
        self.g_div_lambda        = args.g_frob_div_lambda

        # Markov-specific
        self.g_sinkhorn          = args.slot_g_sinkhorn
        self.g_sinkhorn_iters    = args.slot_g_sinkhorn_iters
        self.g_sinkhorn_eps      = args.slot_g_sinkhorn_eps

        # Kernel-specific
        self.kernel_rank         = args.slot_kernel_rank if args.slot_kernel_rank > 0 else self.K
        self.kernel_row_norm     = args.slot_kernel_row_stoch
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
        if self.g_mode in ("kernel", "gw"):
            self.U_param = nn.Parameter(torch.empty(self.M, self.K, self.kernel_rank))
            nn_init.xavier_uniform_(self.U_param)
            self.G_param = None
        elif self.g_mode == "markov":
            self.G_param = nn.Parameter(torch.empty(self.M, self.K, self.K))
            nn_init.xavier_uniform_(self.G_param)
            self.U_param = None
        else:
            raise ValueError(f"Unknown slot_g_mode: {self.g_mode}")

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
        return (1.0 - P).clamp_min(0.0)

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
        #import pdb ; pdb.set_trace()
        return M if G.dim() == 4 else M.squeeze(0)

    @staticmethod 
    def alpha_from_gw(gw_val:torch.Tensor, sigma:float) -> torch.Tensor:
        """
            gw_val : [B,H] 
            return alpha : [B,H,1,1] in (0,1] 
        """
        sigma = max(float(sigma), 1e-6)
        scores = torch.exp(- (gw_val / sigma) ** 2)
        alpha = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # softmax-like
        return alpha

    @staticmethod 
    def sharpen_Q(Q: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Sharpen slot prior Q using alpha weights over M global affinities.

        Args:
            Q: [B,H,N,N] prior affinity (row-stochastic expected)
            alpha: [B,H,M] weights per head for M global affinities
            eps: numerical stability

        Returns:
            Q_hat: [B,H,N,N] sharpened adjacency
        """
        # (1) 차원 확장
        Q_exp = Q.unsqueeze(2)  # [B,H,1,N,N]
        alpha_exp = alpha.unsqueeze(-1).unsqueeze(-1)  # [B,H,M,1,1]

        # (2) slot별 sharpen 결과
        Q_slotwise = (Q_exp.clamp_min(eps)) ** alpha_exp  # [B,H,M,N,N]

        # (3) alpha로 가중합 (M 차원을 reduce)
        Q_hat = (alpha_exp * Q_slotwise).sum(dim=2)  # [B,H,N,N]

        # (4) row normalization (확률 분포 보장)
        Q_hat = Q_hat / Q_hat.sum(dim=-1, keepdim=True).clamp_min(eps)

        return Q_hat
    
    
    @staticmethod
    def _sinkhorn_ot(a, b, C, eps = 0.05, iters= 30, tol=1e-6):
        """
            a: [B,H,N], b:[B,H,K], C:[B,H,N,K]
            return \Pi : [B,H,N,K]
        """
        B, HM, N, K = C.shape 
        Kmat = torch.exp(-C/max(eps, 1e-6)).clamp_min(1e-12)
        u = torch.ones(B,HM,N,device=C.device) / N
        v = torch.ones(B,HM,K,device=C.device) / K
        for _ in range(iters):
            Kv = torch.einsum("bhnk,bhk->bhn",Kmat,v) + tol 
            u = a / Kv 
            KTu = torch.einsum("bhnk,bhn->bhk",Kmat,u) + tol 
            v = b / KTu 
        Pi = (u.unsqueeze(-1) * Kmat) * v.unsqueeze(-2)
        return Pi 
    @staticmethod
    def _gw_cost_matrix(Dx:torch.Tensor,Dy:torch.Tensor,Pi:torch.Tensor) -> torch.Tensor:
        Dx2,Dy2 = Dx ** 2 , Dy ** 2
        mass_j = Pi.sum(dim=-1)
        term1 = torch.einsum("bhij,bhj->bhi",Dx2,mass_j).unsqueeze(-1) # [B,H,N,1]

        mass_l = Pi.sum(dim=-2)
        term2 = torch.einsum("bhkl,bhl->bhk",Dy2,mass_l).unsqueeze(-2) # [B,H,1,K]

        cross = torch.einsum("bhij,bhjl,bhkl->bhik", Dx, Pi, Dy) # [B,H,N,K]
        C = term1 + term2 -2.0 * cross 
        return C.clamp_min(0.0)

    @staticmethod
    def _entropic_gw(DP: torch.tensor, DG:torch.tensor, a:torch.tensor,b:torch.tensor,eps: float = 0.05, outer_iters: int = 10, sinkhorn_iters: int = 30, tol: float = 1e-6):
        Pi = torch.einsum("bhn,bhk->bhnk",a,b)
        for _ in range(outer_iters):
            C = BasisSlotAffinityGAT._gw_cost_matrix(DP, DG, Pi)
            Pi = BasisSlotAffinityGAT._sinkhorn_ot(a,b,C,eps=eps,iters=sinkhorn_iters,tol=tol)

        gw_val = (C * Pi).sum(dim=-2)

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

        if self.g_mode == "markov":
            G = F.softmax(self.G_param / max(self.g_temp, 1e-6), dim=-1)  # row-stochastic
            if self.g_sinkhorn:
                G = self._bistochastic_sinkhorn(G, iters=self.g_sinkhorn_iters, eps=self.g_sinkhorn_eps)
            if self.g_diag_zero:
                G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))

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
            if self.g_ent_lambda > 0.0:
                row_ent = -(G.clamp_min(eps) * G.clamp_min(eps).log()).sum(dim=-1).mean()
                g_reg_terms.append(self.g_ent_lambda * row_ent)

        elif self.g_mode in ("kernel","gw"):
            U = F.softplus(self.U_param)  # [H,K,R], nonnegative for positive affinities
            G = torch.einsum("mkr,mjr->mkj", U, U)  # [H,K,K], symmetric & PSD
            if self.g_diag_zero:
                G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
            if self.kernel_row_norm:
                G = G / (G.sum(dim=-1, keepdim=True).clamp_min(eps))

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
        return G, g_reg, L

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
        eps = 1e-8
        if self.g_mode == "markov":
            G = F.softmax(self.G_param / max(self.g_temp, 1e-6), dim=-1)
            if self.g_sinkhorn:
                G = self._bistochastic_sinkhorn(G, iters=self.g_sinkhorn_iters, eps=self.g_sinkhorn_eps)
            if self.g_diag_zero:
                G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
            return G
        else:
            U = F.softplus(self.U_param)
            G = torch.einsum("hkr,hjr->hkj", U, U)
            if self.g_diag_zero:
                G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
            if self.kernel_row_norm:
                G = G / (G.sum(dim=-1, keepdim=True).clamp_min(eps))
            return G

    def export_G_numpy(self):
        G = self._current_G()
        return G.detach().cpu().numpy()

    def export_slot_cost(self):
        G = self._current_G() 
        M = self.cosine_slot_cost(G)
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
        G, g_reg, L = self._make_G_and_regs(S)      # G:[H,K,K]
        
        A_slot = torch.einsum("bnk,mkl,bjl->bmnj", S, G, S)  # [B,H,N,N]
        if self.no_self_loop:
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, 1, N, N)
            A_slot = A_slot.masked_fill(eye, float("-inf"))
        Q = self._row_softmax(A_slot, temperature=self.tau_slot)        # [B,H,N,N]

        # (4) Slot-related regularizers
        reg_terms = []
        if self.slot_orth_lambda > 0.0:
            StS = torch.einsum("bnk,bnl->bkl", S, S) / max(float(N), 1.0)  # [B,K,K]
            offdiag = StS - torch.diag_embed(torch.diagonal(StS, dim1=-2, dim2=-1))
            reg_terms.append(self.slot_orth_lambda * (offdiag ** 2).mean())

        if self.slot_usage_lambda > 0.0:
            u = S.mean(dim=1)                               # [B,K], 평균 슬롯 사용률
            u = u / (u.sum(dim=-1, keepdim=True) + eps)     # 정규화
            uniform = torch.full_like(u, 1.0 / self.K)      # 균등 분포
            usage_kl = (u.clamp_min(eps) * (u.clamp_min(eps).log() - uniform.log())).sum(dim=-1).mean()
            reg_terms.append(self.slot_usage_lambda * usage_kl)
            
        if (self.g_mode in ("kernel","gw")) and (self.laplacian_lambda > 0.0):
            reg_terms.append(self.laplacian_lambda * self._laplacian_smoothness(S, L))

        total_reg = torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0, device=device)
        total_reg = total_reg + g_reg

        DG = BasisSlotAffinityGAT.cosine_slot_cost(G)
        DG = DG.unsqueeze(0).expand(B, -1, -1, -1)          # [B,M,K,K]
        deg_G = 0.5 * (G.sum(dim=-1) + G.sum(dim=-2))       # [M,K]
        b = deg_G / deg_G.sum(dim=-1, keepdim=True).clamp_min(eps)  # [M,K]
        b = b.unsqueeze(0).expand(B, -1, -1)                # [B,M,K]

        bias_log = torch.log(Q.clamp_min(eps))
        return bias_log, Q, total_reg, DG, b