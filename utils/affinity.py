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
    def __init__(self, args, input_dim: int, k_basis: int, k_slots: int = None):
        super().__init__()
        self.args = args
        self.D = int(input_dim)
        self.H = int(args.n_heads)                 # number of heads (basis)
        self.K = int(k_slots) if k_slots is not None else self.H  # slots (independent from H, fallback to H)

        # ---------------- Hyperparameters ----------------
        self.no_self_loop        = args.no_self_loop
        self.tau_aff             = args.slot_aff_temp          # (currently unused) sample affinity temperature
        self.tau_slot            = args.slot_graph_temp        # slot graph temperature
        self.align_kl_lambda     = args.slot_align_kl_lambda   # (currently unused)
        self.slot_orth_lambda    = args.slot_orth_lambda
        self.scale               = 1.0 / math.sqrt(self.D)
        self.use_l2norm          = args.affinity_l2norm

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

        self.slot_proj = nn.Linear(self.D, self.K)
        nn_init.xavier_uniform_(self.slot_proj.weight)
        nn_init.zeros_(self.slot_proj.bias)

        # Global slot–slot matrix params
        if self.g_mode == "kernel":
            self.U_param = nn.Parameter(torch.empty(self.H, self.K, self.kernel_rank))
            nn_init.xavier_uniform_(self.U_param)
            self.G_param = None
        elif self.g_mode == "markov":
            self.G_param = nn.Parameter(torch.empty(self.H, self.K, self.K))
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

        elif self.g_mode == "kernel":
            U = F.softplus(self.U_param)  # [H,K,R], nonnegative for positive affinities
            G = torch.einsum("hkr,hjr->hkj", U, U)  # [H,K,K], symmetric & PSD
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
        S: [B,N,K], L: [H,K,K]
        return scalar mean_bh Tr(S_b^T L_h S_b)
        """
        if L is None:
            return torch.tensor(0.0, device=S.device)
        y = torch.einsum("bnk,hkj->bhnj", S, L)   # [B,H,N,K]
        S_exp = S.unsqueeze(1)                    # [B,1,N,K]
        term_bh = (S_exp * y).sum(dim=(-1, -2))   # [B,H]
        return term_bh.mean()

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
        elif self.g_mode == "kernel":
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

    # ---------------- Forward ----------------
    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor):
        B, N, D = name_value_embeddings.shape
        device = name_value_embeddings.device
        eps = 1e-8

        # (1) Fusion embedding
        z = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim=-1))  # [B,N,D]

        # (2) Slot assignment S: [B,N,K]
        S_logits = self.slot_proj(z)
        S = F.softmax(S_logits, dim=-1)
        # (3) Global slot graph G & Q(x)
        G, g_reg, L = self._make_G_and_regs(S)      # G:[H,K,K]
        A_slot = torch.einsum("bnk,hkl,bml->bhnm", S, G, S)  # [B,H,N,N]
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

        if getattr(self, "slot_usage_lambda", 0.0) > 0.0:
            u = S.mean(dim=1)                               # [B,K], 평균 슬롯 사용률
            u = u / (u.sum(dim=-1, keepdim=True) + eps)     # 정규화
            uniform = torch.full_like(u, 1.0 / self.K)      # 균등 분포
            usage_kl = (u.clamp_min(eps) * (u.clamp_min(eps).log() - uniform.log())).sum(dim=-1).mean()
            reg_terms.append(self.slot_usage_lambda * usage_kl)
            
        if (self.g_mode == "kernel") and (self.laplacian_lambda > 0.0):
            reg_terms.append(self.laplacian_lambda * self._laplacian_smoothness(S, L))

        total_reg = torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0, device=device)
        total_reg = total_reg + g_reg

        # (5) For GAT bias
        bias_log = torch.log(Q.clamp_min(eps)).detach()     # [B,H,N,N]

        return bias_log, Q, total_reg