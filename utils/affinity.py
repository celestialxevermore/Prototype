import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from entmax import entmax_bisect as entmax
import math



# utils/affinity.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

class BasisSlotAffinityGAT(nn.Module):
    """
    Returns:
        bias_log: [B,H,N,N]  - log Q (detach) : GAT bias 용
        Q       : [B,H,N,N]  - sample-conditional global prior
        total_reg: scalar    - (G/S 관련 정규화 합)
    """
    def __init__(self, args, input_dim: int, k_basis: int, k_slots: int = None):
        super().__init__()
        self.args = args
        self.D = int(input_dim)
        self.H = int(k_basis)                 # heads
        self.K = int(k_slots or k_basis)      # slots (기본 heads와 같게)

        # temps
        self.tau_aff  = float(getattr(args, "slot_aff_temp", 0.5))
        self.tau_slot = float(getattr(args, "slot_graph_temp", 0.5))

        # G constraints (공통)
        self.g_mode        = str(getattr(args, "slot_g_mode", "markov"))  # "markov" | "kernel"
        self.g_diag_zero   = bool(getattr(args, "slot_g_diag_zero", True))
        self.g_sparse_l1   = float(getattr(args, "slot_g_sparse_l1", 0.0))
        self.g_ent_lambda  = float(getattr(args, "slot_g_ent_lambda", 0.0))
        self.g_temp        = float(getattr(args, "slot_g_temp", 1.0))
        self.g_frob_div_lambda = float(getattr(args, "g_frob_div_lambda", 0.02))

        # Markov 전용
        self.g_sinkhorn       = bool(getattr(args, "slot_g_sinkhorn", False))
        self.g_sinkhorn_iters = int(getattr(args, "slot_g_sinkhorn_iters", 10))
        self.g_sinkhorn_eps   = float(getattr(args, "slot_g_sinkhorn_eps", 1e-6))

        # Kernel 전용
        _kr = getattr(args, "slot_kernel_rank", None)
        self.kernel_rank = int(_kr if (_kr is not None and _kr > 0) else self.K)
        self.kernel_row_norm   = bool(getattr(args, "slot_kernel_row_stoch", False))
        self.laplacian_lambda  = float(getattr(args, "slot_laplacian_lambda", 0.0))

        # S 관련 규제
        self.slot_orth_lambda  = float(getattr(args, "slot_orth_lambda", 0.1))
        self.slot_usage_lambda = float(getattr(args, "slot_usage_lambda", 0.1))

        # 융합 임베딩 -> S
        self.fusion_mlp = nn.Linear(self.D * 2, self.D)
        nn_init.xavier_uniform_(self.fusion_mlp.weight); nn_init.zeros_(self.fusion_mlp.bias)

        self.slot_proj = nn.Linear(self.D, self.K)
        nn_init.xavier_uniform_(self.slot_proj.weight); nn_init.zeros_(self.slot_proj.bias)

        # G 파라미터
        if self.g_mode == "markov":
            self.G_param = nn.Parameter(torch.zeros(self.H, self.K, self.K))
            nn_init.xavier_uniform_(self.G_param)
        elif self.g_mode == "kernel":
            self.U_param = nn.Parameter(torch.empty(self.H, self.K, self.kernel_rank))
            nn_init.xavier_uniform_(self.U_param)
        else:
            raise ValueError(f"Unknown slot_g_mode: {self.g_mode}")

        # 옵션
        self.no_self_loop = bool(getattr(args, "no_self_loop", False))

    @staticmethod
    def _row_softmax(x, temperature=1.0, mask=None):
        if mask is not None:
            x = x.masked_fill(mask, float('-inf'))
        t = max(temperature, 1e-6)
        return F.softmax(x / t, dim=-1)

    @staticmethod
    def _bistochastic_sinkhorn(M, iters=10, eps=1e-6):
        # M: [H,K,K] (양수 가정)
        M = M.clamp_min(eps)
        for _ in range(iters):
            M = M / (M.sum(dim=-1, keepdim=True) + eps)  # row normalize
            M = M / (M.sum(dim=-2, keepdim=True) + eps)  # col normalize
        return M

    # ==== G 생성(+정규화)와 그에 따른 정규화항 ====
    def _make_G_and_regs(self, S):
        """
        S: [B,N,K]
        Return:
          G: [H,K,K]  - 제약 적용된 전역 행렬
          g_reg: scalar
          L: [H,K,K] or None (kernel 모드일 때 라플라시안)
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

            if self.g_frob_div_lambda > 0.0:
                H = G.size(0)
                if H > 1:
                    V = F.normalize(G.reshape(H, -1), p=2, dim=1, eps=1e-8)
                    Gram = torch.matmul(V, V.t())
                    off_mean = (Gram.sum() - torch.diag(Gram).sum()) / (H * (H - 1))
                    g_reg_terms.append(self.g_frob_div_lambda * off_mean)

            if self.g_sparse_l1 > 0.0:
                off = G * (1.0 - torch.eye(self.K, device=G.device).view(1, self.K, self.K))
                g_reg_terms.append(self.g_sparse_l1 * off.abs().mean())
            if self.g_ent_lambda != 0.0:
                row_ent = -(G.clamp_min(eps) * G.clamp_min(eps).log()).sum(dim=-1).mean()
                g_reg_terms.append(self.g_ent_lambda * row_ent)

        else:  # "kernel"
            #U = F.softplus(self.U_param)                            # [H,K,R]  (비음수)
            G = torch.einsum("hkr,hjr->hkj", self.U_param, self.U_param)                  # [H,K,K]  (PSD)
            if self.g_diag_zero:
                G = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
            if self.kernel_row_norm:
                G = G / (G.sum(dim=-1, keepdim=True).clamp_min(eps))

            if self.g_frob_div_lambda > 0.0:
                H = G.size(0)
                if H > 1:
                    V = F.normalize(G.reshape(H, -1), p=2, dim=1, eps=1e-8)
                    Gram = torch.matmul(V, V.t())
                    off_mean = (Gram.sum() - torch.diag(Gram).sum()) / (H * (H - 1))
                    g_reg_terms.append(self.g_frob_div_lambda * off_mean)

            if self.g_sparse_l1 > 0.0:
                off = G * (1.0 - torch.eye(self.K, device=G.device).view(1, self.K, self.K))
                g_reg_terms.append(self.g_sparse_l1 * off.abs().mean())

            if self.laplacian_lambda > 0.0:
                D = torch.diag_embed(G.sum(dim=-1))
                L = D - G

        g_reg = (torch.stack(g_reg_terms).sum() if g_reg_terms else torch.tensor(0.0, device=S.device))
        return G, g_reg, L

    # ==== 라플라시안 스무딩 ====
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

    # ==== 현재 G 반환 (시각화/디버깅용) ====
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
        return self._current_G().detach().cpu().numpy()

    # ==== 전방 ====
    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor):
        B, N, D = name_value_embeddings.shape
        device = name_value_embeddings.device
        eps = 1e-8

        # (1) 융합 -> z
        z = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim=-1))  # [B,N,D]

        # (2) 슬롯 할당 S
        S_logits = self.slot_proj(z)                # [B,N,K]
        S = F.softmax(S_logits, dim=-1)

        # (3) G + Q
        G, g_reg, L = self._make_G_and_regs(S)      # G:[H,K,K]
        A_slot = torch.einsum("bnk,hkl,bml->bhnm", S, G, S)  # [B,H,N,N]
        if self.no_self_loop:
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, 1, N, N)
            A_slot = A_slot.masked_fill(eye, float("-inf"))
        Q = self._row_softmax(A_slot, temperature=self.tau_slot)        # [B,H,N,N]

        # (4) S 관련 규제
        reg_terms = []
        if self.slot_orth_lambda > 0.0:
            StS = torch.einsum("bnk,bnl->bkl", S, S) / max(float(N), 1.0)  # [B,K,K]
            offdiag = StS - torch.diag_embed(torch.diagonal(StS, dim1=-2, dim2=-1))
            reg_terms.append(self.slot_orth_lambda * (offdiag ** 2).mean())
        if self.slot_usage_lambda > 0.0:
            u = S.mean(dim=1)
            u = u / (u.sum(dim=-1, keepdim=True) + eps)
            uniform = torch.full_like(u, 1.0 / self.K)
            usage_kl = (u.clamp_min(eps) * (u.clamp_min(eps).log() - uniform.log())).sum(dim=-1).mean()
            reg_terms.append(self.slot_usage_lambda * usage_kl)

        if (self.g_mode == "kernel") and (self.laplacian_lambda > 0.0):
            reg_terms.append(self.laplacian_lambda * self._laplacian_smoothness(S, L))

        total_reg = (torch.stack(reg_terms).sum() if reg_terms else torch.tensor(0.0, device=device)) + g_reg

        # (5) GAT 바이어스 용 logQ (detach)
        bias_log = torch.log(Q.clamp_min(eps)).detach()
        return bias_log, Q, total_reg
