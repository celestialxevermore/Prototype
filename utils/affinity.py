import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from entmax import entmax_bisect as entmax
import math
# class BasisAffinityGAT(nn.Module):
#     def __init__(self, args, input_dim : int, k_basis : int):
#         super().__init__()
#         self.args = args 
#         self.input_dim = input_dim 
#         self.k_basis = k_basis 

#         r = input_dim 
#         self.W_q = nn.Parameter(torch.empty(k_basis, input_dim, r))
#         self.W_k = nn.Parameter(torch.empty(k_basis, input_dim , r))
#         self.a = nn.Parameter(torch.empty(k_basis, 2 * r, 1))
#         for p in (self.W_q, self.W_k, self.a):
#             nn_init.xavier_uniform_(p)
#         self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
#         self.fusion_mlp = nn.Linear(input_dim * 2, input_dim)
#         nn_init.xavier_uniform_(self.fusion_mlp.weight); nn_init.zeros_(self.fusion_mlp.bias)


#         self.momentum = getattr(args, "basis_ema_momentum", 0.99)
#         self.register_buffer("alpha_ema", torch.tensor([]), persistent= False)

#     def _maybe_reset_ema(self, N:int, device):
#         if self.alpha_ema.numel() == 0 or self.alpha_ema.shape != (self.k_basis, N, N):
#             self.alpha_ema = torch.zeros(self.k_basis, N, N, device = device)

#     def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor) -> torch.Tensor:
#         '''
#             Args:
#                 desc_embeddings : [B, N, D]
#                 name_value_embeddings: [B, N, D]
#             Returns:
#                 bias_log : [B, K, N, N] (Var <-> Var log(alpha + eps))
#         '''
#         B, N, D = name_value_embeddings.shape 
#         device = name_value_embeddings.device 

#         self._maybe_reset_ema(N, device)

        
#         fused = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim = -1))
        
#         q = torch.einsum("bnd,kdf->bknf", fused, self.W_q)
#         k = torch.einsum("bnd,kdf->bknf", fused, self.W_k)
#         q_exp = q.unsqueeze(3); k_exp = k.unsqueeze(2)
#         feat = torch.cat([q_exp.expand(-1, -1, -1, N, -1), k_exp.expand(-1, -1, N, -1, -1)], dim = -1)
#         h = self.leaky_relu(feat)
#         logits = torch.einsum('bknmf,kfo->bkno', h, self.a).squeeze(-1) # [B,K,N,N]
        
#         if getattr(self.args, "no_self_loop", False):
#             eye = torch.eye(N, device = device, dtype=torch.bool).view(1, 1, N, N)
#             logits = logits.masked_fill(eye, -1e9)


#         alpha = torch.softmax(logits, dim = -1)
#         alpha_batch_mean = alpha.mean(dim=0, keepdim=False)

#         with torch.no_grad():
#             if (self.alpha_ema.numel() == 0) or (self.alpha_ema.shape != alpha_batch_mean.shape):
#                 self.alpha_ema = alpha_batch_mean.detach().to(device)
#             else:
#                 # in-place EMA (권장)
#                 self.alpha_ema.mul_(self.momentum).add_((1.0 - self.momentum) * alpha_batch_mean.detach())
#                 # 또는 out-of-place 한 줄짜리(위와 동등, 이 줄만 쓰고 위 in-place는 지워):
#                 # self.alpha_ema = self.momentum * self.alpha_ema + (1.0 - self.momentum) * alpha_batch_mean.detach()
#         eps = 1e-6
#         bias_log = torch.log(self.alpha_ema.clamp_min(eps)).detach()
#         bias_log = bias_log.unsqueeze(0).expand(B, -1, -1, -1)
#         return bias_log, alpha
class BasisAffinityGAT(nn.Module):
    def __init__(self, args, input_dim: int, k_basis: int):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.k_basis = k_basis

        r = input_dim
        # per-basis projection
        self.W_q = nn.Parameter(torch.empty(k_basis, input_dim, r))
        self.W_k = nn.Parameter(torch.empty(k_basis, input_dim, r))
        # (레거시) concat-MLP용 파라미터. 체크포인트 호환을 위해 유지하지만 사용하지는 않음.
        self.a = nn.Parameter(torch.empty(k_basis, 2 * r, 1))

        for p in (self.W_q, self.W_k, self.a):
            nn_init.xavier_uniform_(p)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.fusion_mlp = nn.Linear(input_dim * 2, input_dim)
        nn_init.xavier_uniform_(self.fusion_mlp.weight); nn_init.zeros_(self.fusion_mlp.bias)

        # dot-product scale
        self.scale = 1.0 / math.sqrt(r)
        self.use_l2norm = bool(getattr(args, "affinity_l2norm", True))  # 기본 True 권장

        # EMA for global anchor
        self.momentum = float(getattr(args, "basis_ema_momentum", 0.99))
        self.register_buffer("alpha_ema", torch.tensor([]), persistent=True)

    def _maybe_reset_ema(self, N: int, device):
        if self.alpha_ema.numel() == 0 or self.alpha_ema.shape != (self.k_basis, N, N):
            self.alpha_ema = torch.zeros(self.k_basis, N, N, device=device)

    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor):
        """
        Args:
            desc_embeddings      : [B, N, D]
            name_value_embeddings: [B, N, D]
        Returns:
            bias_log : [B, K, N, N]  (Var-Var log prob; no-grad anchor)
            alpha    : [B, K, N, N]  (per-batch affinity prob; with-grad)
        """
        B, N, D = name_value_embeddings.shape
        device = name_value_embeddings.device
        self._maybe_reset_ema(N, device)

        # 1) fuse desc & value
        fused = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim=-1))  # [B,N,D]

        # 2) per-basis Q/K (bmm with per-basis weights)
        #    q,k: [B,K,N,r]
        q = torch.einsum("bnd,kdf->bknf", fused, self.W_q)
        k = torch.einsum("bnd,kdf->bknf", fused, self.W_k)

        # (옵션) L2 normalize for stability
        if self.use_l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # 3) scaled dot-product logits: [B,K,N,N]
        logits = torch.einsum("bknf,bkmf->bknm", q, k) * self.scale

        # 4) (옵션) 자기 자신 금지
        if bool(getattr(self.args, "no_self_loop", False)):
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, 1, N, N)
            logits = logits.masked_fill(eye, float("-inf"))

        # 5) softmax over j-dim
        alpha = torch.softmax(logits, dim=-1)  # [B,K,N,N]

        if self.training:
            # 6) batch mean -> EMA anchor (no-grad)
            alpha_batch_mean = alpha.mean(dim=0, keepdim=False)  # [K,N,N]
            with torch.no_grad():
                if (self.alpha_ema.numel() == 0) or (self.alpha_ema.shape != alpha_batch_mean.shape):
                    self.alpha_ema = alpha_batch_mean.detach().to(device)
                else:
                    self.alpha_ema.mul_(self.momentum).add_((1.0 - self.momentum) * alpha_batch_mean.detach())

        # 7) return log prob (anchor) + current prob
        eps = 1e-6
        bias_log = torch.log(self.alpha_ema.clamp_min(eps)).detach()  # [K,N,N]
        bias_log = bias_log.unsqueeze(0).expand(B, -1, -1, -1)        # [B,K,N,N]
        return bias_log, alpha



class HeadAffinityGAT(nn.Module):
    """
        desc + name_Value 
        head_k의 Var <-> Var affinity logit S_k(b) \ in R^{B x N x N}을 독립적으로 만드는 GAT. 
    """
    def __init__(self, input_dim: int, k_basis : int, dropout_rate: float = 0.1, mask_symmetric: bool = False):
        super().__init__()
        self.input_dim = input_dim 
        self.k_basis = k_basis 
        self.mask_symmetric = mask_symmetric 

        self.fusion_mlp = nn.Linear(input_dim * 2, input_dim)
        nn_init.xavier_uniform_(self.fusion_mlp.weight)
        if self.fusion_mlp.bias is not None:
            nn_init.zeros_(self.fusion_mlp.bias)
        self.W = nn.Parameter(torch.empty(k_basis, input_dim, input_dim))
        self.u_left = nn.Parameter(torch.empty(k_basis, input_dim))
        self.u_right = nn.Parameter(torch.empty(k_basis, input_dim))
        self.b = nn.Parameter(torch.zeros(k_basis))
        nn_init.xavier_uniform_(self.W)
        nn_init.xavier_uniform_(self.u_left.unsqueeze(-1))
        nn_init.xavier_uniform_(self.u_right.unsqueeze(-1))
    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor) -> torch.Tensor:
        fused = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim=-1))
        z = torch.einsum("bnd,kd->bkn", fused, self.W)
        bilinear = torch.einsum("bknf,bmf->bknm", z, fused)
        l = torch.einsum("bnd,kd->bkn", fused, self.u_left)
        r = torch.einsum("bnd,kd->bkn", fused, self.u_right)
        logits = bilinear + l.unsqueeze(-1) + r.unsqueeze(-2) + self.b.view(1, self.k_basis, 1, 1)

        return torch.tanh(logits)


class RelationQueryScorer(nn.Module):
    """
    H(=k_basis)개의 관계 쿼리로 head별 Var-Var 마스크 예측.
    phi([Ei||Ej])를 head별 쿼리 q_h와 내적해서 score → sigmoid.
    입력: E [B,N,input_dim], 출력: M [B,k_basis,N,N]
    """
    def __init__(
        self,
        rel_input_dim: int,
        k_basis: int = 8,
        rel_hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        mask_symmetric: bool = False,
        no_self_loop: bool = True,
    ):
        super().__init__()
        self.rel_input_dim = rel_input_dim
        self.k_basis = k_basis
        self.rel_hidden_dim = rel_hidden_dim
        self.dropout_rate = dropout_rate
        self.mask_symmetric = mask_symmetric    
        self.no_self_loop = no_self_loop

        self.phi = nn.Sequential(
            nn.Linear(2 * self.rel_input_dim, self.rel_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.rel_hidden_dim, self.rel_input_dim),  # pair → input_dim 특징
        )
        # H개의 쿼리 벡터
        self.q = nn.Parameter(torch.randn(self.k_basis, self.rel_input_dim) * (1.0 / (self.rel_input_dim ** 0.5)))

        # init
        for m in self.phi:
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=0.0)
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, N, D]
        B, N, D = E.shape
        Ei = E.unsqueeze(2).expand(B, N, N, D)                 # [B,N,N,D]
        Ej = E.unsqueeze(1).expand(B, N, N, D)                 # [B,N,N,D]
        feat = self.phi(torch.cat([Ei, Ej], dim=-1))           # [B,N,N,D]
        logits = torch.einsum('bijd,hd->bijh', feat, self.q)   # [B,N,N,k_basis]
        M = torch.sigmoid(logits).permute(0, 3, 1, 2).contiguous()  # [B,k_basis,N,N]

        if self.mask_symmetric:
            M = 0.5 * (M + M.transpose(-1, -2))
        if self.no_self_loop:
            eye = torch.eye(N, device=E.device).view(1, 1, N, N)
            M = M * (1.0 - eye)
        return M