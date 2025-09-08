import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from entmax import entmax_bisect as entmax

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
        if self.mask_symmetric:
            logits = 0.5 * (logits + logits.transpose(-1,-2))
        return torch.tanh(logits)

class BasisAffinityGAT(nn.Module):
    def __init__(self, args, input_dim : int, k_basis : int, mask_symmetric: bool = True):
        super().__init__()
        self.args = args 
        self.input_dim = input_dim 
        self.k_basis = k_basis 
        self.mask_symmetric = mask_symmetric 

        r = input_dim 
        self.W_q = nn.Parameter(torch.empty(k_basis, input_dim, r))
        self.W_k = nn.Parameter(torch.empty(k_basis, input_dim , r))
        self.a = nn.Parameter(torch.empty(k_basis, 2 * r, 1))
        for p in (self.W_q, self.W_k, self.a):
            nn_init.xavier_uniform_(p)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.fusion_mlp = nn.Linear(input_dim * 2, input_dim)
        nn_init.xavier_uniform_(self.fusion_mlp.weight); nn_init.zeros_(self.fusion_mlp.bias)
        self.momentum = getattr(args, "basis_ema_momentum", 0.99)
        self.register_buffer("alpha_ema", torch.tensor([]))
    def _maybe_reset_ema(self, N:int, device):
        if self.alpha_ema.numel() == 0 or self.alpha_ema.shape != (self.k_basis, N, N):
            self.alpha_ema = torch.zeros(self.k_basis, N, N, device = device)
    def forward(self, desc_embeddings: torch.Tensor, name_value_embeddings: torch.Tensor) -> torch.Tensor:
        '''
            Args:
                desc_embeddings : [B, N, D]
                name_value_embeddings: [B, N, D]
            Returns:
                bias_log : [B, K, N, N] (Var <-> Var log(alpha + eps))
        '''
        B, N, D = name_value_embeddings.shape 
        device = name_value_embeddings.device 
        fused = self.fusion_mlp(torch.cat([desc_embeddings, name_value_embeddings], dim = -1))
        q = torch.einsum("bnd,kdf->bknf", fused, self.W_q)
        k = torch.einsum("bnd,kdf->bknf", fused, self.W_k)
        q_exp = q.unsqueeze(3); k_exp = k.unsqueeze(2)
        feat = torch.cat([q_exp.expand(-1, -1, -1, N, -1), k_exp.expand(-1, -1, N, -1, -1)], dim = -1)
        h = self.leaky_relu(feat)
        logits = torch.einsum('bknmf,kfo=>bkno', h, self.a).squeeze(-1) # [B,K,N,N]
        if self.mask_symmetric:
            logits = 0.5 * (logits + logits.transpose(-1, -2))
        if getattr(self.args, "no_self_loop", False):
            eye = torch.eye(N, device = device).view(1, 1, N, N)
            logits = logits * (1.0 - eye)
        alpha = torch.softmax(logits, dim = -1)
        alpha_batch_mean = alpha.mean(dim=0, keepdim=False)
        self._maybe_reset_ema(N, device)
        with torch.no_grad():
            self.alpha_ema = self.momentum * self.alpha_ema + (1.0 - self.momentum) * alpha_batch_mean
        eps = 1e-6 
        bias_log = torch.log(self.alpha_ema.clamp_min(eps)).detach() 
        bias_log = bias_log.unsqueeze(0).expand(B, -1, -1, -1)
        return bias_log, alpha  


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

class FSTHeadwiseEntmaxScorer(nn.Module):
    """
    (3.1 방식, head별 완전 분리, Linear만 사용)
    - 입력 E는 이미 외부 FFN1(프로젝션)을 통과했다고 가정.
    - 각 head h: FFN2_h: [Ei||Ej] -> R^2  → entmax(2) → FFN3_h: R^2 -> R^1
    입력:
      E: [B, N, input_dim]
    출력:
      M: [B, k_basis, N, N]   # 실수 가중치 (entmax만 사용; sigmoid 없음)
    """
    def __init__(
        self,
        rel_input_dim: int,            # == rel_proj_dim
        rel_hidden_dim: int = 128,     # == rel_hidden_dim
        k_basis: int = 8,
        dropout_rate: float = 0.1,
        mask_symmetric: bool = False,
        no_self_loop: bool = True,
        alpha: float = 1.5,        # Entmax α
    ):
        super().__init__()
        self.rel_input_dim      = rel_input_dim
        self.rel_hidden_dim     = rel_hidden_dim
        self.k_basis        = k_basis
        self.dropout_rate   = dropout_rate
        self.mask_symmetric = mask_symmetric
        self.no_self_loop   = no_self_loop
        self.alpha          = alpha

        # head별 FFN2 (2D -> hidden -> 2) 와 FFN3 (2 -> 1)
        self.ffn2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * rel_input_dim, rel_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(rel_hidden_dim, 2),
            ) for _ in range(k_basis)
        ])
        self.ffn3 = nn.ModuleList([
            nn.Linear(2, 1) for _ in range(k_basis)
        ])

        self._init_weights()

    def _init_weights(self):
        for block in (self.ffn2, self.ffn3):
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn_init.kaiming_uniform_(m.weight, a=0.0)
                    if m.bias is not None:
                        nn_init.zeros_(m.bias)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, N, D]
        B, N, D = E.shape
        assert D == self.rel_input_dim, f"expected input_dim={self.rel_input_dim}, got {D}"

        # pair: [B, N, N, 2D]
        Ei   = E.unsqueeze(2).expand(B, N, N, D)
        Ej   = E.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([Ei, Ej], dim=-1)

        outs = []
        for h in range(self.k_basis):
            logits2 = self.ffn2[h](pair)                      # [B,N,N,2]
            p2      = entmax(logits2, alpha=self.alpha, dim=-1)  # [B,N,N,2]
            a       = self.ffn3[h](p2).squeeze(-1)            # [B,N,N]
            outs.append(a)

        M = torch.stack(outs, dim=1)                          # [B,k_basis,N,N]

        if self.mask_symmetric:
            M = 0.5 * (M + M.transpose(-1, -2))
        if self.no_self_loop:
            eye = torch.eye(N, device=E.device).view(1, 1, N, N)
            M = M * (1.0 - eye)

        return M