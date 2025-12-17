import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as nn_init 
import math 
import pdb
import numpy as np 
import logging
import pandas as pd
from ot.batch import solve_gromov_batch
import os
import matplotlib.pyplot as plt



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
        self.M = int(n_graphs)
        self.K = int(n_nodes)
        self.D = self.args.input_dim
        self.node_embeddings = nn.Parameter(torch.empty(self.M, self.K, self.D))
        self.struct_mode = args.lcg_struct_type 
        self.struct_dim = args.struct_hidden_dim

        if self.struct_mode == 'projection':
            self.q_proj = nn.Linear(self.D, self.struct_dim)
            self.k_proj = nn.Linear(self.D, self.struct_dim)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
        elif self.struct_mode == 'static':
            self.adj_param = nn.Parameter(torch.randn(self.M, self.K, self.K))
        elif self.struct_mode == 'residual':
            self.bias_param = nn.Parameter(torch.zeros(self.M, self.K, self.K))
        
        with torch.no_grad():
            self.node_embeddings.data.normal_(0, 0.6)

    def get_structure(self, node_embeddings=None):
        if node_embeddings is None:
            node_embeddings = self.node_embeddings
            
        if self.struct_mode == "projection":
            Q = self.q_proj(node_embeddings)
            K = self.k_proj(node_embeddings)
            scale_factor = math.sqrt(self.struct_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale_factor
            attn = torch.softmax(scores, dim = -1)
            return 1.0 - attn 
        elif self.struct_mode == 'static':
            adj = torch.sigmoid(self.adj_param)
            structure = 1.0 - adj
            if node_embeddings.dim() == 4:
                B = node_embeddings.shape[0]
                structure = structure.unsqueeze(0).expand(B, -1, -1, -1)
            return structure
        elif self.struct_mode == 'residual':
            dist_sq = torch.cdist(node_embeddings, node_embeddings, p = 2) ** 2
            dist_norm = dist_sq / self.D
            if node_embeddings.dim() == 4:
                B = node_embeddings.shape[0]
                bias = self.bias_param.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                bias = self.bias_param
            val = dist_norm + bias
            return 1.0 - torch.exp(-val)
        else:
            raise ValueError(f"Invalid structure mode : {self.struct_mode}")

    def forward(self):
        return self.node_embeddings, self.get_structure()


class GraphQuantizer(nn.Module):
    """
        perform graph-level quantization using FGW distances. 
        Selects one latent composite graph per head and applies VQ-stype stop-gradient update. 
    """
    def __init__(self, args, alpha=0.5, tau = 0.1, eps=0.05, outer_iters=20, sinkhorn_iters=200):
        super().__init__() 
        self.args = args 
        self.alpha = alpha 
        self.reg = args.reg
        self.vq_beta = self.args.vq_beta 
        self.tau = args.tau
        self.soft_tau = args.soft_tau
        
        self.logger = logging.getLogger("my_experiment_logger")
        self.logger_name = "my_experiment_logger"
        self.register_buffer("has_printed_initial_weights", torch.tensor(False), persistent=False)
        self.register_buffer('log_step', torch.tensor(0), persistent=False) 
        self.log_interval = 50 
        self.last_pi = None 
        self.last_plan = None
        self.ent_reg = self.args.entropy_reg

    def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
        B, N, D = src_feat.shape 
        dist_sq = torch.cdist(src_feat, tgt_feat, p=2) ** 2 
        
        with torch.no_grad():
            raw_min = dist_sq.min(dim=1, keepdim=True)[0]
            raw_dist_max = dist_sq.max().item()
            raw_dist_mean = dist_sq.mean().item()
            scale_factor = dist_sq.max() + 1e-8 

        dist_norm = dist_sq / scale_factor

        if self.log_step % self.log_interval == 0 and src_feat.requires_grad:
             with torch.no_grad():
                self.logger.info(f"\n[RAW DIST CHECK] Step {self.log_step}")
                self.logger.info(f"   >>> Raw Dist^2 Mean: {raw_dist_mean:.1f} | Max: {raw_dist_max:.1f}")
                self.logger.info(f"   >>> Adaptive Scale Factor: {scale_factor.item():.1f}")
                self.logger.info(f"   >>> Normalized Dist Mean: {dist_norm.mean().item():.4f} (Target ~0.3-0.5)")
                self.logger.info(f"   >>> Current tau: {self.tau}")

        M_cost = 1.0 - torch.exp(-dist_norm / self.tau)

        a = torch.ones(src_feat.shape[0], src_feat.shape[1], device=src_feat.device) / src_feat.shape[1]
        b = torch.ones(tgt_feat.shape[0], tgt_feat.shape[1], device=tgt_feat.device) / tgt_feat.shape[1]

        result = solve_gromov_batch(
            src_str, tgt_str, M=M_cost, alpha=self.alpha, reg=self.reg, a=a, b=b, 
            max_iter=10, tol=1e-3, grad='envelope'
        )
        
        if self.log_step % self.log_interval == 0:
            with torch.no_grad():
                if hasattr(result, 'plan') and result.plan is not None:
                    T = result.plan.detach()
                    feature_term = (M_cost * T).sum(dim=(1,2)).mean().item()
                    total_val = result.value.mean().item()
                    struct_term = (total_val - (1 - self.alpha) * feature_term) / (self.alpha + 1e-9)
                    ratio = feature_term / (struct_term + 1e-9)
                    
                    if src_feat.requires_grad: 
                        self.logger.info(f"\n[DIAGNOSTIC] Step {self.log_step.item()}")
                        self.logger.info(f"   >>> (1) Cost Scale | Feat: {feature_term:.6f} vs Struct: {struct_term:.6f} | Ratio: {ratio:.4f}")
                        
                        if ratio < 0.05: 
                            self.logger.warning("       ⚠️ Feature Cost is STILL too small! Check scale_factor or Decrease tau.")
                        elif ratio > 20.0:
                             self.logger.warning("       ⚠️ Feature Cost is too Large! Increase tau.")
                
        return result.value, result.plan

    def forward(self, source_struct, source_feat, lcg_struct, lcg_feat, batch):
            """
            샘플별로 독립적인 FGW 계산 및 pi 생성 + barycentric residual로
            sample-conditioned LCG view 생성

            Args:
                source_struct: [B, N, N]
                source_feat:   [B, N, D]
                lcg_struct:    [M, K, K]
                lcg_feat:      [M, K, D]

            Returns:
                Fy_res_batch:      [B, M, K, D] - 샘플별 LCG features (barycentric residual 포함)
                lcg_struct_batch:  [B, M, K, K] - (전역 구조를 배치 차원으로 broadcast)
                pi:                [B, M]       - 샘플별 soft assignment
                vq_loss:           scalar
            """
            B, N, D = source_feat.shape
            M, K, _ = lcg_feat.shape

            # =========================================================================
            # Step 1: LCG를 배치 차원으로 복제 → [B, M, K, D], [B, M, K, K]
            # =========================================================================
            lcg_feat_batch   = lcg_feat.unsqueeze(0).expand(B, M, K, D)      # [B, M, K, D]
            lcg_struct_batch = lcg_struct.unsqueeze(0).expand(B, M, K, K)    # [B, M, K, K]

            # =========================================================================
            # Step 2: 샘플별로 M개의 LCG와 비교하기 위해 flatten
            # =========================================================================
            src_feat_exp = source_feat.unsqueeze(1).expand(B, M, N, D).reshape(B * M, N, D)
            src_str_exp  = source_struct.unsqueeze(1).expand(B, M, N, N).reshape(B * M, N, N)
            lcg_feat_exp = lcg_feat_batch.reshape(B * M, K, D)
            lcg_str_exp  = lcg_struct_batch.reshape(B * M, K, K)

            self.log_step += 1

            # =========================================================================
            # Step 3: Commitment FGW (encoder + codebook 둘 다 grad) → d_commit, plan_commit
            # =========================================================================
            dist_commit, plan_commit = self.compute_fgw(
                src_feat_exp,
                src_str_exp,
                lcg_feat_exp,
                lcg_str_exp
            )   # dist_commit: [B*M], plan_commit: [B*M, N, K]

            d_commit = dist_commit.reshape(B, M)  # [B, M]

            # =========================================================================
            # Step 4: 샘플별 soft assignment pi[b,m]
            # =========================================================================
            pi = torch.softmax(-d_commit / self.soft_tau, dim=1)  # [B, M]

            # =========================================================================
            # Step 4.5: Entropy regularization on pi (avoid too hard / too uniform)
            # =========================================================================
            pi_safe = pi.clamp_min(1e-12)
            H_per = -(pi_safe * pi_safe.log()).sum(dim=1)
            H_b = H_per.mean()
            HMax = math.log(M)

            # 원하는 entropy 구간 : [H_low, H_high]
            # - H_low : Too hard 방지 
            H_low = 0.2 * HMax 
            H_high = 0.8 * HMax 
            pen_low = F.relu(H_low - H_b) ** 2 
            pen_high = F.relu(H_b - H_high) ** 2
            entropy_reg = pen_low + pen_high
            # Logging
            if self.log_step % self.log_interval == 0:
                with torch.no_grad():
                    step = int(self.log_step)

                    # ---- distance 통계 ----
                    d_mean = d_commit.mean().item()
                    d_std  = d_commit.std().item()

                    # ---- entropy 통계 (배치 평균 + min/max) ----
                    H_mean = H_per.mean().item()
                    H_min  = H_per.min().item()
                    H_max  = H_per.max().item()

                    # ---- max probability 통계 (얼마나 one-hot에 가까운지) ----
                    max_p_per_sample = pi.max(dim=1)[0]          # [B]
                    max_p_mean = max_p_per_sample.mean().item()
                    max_p_min  = max_p_per_sample.min().item()
                    max_p_max  = max_p_per_sample.max().item()

                    # ---- entropy reg 상태 ----
                    ent_weight = float(getattr(self, "ent_reg", 0.0))
                    self.logger.info(f"\n[SAMPLE-WISE FGW] Step {step}")
                    self.logger.info(f"   >>> Distance Stats | Mean: {d_mean:.6f} | Std: {d_std:.6f}")
                    self.logger.info(f"   >>> Entropy (mean/min/max): "
                                    f"{H_mean:.4f} / {H_min:.4f} / {H_max:.4f} (H_max={HMax:.3f})")
                    self.logger.info(f"   >>> Max p(pi) per sample (mean/min/max): "
                                    f"{max_p_mean:.4f} / {max_p_min:.4f} / {max_p_max:.4f}")
                    self.logger.info(f"   >>> EntropyReg | weight: {ent_weight:.3e} | "
                                    f"pen_low: {pen_low.item():.3e} | pen_high: {pen_high.item():.3e} | "
                                    f"total_pen: {entropy_reg.item():.3e}")
                    # 기존 샘플 0,1의 분포도 그대로 유지
                    self.logger.info(f"   >>> Sample 0 Pi: {pi[0].detach().cpu().numpy().round(4)}")
                    if B > 1:
                        self.logger.info(f"   >>> Sample 1 Pi: {pi[1].detach().cpu().numpy().round(4)}")


            # =========================================================================
            # Step 5: Codebook FGW (encoder detach → codebook 전용 loss)
            # =========================================================================
            dist_codebook, _ = self.compute_fgw(
                src_feat_exp.detach(),
                src_str_exp.detach(),
                lcg_feat_exp,
                lcg_str_exp
            )
            d_codebook = dist_codebook.reshape(B, M)  # [B, M]

            # =========================================================================
            # Step 6: VQ Loss 계산 (pi는 weight 역할만, grad는 안 받음)
            # =========================================================================
            pi_detached = pi.detach()
            self.last_pi = pi_detached

            if plan_commit is not None:
                self.last_plan = plan_commit.detach().reshape(B, M, N, K)

            loss_codebook   = (pi_detached * d_codebook).sum(dim=1).mean()
            loss_commitment = (pi_detached * d_commit).sum(dim=1).mean()
            vq_loss         = loss_codebook + self.vq_beta * loss_commitment + self.ent_reg * entropy_reg

            # =========================================================================
            # Step 7 (추가): barycentric pushforward로 sample-conditioned LCG view 만들기
            #   - plan_commit: [B*M, N, K] → [B, M, N, K]
            #   - src_feat_exp: [B*M, N, D] → [B, M, N, D]
            #   Fy_res_batch[b,m,k,:] = lcg_feat_batch[b,m,k,:]
            #                         + Σ_i Π[b,m,i,k] * x[b,m,i,:] / Σ_i Π[b,m,i,k]
            # =========================================================================
            Fy_res_batch = lcg_feat_batch  # fallback (plan이 없는 경우)

            if plan_commit is not None:
                # (1) transport plan reshape & detach (plan에 대해서는 grad 안 태움)
                Pi_bmnk = plan_commit.detach().reshape(B, M, N, K)   # [B, M, N, K]

                # (2) source feature도 [B,M,N,D]로 복원
                src_feat_bmn = src_feat_exp.reshape(B, M, N, D)      # [B, M, N, D]

                # (3) column-wise barycenter: Π^T X  (N축 sum → K,D)
                #     num: [B,M,K,D], denom: [B,M,K,1]
                num   = torch.einsum("bmnk,bmnd->bmkd", Pi_bmnk, src_feat_bmn)
                denom = Pi_bmnk.sum(dim=2, keepdim=False).unsqueeze(-1).clamp_min(1e-8)

                bary  = num / denom  # [B, M, K, D]

                # (4) 전역 코드북 + residual → 샘플별 LCG view
                Fy_res_batch = lcg_feat_batch + bary   # [B, M, K, D]

            # =========================================================================
            # 최종 출력: sample-conditioned LCG view + 구조 + pi + vq_loss
            # =========================================================================
            return Fy_res_batch, lcg_struct_batch, pi, vq_loss