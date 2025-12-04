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

        if self.struct_mode == 'projection':
            # [Option A] Dynamic Attention
            self.struct_dim = 64 
            self.q_proj = nn.Linear(self.D, self.struct_dim)
            self.k_proj = nn.Linear(self.D, self.struct_dim)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
        elif self.struct_mode == 'static':
            self.adj_param = nn.Parameter(torch.randn(self.M, self.K, self.K))
        elif self.struct_mode == 'residual':
            self.bias_param = nn.Parameter(torch.zeros(self.M, self.K, self.K))
        # copy the KMeans description embeddings. 
        with torch.no_grad():
            self.node_embeddings.data.normal_(0, 0.6)
    def get_structure(self):
        if self.struct_mode == "projection":
            # 1. Projection
            Q = self.q_proj(self.node_embeddings)
            K = self.k_proj(self.node_embeddings)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / 10.0
            attn = torch.softmax(scores, dim = -1)
            return 1.0 - attn 
        elif self.struct_mode == 'static':
            adj = torch.sigmoid(self.adj_param)
            return 1.0 - adj 
        elif self.struct_mode == 'residual':
            dist_sq = torch.cdist(self.node_embeddings, self.node_embeddings, p = 2) ** 2
            dist_norm = dist_sq / self.D 
            val = dist_norm + self.bias_param 
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
    def __init__(self, args, alpha = 0.5, eps = 0.05, outer_iters = 20 , sinkhorn_iters = 200):
        super().__init__() 
        self.args = args 
        self.alpha = alpha 
        self.reg = args.reg
        self.vq_beta = self.args.vq_beta 
        self.tau = 1000.0
        self.soft_tau = 0.0001
        # --- [2. Logger 초기화] --- 
        self.logger = logging.getLogger("my_experiment_logger")
        self.logger_name = "my_experiment_logger"
        self.register_buffer("has_printed_initial_weights", torch.tensor(False), persistent=False)
        self.register_buffer('log_step', torch.tensor(0), persistent=False) 
        self.log_interval = 50 
        self.last_pi = None 
        self.last_pln = None

    def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
        B, N, D = src_feat.shape 

        dist_sq = torch.cdist(src_feat, tgt_feat, p = 2) ** 2 
        if self.log_step % self.log_interval == 0 and src_feat.requires_grad:
             with torch.no_grad():
                raw_dist_mean = dist_sq.mean().item()
                raw_dist_max = dist_sq.max().item()
                denominator = D * self.tau
                print(f"\n[RAW DIST CHECK] Step {self.log_step}")
                print(f"   >>> Raw Dist^2 Mean: {raw_dist_mean:.1f} | Max: {raw_dist_max:.1f}")
                print(f"   >>> Denominator (D*tau): {denominator:.1f} (Current tau={self.tau})")
                if raw_dist_mean > denominator:
                    print("   ⚠️ Raw Distance is larger than Denominator! Increase self.tau massively.")
        M_cost = 1.0 - torch.exp(-dist_sq / (D * self.tau))

        a = torch.ones(src_feat.shape[0], src_feat.shape[1], device = src_feat.device) / src_feat.shape[1]
        b = torch.ones(tgt_feat.shape[0], tgt_feat.shape[1], device = tgt_feat.device) / tgt_feat.shape[1]

        # Solve OT 
        result = solve_gromov_batch(
            src_str, tgt_str, M = M_cost, alpha = self.alpha, reg = self.reg, a = a, b = b, max_iter = 10, tol = 1e-3, grad = 'envelope'
        )
        
        if self.log_step % self.log_interval == 0:
            with torch.no_grad():
                if hasattr(result, 'plan') and result.plan is not None:
                    T = result.plan.detach()
                    # [Syntax 수정] dim=(1,2)
                    feature_term = (M_cost * T).sum(dim=(1,2)).mean().item()
                    total_val = result.value.mean().item()
                    
                    # Struct Term 역산
                    struct_term = (total_val - (1 - self.alpha) * feature_term) / (self.alpha + 1e-9)
                    ratio = feature_term / (struct_term + 1e-9)
                    
                    # 중복 출력을 막기 위해 src_feat의 requires_grad 여부로 메인 호출(d_commit)만 찍도록 함
                    if src_feat.requires_grad: 
                        print(f"\n[DIAGNOSTIC] Step {self.log_step.item()}")
                        print(f"   >>> (1) Cost Scale | Feat: {feature_term:.6f} vs Struct: {struct_term:.6f} | Ratio: {ratio:.4f}")
                        
                        if ratio < 0.05: 
                            print("       ⚠️ Feature Cost is too small! Decrease self.tau.")
                        elif ratio > 20.0:
                             print("       ⚠️ Feature Cost is too Large! Increase self.tau.")
                
        # value = Distance(Gradient 0), plan=Transport Matrix
        return result.value, result.plan
    def forward(self, source_struct, source_feat, lcg_struct, lcg_feat, batch):
        """
            Source : [B, N, D] / [B, N, N]
            lcg : [M, K, D] / [M, K, K]
        """
        B, N, D = source_feat.shape 
        M, K, _ = lcg_feat.shape 
        # Source
        src_feat_exp = source_feat.unsqueeze(1).expand(B, M, N, D).reshape(B*M, N, D)
        src_str_exp = source_struct.unsqueeze(1).expand(B, M, N, N).reshape(B*M, N, N)
        # LCG 
        lcg_feat_exp = lcg_feat.unsqueeze(0).expand(B, M, K, D).reshape(B*M, K, D)
        lcg_str_exp = lcg_struct.unsqueeze(0).expand(B, M, K, K).reshape(B*M, K, K)
        self.log_step +=1 
        # ======
        # Step 1. Commitment & Assignment 
        # LCG : Stop gradient Source -> LCG 
        # ======
        dist_commit , plan_commit = self.compute_fgw(src_feat_exp, src_str_exp, lcg_feat_exp, lcg_str_exp)
        # Distance [B * M] -> [B, M]
        d_commit = dist_commit.reshape(B, M)
        
        # Soft Assignment (Coordinate) pi 
        pi = torch.softmax(-d_commit / self.soft_tau, dim = 1) # [B, M]
        # [진단] Distance Stats & Entropy (d_commit 기준)
        if self.log_step % self.log_interval == 0:
            with torch.no_grad():
                d_mean = d_commit.mean().item()
                d_std = d_commit.std().item()
                
                # 현재 Temperature 적용된 Softmax 분포 확인
                pi_temp = torch.softmax(-d_commit / self.soft_tau, dim=1)
                entropy = -(pi_temp * torch.log(pi_temp + 1e-9)).sum(dim=1).mean().item()
                
                print(f"   >>> (2) Dist Stats | Mean: {d_mean:.6f} | Std: {d_std:.6f} (Target > {self.soft_tau})")
                print(f"   >>> (3) Entropy    | {entropy:.4f} (Max Uniform: {np.log(M):.3f})")
                print(f"   >>> (4) Sample Pi (Top 5 Samples):")
                print(pi_temp[:5].detach().cpu().numpy().round(5))
        
        # ======
        # Step 3. Codebook Loss 
        # ======

        dist_codebook, _ = self.compute_fgw(src_feat_exp.detach(), src_str_exp.detach(), lcg_feat_exp, lcg_str_exp)
        d_codebook = dist_codebook.reshape(B, M)

        # ======
        # Step 4. Total VQ Loss 
        # ======
        pi_detached = pi.detach() 
        self.last_pi = pi.detach() 
        # plan: [B*M, N, K] -> [B, M, N, K]로 변환하여 저장
        if plan_commit is not None:
            self.last_plan = plan_commit.detach().reshape(B, M, N, K)
        loss_codebook = (pi_detached * d_codebook).sum(dim=1).mean() 
        loss_commitment = (pi_detached * d_commit).sum(dim=1).mean() 
        vq_loss = loss_codebook + self.vq_beta * loss_commitment 
        

        return lcg_feat, lcg_struct, pi, vq_loss