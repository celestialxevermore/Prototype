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
        self.tau = 10.0
    
        # --- [2. Logger 초기화] --- 
        self.logger = logging.getLogger("my_experiment_logger")
        self.logger_name = "my_experiment_logger"
        self.register_buffer("has_printed_initial_weights", torch.tensor(False), persistent=False)
        self.register_buffer('log_step', torch.tensor(0), persistent=False) 
        self.log_interval = 50 

    def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
        B, N, D = src_feat.shape 

        dist_sq = torch.cdist(src_feat, tgt_feat, p = 2) ** 2 
        M_cost = 1.0 - torch.exp(-dist_sq / (D * self.tau))

        a = torch.ones(src_feat.shape[0], src_feat.shape[1], device = src_feat.device) / src_feat.shape[1]
        b = torch.ones(tgt_feat.shape[0], tgt_feat.shape[1], device = tgt_feat.device) / tgt_feat.shape[1]

        # Solve OT 
        result = solve_gromov_batch(
            src_str, tgt_str, M = M_cost, alpha = self.alpha, reg = self.reg, a = a, b = b, max_iter = 10, tol = 1e-3
        )
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

        # ======
        # Step 1. Commitment & Assignment 
        # LCG : Stop gradient Source -> LCG 
        # ======
        dist_commit , plan_commit = self.compute_fgw(src_feat_exp, src_str_exp, lcg_feat_exp.detach(), lcg_str_exp.detach())
        # Distance [B * M] -> [B, M]
        d_commit = dist_commit.reshape(B, M)

        # Soft Assignment (Coordinate) pi 
        pi = torch.softmax(-d_commit, dim = 1) # [B, M]

        # ======
        # Step 2. Reconstruction & STE 
        # ====== 

        # # Plan [B*M, N, K] -> [B, M, N, K]
        # T = plan_commit.reshape(B, M, N, K)
        # lcg_feat_view = lcg_feat.unsqueeze(0) # [1, M, K, D]
        # F_quant = torch.einsum('bmnk, bmkd -> bmnd', T, lcg_feat_view) * K
        # F_quant_ste = source_feat.unsqueeze(1) + (F_quant - source_feat.unsqueeze(1)).detach() 

        # ======
        # Step 3. Codebook Loss 
        # ======

        dist_codebook, _ = self.compute_fgw(src_feat_exp.detach(), src_str_exp.detach(), lcg_feat_exp, lcg_str_exp)
        d_codebook = dist_codebook.reshape(B, M)

        # ======
        # Step 4. Total VQ Loss 
        # ======
        pi_detached = pi.detach() 
        loss_codebook = (pi_detached * d_codebook).sum(dim=1).mean() 
        loss_commitment = (pi_detached * d_commit).sum(dim=1).mean() 
        vq_loss = loss_codebook + self.vq_beta * loss_commitment 
        # [로깅 로직 강화]
        # if self.training:
        #     self.log_step += 1
            
        #     if self.log_step % self.log_interval == 0:
        #         # [핵심] 로거를 이 시점에 호출해야 파일 핸들러가 확실히 붙어 있음
        #         logger = logging.getLogger(self.logger_name)
                
        #         with torch.no_grad():
        #             # Plan T Sample (Batch 0, Graph 0)
        #             T_sample = plan_commit.reshape(B, M, N, K)[0, 0]
                    
        #             t_max = T_sample.max().item()
        #             t_mean = T_sample.mean().item()
        #             sharpness = t_max / (t_mean + 1e-9)

        #             # 메시지 리스트 생성
        #             lines = []
        #             lines.append(f"\n[GQ Step {self.log_step}] VQ Loss Breakdown:")
        #             lines.append(f"  Total: {vq_loss.item():.4f} | Codebook: {loss_codebook.item():.4f} | Commit: {loss_commitment.item():.4f}")
        #             lines.append(f"  Plan Sharpness: {sharpness:.2f} | Max: {t_max:.4f}")
        #             lines.append(f"  Full Plan Matrix (Rows=Source({N}), Cols=LCG({K})):")
                    
        #             # 헤더 (LCG Node Index)
        #             header = "       " + " ".join([f" L{k:<4}" for k in range(K)])
        #             lines.append(header)
                    
        #             # 행 출력 (Source Node 전체 Loop)
        #             for i in range(N):
        #                 row = T_sample[i].tolist()
        #                 # 소수점 4자리, 0.01 미만은 흐리게 처리하지 않고 값 그대로 출력 (or 점)
        #                 # 연구자님 요청대로 다 보이게:
        #                 row_str = " ".join([f"{x:.4f}" if x > 0.001 else " .... " for x in row])
        #                 lines.append(f"  Src{i:<2}: {row_str}")
                    
        #             final_log = "\n".join(lines)
                    
        #             # 1. 파일에 쓰기
        #             logger.info(final_log)
                    
        #             # 2. 터미널에도 혹시 모르니 강제 출력 (디버깅용)
        #             # print(final_log) 

        return lcg_feat, lcg_struct, pi, vq_loss