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
from utils.semirelaxed_fgw_batch import solve_semirelaxed_gromov_batch
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
            if node_embeddings is None:
                node_embeddings = self.node_embeddings  # (M, K, D)
            
            structures = []
            for m in range(self.M):
                Q = self.q_projs[m](node_embeddings[m])  # (K, struct_dim)
                K = self.k_projs[m](node_embeddings[m])  # (K, struct_dim)
                scale = math.sqrt(self.struct_dim)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
                attn = torch.softmax(scores, dim=-1)
                structures.append(1.0 - attn)
            
            structure = torch.stack(structures, dim=0)  # (M, K, K)
            
            if node_embeddings.dim() == 4:
                B = node_embeddings.shape[0]
                structure = structure.unsqueeze(0).expand(B, -1, -1, -1)
            return structure
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
        self.orth_reg = getattr(args, 'orth_reg', 0.1)
        self.div_reg = getattr(args, 'div_reg', 0.1)

        self.div_reg = getattr(args, 'div_reg', 0.0)

        self.logger = logging.getLogger("my_experiment_logger")
        self.logger_name = "my_experiment_logger"
        self.register_buffer("has_printed_initial_weights", torch.tensor(False), persistent=False)
        self.register_buffer('log_step', torch.tensor(0), persistent=False) 
        self.log_interval = 50 
        self.last_pi = None 
        self.last_plan = None
        self.ent_reg = self.args.entropy_reg
        self.current_epoch = 0

        M = args.n_graphs 
        self.register_buffer("usage_count", torch.zeros(M))
        self.register_buffer("reset_counter", torch.tensor(0))
        self.dead_threshold = 0.05
        self.reset_noise = 0.01

    def reset_dead(self, latent_graph):
        M = self.usage_count.shape[0]
        avg_usage = self.usage_count.sum() / M
        dead_mask = self.usage_count < (avg_usage * self.dead_threshold)
        if not dead_mask.any():
            self.logger.info(f"   [Dead Reset] No dead codes. Usage: {self.usage_count.cpu().numpy().round(1)}")
            self.usage_count.zero_()
            return 0
        dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()
        self.logger.info(f"   [Dead Reset] Dead: {dead_indices}")
        self.logger.info(f"   [Dead Reset] Usage: {self.usage_count.cpu().numpy().round(1)}")
        
        with torch.no_grad():
            for d_idx in dead_indices:
                latent_graph.node_embeddings.data[d_idx] = latent_graph.init_node_embeddings[d_idx].clone()
                if hasattr(latent_graph, 'adj_param') and hasattr(latent_graph, 'init_adj_param'):
                    latent_graph.adj_param.data[d_idx] = latent_graph.init_adj_param[d_idx].clone()
                self.logger.info(f"   [Dead Reset] LCG {d_idx} -> restored to init position")
        
        self.usage_count.zero_()
        return len(dead_indices)



    def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
        B, N, D = src_feat.shape

        if self.args.feat_distance == "cosine":
            src_norm = F.normalize(src_feat, dim=-1)
            tgt_norm = F.normalize(tgt_feat, dim=-1)
            cos_sim = torch.bmm(src_norm, tgt_norm.transpose(1, 2))
            M_cost = 1.0 - torch.exp(-(1.0 - cos_sim))
        elif self.args.feat_distance == "l2":
            dist_sq = torch.cdist(src_feat, tgt_feat, p=2) ** 2
            M_raw = dist_sq / float(D)
            with torch.no_grad():
                q90 = torch.quantile(M_raw.detach().flatten(), 0.9).clamp_min(1e-8)
            M_cost = M_raw / q90


        with torch.no_grad():
            feat_mean = M_cost.detach().mean()
            struct_mean_src = src_str.detach().mean()
            struct_mean_tgt = tgt_str.detach().mean()
            struct_mean = (struct_mean_src + struct_mean_tgt) / 2
            scale_ratio = feat_mean / struct_mean.clamp_min(1e-8)

        src_str_scaled = src_str * scale_ratio
        tgt_str_scaled = tgt_str * scale_ratio

        # ====== Logging ======
        if self.log_step % self.log_interval == 0 and src_feat.requires_grad:
            with torch.no_grad():
                self.logger.info(f"\n[COST/RANGE CHECK] Step {int(self.log_step)}")

                y = M_cost.detach().float().flatten()
                qy = torch.quantile(y, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=y.device))
                self.logger.info(f"  [dist_sq/D] mean={y.mean().item():.4f} std={y.std().item():.4f} max={y.max().item():.4f}")
                self.logger.info(f"    q00={qy[0].item():.4f} q50={qy[1].item():.4f} q90={qy[2].item():.4f} "
                                f"q95={qy[3].item():.4f} q99={qy[4].item():.4f} q100={qy[5].item():.4f}")

                # ---- Pre-scaling struct ----
                cs_pre = src_str.detach().float().flatten()
                ct_pre = tgt_str.detach().float().flatten()
                self.logger.info(f"  [PRE-SCALE] src_str mean={cs_pre.mean().item():.4f} | tgt_str mean={ct_pre.mean().item():.4f}")

                # ---- Post-scaling struct ----
                cs = src_str_scaled.detach().float().flatten()
                ct = tgt_str_scaled.detach().float().flatten()
                self.logger.info(f"  [POST-SCALE] src_str mean={cs.mean().item():.4f} | tgt_str mean={ct.mean().item():.4f} | scale_ratio={scale_ratio.item():.4f}")

                qcs = torch.quantile(cs, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=cs.device))
                self.logger.info(f"  [CS=src_str] mean={cs.mean().item():.4f} std={cs.std().item():.4f} "
                                f"min={cs.min().item():.4f} max={cs.max().item():.4f}")
                self.logger.info(f"    q00={qcs[0].item():.4f} q50={qcs[1].item():.4f} q90={qcs[2].item():.4f} "
                                f"q95={qcs[3].item():.4f} q99={qcs[4].item():.4f} q100={qcs[5].item():.4f}")

                qct = torch.quantile(ct, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=ct.device))
                self.logger.info(f"  [CT=tgt_str] mean={ct.mean().item():.4f} std={ct.std().item():.4f} "
                                f"min={ct.min().item():.4f} max={ct.max().item():.4f}")
                self.logger.info(f"    q00={qct[0].item():.4f} q50={qct[1].item():.4f} q90={qct[2].item():.4f} "
                                f"q95={qct[3].item():.4f} q99={qct[4].item():.4f} q100={qct[5].item():.4f}")

        a = torch.ones(src_feat.shape[0], src_feat.shape[1], device=src_feat.device) / src_feat.shape[1]
        b = torch.ones(tgt_feat.shape[0], tgt_feat.shape[1], device=tgt_feat.device) / tgt_feat.shape[1]
        # For more precise computation 
        # result = solve_gromov_batch(
        #     src_str_scaled, tgt_str_scaled, M=M_cost, alpha=self.alpha, reg=self.reg, a=a, b=b,
        #     max_iter=10, tol=1e-3, grad='envelope'
        # )


        result = solve_gromov_batch(
            src_str_scaled, tgt_str_scaled, M=M_cost, alpha=self.alpha, reg=self.reg, a=a, b=b,
            max_iter=5, max_iter_inner= 20, tol=1e-3, grad='envelope'
        )

        if self.log_step % self.log_interval == 0:
            with torch.no_grad():
                if hasattr(result, 'plan') and result.plan is not None:
                    T = result.plan.detach()
                    feat_per_pair = (M_cost * T).sum(dim=(1,2))
                    total_per_pair = result.value.detach()
                    grov_per_pair = (total_per_pair - (1 - self.alpha) * feat_per_pair) / (self.alpha + 1e-9)
                    
                    feature_term = feat_per_pair.mean().item()
                    struct_term = grov_per_pair.mean().item()
                    ratio = feature_term / (struct_term + 1e-9)
                    
                    if src_feat.requires_grad:
                        self.logger.info(f"\n[DIAGNOSTIC] Step {self.log_step.item()}")
                        self.logger.info(f"   >>> (1) Cost Scale | Feat: {feature_term:.6f} vs Struct: {struct_term:.6f} | Ratio: {ratio:.4f}")
                        
                        M_lcg = 8
                        if feat_per_pair.shape[0] >= M_lcg:
                            f_sample0 = feat_per_pair[:M_lcg].cpu().numpy()
                            g_sample0 = grov_per_pair[:M_lcg].cpu().numpy()
                            t_sample0 = total_per_pair[:M_lcg].cpu().numpy()
                            self.logger.info(f"   >>> Per-LCG (sample 0):")
                            self.logger.info(f"       feat:  {np.array2string(f_sample0, precision=4)}")
                            self.logger.info(f"       grov:  {np.array2string(g_sample0, precision=4)}")
                            self.logger.info(f"       total: {np.array2string(t_sample0, precision=4)}")
                            self.logger.info(f"       feat range: {f_sample0.max()-f_sample0.min():.6f}")
                            self.logger.info(f"       grov range: {g_sample0.max()-g_sample0.min():.6f}")
                            
                            raw_mcost = M_cost[:M_lcg]
                            mcost_means = raw_mcost.mean(dim=(1,2))
                            self.logger.info(f"       M_cost mean per LCG: {mcost_means.cpu().numpy().round(4)}")
                            self.logger.info(f"       M_cost range: {(mcost_means.max()-mcost_means.min()).item():.6f}")

                            raw_T = T[:M_lcg]
                            T_ent = -(raw_T * (raw_T+1e-12).log()).sum(dim=(1,2))
                            self.logger.info(f"       Plan entropy per LCG: {T_ent.cpu().numpy().round(4)}")
                        
                        if ratio < 0.05:
                            self.logger.warning("       ⚠️ Feature Cost is STILL too small!")
                        elif ratio > 20.0:
                            self.logger.warning("       ⚠️ Feature Cost is too Large!")
                        # pdb 추가
                        # if self.log_step >= 100:
                        #     import pdb; pdb.set_trace()
        return result.value, result.plan

    def diversifying_loss(self, coordinates, labels):
        distance = (coordinates.unsqueeze(1) - coordinates.unsqueeze(0)).abs().sum(dim=2)  # [B, B]
        label_similarity = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
        positive_loss = (distance * label_similarity).sum() / distance.sum().clamp_min(1e-8)
        return positive_loss

    def get_diversifying_loss(self, coordinates, y):
        return self.diversifying_loss(coordinates, y)


    def forward(self, source_struct, source_feat, lcg_struct, lcg_feat, batch):
        """
        End-to-end FGW routing. VQ loss 없이 task loss만으로 LCG 분화.

        Args:
            source_struct: [B, N, N]
            source_feat:   [B, N, D]
            lcg_struct:    [M, K, K]
            lcg_feat:      [M, K, D]

        Returns:
            Fy_res_batch:      [B, M, K, D]
            lcg_struct_batch:  [B, M, K, K]
            pi:                [B, M]
            dummy_loss:        scalar (0.0, 인터페이스 호환용)
        """
        B, N, D = source_feat.shape
        M, K, _ = lcg_feat.shape
        src_idx = batch.get('src_idx', -1)
        if isinstance(src_idx, torch.Tensor):
            src_idx = src_idx.item()

        # =========================================================================
        # Step 1: LCG를 배치 차원으로 복제
        # =========================================================================
        lcg_feat_batch   = lcg_feat.unsqueeze(0).expand(B, M, K, D)
        lcg_struct_batch = lcg_struct.unsqueeze(0).expand(B, M, K, K)

        # =========================================================================
        # Step 2: flatten for FGW
        # =========================================================================
        src_feat_exp = source_feat.unsqueeze(1).expand(B, M, N, D).reshape(B * M, N, D)
        src_str_exp  = source_struct.unsqueeze(1).expand(B, M, N, N).reshape(B * M, N, N)
        lcg_feat_exp = lcg_feat_batch.reshape(B * M, K, D)
        lcg_str_exp  = lcg_struct_batch.reshape(B * M, K, K)

        self.log_step += 1

        # =========================================================================
        # Step 3: FGW 1회 (encoder + LCG 둘 다 grad)
        # =========================================================================
        dist, plan = self.compute_fgw(
            src_feat_exp, src_str_exp, lcg_feat_exp, lcg_str_exp
        )
        d_commit = dist.reshape(B, M)

        # =========================================================================
        # Step 4: Soft coordinate
        # =========================================================================
        pi = torch.softmax(-d_commit / self.soft_tau, dim=1)
        

        p = pi.mean(dim=0)
        H_max = math.log(M)
        H_p = -(p * (p + 1e-9).log()).sum() / H_max
        pi_safe = pi.clamp_min(1e-12)
        H_s = -(pi_safe * pi_safe.log()).sum(dim=1).mean() / H_max
        #reg_loss = 1.0 * (1 - H_p) ** 2 + 0.1 * H_s
        reg_loss = 1.0 * (1 - H_p) ** 2 + 0.5 * H_s


        

        # =========================================================================
        # Logging
        # =========================================================================
        if self.log_step % self.log_interval == 0:
            with torch.no_grad():
                step = int(self.log_step)
                d_min, d_max = d_commit.min().item(), d_commit.max().item()
                d_mean, d_std = d_commit.mean().item(), d_commit.std().item()
                per_range = d_commit.max(dim=1).values - d_commit.min(dim=1).values

                self.logger.info(f"\n[DCOMMIT] Step {step} | Epoch {self.current_epoch}")
                self.logger.info(f"   >>> d_commit: mean={d_mean:.6f}, std={d_std:.6f}, min={d_min:.6f}, max={d_max:.6f}")
                self.logger.info(f"   >>> per-sample gap: mean={per_range.mean():.6f}, min={per_range.min():.6f}, max={per_range.max():.6f}")

                logits = -d_commit / max(1e-8, self.soft_tau)
                top2 = logits.topk(k=min(2, M), dim=1).values
                if top2.shape[1] == 2:
                    gap = top2[:, 0] - top2[:, 1]
                    self.logger.info(f"   >>> logits gap: mean={gap.mean():.6f}, min={gap.min():.6f}, max={gap.max():.6f}")

                bshow = min(B, 3)
                self.logger.info(f"   >>> d_commit[0:{bshow}]: {d_commit[:bshow].detach().cpu().numpy().round(6)}")

                pi_np = pi.detach().cpu().numpy()
                top1 = pi_np.max(axis=1)
                argmax_counts = np.bincount(pi_np.argmax(axis=1), minlength=M)
                p = pi.detach().mean(dim=0)

                H_max = math.log(M)
                pi_safe = pi.detach().clamp_min(1e-12)
                H_s = -(pi_safe * pi_safe.log()).sum(dim=1).mean() / H_max
                H_p = -(p * (p + 1e-9).log()).sum() / H_max

                src_name = f"src={src_idx}" if src_idx is not None else "src=?"
                self.logger.info(
                    f"[ROUTING] Step {step} | Epoch {self.current_epoch} | {src_name} | "
                    f"tau={self.soft_tau:.4f} | H_s={H_s:.3f} H_p={H_p:.3f} | "
                    f"top1={top1.mean():.3f}({top1.min():.3f}~{top1.max():.3f}) | "
                    f"p={np.array2string(p.cpu().numpy(), precision=3)} | "
                    f"argmax={argmax_counts}"
                )
                for s in range(min(2, B)):
                    best = pi_np[s].argmax()
                    self.logger.info(f"   >>> Sample {s} Pi: {np.array2string(pi_np[s], precision=4)} | best=LCG{best}({pi_np[s,best]:.3f})")
                # self.logger.info(
                #     f"[REG] Dead={dead_penalty:.4f} | Margin={margin_loss:.4f} | Orth={orth_loss:.4f} | "
                #     f"weighted_orth={self.orth_reg * orth_loss:.4f} | total_reg={reg_loss:.4f}"
                # )
        # =========================================================================
        # Step 5: Barycentric pushforward
        # =========================================================================
        self.last_pi = pi.detach()
        if plan is not None:
            self.last_plan = plan.detach().reshape(B, M, N, K)

        Fy_res_batch = lcg_feat_batch

        # if plan is not None:
        #     Pi_bmnk = plan.detach().reshape(B, M, N, K)
        #     src_feat_bmn = src_feat_exp.reshape(B, M, N, D)

        #     num   = torch.einsum("bmnk,bmnd->bmkd", Pi_bmnk, src_feat_bmn)
        #     denom = Pi_bmnk.sum(dim=2, keepdim=False).unsqueeze(-1).clamp_min(1e-8)
        #     bary  = num / denom

        #     with torch.no_grad():
        #         scale = lcg_feat_batch.norm(dim=-1).mean() / bary.norm(dim=-1).mean().clamp_min(1e-8)
        #     Fy_res_batch = lcg_feat_batch + scale * bary

        #     if self.log_step % self.log_interval == 0:
        #         with torch.no_grad():
        #             lcg_norm = lcg_feat_batch.norm(dim=-1).mean().item()
        #             bary_norm = (scale * bary).norm(dim=-1).mean().item()
        #             self.logger.info(f"[SCALE] lcg={lcg_norm:.4f} | bary={bary_norm:.4f} | scale={scale:.4f}")

        # =========================================================================
        # Return (vq_loss=0으로 인터페이스 호환)
        # =========================================================================

        return Fy_res_batch, lcg_struct_batch, pi, reg_loss