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
        self.current_epoch = 0
    def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
        B, N, D = src_feat.shape 
        dist_sq = torch.cdist(src_feat, tgt_feat, p=2) ** 2 
        
        with torch.no_grad():
            raw_min = dist_sq.min(dim=1, keepdim=True)[0]
            raw_dist_max = dist_sq.max().item()
            raw_dist_mean = dist_sq.mean().item()
            scale_factor = dist_sq.max() + 1e-8 

        M_raw = dist_sq / float(D)

        with torch.no_grad():
            q90 = torch.quantile(M_raw.detach().flatten(), 0.9).clamp_min(1e-8)


        M_cost = M_raw / q90

        # ====== put this block inside compute_fgw() after you define M_cost ======
        if self.log_step % self.log_interval == 0 and src_feat.requires_grad:
            with torch.no_grad():
                self.logger.info(f"\n[COST/RANGE CHECK] Step {int(self.log_step)}")

                # ---- dist_sq ----
                x = dist_sq.detach().float().flatten()
                qs = torch.quantile(x, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=x.device))
                self.logger.info(f"  [dist_sq] mean={x.mean().item():.1f} std={x.std().item():.1f} max={x.max().item():.1f}")
                self.logger.info(f"    q00={qs[0].item():.1f} q50={qs[1].item():.1f} q90={qs[2].item():.1f} "
                                f"q95={qs[3].item():.1f} q99={qs[4].item():.1f} q100={qs[5].item():.1f}")

                # ---- dist_norm (=M_cost if you use it) ----
                y = M_cost.detach().float().flatten()
                qy = torch.quantile(y, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=y.device))
                self.logger.info(f"  [dist_sq/D] mean={y.mean().item():.4f} std={y.std().item():.4f} max={y.max().item():.4f}")
                self.logger.info(f"    q00={qy[0].item():.4f} q50={qy[1].item():.4f} q90={qy[2].item():.4f} "
                                f"q95={qy[3].item():.4f} q99={qy[4].item():.4f} q100={qy[5].item():.4f}")

                # ---- src_str / tgt_str ----
                cs = src_str.detach().float().flatten()
                qcs = torch.quantile(cs, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=cs.device))
                self.logger.info(f"  [CS=src_str] mean={cs.mean().item():.4f} std={cs.std().item():.4f} "
                                f"min={cs.min().item():.4f} max={cs.max().item():.4f}")
                self.logger.info(f"    q00={qcs[0].item():.4f} q50={qcs[1].item():.4f} q90={qcs[2].item():.4f} "
                                f"q95={qcs[3].item():.4f} q99={qcs[4].item():.4f} q100={qcs[5].item():.4f}")

                ct = tgt_str.detach().float().flatten()
                qct = torch.quantile(ct, torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], device=ct.device))
                self.logger.info(f"  [CT=tgt_str] mean={ct.mean().item():.4f} std={ct.std().item():.4f} "
                                f"min={ct.min().item():.4f} max={ct.max().item():.4f}")
                self.logger.info(f"    q00={qct[0].item():.4f} q50={qct[1].item():.4f} q90={qct[2].item():.4f} "
                                f"q95={qct[3].item():.4f} q99={qct[4].item():.4f} q100={qct[5].item():.4f}")




        
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

            if self.log_step % self.log_interval == 0:
                with torch.no_grad():
                    step = int(self.log_step)

                    # -------------------------
                    # (A) d_commit scale stats
                    # -------------------------
                    d_min = d_commit.min().item()
                    d_max = d_commit.max().item()
                    d_mean = d_commit.mean().item()
                    d_std = d_commit.std().item()

                    # per-sample range (how separable per sample)
                    per_range = (d_commit.max(dim=1).values - d_commit.min(dim=1).values)
                    pr_mean = per_range.mean().item()
                    pr_min  = per_range.min().item()
                    pr_max  = per_range.max().item()

                    self.logger.info(
                        f"\n[DCOMMIT] Step {step} | Epoch {self.current_epoch}"
                    )
                    self.logger.info(
                        f"   >>> d_commit: mean={d_mean:.6f}, std={d_std:.6f}, min={d_min:.6f}, max={d_max:.6f}"
                    )
                    self.logger.info(
                        f"   >>> per-sample (max-min): mean={pr_mean:.6f}, min={pr_min:.6f}, max={pr_max:.6f}"
                    )

                    # -------------------------
                    # (B) what tau does to logits
                    # -------------------------
                    tau_now = float(getattr(self, "soft_tau", 1.0))  # 지금 쓰는 tau
                    logits = -d_commit / max(1e-8, tau_now)

                    # top1-top2 gap in logits (bigger gap => sharper pi)
                    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values  # [B,2] (if M>=2)
                    if top2.shape[1] == 2:
                        gap = (top2[:, 0] - top2[:, 1])  # [B]
                        self.logger.info(
                            f"   >>> logits=-d/tau (tau={tau_now:.6f}): gap(top1-top2) mean={gap.mean().item():.6f}, "
                            f"min={gap.min().item():.6f}, max={gap.max().item():.6f}"
                        )
                        # exp(-gap) is roughly "how close" 2nd is to 1st in softmax ratio
                        ratio = torch.exp(-gap).clamp_max(1e6)
                        self.logger.info(
                            f"   >>> approx exp(-gap): mean={ratio.mean().item():.6f}, min={ratio.min().item():.6f}, max={ratio.max().item():.6f}"
                        )

                    # show a few rows
                    bshow = min(d_commit.shape[0], 3)
                    self.logger.info(f"   >>> d_commit[0:{bshow}]: {d_commit[:bshow].detach().cpu().numpy().round(6)}")
            # # =========================================================================
            # # Step 4: 샘플별 soft assignment pi[b,m]
            # # =========================================================================
            warmup_epochs = int(getattr(self.args, "warmup_epochs", 20))
            rampup_epochs = int(getattr(self.args, "rampup_epochs", 40))
            soft_tau_start = float(getattr(self.args, "soft_tau_start", float(self.soft_tau)))
            soft_tau_end = float(getattr(self.args, "soft_tau_end", float(self.soft_tau)))
            if self.current_epoch < warmup_epochs:
                prog = 0.0 
            else:
                t = (self.current_epoch - warmup_epochs) / max(1, rampup_epochs)
                prog = float(max(0.0, min(1.0, t)))
            soft_tau_now = soft_tau_start + (soft_tau_end - soft_tau_start) * prog 
            soft_tau_now = max(1e-8, soft_tau_now) 

            pi = torch.softmax(-d_commit / self.soft_tau, dim=1)  # [B, M]

            # =========================================================================
            # Step 4.5: Entropy regularization on pi (avoid too hard / too uniform)
            # =========================================================================
            pi_safe = pi.clamp_min(1e-12)
            H_sample = -(pi_safe * pi_safe.log()).sum(dim=1)
            H_sample_mean = H_sample.mean() 

            pi_bar = pi.mean(dim = 0)
            H_max_val = math.log(M)
            
            
            H_batch = -(pi_bar * (pi_bar + 1e-9).log()).sum()
            H_s_norm = (H_sample_mean / H_max_val)
            H_p_norm = (H_batch / H_max_val)
            #u = 1.0 / M 

            lambda_s_max = float(getattr(self.args, "lambda_s_max", 2.0))
            lambda_p_max = float(getattr(self.args, "lambda_p_max", 1.0))  # <-- 새로 추가
            lambda_p_min = float(getattr(self.args, "lambda_p_min", lambda_p_max))

            lambda_s = lambda_s_max * prog
            lambda_p = lambda_p_max - (lambda_p_max - lambda_p_min) * prog

            # KL_pop / KL_norm 제거, 대신 H_p_norm 사용
            entropy_reg = (lambda_p * H_p_norm) - (lambda_s * H_s_norm)
        
            
            reg_mode = (
                f"tau={soft_tau_now:.4g} (start={soft_tau_start:.4g}->end={soft_tau_end:.4g}, prog={prog:.2f}) | "
                f"lamS={lambda_s:.3f}, lamB={lambda_p:.3f}"
            )

            # Logging
            if self.log_step % self.log_interval == 0:
                with torch.no_grad():
                    step = int(self.log_step)

                    d_mean = d_commit.mean().item()
                    d_std  = d_commit.std().item()

                    H_mean = H_sample.mean().item()
                    H_min  = H_sample.min().item()
                    H_max  = H_sample.max().item()

                    max_p_per_sample = pi.max(dim=1)[0]
                    max_p_mean = max_p_per_sample.mean().item()
                    max_p_min  = max_p_per_sample.min().item()
                    max_p_max  = max_p_per_sample.max().item()

                    ent_weight = float(getattr(self, "ent_reg", 0.0))

                    self.logger.info(f"\n[SAMPLE-WISE FGW] Step {step} | Epoch {self.current_epoch} | Mode: {reg_mode}")
                    self.logger.info(f"   >>> Distance Stats | Mean: {d_mean:.6f} | Std: {d_std:.6f}")
                    self.logger.info(
                        f"   >>> Entropy (mean/min/max): {H_mean:.4f} / {H_min:.4f} / {H_max:.4f} (H_max={H_max_val:.3f})"
                    )
                    self.logger.info(
                        f"   >>> Max p(pi) per sample (mean/min/max): {max_p_mean:.4f} / {max_p_min:.4f} / {max_p_max:.4f}"
                    )

                    self.logger.info(f"   >>> [MI-Reg Stats] (Norm 0~1)")
                    self.logger.info(f"       (1) Sample Sharpness (H_s): {H_s_norm.item():.4f} (Goal: Low)")
                    self.logger.info(f"       (2) Pop Diversity  (H_p): {H_p_norm.item():.4f} (Goal: High)")
                    self.logger.info(f"       -> Reg Value: {entropy_reg.item():.4f} (weight: {ent_weight:.3e})")

                    self.logger.info(f"   >>> Sample 0 Pi: {pi[0].detach().cpu().numpy().round(4)}")
                    self.logger.info(f"   >>> Sample 1 Pi: {pi[1].detach().cpu().numpy().round(4)}")
                    
                    with torch.no_grad():
                        nodes = lcg_feat_exp[:M]  # [M, K, D] — 첫 샘플의 M개 LCG
                        centroids = nodes.mean(dim=1)  # [M, D]
                        cdist_mat = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # [M, M]
                        mask = ~torch.eye(M, dtype=torch.bool, device=cdist_mat.device)
                        off_diag = cdist_mat[mask]
                        self.logger.info(
                            f"   >>> LCG centroid dist: mean={off_diag.mean():.4f}, "
                            f"min={off_diag.min():.4f}, max={off_diag.max():.4f}"
                        )
                    self.logger.info(f"   prog : {prog:.3f}")
                    self.logger.info(
                        f"       soft_tau_now={soft_tau_now:.6f} | "
                        f"lambdas: lamS={lambda_s:.3f}, lamP={lambda_p:.3f} | "
                        f"Hs(raw)={H_sample_mean.item():.4f}, Hs(norm)={H_s_norm.item():.4f} | "
                        f"Hp(raw)={H_batch.item():.4f}, Hp(norm)={H_p_norm.item():.4f}"
                    )
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
            vq_loss         = loss_codebook + self.vq_beta * loss_commitment - self.ent_reg * entropy_reg
            

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