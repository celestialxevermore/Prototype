import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as nn_init 
import math 
import pdb
import numpy as np 
import logging
import pandas as pd
class FGWUtils:
    """ 
        Differentiable Fused Gromov-Wasserstein utilities.
        - Pure math layer (no parameters)
    """
    @staticmethod 
    def _sinkhorn_ot(a, b, C, eps = 0.05, iters= 100, tolerance = 1e-6):
        """
            a : [B, H, N] (distribution over N features of H attention heads)
            b : [B, M, K] (distribution over K features of M latent composite graph)
            C : [B, H, M, N, K] Cost matrix of Head H x LCG M 

            return \Pi : [B, H, M, N, K] 
        """
        B, H,  M, N, K = C.shape 
        Kmat = torch.exp(-C / max(eps, 1e-6)).clamp_min(1e-12) # [B,H,M,N,K]
        # --- [❗️❗️ 5. Kmat 계산 직후 로깅 ❗️❗️] ---
        Kmat_pre_exp = -C / max(eps, 1e-6)
        Kmat = torch.exp(Kmat_pre_exp).clamp_min(1e-12)
        logger = logging.getLogger("my_experiment_logger")
        # C.mean()은 reconstruct_FGW와 차원(M)이 달라서 값이 다를 수 있습니다.
        logger.info(f"--- 🩺 FGWUtils (_sinkhorn_ot) ---")
        logger.info(f"  [Input] C.mean: {C.mean().item():.4f}, eps: {eps}")
        logger.info(f"  [Pre-Exp] (-C/eps).mean: {Kmat_pre_exp.mean().item():.4f}")
        logger.info(f"  [Kmat] Kmat.mean: {Kmat.mean().item():.4E}, Kmat.max: {Kmat.max().item():.4E}")
        # Initialization 
        u = torch.ones(B, H, M, N, device=C.device) / N # Head Marginal distribution 
        v = torch.ones(B, H, M, K, device=C.device) / K # Latent Composite Graph Marginal distribution 
        epsilon_tolerance = 1e-9
        for _ in range(iters):
            Kv = torch.einsum("bhmnk,bhmk->bhmn",Kmat,v) + epsilon_tolerance 
            u = a.unsqueeze(2) / Kv 

            KTu = torch.einsum("bhmnk,bhmn->bhmk", Kmat, u) + epsilon_tolerance 
            v = b.unsqueeze(1) / KTu 
        Pi = (u.unsqueeze(-1) * Kmat) * v.unsqueeze(-2) # Transport Plan Head H x Latent composite M 
        return Pi

    @staticmethod 
    def _pairwise_feature_cost(Fx, Fy):
        """
            Fx : [B, H, N, D]
            Fy : [M, K, D]
            Return : C_feat [B, H, M, N, K]
        """
        if Fy.dim() == 3 : # global dictionary (assign)
            B, H, N, D = Fx.shape
            M, K, _ = Fy.shape
            # Broadcast feature tensors 
            Fx_ = Fx.unsqueeze(2).unsqueeze(-2) #[B, H, 1, N, 1, D]
            Fy_ = Fy.unsqueeze(0).unsqueeze(0).unsqueeze(3) # [1, 1, M, 1, K, D]
            return ((Fx_ - Fy_) ** 2).sum(dim=-1) #[B, H, M, N, K]
        else: # [B, H, K ,D] (reconstruct)
            B, H, N, D = Fx.shape
            _, _, K, _ = Fy.shape 
            Fx_ = Fx.unsqueeze(-2) # [B, H, N, 1, D]
            Fy_ = Fy.unsqueeze(2) # [B, H, 1, K ,D]
            return ((Fx_ - Fy_) ** 2).sum(-1) # [B, H, N, K]

    @staticmethod 
    def _Gromov_Wasserstein_cost(Dx, Dy, Pi):
        """
        Dx : [B, H, N, N]
        Dy : [M, K, K] (assign) or [B, H, K, K]
        Pi : [B, H, M, N, K] (assign - current transport plan) or [B, H, N, K] (reconstruct - current transport plan)
        Returns:
            C_gw : [B, H, M, N, K]
        """

        if Dy.dim() == 3: # (assign)
            B, H, M, N, K = Pi.shape 
            Dx_ = Dx.unsqueeze(2).unsqueeze(-1).unsqueeze(-1) # [B, H, 1, N, N, 1, 1]
            Dy_ = Dy.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3) # [1, 1, M, 1, 1, K, K]
            diff2 = (Dx_ - Dy_) ** 2 
            
            return torch.einsum("bhmnNkK,bhmNK->bhmnk", diff2, Pi)
        else: # (reconstruct)
            B, H, N, K = Pi.shape 
            Dx_ = Dx.unsqueeze(-1).unsqueeze(-1) # [B, H, N, N, 1, 1]
            Dy_ = Dy.unsqueeze(2).unsqueeze(2) # [B, H, 1, 1, K, K]
            diff2 = (Dx_ - Dy_) ** 2 

            return torch.einsum("bhnNkK,bhNK->bhnk", diff2, Pi)
    
    # ---- 1. assign (no_grad) ---- 
    @staticmethod 
    def assign_FGW(Fx, Fy, Dx, Dy, a, b, alpha = 0.5, eps = 0.05, outer_iters = 10, sinkhorn_iters = 30, do_log : bool = False):
        """
            assign phase : Select the nearest latent composite graph on each (B, H)
            Fx : [B, H, N, D]
            Fy : [M, K, D]
            Dx : [B, H, N, N]
            Dy : [M, K, K]
            Return : 
            fgw_values : [B, H, M]
            Pi : [B, H, M, N, K]
        """
        # assign_FGW 
        logger = logging.getLogger("my_experiment_logger")
        with torch.no_grad():
            Pi = torch.einsum("bhn,bmk->bhmnk",a,b)
            for _ in range(outer_iters):
                C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy)
                C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy, Pi)

                if _ == outer_iters -1 and do_log:
                    # ⬇️ --- [ ✨ "타이틀" 수정됨 ✨ ] --- ⬇️
                    logger.info(f"--- 🩺 FGWUtils (assign_FGW) ---") 
                    logger.info(f"   (alpha: {alpha}, eps: {eps})")

                    # ⬇️ --- [ ✨ "입력 재료" 로깅 추가됨 ✨ ] --- ⬇️
                    logger.info("--- 🩺 FGW Inputs (Source) ---")
                    logger.info(f"   [Fx]     mean: {Fx.mean().item():.4f}, std: {Fx.std().item():.4f}, L2_norm_sq: {(Fx**2).sum(-1).mean().item():.4f}")
                    logger.info(f"   [Dx]     mean: {Dx.mean().item():.4f}, std: {Dx.std().item():.4f}, min: {Dx.min().item():.4f}, max: {Dx.max().item():.4f}")
                    logger.info("--- 🩺 FGW Inputs (Target_All) ---") # 'Selected' -> 'All'
                    
                    # ⬇️ --- [ ✨ "변수명" 수정됨 (Fy_sel -> Fy, Dy_sel -> Dy) ✨ ] --- ⬇️
                    logger.info(f"   [Fy]     mean: {Fy.mean().item():.4f}, std: {Fy.std().item():.4f}, L2_norm_sq: {(Fy**2).sum(-1).mean().item():.4f}")
                    logger.info(f"   [Dy]     mean: {Dy.mean().item():.4f}, std: {Dy.std().item():.4f}, min: {Dy.min().item():.4f}, max: {Dy.max().item():.4f}")
                    # ⬆️ --- [ ✨ 로깅 코드 끝 ✨ ] --- ⬆️

                    # (Pre-Alpha) C_feat와 C_gw의 원본 통계
                    logger.info("--- 🩺 FGW Cost Components (Pre-Alpha) ---")
                    logger.info(f"   [C_feat] mean: {C_feat.mean().item():.4f}, std: {C_feat.std().item():.4f}, max: {C_feat.max().item():.4f}")
                    logger.info(f"   [C_gw]   mean: {C_gw.mean().item():.4f}, std: {C_gw.std().item():.4f}, max: {C_gw.max().item():.4f}")
                    

                # # ⬆️ --- [ ✨ 로깅 코드 끝 ✨ ] --- ⬆️
                C = (1 - alpha) * C_feat + alpha * C_gw
                Pi = FGWUtils._sinkhorn_ot(a, b, C, eps = eps, iters = sinkhorn_iters)
            fgw_values = (C * Pi).sum((-2, -1))
        return Pi, fgw_values
    # ---- 2. reconstruct (grad_on) ---- 
    @staticmethod 
    def reconstruct_FGW(Fx, Fy_sel, Dx, Dy_sel, a, b, 
                        alpha = 0.5, eps = 0.05, 
                        outer_iters = 10, sinkhorn_iters = 30,
                        map_encoder_output: bool = False, do_log : bool = False):
        logger = logging.getLogger("my_experiment_logger")
        # --- 1. Pi, fgw_loss 계산 (기존과 동일) ---
        Pi = torch.einsum("bhn,bhk->bhnk", a, b)
        for _ in range(outer_iters):
            C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy_sel)
            C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy_sel, Pi)

            # ⬇️ --- [ ✨ 로깅 코드 수정/병합됨 ✨ ] --- ⬇️
            if _ == outer_iters -1 and do_log:
                logger.info(f"--- 🩺 FGWUtils (reconstruct_FGW) ---")
                logger.info(f"   (alpha: {alpha}, eps: {eps})")

                # ⬇️ --- [ ✨ "입력 재료" 로깅 추가됨 ✨ ] --- ⬇️
                logger.info("--- 🩺 FGW Inputs (Source) ---")
                logger.info(f"   [Fx]     mean: {Fx.mean().item():.4f}, std: {Fx.std().item():.4f}, L2_norm_sq: {(Fx**2).sum(-1).mean().item():.4f}")
                logger.info(f"   [Dx]     mean: {Dx.mean().item():.4f}, std: {Dx.std().item():.4f}, min: {Dx.min().item():.4f}, max: {Dx.max().item():.4f}")
                logger.info("--- 🩺 FGW Inputs (Target_Selected) ---")
                logger.info(f"   [Fy_sel] mean: {Fy_sel.mean().item():.4f}, std: {Fy_sel.std().item():.4f}, L2_norm_sq: {(Fy_sel**2).sum(-1).mean().item():.4f}")
                logger.info(f"   [Dy_sel] mean: {Dy_sel.mean().item():.4f}, std: {Dy_sel.std().item():.4f}, min: {Dy_sel.min().item():.4f}, max: {Dy_sel.max().item():.4f}")
                # ⬆️ --- [ ✨ 로깅 코드 끝 ✨ ] --- ⬆️

                # (Pre-Alpha) C_feat와 C_gw의 원본 통계
                logger.info("--- 🩺 FGW Cost Components (Pre-Alpha) ---")
                logger.info(f"   [C_feat] mean: {C_feat.mean().item():.4f}, std: {C_feat.std().item():.4f}, max: {C_feat.max().item():.4f}")
                logger.info(f"   [C_gw]   mean: {C_gw.mean().item():.4f}, std: {C_gw.std().item():.4f}, max: {C_gw.max().item():.4f}")
                
            C = (1 - alpha) * C_feat + alpha * C_gw
           
            Pi = FGWUtils._sinkhorn_ot(a, b, C.unsqueeze(2), eps = eps, iters = sinkhorn_iters)[:, :, 0]
        
        fgw_loss = (C * Pi).sum((-2, -1)).mean() 

        '''
            b return updates
        '''
        b_updated = Pi.sum(dim=-2)

        # --- 2. [수정] 플래그에 따라 반환값 분기 ---
        if map_encoder_output:
            # === Path (b)용: 그래디언트가 Fx로 흘러야 함 ===
            Pi_detached = Pi.detach()
            
            # 1. Fx_mapped 계산
            # [32, 1, 13, 8] * [32, 1, 13, 768] -> [32, 1, 8, 768]
            Fx_mapped = torch.einsum("bhnk,bhnd->bhkd", Pi_detached, Fx)
            
            # b_k_for_Fx shape을 [B, H, K, 1]로 만듭니다
            
            # Pi_detached.sum(dim=-2) -> [32, 1, 8] (B, H, K)
            b_k_for_Fx = Pi_detached.sum(dim=-2) 
            
            # b_k_for_Fx.unsqueeze(-1) -> [32, 1, 8, 1] (B, H, K, 1)
            b_k_for_Fx = b_k_for_Fx.unsqueeze(-1) 
            
            # [32, 1, 8, 768] / [32, 1, 8, 1] -> 브로드캐스팅 성공
            Fx_mapped = Fx_mapped / (b_k_for_Fx + 1e-8) 
            # ------------------------------------
            
            Fy_res = Fx_mapped # Fx와 그래디언트가 연결됨
            
            # 2. Dx_mapped 계산
            Dx_mapped = torch.einsum("bhnk,bhml,bhnm->bhkl", Pi_detached, Pi_detached, Dx)
            
            b_k_for_Dx = Pi_detached.sum(dim=-2, keepdim=True) # [B, H, 1, K]
            b_kl_denom = torch.einsum("bhnk,bhml->bhkl", b_k_for_Dx, b_k_for_Dx.transpose(-1, -2)) # [B, H, K, K]

            Dy_res = Dx_mapped / (b_kl_denom + 1e-8) # Dx와 그래디언트가 연결됨
            
        else:
            # === Path (a)용: 그래디언트가 Fy_sel로 흘러야 함 ===
            Pi_detached = Pi.detach()
            Fy_res = Fy_sel
            Dy_res = Dy_sel 

        return Fy_res, Dy_res, fgw_loss, b_updated, Pi_detached


    @staticmethod
    def _fused_Gromov_Wasserstein(Fx, Fy, Dx, Dy, alpha=0.5):
        """
        Compute fused cost C_fused = (1 - α)*C_feat + α*C_gw
        Fx: [B, H, N, D]
        Fy: [M, K, D]
        Dx: [B, H, N, N]
        Dy: [M, K, K]
        Return: C_fused [B, H, M, N, K]
        """
        # 1. Feature-level cost
        C_feat = FGWUtils._pairwise_feature_cost(Fx, Fy)
        # 2. Uniform initial Pi
        Pi_init = torch.ones_like(C_feat) / (C_feat.size(-2) * C_feat.size(-1))
        # 3. Structure-level cost
        C_gw = FGWUtils._Gromov_Wasserstein_cost(Dx, Dy, Pi_init)
        # 4. Combine
        C_fused = (1 - alpha) * C_feat + alpha * C_gw
        return C_fused 

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
        self.H = int(args.num_basis_heads)
        self.M = int(n_graphs)
        self.K = int(n_nodes)
        self.D = self.args.input_dim // self.args.num_basis_heads
        self.node_embeddings = nn.Parameter(torch.empty(self.M, self.K, self.D))
        nn_init.xavier_uniform_(self.node_embeddings)

        self.node_embedding_grad_stats = []
        if self.node_embeddings.requires_grad:
            self.node_embeddings.register_hook(self._save_grad_stats)
    @staticmethod 
    def _row_softmax(x, temperature = 1.0, mask=None):
        if mask is not None:
            x = x.masked_fill(mask, float("-inf"))
        t = max(temperature, 1e-6)
        return F.softmax(x / t, dim=-1)
    @staticmethod 
    def _bistochastic_sinkhorn(M, iters=20, eps=1e-6):
        M = M.clamp_min(eps)
        for _ in range(iters):
            M = M / (M.sum(dim=-1, keepdim=True) + eps)
            M = M / (M.sum(dim=-2, keepdim=True) + eps)
        return M 
                   
    def _save_grad_stats(self, grad):
        """
        [수정됨]
        self.node_embeddings.register_hook()에 의해 호출되는 콜백 함수입니다.
        M개의 LCG별로 'Abs Mean' 그래디언트를 계산하여 딕셔너리에 저장합니다.
        """
        self.lcg_grad_stats = {} # 👈 매 스텝 초기화
        if grad is not None:
            grad_data = grad.detach()
            M = grad_data.shape[0] # LCG 개수 (M=8)
            
            for m in range(M):
                # LCG m에 해당하는 그래디언트 슬라이스 [K, D]
                grad_slice = grad_data[m, :, :]
                
                # 이 LCG의 평균 그래디언트 크기
                abs_mean = grad_slice.abs().mean().item()
                
                # 딕셔너리에 LCG 인덱스(m)와 그래디언트 크기(abs_mean) 저장
                self.lcg_grad_stats[m] = abs_mean
        else:
            # 그래디언트가 없는 경우
            for m in range(self.M):
                self.lcg_grad_stats[m] = 0.0
    @staticmethod 
    def normalize_affinity(P:torch.Tensor, sym:bool=True, eps:float = 1e-8) -> torch.Tensor:
        f"""
            Args:
                P : Graph Attention Affinity 
            Return:
                Symetric Graph Attention Affinity tilde P
        """
        if sym:
            P = 0.5 * (P + P.transpose(-1,-2))
        P = P.clamp_min(eps)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        return P 
    @staticmethod 
    def affinity_to_distance(P:torch.Tensor) -> torch.Tensor:
        return (1.0 - P).clamp_min(0.0) 
    @staticmethod 
    def cosine_slot_cost(G : torch.Tensor, eps : float = 1e-8) -> torch.Tensor:
        diag = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(eps)
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2)).clamp_min(eps)
        cosG = (G / denom).clamp(-1.0, 1.0)
        M = (1.0 - cosG).clamp_min(0.0)
        M = M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
        return M 
    @staticmethod
    def cosine_slot_cost_from_U(U):
        """
        Compute cosine-based affinity matrix from node embeddings U
        Supports both:
            U [M, K, D]  (no batch/head)
            U [B, H, K, D] (with batch & head)
        Returns:
            Dy [M, K, K] or [B, H, K, K]
        """
        U_norm = F.normalize(U, p=2, dim=-1)

        if U.dim() == 3:
            # [M, K, D]
            cosine_similarity = torch.einsum("mrd,mjd->mrj", U_norm, U_norm)
        elif U.dim() == 4:
            # [B, H, K, D]
            cosine_similarity = torch.einsum("bhkd,bhjd->bhkj", U_norm, U_norm)
        else:
            raise ValueError(f"Unexpected input shape for cosine_slot_cost_from_U: {U.shape}")

        # enforce symmetry
        cosine_similarity = 0.5 * (cosine_similarity + cosine_similarity.transpose(-1, -2))
        cosine_similarity = (1.0 - cosine_similarity) / 2.0 
        return cosine_similarity

    def forward(self):
        """ 
            Returns:
                node_embeddings: [M, K, D]
                Dy : [M, K, K] cosine-based structural distances for each latent graph 
                LCG_diversifying_loss : (Scalar) Loss to encourage diversify among the latent composite graph
        """

        Dy_affinity = self.cosine_slot_cost_from_U(self.node_embeddings)
        diversify_loss = None
        if getattr(self.args, "lcg_diversifying_loss", False) is True:
            
            # --- [ ❗️ 1. (수정) M개 LCG의 "중심점" 계산 ❗️ ] ---
            # (기존 K개 노드 계산에서 M개 그래프 중심 계산으로 변경)
            
            # 1. M개 LCG의 중심점(center)을 계산합니다. [M, D]
            # (각 LCG의 K개 노드 임베딩을 평균냅니다)
            lcg_centers = self.node_embeddings.mean(dim=1) # [M, D]
            
            # 2. M개 중심점 간의 쌍별 거리(pairwise distances)를 계산합니다.
            # (M, 1, D) vs (1, M, D) -> [M, M]
            C1, C2 = lcg_centers.unsqueeze(1), lcg_centers.unsqueeze(0)
            pdist_sq = (C1 - C2).pow(2).sum(-1) # [M, M]

            # 3. lcg_hinge_margin_sq (기존 인자 재사용)
            # (M-그래프용으로는 1.0이 너무 작을 수 있으니, 
            #  --lcg_hinge_margin_sq 4.0 처럼 큰 값을 주는 것을 권장합니다)
            margin_sq = getattr(self.args, "lcg_hinge_margin_sq", 1.0) 

            # 4. Hinge Loss 
            loss_matrix = torch.clamp_min(margin_sq - pdist_sq, 0.0)

            # 5. Diagonal masking (M x M 크기로 변경)
            identity_mask = torch.eye(self.M, device = self.node_embeddings.device, dtype=torch.bool)
            loss_matrix.masked_fill_(identity_mask, 0)

            # 6. non-diagonal loss calculation (M x (M-1) 크기로 변경)
            num_pairs = self.M * (self.M - 1)
            diversify_loss = loss_matrix.sum() / (num_pairs + 1e-8)
            # --- [ ❗️ 수정 끝 ❗️ ] ---

        return self.node_embeddings, Dy_affinity, diversify_loss

class GraphQuantizer(nn.Module):
    """
        perform graph-level quantization using FGW distances. 
        Selects one latent composite graph per head and applies VQ-stype stop-gradient update. 
    """
    def __init__(self, args, alpha = 0.5, eps = 0.05, outer_iters = 20 , sinkhorn_iters = 40):
        super().__init__() 
        self.args = args 
        self.alpha = alpha 
        self.vq_beta = self.args.vq_beta 
        self.eps = eps 
        self.outer_iters = outer_iters 
        self.sinkhorn_iters = sinkhorn_iters 
        self.additional_FGW = self.args.additional_FGW
        self.eps_assign = 0.1

        # --- [2. Logger 초기화] --- 
        self.logger = logging.getLogger("my_experiment_logger")
        self.register_buffer("has_printed_initial_weights", torch.tensor(False), persistent=False)

        
    def forward(self, P_affinity : torch.Tensor, basis_outputs : torch.Tensor, latent_graph : torch.Tensor, batch : dict):
        """
            Args:
                P_affinity : [B, H, N, N]
                basis_outputs : [B, N+1, H, D/H]
                latent_graph : LatentCompositeGraph instance 
                detach_encoder : bool (True -> encoder stopgrad, False -> dictionary stopgrad)       
            Returns:
                assign_idx : [B, H] index of selected latent composite graph per head
        """
        # (0) dimension 
        B, H, N, _ = P_affinity.shape 
        
        # (1) Source graph & affinity
        Fx = basis_outputs[:, 1:, :, :].permute(0, 2, 1, 3) #[B, H, N, D/H]

        logger = logging.getLogger("my_experiment_logger")
        if not hasattr(self, 'gq_step_counter'): self.gq_step_counter = 0
        self.gq_step_counter += 1

        do_log = (self.gq_step_counter % 10 == 1)
        if do_log:
                
            logger.info(f"\n" + "="*60)
            logger.info(f"--- 📊 GraphQuantizer Inputs (Step: {self.gq_step_counter}) ---")
            logger.info("--- 🩺 (Source - 1. Raw Affinity) ---")
            logger.info(f"   [P_affinity] (Raw GAT Attn) mean: {P_affinity.mean().item():.4f}, std: {P_affinity.std().item():.4f}, min: {P_affinity.min().item():.4f}, max: {P_affinity.max().item():.4f}")

        Dx_affinity = LatentCompositeGraph.normalize_affinity(P_affinity)
        Dx = LatentCompositeGraph.affinity_to_distance(Dx_affinity)
        # (2) Latent composite graph & affinity 
        
        Fy, Dy, lcg_diversifying_loss = latent_graph()

        # (3) Uniform marginals 
        a = torch.ones(B, H, N, device = Fx.device) / N 
        b = torch.ones(B, latent_graph.M, latent_graph.K, device=Fx.device) / latent_graph.K 

        
        # reconstruct_FGW 내부의 로그와 겹칠 수 있으므로 50~100 스텝마다 한 번씩만 확인
        if do_log: 
            logger.info("--- 🩺 (Source - 2. Processed Inputs) ---")
            logger.info(f"   [Fx] (Sample)  mean: {Fx.mean().item():.4f}, std: {Fx.std().item():.4f}, L2_norm_sq: {(Fx**2).sum(-1).mean().item():.4f}")
            # "After" 통계
            logger.info(f"   [Dx] (After Norm & 1-P) mean: {Dx.mean().item():.4f}, std: {Dx.std().item():.4f}, min: {Dx.min().item():.4f}, max: {Dx.max().item():.4f}")
            logger.info("--- 🩺 (Target Codebook - All) ---")
            logger.info(f"   [Fy] (All LCG) mean: {Fy.mean().item():.4f}, std: {Fy.std().item():.4f}, L2_norm_sq: {(Fy**2).sum(-1).mean().item():.4f}")
            logger.info(f"   [Dy] (All LCG) mean: {Dy.mean().item():.4f}, std: {Dy.std().item():.4f}, min: {Dy.min().item():.4f}, max: {Dy.max().item():.4f}")
            
            # --- [ ❗️ 3. 수정된 로그: LCG별 그래디언트 값 ❗️ ] ---
            logger.info("  [📉 GRAD STATS (Per-LCG Abs Mean)]")
            if hasattr(latent_graph, 'lcg_grad_stats') and latent_graph.lcg_grad_stats:
                stats_dict = latent_graph.lcg_grad_stats
                
                dead_indices = []
                active_indices = []
                
                # M (LCG 개수)은 latent_graph에서 가져옵니다.
                M = latent_graph.M
                
                for m in range(M):
                    # 딕셔너리에서 m번째 LCG의 그래디언트 크기 가져오기
                    abs_mean = stats_dict.get(m, 0.0) 
                    
                    if abs_mean > 1e-10:
                        # ❗️ 활성화된 LCG (선택됨)
                        logger.info(f"   ✅ LCG {m} (Selected): Abs Mean = {abs_mean:.6e}")
                        active_indices.append(m)
                    else:
                        # ❗️ 죽은 LCG (선택 안 됨)
                        logger.info(f"   ❌ LCG {m} (Unused):   Abs Mean = {abs_mean:.6e}")
                        dead_indices.append(m)
                
                # 요약 로그
                if dead_indices and len(dead_indices) < M: # 일부만 죽었을 때
                     logger.warning(f"   ❗️ CODEBOOK COLLAPSE: LCGs {dead_indices} received NO gradients.")
                elif len(dead_indices) == M: # 전부 죽었을 때
                    logger.warning("   ❗️ WARNING: ALL LCGs received NO gradients.")
                else: # 전부 살아있을 때
                    logger.info("   ✅ All LCGs received gradients.")

            else:
                logger.info("   (No gradient stats from previous step yet)")
            # --- [ 수정 끝 ] ---
            
            logger.info("="*60 + "\n")
            
        # ⬆️ --- [ ✨ 로깅 코드 끝 ✨ ] --- ⬆️
        # ---- Step 1. Fused Gromov-Wasserstein distance computation for selection ---- 
        with torch.no_grad():
            Pi_all, fgw_values = FGWUtils.assign_FGW(
                Fx, Fy, Dx, Dy, a, b,
                alpha = self.alpha, eps = self.eps_assign, 
                outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters , do_log = do_log
            )
            assign_idx = torch.argmin(fgw_values, dim=-1)
        
        # ---- Step 2. Gather selected latent graphs ---- 
        Fy_sel = Fy[assign_idx] # [B, H, K, D]
        Dy_sel = Dy[assign_idx] # [B, H, K, K]
        b_sel = b[torch.arange(B).unsqueeze(1), assign_idx]

        Fy_res_sel_detached, Dy_res_sel_detached, loss_dict, _, Pi_dict = FGWUtils.reconstruct_FGW(
            Fx.detach(), Fy_sel, Dx.detach(), Dy_sel, a, b_sel, 
            alpha = self.alpha, eps = self.eps_assign, 
            outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters, map_encoder_output = False, do_log = do_log
        )

        Fy_res_sel_live, Dy_res_sel_live, loss_enc, b_sel_updated , Pi_enc = FGWUtils.reconstruct_FGW(
            Fx, Fy_sel.detach(), Dx, Dy_sel.detach(), a, b_sel, 
            alpha = self.alpha, eps = self.eps_assign, 
            outer_iters = self.outer_iters, sinkhorn_iters = self.sinkhorn_iters, map_encoder_output = True, do_log = do_log
        )

        Fy_res_sel_ste = Fy_res_sel_live + (Fy_res_sel_detached - Fy_res_sel_live).detach() 
        Dy_res_sel_ste = Dy_res_sel_live + (Dy_res_sel_detached - Dy_res_sel_live).detach()
        # --- 4. 최종 STE 출력 (디코더 입력) 상태 확인 ---
        if not hasattr(self, 'step_counter'): self.step_counter = 0
        self.step_counter += 1
        if self.step_counter % 100 == 0: # 100 스텝마다
            self.logger.info("\n" + "*"*50)
            self.logger.info(f"--- 📊 STATS COMPARE (Step: {self.step_counter}) ---")
            avg_Fx_norm_sq = (Fx.detach()**2).sum(dim=-1).mean().item()
            avg_Fy_norm_sq = (Fy.detach()**2).sum(dim=-1).mean().item() 

            self.logger.info("  [CODEBOOK (Fy)]")
            self.logger.info(f"   min: {Fy.min().item():.6f}, max: {Fy.max().item():.6f}, std: {Fy.std().item():.6f}")
            self.logger.info(f"  Avg L2 norm^2 : {avg_Fy_norm_sq:.6f}")
            self.logger.info("  [ENCODER (Fx)]")
            self.logger.info(f"   min: {Fx.min().item():.6f}, max: {Fx.max().item():.6f}, std: {Fx.std().item():.6f}")
            self.logger.info(f"  Avg L2 Norm^2: {avg_Fx_norm_sq:.6f}")
            
            self.logger.info("  [Updated LCG Marginal (b)]")
            self.logger.info(f"   mean: {b_sel_updated.mean().item():.6f}, std: {b_sel_updated.std().item():.6f} b shape : {b.shape}")
            self.logger.info(f"   (Sample 0, H 0): {b_sel_updated[0, 0, :].detach().cpu().numpy()}")
            

            B = Pi_enc.shape[0] 

            num_samples_to_log   = min(10, B)
            for i in range(num_samples_to_log):
                # Pi_enc shape이 [B, H, N, K]이므로 [i, 0]으로 접근
                Pi_sample = Pi_enc[i, 0].detach().cpu().numpy() 
                
                self.logger.info(f"--- (Sample {i}, Head 0) Pi Matrix (N x K) ---")
                self.logger.info(f"    (Shape: {Pi_sample.shape})") 
                
                # ... (개별 샘플 통계 로깅은 동일) ...
                lcg_node_importance = Pi_sample.sum(axis=0)
                self.logger.info(f"   > LCG Node Importance (Sum over N): {np.array2string(lcg_node_importance, precision=4)}")
                source_importance = Pi_sample.sum(axis=1)
                self.logger.info(f"   > Source Node Importance (Sum over K): {np.array2string(source_importance, precision=4)}")
                self.logger.info("   > Sampled Pi Matrix (N x K):")
                self.logger.info(np.array2string(Pi_sample, precision=3, max_line_width=200, suppress_small=True))
            
            self.logger.info("  [FGW 최종 성적표 (Head 0)]")
            batch_mean_distances = fgw_values[:, 0, :].mean(dim=0).cpu().numpy()
            self.logger.info(f"   Avg Distances (Batch): {np.array2string(batch_mean_distances, precision=4)}")
            self.logger.info(f"   Distances (Sample 0):  {np.array2string(fgw_values[0, 0, :].cpu().numpy(), precision=4)}")
            
            self.logger.info("  [🏆 Selected LCGs (by Source Index, Head 0)]")
           # --- [ ❗️❗️ 여기가 버그 수정 ❗️❗️ ] ---
            if 'src_idx' in batch:
                try:
                    # src_idx는 int (예: 1)
                    src_idx_int = batch['src_idx'] 
                    # assign_idx는 [B, H] (예: [32, 1])
                    B = assign_idx.shape[0]
                    
                    # src_indices를 [1, 1, 1, ..., 1] (길이 B) 배열로 생성
                    src_indices = np.full(B, src_idx_int) 
                    
                    assigned_indices_head0 = assign_idx[:, 0].cpu().numpy()
                    
                    df = pd.DataFrame({'Source_Idx': src_indices, 'LCG_Idx': assigned_indices_head0})
                    assignment_summary = df.groupby('Source_Idx')['LCG_Idx'].value_counts().unstack(fill_value=0)
                    self.logger.info(f"\n{assignment_summary.to_string()}")
                    
                except Exception as e:
                    self.logger.warning(f"   Pandas 요약 실패 ({e}).")
            # --- [ 버그 수정 끝 ] ---
            else:
                self.logger.info(f"   (Batch 'src_idx' not found) Raw (H0): {assign_idx[:, 0].cpu().numpy()}") 

            # ... (loss_dict, loss_enc, lcg_div_loss 로깅은 동일) ...
            self.logger.info("*"*50 + "\n")
        # --- [ 수정 끝 ] ---
        B, H = assign_idx.shape 
        M, K, D = Fy.shape

        
        # ---- Step 5. Merge results ----
        Fy_res_all = Fy_res_sel_ste.unsqueeze(2)
        Dy_res_all = Dy_res_sel_ste.unsqueeze(2)
        Ay_res_all = 1.0 - Dy_res_all.clamp(0.0, 1.0)
        fgw_loss = loss_dict.mean() + self.vq_beta * loss_enc.mean()
        self.logger.info(f"loss_dict: {loss_dict.item():.6f}, loss_enc: {loss_enc.item():.6f}")

        if getattr(self.args, "lcg_diversifying_loss", False) is True:
            div_alpha = getattr(self.args, "lcg_div_alpha", 10)
            fgw_loss += div_alpha * lcg_diversifying_loss
        return Fy_res_all, Ay_res_all, fgw_loss, assign_idx