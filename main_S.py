import torch
#torch.cuda.set_device(0)
#torch.use_deterministic_algorithms(False)
import os
import random, time
import argparse
import pandas as pd
import pdb, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from utils.util import setup_logger, format_time, fix_seed, prepare_results_, save_results_, wrap_up_results_, make_warmup_cosine_epochs, make_warmup_cosine_steps, current_lr, build_epoch_scheduler
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from sklearn.model_selection import StratifiedKFold
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_S import Model
from utils.coord_Kmeans import compute_coordinate_centroids_auto
#from main_G import final_test_evaluate  # few-shot 학습/테스트 루틴 사용
import psutil
from utils.visualization import visualize_model_structure
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import shutil

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=192)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--source_data', nargs='+',
                        default=['Heart_disease_statlog', 'Cardiovascular_Disease_Dataset', 'heart_target_3', 'heart_target_4'],
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart',
                                 'heart_target_1','heart_target_2','heart_target_3','heart_target_4','myocardial',
                                 'cleveland','heart_statlog','hungarian','switzerland','breast','magic_telescope',
                                 'forest_covertype_sampled','higgs_sampled','Cardiovascular_Disease_Dataset','Heart_disease_statlog'])
    parser.add_argument('--target_data', type=str, default='heart')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--source_lr_few', type=float, default=0.00001)
    parser.add_argument('--llm_model', type=str, default='gpt2_mean',
                        choices=['gpt2_mean','gpt2_auto','sentence-bert','bio-bert','bio-clinical-bert','bio-llama',
                                 'new','LLAMA_mean','LLAMA_auto'])
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--des', type=str, help='experimental memo')
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],
                        help='List of baselines to use. Leave empty to use only our model.')
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")
    parser.add_argument('--del_feat', nargs='+', default=[],
                        help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
    parser.add_argument('--use_target_head', type=bool, default=False)
    # MODELS : coord_kmeans
    parser.add_argument('--coord_softmax_temp', type=float, default=0.5, help='Coordinator softmax temperature (lower = sharper).')
    parser.add_argument('--coord_reg_lambda', type=float, default=0.2, help='Weight of KL(coord) regularizer during Few-shot.')
    parser.add_argument('--coord_target_mode', type=str, default='soft', choices=['soft', 'hard'], help='Centroid target mode for coordinate regularization.')
    parser.add_argument('--coord_tau', type=float, default=0.3,help='Temperature for soft centroid mixing (soft target).')
    # MODELS : latent Composite Graph specs
    '''
        Latent Composite Graph Configuration
    '''
    parser.add_argument("--n_graphs", type=int, default=8, help="Global slot space number M")
    parser.add_argument("--n_nodes", type = int , default = 8, help = "Global node embedding numbers")
    parser.add_argument("--graph_dim", type = int, default = 768, help = "Global node embedding dimensions")
    parser.add_argument('--fgw_alpha', type = float, default = 1)
    parser.add_argument('--alpha' , type = float, default = 0.9)
    parser.add_argument('--eps', type = float , default = 0.01)
    parser.add_argument('--reg', type = float, default = 0.01)
    parser.add_argument('--lcg_div_alpha', type = float, default = 10)
    parser.add_argument('--vq_beta', type = float, default = 0.3)
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--kl_gamma', type = float, default = 2.0)
    parser.add_argument('--additional_FGW',action = 'store_true')
    parser.add_argument('--diversifying_loss', action='store_true', help = "diversifying the latent composite graph affinity")
    parser.add_argument('--lcg_diversifying_loss', action='store_true', help = "diversifying the latent composite graph affinity")
    parser.add_argument('--lcg_hinge_margin_sq', type = float, default = 1.0)
    parser.add_argument('--lcg_strategy', type = str, default = 'round_robin', choices = ['hierarchical', 'round_robin', 'sequential'])
    parser.add_argument('--lcg_struct_type', type = str, default = 'projection', choices = ['projection', 'static', ' residual'])
    '''
        Basis GAT Configuration
    '''
    parser.add_argument('--num_shared_layers', type=int, default=2, help = "Number of SharedGAT layers")
    parser.add_argument('--num_basis_layers', type=int, default=2, help= 'Number of stacked BasisGAT layers.')
    parser.add_argument('--basis_type', type=str, choices=['mul','ind'], default = 'ind')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--edge_type', default='mlp', choices=['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default='carte', choices=['carte','carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices=['gat_v1','att','gat_v2','gate'])

    # Experiments Resampling
    parser.add_argument('--support_resamples', type=int, default=1, help='How many support resamples per seed')
    parser.add_argument('--warmup_ratio', type=float, default=0.06,
                    help='Warmup steps/epochs ratio (0~1)')
    parser.add_argument('--min_lr_mult', type=float, default=0.10,
                        help='Final LR multiplier vs. base LR for cosine annealing (e.g., 0.1 means 10% of base)')
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args

def init_lcg(args, model, loaders, device, strategy='hierarchical', injection_scale=1.0):
    import logging
    from sklearn.cluster import KMeans
    # 로거 설정 (메인에 있다면 가져옴)
    logger = logging.getLogger("my_experiment_logger")
    logger.info(f"\n{'='*20} [Bridge] LCG INIT from Pre-trained CLS {'='*20}")
    all_cls_tokens = [] 
    model.eval()
    max_samples = 50000
    logger.info(f"[LCG Init] Starts. Strategy: {strategy}, Injection: {injection_scale}")
    
    with torch.no_grad():
        for src_name, loader in loaders.items():
            logger.info(f"   - Collecting from: {src_name}")
            for i, batch in enumerate(loader):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                name_embs, val_embs = [], [] 
                if 'cat_name_embeddings' in batch: name_embs.append(batch['cat_name_embeddings'])
                if 'num_name_embeddings' in batch: name_embs.append(batch['num_name_embeddings'])
                if 'cat_value_embeddings' in batch: val_embs.append(batch['cat_value_embeddings'])
                if 'num_prompt_embeddings' in batch: val_embs.append(batch['num_prompt_embeddings'])
                if not name_embs: continue 
                name = torch.cat(name_embs, dim = 1)
                val = torch.cat(val_embs, dim = 1)
                x_basis = torch.cat([model.basis_cls.expand(val.size(0), 1, model.input_dim), val], dim = 1)
                for l in range(model.num_basis_layers):
                    norm_x = model.basis_layer_norms[l](x_basis)
                    basis_outputs, _ = model.basis_layers[l](name, norm_x)
                    x_basis = x_basis + basis_outputs.reshape(x_basis.size(0), x_basis.size(1), model.input_dim)
                cls_token = x_basis[:, 0, :].cpu().numpy() 
                all_cls_tokens.append(cls_token)
                if len(all_cls_tokens) * cls_token.shape[0] >= max_samples: break 
            if len(all_cls_tokens) * cls_token.shape[0] >= max_samples: break
    data_pool = np.concatenate(all_cls_tokens, axis = 0)
    logger.info(f">> Collected  {data_pool.shape[0]} CLS samples.")
    
    # 2. KMeans
    M = model.latent_graph.M 
    K = model.latent_graph.K
    D = model.latent_graph.D 
    kmeans = KMeans(n_clusters = M * K, n_init = 10, random_state = 42).fit(data_pool)
    centers = torch.tensor(kmeans.cluster_centers_, dtype = torch.float32)

    # 3. Stregegy Assignemtn 
    final_centroids = torch.zeros(M, K, D)
    if strategy == 'round_robin':
        final_centroids = centers.view(K, M, D).transpose(0, 1).contiguous() 
    else: 
        final_centroids = centers.view(M, K, D)
    
    # 4. Variance Injection 
    src_std = np.std(data_pool, axis=0)
    noise = torch.randn_like(final_centroids) * torch.tensor(src_std) * injection_scale
    final_init = final_centroids + noise
    
    # 5. Update Parameter
    with torch.no_grad():
        model.latent_graph.node_embeddings.data.copy_(final_init.to(device))
        
    logger.info(f">> ✅ LCG Parameters Updated. (Strategy: {strategy})")

class _DummySet:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n

class MultiSourceStepLoader:
    """
    여러 DataLoader를 받아 스텝마다 하나의 소스에서 배치를 꺼냄.
    각 배치에 'src_idx'를 주입.
    len(self)  = 모든 소스 배치 수 합
    dataset.len = 모든 소스 샘플 수 합 (평균 손실 계산용)
    """
    def __init__(self, loaders, mode='random', seed=42, src_idx=None):
        self.loaders = loaders
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.src_idx = src_idx

        self._blens = [len(dl) for dl in loaders]             # 각 로더의 배치 수
        self._steps = int(np.sum(self._blens))                # 총 스텝 수
        self._ns    = [len(dl.dataset) for dl in loaders]     # 각 로더의 샘플 수
        self._dataset = _DummySet(int(np.sum(self._ns)))      # 학습 루프에서 평균 계산용

    def __len__(self):
        return self._steps

    @property
    def dataset(self):
        return self._dataset

    def __iter__(self):
        iters = [iter(dl) for dl in self.loaders]
        pos   = [0 for _ in self.loaders]
        k = len(self.loaders)

        for s in range(self._steps):
            idx = s % k if self.mode == 'round' else int(self.rng.integers(0, k))
            if pos[idx] >= self._blens[idx]:
                iters[idx] = iter(self.loaders[idx])
                pos[idx] = 0
            batch = next(iters[idx])
            pos[idx] += 1
            src_i = self.src_idx if self.src_idx is not None else idx
            # 배치가 dict라고 가정 (prepare_embedding_dataloaders 출력과 일치)
            batch['src_idx'] = src_i
            yield batch


# ===== Variable-length time-series accumulation helpers =====
def init_accum():
    # returns a dict holding sum and count arrays
    return {'sum': None, 'cnt': None}

def accum(acc, values):
    import numpy as np
    arr = np.asarray(values, dtype=np.float32)
    if acc['sum'] is None:
        acc['sum'] = arr.copy()
        acc['cnt'] = np.ones_like(arr, dtype=np.int32)
        return acc

    # pad to same length
    max_len = max(len(acc['sum']), len(arr))
    if len(acc['sum']) < max_len:
        acc['sum'] = np.pad(acc['sum'], (0, max_len - len(acc['sum'])), constant_values=0.0)
        acc['cnt'] = np.pad(acc['cnt'], (0, max_len - len(acc['cnt'])), constant_values=0)

    if len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

    mask = np.isfinite(arr)
    acc['sum'][mask] += arr[mask]
    acc['cnt'][mask] += 1
    return acc

def finalize_mean(acc):
    import numpy as np
    if acc['sum'] is None:
        return []
    cnt = np.maximum(acc['cnt'], 1)
    return (acc['sum'] / cnt).tolist()


def make_step(loaders, mode='random', seed=42):
    return MultiSourceStepLoader(loaders, mode=mode, seed=seed)



def load_one(args, name):
    res = prepare_embedding_dataloaders(args, name)
    train_loader, val_loader, test_loader = res['loaders']
    num_classes = res['num_classes']
    return train_loader, val_loader, test_loader, num_classes


def find_optimal_threshold(y_true, y_pred):
    # y_pred: sigmoid 결과(확률)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    idx = int(np.argmax(f1s))
    return thresholds[idx] if idx < len(thresholds) else thresholds[-1]



def final_test_evaluate(model, test_loader, criterion, device, is_binary, threshold=None, mode="Full", args=None):
    """
    학습이 끝난 뒤, Test 로더에 대해 최종 성능을 측정.
    [수정 완료] Phase 2(Joint)와 Phase 3(Adaptation)에서는 Global 결과를 반환.
    """
    logger = logging.getLogger("my_experiment_logger")

    # 1. 함수 매핑 (Dual)
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    # 2. 평가 실행 (Binary / Multi-class 공통 구조화)
    # evaluate_func가 항상 ((loss_g, true, pred_g), (loss_l, true, pred_l)) 형태를 반환한다고 가정
    (loss_g, true_g, pred_g), (loss_l, true_l, pred_l) = evaluate_func(model, test_loader, criterion, device)

    # -----------------------------------------------------------
    # [핵심 수정] Global 선택 기준 로직 통일
    # -----------------------------------------------------------
    # 1. Target Adaptation (Few) 이거나
    # 2. Pretrain 중 Joint Learning (use_lcg=True) 인 경우
    # -> Global이 메인
    
    use_lcg_flag = getattr(args, 'use_lcg', False) if args is not None else False
    use_global = (mode == 'Few') or use_lcg_flag

    if use_global:
        # [Global 기준]
        test_loss = loss_g
        y_true_test = true_g
        y_pred_test = pred_g
        
        # 참고용 로그
        if is_binary:
            auc_l = roc_auc_score(true_l, pred_l)
            logger.info(f"[Test Check] Mode={mode} (Global Selected). Local AUC: {auc_l:.4f} (Ref)")
        else:
            logger.info(f"[Test Check] Mode={mode} (Global Selected). Local results ignored in log.")

    else:
        # [Local 기준] (Phase 1 Vanilla GAT)
        test_loss = loss_l
        y_true_test = true_l
        y_pred_test = pred_l
        
        # 참고용 로그
        if is_binary:
            auc_g = roc_auc_score(true_g, pred_g)
            logger.info(f"[Test Check] Mode={mode} (Local Selected). Global AUC: {auc_g:.4f} (Ref)")
        else:
            logger.info(f"[Test Check] Mode={mode} (Local Selected). Global results ignored in log.")

    # 3. Metric 계산 (선택된 y_pred_test 사용)
    if is_binary:
        test_auc = roc_auc_score(y_true_test, y_pred_test)
        
        # Threshold 적용
        if threshold is None:
            # Threshold가 안 넘어왔으면 최적 threshold 찾기 (Test set 기준이라 좀 그렇지만, fallback)
            # 보통은 학습 때 구한 threshold를 넘겨줘야 함.
            threshold = 0.5 
        
        y_pred_test_bin = (y_pred_test > threshold).astype(int)
        
        test_precision = precision_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_f1 = f1_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_acc = accuracy_score(y_true_test, y_pred_test_bin)
        
    else:
        # Multi-class Metrics
        n_classes = y_pred_test.shape[1]
        y_true_test_bin = label_binarize(y_true_test, classes=range(n_classes))
        test_auc = roc_auc_score(y_true_test_bin, y_pred_test, multi_class='ovr', average='macro')
        
        preds_argmax = y_pred_test.argmax(axis=1)
        test_precision = precision_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_recall = recall_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_f1 = f1_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_acc = accuracy_score(y_true_test, preds_argmax)

    logger.info(f"[Test Final] Mode: {mode} | Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    return test_loss, test_auc, test_precision, test_recall, test_f1, test_acc, y_true_test, y_pred_test


def train_and_validate(args, model, train_loader, val_loader,
                       criterion, optimizer, device, epochs,
                       is_binary, patience=10, mode="Full", scheduler=None, warmup_epochs=0):
    """
    Train + Validation을 진행하며, 
    - Phase 1 (Vanilla): Local 성능 기준
    - Phase 2 (Joint) & Phase 3 (Adaptation): Global 성능 기준
    으로 모니터링하고 Best Model을 저장함. (생략 없는 Full Version)
    """
    
    # --- 1. Logger Setup ---
    logger_name = "my_experiment_logger" 
    logger = logging.getLogger(logger_name)
    
    # --- 2. Metrics Storage ---
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []

    # --- 3. Function Setup ---
    train_func = binary_train if is_binary else multi_train
    # Binary/Multi 모두 Dual Output((Global_Res), (Local_Res))을 리턴한다고 가정
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    # --- 4. Init State ---
    best_val_auc = 0.0
    best_epoch = 0
    no_improve = 0
    warmup_epochs = int(warmup_epochs)
    best_threshold = 0.5
    best_model_state = None

    # --- 5. Checkpoint Directory Setup ---
    src_tag = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = (
        f"ngraphs-{args.n_graphs}"
        f"_nnodes-{args.n_nodes}"
        f"_gdim-{args.graph_dim}"
        f"_nbasis-{args.num_basis_layers}"
        f"_basis-{args.basis_type}"
        f"_attn-{args.attn_type}"
        f"_fgw_alpha-{args.fgw_alpha}"
        f"_vq_beta-{args.vq_beta}"
        f"_kl_gamma-{args.kl_gamma}"
        f"_target_data-{args.target_data}"
        f"_description-{args.des}"
    )
    checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file_path = os.path.join(checkpoint_dir, f"train_log.log")

    # File Handler Setup
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Logger Handler Check
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        logger.addHandler(file_handler)
        logger.info(f"--- Log file initialized ({mode} mode). Saving stats to: {log_file_path} ---")

    # --- 6. Training Loop ---
    print(f"\n>>> [Start {mode} Training] Total Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        
        # [Logic] 모델에게 현재 Epoch 주입 (Switching Logic용)
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch + 1
            if hasattr(model, 'switch_epoch') and model.current_epoch == model.switch_epoch:
                 logger.info(f"\n>>> [PHASE CHANGE] Epoch {model.current_epoch}: Detach OFF & Global Inference ON\n")

        # [검증] 학습 전 LCG 상태 저장
        if hasattr(model, 'latent_graph'):
            before_lcg = model.latent_graph.node_embeddings.clone().detach()

        # -------- Train Step --------
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # [검증] LCG 파라미터 변화량 체크
        if hasattr(model, 'latent_graph'):
            after_lcg = model.latent_graph.node_embeddings.detach()
            diff = (after_lcg - before_lcg).abs().mean().item()
            fgw_val = getattr(model, 'fgw_loss', torch.tensor(0.0)).item()
            
            logger.info(f"[Epoch {epoch+1}] LCG Diff: {diff:.6f} | FGW Loss: {fgw_val:.4f}")
            
            if diff == 0.0 and epoch > 0 and mode != 'Few':
                logger.warning("🚨 WARNING: LCG Parameters did NOT change! Check optimizer.")

        # Scheduler Step
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception as e:
                logger.warning(f"[Scheduler] step() failed at epoch {epoch+1}: {e}")

        # LR Logging
        try:
            curr_lr = optimizer.param_groups[0]['lr']
        except Exception:
            curr_lr = None

        # -------- Evaluate (Dual Evaluation & Criteria Switching) --------
        
        # Global 기준 저장 여부 판단 (Phase 2 or 3)
        use_global_criteria = (mode == 'Few') or getattr(args, 'use_lcg', False)

        if is_binary:
            # 1. Trainset Eval (Dual)
            (tr_loss_g, tr_true_g, tr_pred_g), (tr_loss_l, tr_true_l, tr_pred_l) = evaluate_func(model, train_loader, criterion, device)
            # 2. Validation Eval (Dual)
            (val_loss_g, val_true, val_pred_g), (val_loss_l, val_true_l, val_pred_l) = evaluate_func(model, val_loader, criterion, device)
            
            # AUC 계산
            auc_tr_g = roc_auc_score(tr_true_g, tr_pred_g)
            auc_val_g = roc_auc_score(val_true, val_pred_g)
            
            auc_tr_l = roc_auc_score(tr_true_l, tr_pred_l)
            auc_val_l = roc_auc_score(val_true_l, val_pred_l)
            
            if use_global_criteria:
                # [Global 기준]
                monitor_auc = auc_val_g
                current_threshold = find_optimal_threshold(val_true, val_pred_g)
                
                # Accuracy (Global)
                y_pred_val_bin = (val_pred_g > current_threshold).astype(int)
                val_acc = accuracy_score(val_true, y_pred_val_bin)
                
                main_val_loss = val_loss_g
                logger_prefix = f"[{mode}/Global Best]"
                
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Global] Loss: {tr_loss_g:.4f} / {val_loss_g:.4f} \n"
                    f"   >>> [Global] AUC : {auc_tr_g:.4f} / {auc_val_g:.4f} | ACC: {val_acc:.4f} | Thr: {current_threshold:.4f}\n"
                    f"   >>> [Local ] Ref AUC: {auc_val_l:.4f}"
                )
                
                # 히스토리 (Global)
                train_auc_cur = auc_tr_g
                train_acc_cur = accuracy_score(tr_true_g, (tr_pred_g > current_threshold).astype(int))

            else:
                # [Local 기준]
                monitor_auc = auc_val_l
                current_threshold = find_optimal_threshold(val_true_l, val_pred_l)
                
                # Accuracy (Local)
                y_pred_val_bin = (val_pred_l > current_threshold).astype(int)
                val_acc = accuracy_score(val_true_l, y_pred_val_bin)
                
                main_val_loss = val_loss_l
                logger_prefix = "[Pretrain/Local Best]"
                
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Local ] Loss: {tr_loss_l:.4f} / {val_loss_l:.4f} \n"
                    f"   >>> [Local ] AUC : {auc_tr_l:.4f} / {auc_val_l:.4f} | ACC: {val_acc:.4f} | Thr: {current_threshold:.4f}\n"
                    f"   >>> [Global] Ref AUC: {auc_val_g:.4f}"
                )
                
                # 히스토리 (Local)
                train_auc_cur = auc_tr_l
                train_acc_cur = accuracy_score(tr_true_l, (tr_pred_l > current_threshold).astype(int))
        else:
            # [Multi-class]
            # multi_evaluate도 Dual Output ((Loss, True, Pred), (Loss, True, Pred))을 리턴한다고 가정
            (tr_loss_g, tr_true_g, tr_pred_g), (tr_loss_l, tr_true_l, tr_pred_l) = evaluate_func(model, train_loader, criterion, device)
            (val_loss_g, val_true, val_pred_g), (val_loss_l, val_true_l, val_pred_l) = evaluate_func(model, val_loader, criterion, device)

            n_cls = val_pred_g.shape[1]
            
            # --- AUC (Global) ---
            y_bin_val_g = label_binarize(val_true, classes=range(n_cls))
            auc_val_g = roc_auc_score(y_bin_val_g, val_pred_g, multi_class='ovr', average='macro')
            
            # --- AUC (Local) ---
            y_bin_val_l = label_binarize(val_true_l, classes=range(n_cls))
            auc_val_l = roc_auc_score(y_bin_val_l, val_pred_l, multi_class='ovr', average='macro')

            if use_global_criteria:
                # [Global 기준]
                monitor_auc = auc_val_g
                preds_val = val_pred_g.argmax(axis=1)
                val_acc = accuracy_score(val_true, preds_val)
                main_val_loss = val_loss_g
                logger_prefix = f"[{mode}/Global Best]"
                
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Global] Loss: {tr_loss_g:.4f} / {val_loss_g:.4f} \n"
                    f"   >>> [Global] AUC : {auc_val_g:.4f} | ACC: {val_acc:.4f}\n"
                    f"   >>> [Local ] Ref AUC: {auc_val_l:.4f}"
                )
                train_auc_cur = 0.0 # Train AUC 계산 생략 시
                train_acc_cur = 0.0
            else:
                # [Local 기준]
                monitor_auc = auc_val_l
                preds_val = val_pred_l.argmax(axis=1)
                val_acc = accuracy_score(val_true_l, preds_val)
                main_val_loss = val_loss_l
                logger_prefix = "[Pretrain/Local Best]"
                
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Local ] Loss: {tr_loss_l:.4f} / {val_loss_l:.4f} \n"
                    f"   >>> [Local ] AUC : {auc_val_l:.4f} | ACC: {val_acc:.4f}\n"
                    f"   >>> [Global] Ref AUC: {auc_val_g:.4f}"
                )
                train_auc_cur = 0.0
                train_acc_cur = 0.0
            
            current_threshold = None

        # --- 공통 저장 로직 ---
        val_losses.append(main_val_loss)
        train_aucs.append(train_auc_cur); val_aucs.append(monitor_auc)
        train_accs.append(train_acc_cur); val_accs.append(val_acc)
        
        logger.info(log_msg)

        # --- 7. Visualization Snapshot ---
        if (epoch + 1) % 10 == 0 or (epoch == 0):
            try:
                if isinstance(train_loader, dict):
                    temp_loaders = train_loader
                else:
                    source_key = list(args.source_data)[0] if isinstance(args.source_data, (list, tuple)) else args.source_data
                    temp_loaders = {source_key: train_loader}

                if hasattr(model, 'latent_graph'):
                    print(f"\n>>> Generating Snapshot for Epoch {epoch+1}...")
                    from utils.visualization import visualize_training_snapshot_v2
                    visualize_training_snapshot_v2(
                        model, temp_loaders, 
                        model.latent_graph, model.latent_graph, 
                        epoch+1, 0,
                        save_dir=checkpoint_dir
                    )
            except Exception as e:
                logger.warning(f"[Visualization Skipped] {e}")

        # --- 8. Best Model Saving (monitor_auc 기준) ---
        if monitor_auc > best_val_auc:
            best_val_auc = monitor_auc
            best_epoch   = epoch
            no_improve   = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"{logger_prefix} New Best AUC: {best_val_auc:.4f} at epoch {epoch+1}")
            
            if current_threshold is not None:
                best_threshold = current_threshold

            ckpt_path = os.path.join(
                checkpoint_dir,
                f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}_S:{args.random_seed}_{experiment_id}.pt"
            )
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': best_val_auc,
                # 나중에 로드해서 확인할 때 헷갈리지 않게 Global 점수도 별도 키로 저장
                'val_auc_global': auc_val_g if is_binary else auc_val_g,
                'threshold': best_threshold,
                'args': args
            }, ckpt_path)
        else:
            if epoch + 1 > warmup_epochs:
                no_improve += 1 
            else: 
                no_improve = 0 
        
        if (epoch + 1 > warmup_epochs) and (no_improve >= patience):
            logger.info(f"[{mode}] Early stopping at epoch {epoch+1} (no improve {patience} epochs)")
            break

    # --- 9. Finish ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning(f"[{mode}] No best_model_state saved; model not updated.")

    return (train_losses, val_losses,
            train_aucs, val_aucs,
            train_precisions, val_precisions,
            train_recalls, val_recalls,
            train_f1s, val_f1s,
            train_accs, val_accs,
            best_epoch, best_val_auc, best_threshold)

# -----------------------------
# 멀티 소스 프리트레인 (per-source patience) + 소스별 테스트 → 평균
# -----------------------------
def pretrain_and_eval_sources(args, model, device, sources, patience=10):
    import shutil
    logger_name = "my_experiment_logger"
    logger = logging.getLogger(logger_name)

    name_to_idx = {name: i for i, name in enumerate(sources)}
    trains, vals, tests, ncs = [], [], [], []
    for name in sources:
        tr, va, te, nc = load_one(args, name)
        trains.append(tr); vals.append(va); tests.append(te); ncs.append(nc)

    if len(set(ncs)) != 1:
        raise ValueError(f"num_classes mismatch across sources: {ncs}")
    args.num_classes = ncs[0]
    args.output_dim  = args.num_classes if args.num_classes > 2 else 1

    # 학습은 다중 소스를 섞어서
    tr_step = make_step(trains, mode='random', seed=args.random_seed)

    # 검증/테스트/개별-학습 평가용
    val_steps   = [MultiSourceStepLoader([vals[i]],   mode='round', seed=args.random_seed, src_idx=i) for i in range(len(vals))]
    test_steps  = [MultiSourceStepLoader([tests[i]],  mode='round', seed=args.random_seed, src_idx=i) for i in range(len(tests))]
    train_steps = [MultiSourceStepLoader([trains[i]], mode='round', seed=args.random_seed, src_idx=i) for i in range(len(trains))]

    is_bin = (args.num_classes == 2)
    crit   = nn.BCEWithLogitsLoss() if is_bin else nn.CrossEntropyLoss()
    
    base_params = [
        p for name, p in model.named_parameters()
        if "latent_graph" not in name and "gnn_experts" not in name and p.requires_grad
    ]
    lcg_params = [
        p for name, p in model.named_parameters()
        if ("latent_graph" in name or "gnn_experts" in name) and p.requires_grad
    ]
    
    gat_lr = args.source_lr * 0.1 
    global_lr = args.source_lr    

    logger.info(f"--- 🚀 [Phase 2] Applying Differential LR ---")
    logger.info(f"   GAT LR: {gat_lr} (Slow)")
    logger.info(f"   Global LR: {global_lr} (Fast)")
    
    opt = optim.Adam(
        [
            {'params': base_params, 'lr': gat_lr}, 
            {'params': lcg_params,  'lr': global_lr} 
        ], 
        weight_decay=1e-5
    )
    
    total_epochs = int(args.train_epochs)
    if total_epochs > 0:
        warmup_epochs = max(1, int(args.warmup_ratio * total_epochs))
        scheduler_ep  = make_warmup_cosine_epochs(opt, total_epochs, warmup_epochs, args.min_lr_mult)
        logger.info(f"[Pretrain] LR schedule: warmup_epochs={warmup_epochs}, final_mult={args.min_lr_mult}")
    else:
        scheduler_ep = None
        logger.info("[Pretrain] Eval-only run. Skipping training.")

    eval_fn  = binary_evaluate if is_bin else multi_evaluate
    train_fn = binary_train    if is_bin else multi_train

    # [수정 1] Mean AUC 기준을 위한 초기화
    best_mean_auc = -1.0 
    no_improve = 0
    best_state = None
    last_best_epoch = -1

    # === 체크포인트 설정 ===
    src_tag   = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = (f"ngraphs-{args.n_graphs}_nnodes-{args.n_nodes}_gdim-{args.graph_dim}_nbasis-{args.num_basis_layers}"
                 f"_basis-{args.basis_type}_attn-{args.attn_type}_fgw_alpha-{args.fgw_alpha}_vq_beta-{args.vq_beta}"
                 f"_kl_gamma-{args.kl_gamma}_target_data-{args.target_data}_description-{args.des}")
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_latest = os.path.join(ckpt_dir, "best.pt")
    ckpt_hist   = os.path.join(ckpt_dir, f"best_{experiment_id}.pt")
    
    log_file_path = os.path.join(ckpt_dir, f"train_log_{experiment_id}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        logger.addHandler(file_handler)
        logger.info(f"--- Log file initialized. Saving to: {log_file_path} ---")

    # === 학습 루프 ===
    for epoch in range(total_epochs):
        _ = train_fn(model, tr_step, crit, opt, device)
        if scheduler_ep is not None: scheduler_ep.step()
        
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch + 1 
            print(model.current_epoch)
            if model.current_epoch == model.switch_epoch:
                logger.info(f">>> [PHASE CHANGE] Epoch {model.current_epoch}: Global Inference ON")
        
        aucs_local = []
        aucs_global = []
        
        for vl in val_steps:
            res_g, res_l = eval_fn(model, vl, crit, device)
            _, y_true_l, y_pred_l = res_l
            _, y_true_g, y_pred_g = res_g
            
            if is_bin:
                score_l = roc_auc_score(y_true_l, y_pred_l)
                score_g = roc_auc_score(y_true_g, y_pred_g)
            else:
                n_cls = y_pred_l.shape[1]
                y_bin_l = label_binarize(y_true_l, classes=range(n_cls))
                y_bin_g = label_binarize(y_true_g, classes=range(n_cls))
                score_l = roc_auc_score(y_bin_l, y_pred_l, multi_class='ovr', average='macro')
                score_g = roc_auc_score(y_bin_g, y_pred_g, multi_class='ovr', average='macro')
            
            aucs_local.append(score_l)
            aucs_global.append(score_g)

        # === [수정 2] Best 갱신 로직 (Mean 기준) ===
        improved = False
        target_aucs = aucs_global if getattr(args, 'use_lcg', False) else aucs_local 
        
        current_lcg_status = getattr(args, 'use_lcg', False)
        print(f"\n[DEBUG CHECK][Epoch {epoch+1}] args.use_lcg: {current_lcg_status} -> Watching: {'Global (LCG)' if current_lcg_status else 'Local (GAT)'}")
        # 현재 평균 계산
        current_mean_auc = float(np.mean(target_aucs))
        
        # 평균이 기존 최고 평균보다 높으면 저장
        if current_mean_auc > best_mean_auc:
            best_mean_auc = current_mean_auc
            improved = True

        # === Logging ===
        mean_auc_l = float(np.mean(aucs_local))
        mean_auc_g = float(np.mean(aucs_global))
        
        log_msg = (
            f"[Pre][Epoch {epoch+1}/{total_epochs}]\n"
            f"   >>> Local (GAT): Mean AUC {mean_auc_l:.4f} | Per-Source: {['%.4f'%x for x in aucs_local]}\n"
            f"   >>> Global(LCG): Mean AUC {mean_auc_g:.4f} | Per-Source: {['%.4f'%x for x in aucs_global]}"
        )
        logger.info(log_msg)
        
        if improved:
            best_state = model.state_dict()
            last_best_epoch = epoch
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                # 저장되는 메타데이터도 Mean 값으로 기록
                'val_auc_mean': best_mean_auc, 
                'val_aucs_per_source': target_aucs,
                'args': args
            }, ckpt_latest)
            try:
                shutil.copyfile(ckpt_latest, ckpt_hist)
            except Exception as e:
                logger.warning(f"History copy failed: {e}")
        else:
            if epoch + 1 > warmup_epochs: 
                no_improve += 1 
            else: 
                no_improve = 0 
            if no_improve >= patience : 
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------------
    # 최종 리포트 (소스별 threshold 산출 -> train/val/test 지표)
    # -----------------------------
    per_train_loss, per_val_loss, per_test_loss = [], [], []
    per_train_auc,  per_val_auc,  per_test_auc  = [], [], []
    per_train_precision, per_val_precision, per_test_precision = [], [], []
    per_train_recall,    per_val_recall,    per_test_recall    = [], [], []
    per_train_f1,        per_val_f1,        per_test_f1        = [], [], []
    per_train_acc,       per_val_acc,       per_test_acc       = [], [], []

    all_y_true_full_list = []
    all_y_pred_full_list = []

    for i in range(len(sources)):
        # <--- [확인] Global 결과 우선 (res_g)
        (val_loss_i, y_true_val_i, y_pred_val_i), _ = eval_fn(model, val_steps[i], crit, device)
        
        if is_bin:
            thr_i = find_optimal_threshold(y_true_val_i, y_pred_val_i)
        else:
            thr_i = None

        # Train
        (train_loss_i, y_true_tr_i, y_pred_tr_i), _ = eval_fn(model, train_steps[i], crit, device)
        
        if is_bin:
            train_auc_i = roc_auc_score(y_true_tr_i, y_pred_tr_i)
            y_bin_tr = (y_pred_tr_i > thr_i).astype(int)
            train_precision_i = precision_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_recall_i    = recall_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_f1_i        = f1_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_acc_i       = accuracy_score(y_true_tr_i, y_bin_tr)
        else:
            n_cls = y_pred_tr_i.shape[1]
            y_bin_tr = label_binarize(y_true_tr_i, classes=range(n_cls))
            train_auc_i = roc_auc_score(y_bin_tr, y_pred_tr_i, multi_class='ovr', average='macro')
            preds_tr = y_pred_tr_i.argmax(axis=1)
            train_precision_i = precision_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_recall_i    = recall_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_f1_i        = f1_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_acc_i       = accuracy_score(y_true_tr_i, preds_tr)

        # Val
        if is_bin:
            val_auc_i = roc_auc_score(y_true_val_i, y_pred_val_i)
            y_bin_val = (y_pred_val_i > thr_i).astype(int)
            val_precision_i = precision_score(y_true_val_i, y_bin_val, zero_division=0)
            val_recall_i    = recall_score(y_true_val_i, y_bin_val, zero_division=0)
            val_f1_i        = f1_score(y_true_val_i, y_bin_val, zero_division=0)
            val_acc_i       = accuracy_score(y_true_val_i, y_bin_val)
        else:
            n_cls = y_pred_val_i.shape[1]
            y_bin_val = label_binarize(y_true_val_i, classes=range(n_cls))
            val_auc_i = roc_auc_score(y_bin_val, y_pred_val_i, multi_class='ovr', average='macro')
            preds_val = y_pred_val_i.argmax(axis=1)
            val_precision_i = precision_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_recall_i    = recall_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_f1_i        = f1_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_acc_i       = accuracy_score(y_true_val_i, preds_val)

        # Test
        (test_loss_i, y_true_te_i, y_pred_te_i), _ = eval_fn(model, test_steps[i], crit, device)
        
        if is_bin:
            test_auc_i = roc_auc_score(y_true_te_i, y_pred_te_i)
            y_bin_te = (y_pred_te_i > thr_i).astype(int)
            test_precision_i = precision_score(y_true_te_i, y_bin_te, zero_division=0)
            test_recall_i    = recall_score(y_true_te_i, y_bin_te, zero_division=0)
            test_f1_i        = f1_score(y_true_te_i, y_bin_te, zero_division=0)
            test_acc_i       = accuracy_score(y_true_te_i, y_bin_te)
        else:
            n_cls = y_pred_te_i.shape[1]
            y_bin_te = label_binarize(y_true_te_i, classes=range(n_cls))
            test_auc_i = roc_auc_score(y_bin_te, y_pred_te_i, multi_class='ovr', average='macro')
            preds_te = y_pred_te_i.argmax(axis=1)
            test_precision_i = precision_score(y_true_te_i, preds_te, average='macro', zero_division=0)
            test_recall_i    = recall_score(y_true_te_i, preds_te, average='macro', zero_division=0)
            test_f1_i        = f1_score(y_true_te_i, preds_te, average='macro', zero_division=0)
            test_acc_i       = accuracy_score(y_true_te_i, preds_te)

        # 누적
        per_train_loss.append(train_loss_i); per_val_loss.append(val_loss_i); per_test_loss.append(test_loss_i)
        per_train_auc.append(train_auc_i);   per_val_auc.append(val_auc_i);   per_test_auc.append(test_auc_i)
        per_train_precision.append(train_precision_i); per_val_precision.append(val_precision_i); per_test_precision.append(test_precision_i)
        per_train_recall.append(train_recall_i);       per_val_recall.append(val_recall_i);       per_test_recall.append(test_recall_i)
        per_train_f1.append(train_f1_i);               per_val_f1.append(val_f1_i);               per_test_f1.append(test_f1_i)
        per_train_acc.append(train_acc_i);             per_val_acc.append(val_acc_i);             per_test_acc.append(test_acc_i)

        all_y_true_full_list.append(y_true_te_i)
        all_y_pred_full_list.append(y_pred_te_i)

    # 평균 집계
    train_losses_full = [float(np.mean(per_train_loss))]
    val_losses_full   = [float(np.mean(per_val_loss))]
    test_losses_full  = [float(np.mean(per_test_loss))]
    train_aucs_full = [float(np.mean(per_train_auc))]
    val_aucs_full   = [float(np.mean(per_val_auc))]
    test_auc_full   = float(np.mean(per_test_auc))
    train_precisions_full = [float(np.mean(per_train_precision))]
    val_precisions_full   = [float(np.mean(per_val_precision))]
    test_precision_full   = float(np.mean(per_test_precision))
    train_recalls_full = [float(np.mean(per_train_recall))]
    val_recalls_full   = [float(np.mean(per_val_recall))]
    test_recall_full   = float(np.mean(per_test_recall))
    train_f1s_full = [float(np.mean(per_train_f1))]
    val_f1s_full   = [float(np.mean(per_val_f1))]
    test_f1_full   = float(np.mean(per_test_f1))
    train_accs_full = [float(np.mean(per_train_acc))]
    val_accs_full   = [float(np.mean(per_val_acc))]
    test_acc_full   = float(np.mean(per_test_acc))
    all_y_true_full = np.concatenate(all_y_true_full_list, axis=0)
    all_y_pred_full = np.concatenate(all_y_pred_full_list, axis=0)
    best_epoch_full = last_best_epoch

    full_pack = dict(
        train_losses_full=train_losses_full,
        val_losses_full=val_losses_full,
        test_losses_full=test_losses_full,
        train_aucs_full=train_aucs_full,
        val_aucs_full=val_aucs_full,
        test_auc_full=test_auc_full,
        train_precisions_full=train_precisions_full,
        val_precisions_full=val_precisions_full,
        test_precision_full=test_precision_full,
        train_recalls_full=train_recalls_full,
        val_recalls_full=val_recalls_full,
        test_recall_full=test_recall_full,
        train_f1s_full=train_f1s_full,
        val_f1s_full=val_f1s_full,
        test_f1_full=test_f1_full,
        train_accs_full=train_accs_full,
        val_accs_full=val_accs_full,
        test_acc_full=test_acc_full,
        all_y_true_full=all_y_true_full,
        all_y_pred_full=all_y_pred_full,
        best_epoch_full=best_epoch_full
    )
    return full_pack
def find_pretrain_ckpt(ckpt_dir: str):
    stable = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(stable):
        return stable
    try:
        cands = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                 if f.startswith("best_") and f.endswith(".pt")]
        return max(cands, key=os.path.getmtime) if cands else None
    except FileNotFoundError:
        return None

def main():
    start_time = time.time()
    args = get_args()

    fix_seed(args.random_seed)
    # 1. "전용 로거"를 가져옵니다.
    logger_name = "my_experiment_logger" 
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False # "복도"로 소리가 새어나가지 않게 함

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    logger.info("--- 💡 Global logger initialized (Console) 💡 ---")
    try:
        ncpu = os.cpu_count() or 1
        p.cpu_affinity(range(1, min(ncpu, 64)))
    except Exception as e:
        logger.warning(f"cpu_affinity not set: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Starting experiment with Multiple-Source : {args.source_data}")
    logger.info(f"Device: {device}")
    logger.info("Preparing Tabular datasets...")

    # 1) 모델 생성
    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.dropout_rate, args.llm_model,
                       experiment_id, mode="Full").to(device)
    model_few  = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.dropout_rate, args.llm_model,
                       experiment_id, mode="Few").to(device)

    # 2) 프리트레인 체크포인트 로드 시도 (고정 best.pt 우선, 없으면 최신 best_*.pt)
    src_tag = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = (
        f"ngraphs-{args.n_graphs}"
        f"_nnodes-{args.n_nodes}"
        f"_gdim-{args.graph_dim}"
        f"_nbasis-{args.num_basis_layers}"
        f"_basis-{args.basis_type}"
        f"_attn-{args.attn_type}"
        f"_fgw_alpha-{args.fgw_alpha}"
        f"_vq_beta-{args.vq_beta}"
        f"_kl_gamma-{args.kl_gamma}"
        f"_target_data-{args.target_data}"
        f"_description-{args.des}"
    )
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    os.makedirs(ckpt_dir, exist_ok = True)
    ckpt_final = os.path.join(ckpt_dir, "best_joint.pt")
    ckpt_vanilla = os.path.join(ckpt_dir, "best_vanilla.pt")
    old_best = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(old_best) and not os.path.exists(ckpt_final):
        shutil.copy(old_best, ckpt_final)
    loaded_pretrain = False
    full_metrics = None

    
    # ==================================================================
    # [Logic] 2-Stage Pre-training Pipeline
    # ==================================================================
    
    if os.path.exists(ckpt_final):
        # Case A: 이미 최종 학습(Phase 2) 완료됨 -> 로드
        logger.info(f"✅ [Pretrain] Found Final Checkpoint: {ckpt_final}")
        ckpt = torch.load(ckpt_final, map_location=device)
        model_full.load_state_dict(ckpt['model_state_dict'])
        loaded_pretrain = True
        
    else:
        # Case B: 학습 필요
        logger.info(f"🚀 [Pretrain] Starting 2-Stage Training Pipeline...")
        
        # Loaders 준비 (Init용)
        all_loaders = {}
        src_list = args.source_data if isinstance(args.source_data, (list, tuple)) else [args.source_data]
        for s in src_list:
            tr, _, _, _ = load_one(args, s)
            all_loaders[s] = tr

        # --- [Step 1] Phase 1: Vanilla GAT ---
        if os.path.exists(ckpt_vanilla):
            logger.info(f"   -> Found Phase 1 Checkpoint. Loading...")
            ckpt = torch.load(ckpt_vanilla, map_location=device)
            model_full.load_state_dict(ckpt['model_state_dict'])
        else:
            logger.info(f"\n{'='*40}\n>>> [Phase 1] Start Vanilla GAT Training (LCG OFF)\n{'='*40}")
            args.use_lcg = False 
            # 학습 실행 (내부에서 best.pt 생성됨)
            _ = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=10)
            
            # 결과 백업
            shutil.copy(os.path.join(ckpt_dir, "best.pt"), ckpt_vanilla)
            logger.info(f"   -> Phase 1 Saved to {ckpt_vanilla}")

        # --- [Step 2] Bridge: LCG Init ---
        logger.info(f"\n{'='*40}\n>>> [Bridge] Initializing LCG from Pre-trained CLS\n{'='*40}")
        init_lcg(
            args, model_full, all_loaders, device, 
            strategy=args.lcg_strategy, injection_scale=0.1
        )

        # --- [Step 3] Phase 2: Joint Training ---
        logger.info(f"\n{'='*40}\n>>> [Phase 2] Start Joint Training (Global ON)\n{'='*40}")
        args.use_lcg = True 
        # 이어서 학습 (Fine-tuning)
        full_metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=10)
        
        # 최종 저장
        shutil.copy(os.path.join(ckpt_dir, "best.pt"), ckpt_final)
        logger.info(f"   -> Phase 2 Saved to {ckpt_final}")
        loaded_pretrain = True # 학습 완료

    # 3) (옵션) 4-shot일 때, 로드된 모델로 소스 리포트만 재평가(eval-only)
    if loaded_pretrain and (args.few_shot == 4 or args.few_shot ==0):
        logger.info("[Full] Using loaded pretrain for source metrics report (eval only).")
        _bak = args.train_epochs
        args.train_epochs = 0
        args.use_lcg = True
        full_metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=0)
        args.train_epochs = _bak
    else:
        full_metrics = None
    # 4) few-shot 적응: 가중치 복사 → freeze 정책 적용

    
    args.use_target_head = True
    args.use_lcg = True 
    model_few.args.use_lcg = True
    model_few.load_state_dict(model_full.state_dict(),strict=False)
    
    logger.info(f"[Few-shot] target = {args.target_data}")
    r_t = prepare_embedding_dataloaders(args, args.target_data)
    train_loader_t, val_loader_t, test_loader_t = r_t['loaders']
    num_classes_t = r_t['num_classes']
    args.num_classes = num_classes_t
    args.output_dim  = num_classes_t if num_classes_t > 2 else 1

    is_binary_t = (args.num_classes == 2)
    crit_t = nn.BCEWithLogitsLoss() if is_binary_t else nn.CrossEntropyLoss()

    if args.few_shot == 0:
        logger.info("\n>>> [Zero-shot] Evaluating pretrained model directly on target test set...")
        
        evaluate_func = binary_evaluate if is_binary_t else multi_evaluate
        
        res_val = evaluate_func(model_full, val_loader_t, crit_t, device)
        
        if isinstance(res_val, tuple) and len(res_val) == 2:
            res_g_val, res_l_val = res_val
            _, y_true_val, y_pred_val = res_g_val # Global 선택
        else:
            _, y_true_val, y_pred_val = res_val
        
        if is_binary_t:
            best_threshold_zero = find_optimal_threshold(y_true_val, y_pred_val)
        else:
            best_threshold_zero = None
        
        # Test set 평가
        # [핵심] mode='Full' (학습된 ghead 사용), args=args (Global 점수 리턴)
        (test_loss_zero, test_auc_zero, test_precision_zero, test_recall_zero, 
         test_f1_zero, test_acc_zero, all_y_true_zero, all_y_pred_zero
        ) = final_test_evaluate(
            model_full, test_loader_t, crit_t, device, is_binary_t, 
            threshold=best_threshold_zero, mode='Full', args=args
        )
        
        logger.info(f"[Zero-shot] Test Results: "
                   f"AUC={test_auc_zero:.4f} ACC={test_acc_zero:.4f} "
                   f"Prec={test_precision_zero:.4f} Rec={test_recall_zero:.4f} F1={test_f1_zero:.4f}")
        
        # 결과 래핑 및 저장 후 종료
        if full_metrics is not None:
            full_ours_results = wrap_up_results_(
                train_losses=full_metrics['train_losses_full'],
                val_losses=full_metrics['val_losses_full'],
                test_losses=full_metrics['test_losses_full'],
                train_aucs=full_metrics['train_aucs_full'],
                val_aucs=full_metrics['val_aucs_full'],
                test_aucs=[full_metrics['test_auc_full']],
                train_precisions=full_metrics['train_precisions_full'],
                val_precisions=full_metrics['val_precisions_full'],
                test_precisions=[full_metrics['test_precision_full']],
                train_recalls=full_metrics['train_recalls_full'],
                val_recalls=full_metrics['val_recalls_full'],
                test_recalls=[full_metrics['test_recall_full']],
                train_f1s=full_metrics['train_f1s_full'],
                val_f1s=full_metrics['val_f1s_full'],
                test_f1s=[full_metrics['test_f1_full']],
                all_y_true=[full_metrics['all_y_true_full']],
                all_y_pred=[full_metrics['all_y_pred_full']],
                best_epoch=full_metrics['best_epoch_full'],
                best_ours_auc=full_metrics['test_auc_full'],
                best_ours_acc=full_metrics['test_acc_full'],
                best_ours_precision=full_metrics['test_precision_full'],
                best_ours_recall=full_metrics['test_recall_full'],
                best_ours_f1=full_metrics['test_f1_full'],
                train_accs=full_metrics['train_accs_full'],
                val_accs=full_metrics['val_accs_full'],
                test_accs=[full_metrics['test_acc_full']]
            )
        else:
            full_ours_results = None
        
        zero_shot_results = wrap_up_results_(
            train_losses=[], val_losses=[], test_losses=[],
            train_aucs=[], val_aucs=[], test_aucs=[test_auc_zero],
            train_precisions=[], val_precisions=[], test_precisions=[test_precision_zero],
            train_recalls=[], val_recalls=[], test_recalls=[test_recall_zero],
            train_f1s=[], val_f1s=[], test_f1s=[test_f1_zero],
            all_y_true=[all_y_true_zero], all_y_pred=[all_y_pred_zero],
            best_epoch=0, best_ours_auc=test_auc_zero, best_ours_acc=test_acc_zero,
            best_ours_precision=test_precision_zero, best_ours_recall=test_recall_zero,
            best_ours_f1=test_f1_zero,
            train_accs=[], val_accs=[], test_accs=[test_acc_zero]
        )
        
        results = prepare_results_(full_ours_results, zero_shot_results)
        
        logger.info("Saving Zero-shot results...")
        import copy
        args_for_save = copy.deepcopy(args)
        if isinstance(args_for_save.source_data, (list, tuple)):
            args_for_save.source_data = "+".join(map(str, args_for_save.source_data))
        else:
            args_for_save.source_data = str(args_for_save.source_data)

        save_results_(args_for_save, results)
        logger.info("Results saved")
        return # 종료!

    model_few.set_freeze_target()

    trainables = [n for n, p in model_few.named_parameters() if p.requires_grad]
    logger.info("Few-shot trainable params:\n" + "\n".join(trainables))
    
    R = int(getattr(args, 'support_resamples', 1))
    logger.info(f"[Few-shot] support resamples R = {R}")

    base_state_cpu = {k: v.cpu() for k, v in model_full.state_dict().items()}

    acc = {
        'train_losses':  init_accum(),
        'val_losses':    init_accum(),
        'train_aucs':    init_accum(),
        'val_aucs':      init_accum(),
        'train_precs':   init_accum(),
        'val_precs':     init_accum(),
        'train_recalls': init_accum(),
        'val_recalls':   init_accum(),
        'train_f1s':     init_accum(),
        'val_f1s':       init_accum(),
        'train_accs':    init_accum(),
        'val_accs':      init_accum(),
    }
    ep_best_epochs = []
    ep_test_metrics = []  # (loss, auc, prec, rec, f1, acc)
    y_true_last, y_pred_last = None, None

    import numpy as _np

    for r in range(R):
        model_few.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()}, strict=False)
        
        model_few.set_freeze_target() 

        gat_params_few = []
        global_params_few = []
        
        for name, p in model_few.named_parameters():
            if not p.requires_grad: continue
            if 'basis' in name: # GAT
                gat_params_few.append(p)
            else: # Global (LCG, Expert, Head)
                global_params_few.append(p)
        gat_lr_few = args.source_lr_few * 0.1 
        global_lr_few = args.source_lr_few #* 50.0 # 0.0005
        
        logger.info(f"[Few-shot][Ep {r+1}] GAT LR: {gat_lr_few} | Global LR: {global_lr_few}")

        optimizer_few = optim.Adam(
            [
                {'params': gat_params_few,    'lr': gat_lr_few},
                {'params': global_params_few, 'lr': global_lr_few}
            ],
            weight_decay=3e-5
        )
        warmup_epochs_few = max(1, int(args.warmup_ratio * args.train_epochs))
        #warmup_epochs_few = 0 
        scheduler_few = make_warmup_cosine_epochs(
            optimizer_few,
            total_epochs=args.train_epochs,
            warmup_epochs=warmup_epochs_few,
            min_lr_mult=args.min_lr_mult
        )
        logger.info(f"[Few-shot][Ep {r+1}] LR schedule: warmup_epochs={warmup_epochs_few}, final_mult={args.min_lr_mult}")

        # ---- support 재샘플: reproducible하게 시드만 살짝 변경 ----
        seed_bak = args.random_seed
        args.random_seed = seed_bak + (r + 1)
        fix_seed(args.random_seed)

        if args.few_shot > 0:
            train_loader_epi = get_few_shot_embedding_samples(train_loader_t, args)
        else:
            train_loader_epi = train_loader_t

        # 시드 복원
        args.random_seed = seed_bak
        fix_seed(args.random_seed)

        # ---- few-shot 학습/검증 ----
        (train_losses_few, val_losses_few,
         train_aucs_few,   val_aucs_few,
         train_precisions_few, val_precisions_few,
         train_recalls_few,    val_recalls_few,
         train_f1s_few,        val_f1s_few,
         train_accs_few,       val_accs_few,
         best_epoch_few, best_val_auc_few, best_threshold_few
        ) = train_and_validate(args, model_few, train_loader_epi, val_loader_t, crit_t,
                               optimizer_few, device, args.train_epochs, is_binary_t, patience=50, mode="Few", scheduler=scheduler_few, warmup_epochs=warmup_epochs_few)

        # ---- 테스트 ----
        (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
         test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
            model_few, test_loader_t, crit_t, device, is_binary_t, threshold=best_threshold_few, mode='Few', args = args
        )

        logger.info(f"[Few-shot][Ep {r+1}/{R}] AUC={test_auc_few:.4f} ACC={test_acc_few:.4f} "
                    f"Prec={test_precision_few:.4f} Rec={test_recall_few:.4f} F1={test_f1_few:.4f}")

        # ---- 누적 합 (가변 길이 안전) ----
        acc['train_losses']  = accum(acc['train_losses'],  train_losses_few)
        acc['val_losses']    = accum(acc['val_losses'],    val_losses_few)
        acc['train_aucs']    = accum(acc['train_aucs'],    train_aucs_few)
        acc['val_aucs']      = accum(acc['val_aucs'],      val_aucs_few)
        acc['train_precs']   = accum(acc['train_precs'],   train_precisions_few)
        acc['val_precs']     = accum(acc['val_precs'],     val_precisions_few)
        acc['train_recalls'] = accum(acc['train_recalls'], train_recalls_few)
        acc['val_recalls']   = accum(acc['val_recalls'],   val_recalls_few)
        acc['train_f1s']     = accum(acc['train_f1s'],     train_f1s_few)
        acc['val_f1s']       = accum(acc['val_f1s'],       val_f1s_few)
        acc['train_accs']    = accum(acc['train_accs'],    train_accs_few)
        acc['val_accs']      = accum(acc['val_accs'],      val_accs_few)

        ep_best_epochs.append(best_epoch_few)
        ep_test_metrics.append((test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few, test_acc_few))

        y_true_last, y_pred_last = all_y_true_few, all_y_pred_few  # 마지막 에피소드 저장

    # ---- 에피소드 평균(시드 내부 평균) ----
    train_losses_few_mean = finalize_mean(acc['train_losses'])
    val_losses_few_mean   = finalize_mean(acc['val_losses'])
    train_aucs_few_mean   = finalize_mean(acc['train_aucs'])
    val_aucs_few_mean     = finalize_mean(acc['val_aucs'])
    train_precs_few_mean  = finalize_mean(acc['train_precs'])
    val_precs_few_mean    = finalize_mean(acc['val_precs'])
    train_recalls_few_mean= finalize_mean(acc['train_recalls'])
    val_recalls_few_mean  = finalize_mean(acc['val_recalls'])
    train_f1s_few_mean    = finalize_mean(acc['train_f1s'])
    val_f1s_few_mean      = finalize_mean(acc['val_f1s'])
    train_accs_few_mean   = finalize_mean(acc['train_accs'])
    val_accs_few_mean     = finalize_mean(acc['val_accs'])

    ep_arr = _np.asarray(ep_test_metrics, dtype=_np.float32)  # [R, 6]
    mean_test_loss, mean_test_auc, mean_test_prec, mean_test_rec, mean_test_f1, mean_test_acc = ep_arr.mean(axis=0).tolist()
    Rf = float(max(len(ep_best_epochs), 1))
    best_epoch_few_mean = int(round(sum(ep_best_epochs) / Rf))

    # 8) 결과 래핑 (소스 리포트는 4샷일 때만 존재)
    if full_metrics is not None:
        full_ours_results = wrap_up_results_(
            train_losses=full_metrics['train_losses_full'],
            val_losses=full_metrics['val_losses_full'],
            test_losses=full_metrics['test_losses_full'],
            train_aucs=full_metrics['train_aucs_full'],
            val_aucs=full_metrics['val_aucs_full'],
            test_aucs=[full_metrics['test_auc_full']],
            train_precisions=full_metrics['train_precisions_full'],
            val_precisions=full_metrics['val_precisions_full'],
            test_precisions=[full_metrics['test_precision_full']],
            train_recalls=full_metrics['train_recalls_full'],
            val_recalls=full_metrics['val_recalls_full'],
            test_recalls=[full_metrics['test_recall_full']],
            train_f1s=full_metrics['train_f1s_full'],
            val_f1s=full_metrics['val_f1s_full'],
            test_f1s=[full_metrics['test_f1_full']],
            all_y_true=[full_metrics['all_y_true_full']],
            all_y_pred=[full_metrics['all_y_pred_full']],
            best_epoch=full_metrics['best_epoch_full'],
            best_ours_auc=full_metrics['test_auc_full'],
            best_ours_acc=full_metrics['test_acc_full'],
            best_ours_precision=full_metrics['test_precision_full'],
            best_ours_recall=full_metrics['test_recall_full'],
            best_ours_f1=full_metrics['test_f1_full'],
            train_accs=full_metrics['train_accs_full'],
            val_accs=full_metrics['val_accs_full'],
            test_accs=[full_metrics['test_acc_full']]
        )
    else:
        full_ours_results = None

    few_ours_results = wrap_up_results_(
        train_losses_few_mean, val_losses_few_mean, [],
        train_aucs_few_mean,   val_aucs_few_mean,   [mean_test_auc],
        train_precs_few_mean,  val_precs_few_mean,  [mean_test_prec],
        train_recalls_few_mean,val_recalls_few_mean,[mean_test_rec],
        train_f1s_few_mean,    val_f1s_few_mean,    [mean_test_f1],
        [y_true_last], [y_pred_last],
        best_epoch_few_mean, mean_test_auc, mean_test_acc,
        mean_test_prec, mean_test_rec, mean_test_f1,
        train_accs=train_accs_few_mean, val_accs=val_accs_few_mean, test_accs=[mean_test_acc]
    )

    results = prepare_results_(full_ours_results, few_ours_results)

    # 9) 저장
    logger.info("Saving results...")
    import copy
    args_for_save = copy.deepcopy(args)
    if isinstance(args_for_save.source_data, (list, tuple)):
        args_for_save.source_data = "+".join(map(str, args_for_save.source_data))
    else:
        args_for_save.source_data = str(args_for_save.source_data)

    save_results_(args_for_save, results)
    logger.info("Results saved")
    logger.info(f"Total experiment time: {format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()