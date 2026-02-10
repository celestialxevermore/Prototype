
#torch.cuda.set_device(0)
#torch.use_deterministic_algorithms(False)
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch
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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from utils.util import setup_logger, format_time, fix_seed, prepare_results_, save_results_, wrap_up_results_, make_warmup_cosine_epochs, make_warmup_cosine_steps, current_lr, build_epoch_scheduler
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from sklearn.model_selection import StratifiedKFold
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_S import Model
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
import wandb 

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--run_tag', type=str, default="", help="Run identifier to avoid checkpoint collisions (e.g., 20251215_132001)")
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=192)
    parser.add_argument('--struct_hidden_dim', type=int, default = 192)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--source_data', nargs='+',
                        default=[ 'Medicaldataset','Cardiovascular_Disease_Dataset', 'Heart_disease_statlog','Erbil_Cardiovascular_Health_Dataset', 'cardio_SAheart', 'heart_failure_clinical_records'],
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart',
                                 'heart_target_1','heart_target_2','heart_target_3','heart_target_4','myocardial',
                                 'cleveland','heart_statlog','hungarian','switzerland','breast','magic_telescope',
                                 'forest_covertype_sampled','higgs_sampled','Cardiovascular_Disease_Dataset','Heart_disease_statlog','Medicaldataset', 'heart_failure_clinical_records','cardio_SAheart', 'Erbil_Cardiovascular_Health_Dataset'])
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
    parser.add_argument('--tau', type = float, default=0.5)
    parser.add_argument('--soft_tau', type = float, default=0.01)
    parser.add_argument('--entropy_reg', type = float, default = 0.01)
    parser.add_argument('--lcg_div_alpha', type = float, default = 10)
    parser.add_argument('--vq_beta', type = float, default = 0.3)
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--kl_gamma', type = float, default = 2.0)
    parser.add_argument('--additional_FGW',action = 'store_true')
    parser.add_argument('--diversifying_loss', action='store_true', help = "diversifying the latent composite graph affinity")
    parser.add_argument('--lcg_diversifying_loss', action='store_true', help = "diversifying the latent composite graph affinity")
    parser.add_argument('--lcg_hinge_margin_sq', type = float, default = 1.0)
    parser.add_argument('--lcg_strategy', type = str, default = 'hierarchical', choices = ['hierarchical', 'round_robin', 'sequential', 'balanced_hierarchical'])
    parser.add_argument('--lcg_struct_type', type = str, default = 'static', choices = ['projection', 'static', ' residual'])
    parser.add_argument('--feat_distance', type = str, default = 'cosine', choices=['cosine','l2'])
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

WANDB_KEYS = [
    "alpha", "tau", "soft_tau", "vq_beta",
    "source_lr", "source_lr_few", "dropout_rate"
]
def wandb_make_serializable_config(args):
    """args -> wandb.config에 안전하게 들어가도록 직렬화 가능한 dict로 변환"""
    cfg = {}
    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            cfg[k] = v
        elif isinstance(v, (list, tuple)):
            cfg[k] = list(v)
        else:
            cfg[k] = str(v)
    return cfg

def wandb_init_and_override_args(args):
    """
    - wandb.init(config=vars(args))로 전체 args를 "초기 기록"은 하되,
    - sweep(agent)로 넘어온 wandb.config 값은 WANDB_KEYS만 args에 덮어쓴다.
    """
    try:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "ProtoLLM-Sweep"),
            entity=os.getenv("WANDB_ENTITY", None),
            config=vars(args),
            name=os.getenv("WANDB_RUN_NAME", None),
        )
    except Exception as e:
        print(f"[wandb] init skipped: {e}")
        return None

    cfg = wandb.config

    # ✅ sweep 값 주입: WANDB_KEYS만!
    for k in WANDB_KEYS:
        if k in cfg:
            try:
                setattr(args, k, cfg[k])
            except Exception:
                pass

    # ✅ wandb.config에도 "WANDB_KEYS만" 다시 반영 (UI에서 보기 좋게)
    try:
        wandb.config.update(
            {k: getattr(args, k) for k in WANDB_KEYS if hasattr(args, k)},
            allow_val_change=True
        )
    except Exception:
        pass

    return run

def wandb_safe_log(d, step=None):
    if wandb.run is None:
        return
    try:
        if step is None:
            wandb.log(d)
        else:
            wandb.log(d, step=step)
    except Exception:
        pass

def wandb_safe_summary_set(d):
    if wandb.run is None:
        return
    try:
        for k, v in d.items():
            wandb.run.summary[k] = v
    except Exception:
        pass


def wandb_update_config_minimal(args):
    """
    wandb.config를 WANDB_KEYS만 업데이트.
    (args 전체를 다 올리지 않음)
    """
    try:
        import wandb
        if wandb.run is None:
            return
        cfg = {k: getattr(args, k) for k in WANDB_KEYS if hasattr(args, k)}
        wandb.config.update(cfg, allow_val_change=True)
    except Exception:
        pass



def init_lcg(args, model, loaders, device, save_dir, strategy='hierarchical', injection_scale=1.0):
    import logging
    from sklearn.cluster import KMeans

    logger = logging.getLogger("my_experiment_logger")
    logger.info(f"\n{'='*20} [Bridge] LCG INIT from Pre-trained CLS {'='*20}")

    temp_seed = args.random_seed 
    random.seed(temp_seed)
    np.random.seed(temp_seed)
    torch.manual_seed(temp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(temp_seed)
    logger.info(f">>> [LCG Init] Seed reset to {temp_seed} for deterministic data sampling.")
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
    
    M = model.latent_graph.M 
    K = model.latent_graph.K
    D = model.latent_graph.D 

    # =========================================================================
    # Strategy Assignment
    # =========================================================================
    if strategy == 'hierarchical':
        logger.info(f">> [Hierarchical] Step 1: Global KMeans with M={M} clusters")
        km_global = KMeans(n_clusters=M, n_init=10, random_state=args.random_seed).fit(data_pool)
        
        final_centroids = np.zeros((M, K, D))
        for m in range(M):
            group = data_pool[km_global.labels_ == m]
            logger.info(f"   LCG {m}: {len(group)} samples in global cluster")
            if len(group) < K:
                supp = data_pool[np.random.choice(len(data_pool), K - len(group))]
                group = np.concatenate([group, supp], axis=0)
            km_local = KMeans(n_clusters=K, n_init=10, random_state=args.random_seed).fit(group)
            final_centroids[m] = km_local.cluster_centers_
        
        final_centroids = torch.tensor(final_centroids, dtype=torch.float32)

    elif strategy == 'balanced_hierarchical':
        logger.info(f">> [Balanced Hierarchical] Step 1: Global KMeans with M={M} clusters")
        km_global = KMeans(n_clusters=M, n_init=10, random_state=args.random_seed).fit(data_pool)
        
        # Balanced assignment: 각 클러스터 최대 크기 제한
        max_per_cluster = int(np.ceil(len(data_pool) / M * 1.3))  # 평균의 1.3배까지 허용
        min_per_cluster = max(K, int(len(data_pool) / M * 0.5))   # 최소 K개 또는 평균의 0.5배
        
        # 모든 샘플에서 각 centroid까지 거리
        centroids_global = km_global.cluster_centers_  # [M, D]
        dists = np.linalg.norm(data_pool[:, None, :] - centroids_global[None, :, :], axis=2)  # [N, M]
        
        # 선호도 순서로 greedy assignment
        assignments = -np.ones(len(data_pool), dtype=int)
        cluster_counts = np.zeros(M, dtype=int)
        
        # 각 샘플의 "확신도" (1등과 2등 거리 차이)가 큰 순서대로 배정
        sorted_dists = np.sort(dists, axis=1)
        confidence = sorted_dists[:, 1] - sorted_dists[:, 0]  # 2등-1등 gap
        order = np.argsort(-confidence)  # 확신이 큰 것부터
        
        for idx in order:
            prefs = np.argsort(dists[idx])  # 가까운 순서
            for pref in prefs:
                if cluster_counts[pref] < max_per_cluster:
                    assignments[idx] = pref
                    cluster_counts[pref] += 1
                    break
        
        logger.info(f">> [Balanced] Cluster sizes: {cluster_counts.tolist()} "
                     f"(target: {len(data_pool)//M}, max: {max_per_cluster})")
        
        # Step 2: 각 balanced cluster 안에서 local KMeans
        final_centroids = np.zeros((M, K, D))
        for m in range(M):
            group = data_pool[assignments == m]
            logger.info(f"   LCG {m}: {len(group)} samples (balanced)")
            if len(group) < K:
                supp = data_pool[np.random.choice(len(data_pool), K - len(group))]
                group = np.concatenate([group, supp], axis=0)
            km_local = KMeans(n_clusters=K, n_init=10, random_state=args.random_seed).fit(group)
            final_centroids[m] = km_local.cluster_centers_
        
        final_centroids = torch.tensor(final_centroids, dtype=torch.float32)
        
    elif strategy == 'round_robin':
        kmeans = KMeans(n_clusters=M * K, n_init=10, random_state=args.random_seed).fit(data_pool)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        final_centroids = centers.view(K, M, D).transpose(0, 1).contiguous()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'hierarchical', 'balanced_hierarchical', or 'round_robin'.")

    # =========================================================================
    # Update node_embeddings
    # =========================================================================
    with torch.no_grad():
        model.latent_graph.node_embeddings.data.copy_(final_centroids.to(device))
        
    logger.info(f">> ✅ LCG Parameters Updated. (Strategy: {strategy})")

    # =========================================================================
    # Initialize adj_param from node embedding distances
    # =========================================================================
    if model.latent_graph.struct_mode == 'static':
        with torch.no_grad():
            node_emb = model.latent_graph.node_embeddings.data
            dist = torch.cdist(node_emb, node_emb, p=2) ** 2
            for m in range(M):
                q90 = torch.quantile(dist[m].flatten(), 0.9).clamp_min(1e-8)
                dist[m] = (dist[m] / q90).clamp_max(1.0)
            
            target = (1.0 - dist).clamp(0.01, 0.99)
            adj_init = torch.log(target / (1.0 - target))
            model.latent_graph.adj_param.data.copy_(adj_init.to(device))
        
        logger.info(f">> ✅ adj_param initialized from node embedding distances.")
        with torch.no_grad():
            ct = 1.0 - torch.sigmoid(model.latent_graph.adj_param.data)
            for m in range(M):
                logger.info(f"   LCG {m}: CT mean={ct[m].mean():.4f}, std={ct[m].std():.4f}")

    # =========================================================================
    # Save initial state for dead code reset
    # =========================================================================
    model.latent_graph.register_buffer(
        'init_node_embeddings', 
        model.latent_graph.node_embeddings.data.clone()
    )
    if model.latent_graph.struct_mode == 'static':
        model.latent_graph.register_buffer(
            'init_adj_param',
            model.latent_graph.adj_param.data.clone()
        )
    logger.info(f">> ✅ Initial LCG state saved for dead code reset")

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

def compute_auprc(y_true, y_pred) -> float:
    # binary: (N,) 또는 (N,1)
    if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
        return float(average_precision_score(y_true, y_pred.reshape(-1)))

    # multi-class: (N,C)
    n_cls = y_pred.shape[1]
    y_bin = label_binarize(y_true, classes=range(n_cls))
    return float(average_precision_score(y_bin, y_pred, average='macro'))



def final_test_evaluate(model, test_loader, criterion, device, is_binary, threshold=None, mode="Full", args=None):
    """
    학습이 끝난 뒤, Test 로더에 대해 최종 성능을 측정.
    [수정 완료] Phase 2(Joint)와 Phase 3(Adaptation)에서는 Global 결과를 반환.
    """
    #pdb.set_trace()
    logger = logging.getLogger("my_experiment_logger")
    from sklearn.metrics import average_precision_score
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
    
    test_auprc = float('nan')
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
        test_auprc = average_precision_score(y_true_test, y_pred_test)

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
        test_auprc = average_precision_score(y_true_test_bin, y_pred_test, average='macro')

    logger.info(f"[Test Final] Mode: {mode} | Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, ACC: {test_acc:.4f}, "
            f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    return test_loss, test_auc, test_auprc, test_precision, test_recall, test_f1, test_acc, y_true_test, y_pred_test

def train_and_validate(args, model, train_loader, val_loader,
                       criterion, optimizer, device, epochs,
                       is_binary, patience=50, mode="Full", scheduler=None, warmup_epochs=0):

    logger_name = "my_experiment_logger"
    logger = logging.getLogger(logger_name)
    from sklearn.metrics import average_precision_score
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    train_auprcs, val_auprcs = [], []
    train_func = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    best_val_auc = 0.0
    best_epoch = 0
    no_improve = 0
    warmup_epochs = int(warmup_epochs)
    best_threshold = 0.5
    best_model_state = None

    src_tag = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = (
        f"ngraphs-{args.n_graphs}"
        f"_nnodes-{args.n_nodes}"
        f"_gdim-{args.graph_dim}"
        f"_nbasis-{args.num_basis_layers}"
        f"_basis-{args.basis_type}"
        f"_attn-{args.attn_type}"
        f"_struct_hidden_dim-{args.struct_hidden_dim}"
        f"_fgw_alpha-{args.fgw_alpha}"
        f"_alpha-{args.alpha}"
        f"_vq_beta-{args.vq_beta}"
        f"_kl_gamma-{args.kl_gamma}"
        f"_tau-{args.tau}"
        f"_target_data-{args.target_data}"
        f"_entropic_reg-{args.entropy_reg}"
        f"_description-{args.des}"
    )
    checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}"
    #checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}/{args.run_tag}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file_path = os.path.join(checkpoint_dir, f"train_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        logger.addHandler(file_handler)
        logger.info(f"--- Log file initialized ({mode} mode). Saving stats to: {log_file_path} ---")

    # ✅ wandb: 시작 시점에 WANDB_KEYS만 기록
    try:
        hp_dict = {f"hp/{k}": getattr(args, k, None) for k in WANDB_KEYS}
        wandb_safe_log({
            "Mode/name": mode,
            "Mode/is_binary": 1 if is_binary else 0,
            "Train/epochs": int(epochs),
            "Train/patience": int(patience),
            "Train/warmup_epochs": int(warmup_epochs),
            **hp_dict
        })
    except Exception:
        pass

    print(f"\n>>> [Start {mode} Training] Total Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()

        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch + 1
            if hasattr(model, 'switch_epoch') and model.current_epoch == model.switch_epoch:
                logger.info(f"\n>>> [PHASE CHANGE] Epoch {model.current_epoch}: Detach OFF & Global Inference ON\n")
                try:
                    wandb_safe_log({
                        f"{mode}/phase_change": 1,
                        f"{mode}/switch_epoch": int(getattr(model, "switch_epoch", -1)),
                    }, step=epoch + 1)
                except Exception:
                    pass

        if hasattr(model, 'latent_graph'):
            before_lcg = model.latent_graph.node_embeddings.clone().detach()

        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        diff = 0.0
        fgw_val = 0.0
        if hasattr(model, 'latent_graph'):
            after_lcg = model.latent_graph.node_embeddings.detach()
            diff = (after_lcg - before_lcg).abs().mean().item()
            fgw_val = getattr(model, 'fgw_loss', torch.tensor(0.0)).item()
            logger.info(f"[Epoch {epoch+1}] LCG Diff: {diff:.6f} | FGW Loss: {fgw_val:.4f}")
            if diff == 0.0 and epoch > 0 and mode != 'Few':
                logger.warning("🚨 WARNING: LCG Parameters did NOT change! Check optimizer.")

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception as e:
                logger.warning(f"[Scheduler] step() failed at epoch {epoch+1}: {e}")

        try:
            curr_lr = optimizer.param_groups[0]['lr']
        except Exception:
            curr_lr = None

        use_global_criteria = (mode == 'Few') or getattr(args, 'use_lcg', False)
        
        monitor_auprc = 0.0 
        train_auprc_cur = 0.0 
        if is_binary:
            (tr_loss_g, tr_true_g, tr_pred_g), (tr_loss_l, tr_true_l, tr_pred_l) = evaluate_func(model, train_loader, criterion, device)
            (val_loss_g, val_true,   val_pred_g), (val_loss_l, val_true_l, val_pred_l) = evaluate_func(model, val_loader, criterion, device)

            auc_tr_g  = roc_auc_score(tr_true_g, tr_pred_g)
            auc_val_g = roc_auc_score(val_true,  val_pred_g)
            auc_tr_l  = roc_auc_score(tr_true_l, tr_pred_l)
            auc_val_l = roc_auc_score(val_true_l,val_pred_l)

            auprc_tr_g  = average_precision_score(np.array(tr_true_g).reshape(-1),  np.array(tr_pred_g).reshape(-1))
            auprc_val_g = average_precision_score(np.array(val_true).reshape(-1),   np.array(val_pred_g).reshape(-1))
            auprc_tr_l  = average_precision_score(np.array(tr_true_l).reshape(-1),  np.array(tr_pred_l).reshape(-1))
            auprc_val_l = average_precision_score(np.array(val_true_l).reshape(-1), np.array(val_pred_l).reshape(-1))

            if use_global_criteria:
                monitor_auc = auc_val_g
                monitor_auprc = auprc_val_g 
                train_auprc_cur = auprc_tr_g 
                current_threshold = find_optimal_threshold(val_true, val_pred_g)
                y_pred_val_bin = (val_pred_g > current_threshold).astype(int)
                val_acc = accuracy_score(val_true, y_pred_val_bin)
                main_val_loss = val_loss_g
                logger_prefix = f"[{mode}/Global Best]"
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Global] Loss : {tr_loss_g:.4f} / {val_loss_g:.4f} \n"
                    f"   >>> [Global] AUC  : {auc_tr_g:.4f} / {auc_val_g:.4f} | ACC: {val_acc:.4f} | Thr: {current_threshold:.4f}\n"
                    f"   >>> [Global] AUPRC: {auprc_tr_g:.4f} / {auprc_val_g:.4f}\n"
                    f"   >>> [Local ] Ref AUC: {auc_val_l:.4f} | Ref AUPRC: {auprc_val_l:.4f}"
                )
                train_auc_cur = auc_tr_g
                train_acc_cur = accuracy_score(tr_true_g, (tr_pred_g > current_threshold).astype(int))
            else:
                monitor_auc = auc_val_l
                monitor_auprc = auprc_val_l 
                train_auprc_cur = auprc_tr_l 
                current_threshold = find_optimal_threshold(val_true_l, val_pred_l)
                y_pred_val_bin = (val_pred_l > current_threshold).astype(int)
                val_acc = accuracy_score(val_true_l, y_pred_val_bin)
                main_val_loss = val_loss_l
                logger_prefix = "[Pretrain/Local Best]"
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Local ] Loss : {tr_loss_l:.4f} / {val_loss_l:.4f} \n"
                    f"   >>> [Local ] AUC  : {auc_tr_l:.4f} / {auc_val_l:.4f} | ACC: {val_acc:.4f} | Thr: {current_threshold:.4f}\n"
                    f"   >>> [Local ] AUPRC: {auprc_tr_l:.4f} / {auprc_val_l:.4f}\n"
                    f"   >>> [Global] Ref AUC: {auc_val_g:.4f} | Ref AUPRC: {auprc_val_g:.4f}"
                )
                train_auc_cur = auc_tr_l
                train_acc_cur = accuracy_score(tr_true_l, (tr_pred_l > current_threshold).astype(int))
        else:
            (tr_loss_g, tr_true_g, tr_pred_g), (tr_loss_l, tr_true_l, tr_pred_l) = evaluate_func(model, train_loader, criterion, device)
            (val_loss_g, val_true,   val_pred_g), (val_loss_l, val_true_l, val_pred_l) = evaluate_func(model, val_loader, criterion, device)

            n_cls = val_pred_g.shape[1]
            y_bin_val_g = label_binarize(val_true, classes=range(n_cls))
            y_bin_val_l = label_binarize(val_true_l, classes=range(n_cls))
            auc_val_g = roc_auc_score(y_bin_val_g, val_pred_g, multi_class='ovr', average='macro')
            auc_val_l = roc_auc_score(y_bin_val_l, val_pred_l, multi_class='ovr', average='macro')

            y_bin_tr_g = label_binarize(tr_true_g, classes=range(n_cls))
            y_bin_tr_l = label_binarize(tr_true_l, classes=range(n_cls))
            auprc_tr_g  = average_precision_score(y_bin_tr_g, tr_pred_g, average='macro')
            auprc_val_g = average_precision_score(y_bin_val_g, val_pred_g, average='macro')
            auprc_tr_l  = average_precision_score(y_bin_tr_l, tr_pred_l, average='macro')
            auprc_val_l = average_precision_score(y_bin_val_l, val_pred_l, average='macro')



            if use_global_criteria:
                monitor_auc = auc_val_g
                monitor_auprc = auprc_val_g 
                train_auprc_cur = auprc_tr_g 

                preds_val = val_pred_g.argmax(axis=1)
                val_acc = accuracy_score(val_true, preds_val)
                main_val_loss = val_loss_g
                logger_prefix = f"[{mode}/Global Best]"
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Global] Loss : {tr_loss_g:.4f} / {val_loss_g:.4f} \n"
                    f"   >>> [Global] AUC  : {auc_val_g:.4f} | ACC: {val_acc:.4f}\n"
                    f"   >>> [Global] AUPRC: {auprc_val_g:.4f}\n"
                    f"   >>> [Local ] Ref AUC: {auc_val_l:.4f} | Ref AUPRC: {auprc_val_l:.4f}"
                )
            else:
                monitor_auc = auc_val_l
                monitor_auprc = auprc_val_l 
                train_auprc_cur = auprc_tr_l 
                preds_val = val_pred_l.argmax(axis=1)
                val_acc = accuracy_score(val_true_l, preds_val)
                main_val_loss = val_loss_l
                logger_prefix = "[Pretrain/Local Best]"
                log_msg = (
                    f"[{mode}][Ep {epoch+1}/{epochs}] LR: {curr_lr:.1e}\n"
                    f"   >>> [Local ] Loss : {tr_loss_l:.4f} / {val_loss_l:.4f} \n"
                    f"   >>> [Local ] AUC  : {auc_val_l:.4f} | ACC: {val_acc:.4f}\n"
                    f"   >>> [Local ] AUPRC: {auprc_val_l:.4f}\n"
                    f"   >>> [Global] Ref AUC: {auc_val_g:.4f} | Ref AUPRC: {auprc_val_g:.4f}"
                )
            current_threshold = None
            train_auc_cur = 0.0
            train_acc_cur = 0.0

        val_losses.append(main_val_loss)
        train_aucs.append(train_auc_cur); val_aucs.append(monitor_auc)
        train_accs.append(train_acc_cur); val_accs.append(val_acc)
        train_auprcs.append(train_auprc_cur); val_auprcs.append(monitor_auprc)
        logger.info(log_msg)

        # ✅ wandb: epoch 로그 (hp는 WANDB_KEYS만)
        try:
            log_dict = {
                f"{mode}/epoch": epoch + 1,
                f"{mode}/train_loss": float(train_loss),
                f"{mode}/val_loss_main": float(main_val_loss),
                f"{mode}/monitor_auc": float(monitor_auc),
                f"{mode}/val_acc": float(val_acc),
                f"{mode}/use_global_criteria": 1 if use_global_criteria else 0,
                f"{mode}/lr": curr_lr,
                f"{mode}/best_val_auc_so_far": float(best_val_auc),
                f"{mode}/no_improve": int(no_improve),
            }

            if hasattr(model, 'latent_graph'):
                log_dict.update({
                    f"{mode}/lcg_diff": float(diff),
                    f"{mode}/fgw_loss": float(fgw_val),
                })

            if is_binary:
                log_dict.update({
                    f"{mode}/auc_val_g": float(auc_val_g),
                    f"{mode}/auc_val_l": float(auc_val_l),
                    f"{mode}/auprc_val_g": float(auprc_val_g),
                    f"{mode}/auprc_val_l": float(auprc_val_l),
                })
                if current_threshold is not None:
                    log_dict[f"{mode}/threshold"] = float(current_threshold)
            else:
                log_dict.update({
                    f"{mode}/auc_val_g": float(auc_val_g),
                    f"{mode}/auc_val_l": float(auc_val_l),
                    f"{mode}/auprc_val_g": float(auprc_val_g),
                    f"{mode}/auprc_val_l": float(auprc_val_l),
                })

            # hp는 "WANDB_KEYS만" (매 epoch 찍고 싶으면 유지)
            log_dict.update({f"hp/{k}": getattr(args, k, None) for k in WANDB_KEYS})

            wandb_safe_log(log_dict, step=epoch + 1)
        except Exception:
            pass

        if monitor_auc > best_val_auc:
            best_val_auc = monitor_auc
            best_epoch = epoch
            no_improve = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"{logger_prefix} New Best AUC: {best_val_auc:.4f} at epoch {epoch+1}")

            if is_binary and (current_threshold is not None):
                best_threshold = current_threshold

            ckpt_path = os.path.join(
                checkpoint_dir,
                f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}_S:{args.random_seed}_{experiment_id}.pt"
            )

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': best_val_auc,
                'val_auprc': float(monitor_auprc),
                'threshold': best_threshold,
                'args': args
            }, ckpt_path)

            try:
                wandb_safe_log({
                    f"{mode}/best_update": 1,
                    f"{mode}/best_epoch": int(best_epoch) + 1,
                    f"{mode}/best_val_auc": float(best_val_auc),
                    f"{mode}/best_val_auprc_at_best_auc": float(monitor_auprc),
                    f"{mode}/best_threshold": float(best_threshold) if best_threshold is not None else None,
                }, step=epoch + 1)
                if wandb.run is not None:
                    wandb.run.summary[f"{mode}/best_val_auc"] = float(best_val_auc)
                    wandb.run.summary[f"{mode}/best_epoch"] = int(best_epoch) + 1
                    wandb.run.summary[f"{mode}/best_val_auprc_at_best_auc"] = float(monitor_auprc)
            except Exception:
                pass
        else:
            if epoch + 1 > warmup_epochs:
                no_improve += 1
            else:
                no_improve = 0

        if (epoch + 1 > warmup_epochs) and (no_improve >= patience):
            logger.info(f"[{mode}] Early stopping at epoch {epoch+1} (no improve {patience} epochs)")
            try:
                wandb_safe_log({
                    f"{mode}/early_stop": 1,
                    f"{mode}/early_stop_epoch": epoch + 1,
                    f"{mode}/best_val_auc": float(best_val_auc),
                    f"{mode}/best_epoch": int(best_epoch) + 1,
                    f"{mode}/best_val_auprc_at_best_auc": float(monitor_auprc),
                }, step=epoch + 1)
            except Exception:
                pass
            break

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
def pretrain_and_eval_sources(args, model, device, sources, patience=20):
    import shutil
    logger_name = "my_experiment_logger"
    logger = logging.getLogger(logger_name)
    #pdb.set_trace()
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
    

    from sklearn.metrics import average_precision_score  

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
    model_sig = (
        f"ngraphs-{args.n_graphs}"
        f"_nnodes-{args.n_nodes}"
        f"_gdim-{args.graph_dim}"
        f"_nbasis-{args.num_basis_layers}"
        f"_basis-{args.basis_type}"
        f"_attn-{args.attn_type}"
        f"_struct_hidden_dim-{args.struct_hidden_dim}"
        f"_fgw_alpha-{args.fgw_alpha}"
        f"_alpha-{args.alpha}"
        f"_vq_beta-{args.vq_beta}"
        f"_kl_gamma-{args.kl_gamma}"
        f"_tau-{args.tau}"
        f"_target_data-{args.target_data}"
        f"_entropic_reg-{args.entropy_reg}"
        f"_description-{args.des}"
    )
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    #ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}/{args.run_tag}"
    
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

    from utils.util import fix_seed 
    fix_seed(args.random_seed)

    # ===========================
    # ✅ wandb: 함수 시작 시점에 현재 HP 한번 로깅 (로직 영향 X)
    # ===========================
    try:
        wandb_safe_log({
            "hp/alpha": getattr(args, "alpha", None),
            "hp/tau": getattr(args, "tau", None),
            "hp/soft_tau": getattr(args, "soft_tau", None),
            "hp/vq_beta": getattr(args, "vq_beta", None),
            "hp/source_lr": getattr(args, "source_lr", None),
            "hp/source_lr_few": getattr(args, "source_lr_few", None),
            "hp/dropout_rate": getattr(args, "dropout_rate", None),
            "Pre/total_epochs": total_epochs,
            "Pre/warmup_epochs": warmup_epochs if total_epochs > 0 else 0,
            "Pre/use_lcg": 1 if getattr(args, "use_lcg", False) else 0,
        })
    except Exception:
        pass
    
    # === 학습 루프 ===
    for epoch in range(total_epochs):
        _ = train_fn(model, tr_step, crit, opt, device)
        if scheduler_ep is not None: 
            scheduler_ep.step()
        if hasattr(model, 'graph_quantizer') and args.use_lcg is True:
            model.graph_quantizer.current_epoch = epoch + 1 
            
            if (epoch + 1) % 5 ==0 and (epoch + 1) >= warmup_epochs:
                n_reset = model.graph_quantizer.reset_dead(model.latent_graph)
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch + 1 
            #print(model.current_epoch)
            if model.current_epoch == model.switch_epoch:
                logger.info(f">>> [PHASE CHANGE] Epoch {model.current_epoch}: Global Inference ON")
        
        aucs_local = []
        aucs_global = []
        auprcs_local = [] 
        auprcs_global = [] 

        for vl in val_steps:
            res_g, res_l = eval_fn(model, vl, crit, device)
            _, y_true_l, y_pred_l = res_l
            _, y_true_g, y_pred_g = res_g
            
            if is_bin:
                score_l = roc_auc_score(y_true_l, y_pred_l)
                score_g = roc_auc_score(y_true_g, y_pred_g)

                ap_l = average_precision_score(y_true_l, y_pred_l)
                ap_g = average_precision_score(y_true_g, y_pred_g)
            else:
                n_cls = y_pred_l.shape[1]
                y_bin_l = label_binarize(y_true_l, classes=range(n_cls))
                y_bin_g = label_binarize(y_true_g, classes=range(n_cls))
                score_l = roc_auc_score(y_bin_l, y_pred_l, multi_class='ovr', average='macro')
                score_g = roc_auc_score(y_bin_g, y_pred_g, multi_class='ovr', average='macro')

                # === [AUPRC 추가] ===
                ap_l = average_precision_score(y_bin_l, y_pred_l, average='macro')
                ap_g = average_precision_score(y_bin_g, y_pred_g, average='macro')

            aucs_local.append(score_l)
            aucs_global.append(score_g)

            auprcs_local.append(ap_l)
            auprcs_global.append(ap_g)

        # === [수정 2] Best 갱신 로직 (Mean 기준) ===
        improved = False
        target_aucs = aucs_global if getattr(args, 'use_lcg', False) else aucs_local 
        # === [AUPRC 추가] ===
        target_auprcs = auprcs_global if getattr(args, 'use_lcg', False) else auprcs_local
        
        current_lcg_status = getattr(args, 'use_lcg', False)
        print(f"\n[DEBUG CHECK][Epoch {epoch+1}] args.use_lcg: {current_lcg_status} -> Watching: {'Global (LCG)' if current_lcg_status else 'Local (GAT)'}")
        # 현재 평균 계산
        current_mean_auc = float(np.mean(target_aucs))
        
        current_mean_auprc = float(np.mean(target_auprcs))
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


        mean_auprc_l = float(np.mean(auprcs_local))
        mean_auprc_g = float(np.mean(auprcs_global))
        log_msg_auprc = (
            f"[Pre][Epoch {epoch+1}/{total_epochs}]\n"
            f"   >>> Local (GAT): Mean AUPRC {mean_auprc_l:.4f} | Per-Source: {['%.4f'%x for x in auprcs_local]}\n"
            f"   >>> Global(LCG): Mean AUPRC {mean_auprc_g:.4f} | Per-Source: {['%.4f'%x for x in auprcs_global]}"
        )
        logger.info(log_msg_auprc)

        # ===========================
        # ✅ wandb: epoch 로그 (로직 영향 X)
        # ===========================
        try:
            lr_gat = opt.param_groups[0]["lr"] if len(opt.param_groups) > 0 else None
            lr_global = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else None
            wandb_safe_log({
                "Pre/mean_auc_local": mean_auc_l,
                "Pre/mean_auc_global": mean_auc_g,
                "Pre/mean_auc_watched": current_mean_auc,
                "Pre/best_mean_auc": best_mean_auc,
                "Pre/improved": 1 if improved else 0,
                "Pre/use_lcg": 1 if getattr(args, "use_lcg", False) else 0,
                "lr/gat": lr_gat,
                "lr/global": lr_global,
                "Pre/epoch": epoch + 1,
                # sweep에서 보고 싶은 hp도 계속 찍어두기
                "hp/alpha": getattr(args, "alpha", None),
                "hp/tau": getattr(args, "tau", None),
                "hp/soft_tau": getattr(args, "soft_tau", None),
                "hp/vq_beta": getattr(args, "vq_beta", None),
                "hp/source_lr": getattr(args, "source_lr", None),
                "hp/dropout_rate": getattr(args, "dropout_rate", None),
            }, step=epoch + 1)
        except Exception:
            pass
        
        if improved:
            #best_state = model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            last_best_epoch = epoch
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc_mean': best_mean_auc, 
                'val_aucs_per_source': target_aucs,
                'val_auprc_mean' : current_mean_auprc,
                'val_auprcs_per_source': target_auprcs,
                'args': args
            }, ckpt_latest)
            try:
                shutil.copyfile(ckpt_latest, ckpt_hist)
            except Exception as e:
                logger.warning(f"History copy failed: {e}")

            # ✅ wandb: best 갱신 이벤트(선택)
            try:
                wandb_safe_log({
                    "Pre/best_epoch": last_best_epoch + 1,
                    "Pre/best_mean_auc": best_mean_auc,
                }, step=epoch + 1)
            except Exception:
                pass
        else:
            if epoch + 1 > warmup_epochs: 
                no_improve += 1 
            else: 
                no_improve = 0 
            if no_improve >= patience : 
                logger.info(f"Early stop at epoch {epoch+1}")

                # ✅ wandb: early stop 표시(선택)
                try:
                    wandb_safe_log({
                        "Pre/early_stop": 1,
                        "Pre/early_stop_epoch": epoch + 1,
                        "Pre/best_mean_auc": best_mean_auc,
                    }, step=epoch + 1)
                except Exception:
                    pass

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
    per_train_auprc,  per_val_auprc,  per_test_auprc  = [], [], []
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
            train_auprc_i = average_precision_score(y_true_tr_i, y_pred_tr_i)
            y_bin_tr = (y_pred_tr_i > thr_i).astype(int)
            train_precision_i = precision_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_recall_i    = recall_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_f1_i        = f1_score(y_true_tr_i, y_bin_tr, zero_division=0)
            train_acc_i       = accuracy_score(y_true_tr_i, y_bin_tr)
        else:
            n_cls = y_pred_tr_i.shape[1]
            y_bin_tr = label_binarize(y_true_tr_i, classes=range(n_cls))
            train_auc_i = roc_auc_score(y_bin_tr, y_pred_tr_i, multi_class='ovr', average='macro')
            train_auprc_i = average_precision_score(y_bin_tr, y_pred_tr_i, average='macro')
            preds_tr = y_pred_tr_i.argmax(axis=1)
            train_precision_i = precision_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_recall_i    = recall_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_f1_i        = f1_score(y_true_tr_i, preds_tr, average='macro', zero_division=0)
            train_acc_i       = accuracy_score(y_true_tr_i, preds_tr)

        # Val
        if is_bin:
            val_auc_i = roc_auc_score(y_true_val_i, y_pred_val_i)
            val_auprc_i = average_precision_score(y_true_val_i, y_pred_val_i)
            y_bin_val = (y_pred_val_i > thr_i).astype(int)
            val_precision_i = precision_score(y_true_val_i, y_bin_val, zero_division=0)
            val_recall_i    = recall_score(y_true_val_i, y_bin_val, zero_division=0)
            val_f1_i        = f1_score(y_true_val_i, y_bin_val, zero_division=0)
            val_acc_i       = accuracy_score(y_true_val_i, y_bin_val)
            
        else:
            n_cls = y_pred_val_i.shape[1]
            y_bin_val = label_binarize(y_true_val_i, classes=range(n_cls))
            val_auc_i = roc_auc_score(y_bin_val, y_pred_val_i, multi_class='ovr', average='macro')
            val_auprc_i = average_precision_score(y_bin_val, y_pred_val_i, average='macro')
            preds_val = y_pred_val_i.argmax(axis=1)
            val_precision_i = precision_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_recall_i    = recall_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_f1_i        = f1_score(y_true_val_i, preds_val, average='macro', zero_division=0)
            val_acc_i       = accuracy_score(y_true_val_i, preds_val)

        # Test
        (test_loss_i, y_true_te_i, y_pred_te_i), _ = eval_fn(model, test_steps[i], crit, device)
        
        if is_bin:
            test_auc_i = roc_auc_score(y_true_te_i, y_pred_te_i)
            test_auprc_i = average_precision_score(y_true_te_i, y_pred_te_i)
            y_bin_te = (y_pred_te_i > thr_i).astype(int)
            test_precision_i = precision_score(y_true_te_i, y_bin_te, zero_division=0)
            test_recall_i    = recall_score(y_true_te_i, y_bin_te, zero_division=0)
            test_f1_i        = f1_score(y_true_te_i, y_bin_te, zero_division=0)
            test_acc_i       = accuracy_score(y_true_te_i, y_bin_te)
        else:
            n_cls = y_pred_te_i.shape[1]
            y_bin_te = label_binarize(y_true_te_i, classes=range(n_cls))
            test_auc_i = roc_auc_score(y_bin_te, y_pred_te_i, multi_class='ovr', average='macro')
            test_auprc_i = average_precision_score(y_bin_te, y_pred_te_i, average='macro')

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
        per_train_auprc.append(train_auprc_i); per_val_auprc.append(val_auprc_i); per_test_auprc.append(test_auprc_i)

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
    train_auprcs_full = [float(np.mean(per_train_auprc))]
    val_auprcs_full   = [float(np.mean(per_val_auprc))]
    test_auprc_full   = float(np.mean(per_test_auprc))
    # ===========================
    # ✅ wandb: 최종 요약 로그(선택)
    # ===========================
    try:
        # 이건 step 없이 찍어도 됨 (summary용으로)
        wandb_safe_log({
            "Pre/final_train_auc_mean": float(np.mean(per_train_auc)) if len(per_train_auc) else None,
            "Pre/final_val_auc_mean": float(np.mean(per_val_auc)) if len(per_val_auc) else None,
            "Pre/final_test_auc_mean": float(np.mean(per_test_auc)) if len(per_test_auc) else None,
            "Pre/final_test_acc_mean": float(np.mean(per_test_acc)) if len(per_test_acc) else None,
            "Pre/best_epoch": int(best_epoch_full) + 1 if best_epoch_full is not None and best_epoch_full >= 0 else None,
            "Pre/best_mean_auc": float(best_mean_auc),
        })
        if wandb.run is not None:
            wandb.run.summary["Pre/best_mean_auc"] = float(best_mean_auc)
            wandb.run.summary["Pre/best_epoch"] = int(best_epoch_full) + 1 if best_epoch_full is not None and best_epoch_full >= 0 else None
    except Exception:
        pass

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
        best_epoch_full=best_epoch_full,
        train_auprcs_full=train_auprcs_full,
        val_auprcs_full=val_auprcs_full,
        test_auprc_full=test_auprc_full
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
    if not getattr(args, "run_tag", ""):
        from datetime import datetime
        args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # 자동 생성
    wandb_init_and_override_args(args)  # ✅ 추가 (sweep 값이 args로 들어옴)
    wandb_update_config_minimal(args)
    # ✅ [wandb 추가 1] 안전 로거/컨피그 업데이트 (로직 영향 X)
    def _wandb_log(d, step=None):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(d, step=step)
        except Exception:
            pass

    def _wandb_summary_set(d):
        try:
            import wandb
            if wandb.run is not None:
                for k, v in d.items():
                    wandb.run.summary[k] = v
        except Exception:
            pass

    # args 전체를 wandb.config에 저장 (가능한 값만/문자열 fallback)
    try:
        import wandb
        if wandb.run is not None:
            cfg = {}
            for k, v in vars(args).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    cfg[k] = v
                elif isinstance(v, (list, tuple)):
                    cfg[k] = list(v)
                else:
                    cfg[k] = str(v)
            wandb.config.update(cfg, allow_val_change=True)
    except Exception:
        pass

    fix_seed(args.random_seed)
    
    # 1. 로거 설정
    logger_name = "my_experiment_logger" 
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    logger.info("--- 💡 Global logger initialized (Console) 💡 ---")
    logger.info(f"[RUN_TAG] run_tag = {args.run_tag}")
    try:
        ncpu = os.cpu_count() or 1
        p.cpu_affinity(range(1, min(ncpu, 64)))
    except Exception as e:
        logger.warning(f"cpu_affinity not set: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Starting experiment with Multiple-Source : {args.source_data}")
    logger.info(f"Device: {device}")
    logger.info("Preparing Tabular datasets...")

    _wandb_log({
        "env/device": str(device),
        "env/use_gpu": int(bool(getattr(args, "use_gpu", False))),
        "data/source_data": "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data),
        "data/target_data": str(getattr(args, "target_data", "")),
        "exp/random_seed": int(getattr(args, "random_seed", -1)),
        "exp/experiment_id": str(experiment_id),
    })

    # 1) 모델 생성
    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.dropout_rate, args.llm_model,
                       experiment_id, mode="Full").to(device)
    model_few  = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.dropout_rate, args.llm_model,
                       experiment_id, mode="Few").to(device)

    # 2) 프리트레인 체크포인트 로드 시도
    src_tag = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = (
        f"ngraphs-{args.n_graphs}"
        f"_nnodes-{args.n_nodes}"
        f"_gdim-{args.graph_dim}"
        f"_nbasis-{args.num_basis_layers}"
        f"_basis-{args.basis_type}"
        f"_attn-{args.attn_type}"
        f"_struct_hidden_dim-{args.struct_hidden_dim}"
        f"_fgw_alpha-{args.fgw_alpha}"
        f"_alpha-{args.alpha}"
        f"_vq_beta-{args.vq_beta}"
        f"_kl_gamma-{args.kl_gamma}"
        f"_tau-{args.tau}"
        f"_target_data-{args.target_data}"
        f"_entropic_reg-{args.entropy_reg}"
        f"_description-{args.des}"
    )
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    #ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}/{args.run_tag}"
    os.makedirs(ckpt_dir, exist_ok = True)
    ckpt_final = os.path.join(ckpt_dir, "best_joint.pt")
    ckpt_vanilla = os.path.join(ckpt_dir, "best_vanilla.pt")
    old_best = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(old_best) and not os.path.exists(ckpt_final):
        shutil.copy(old_best, ckpt_final)
    loaded_pretrain = False
    full_metrics = None

    # ✅ [wandb 추가 3] 경로/세팅 로그 (로직 영향 X)
    _wandb_log({
        "ckpt/pre_dir": ckpt_dir,
        "ckpt/final_exists": int(os.path.exists(ckpt_final)),
        "ckpt/vanilla_exists": int(os.path.exists(ckpt_vanilla)),
    })

    # ==================================================================
    # [Logic] 2-Stage Pre-training Pipeline
    # ==================================================================
    
    if os.path.exists(ckpt_final):
        # Case A: 이미 최종 학습(Phase 2) 완료됨 -> 로드
        logger.info(f"✅ [Pretrain] Found Final Checkpoint: {ckpt_final}")
        ckpt = torch.load(ckpt_final, map_location=device)
        model_full.load_state_dict(ckpt['model_state_dict'])
        loaded_pretrain = True

        # ✅ wandb
        _wandb_log({"pretrain/loaded_final_ckpt": 1})
        
    else:
        # Case B: 학습 필요
        logger.info(f"🚀 [Pretrain] Starting 2-Stage Training Pipeline...")
        _wandb_log({"pretrain/loaded_final_ckpt": 0})

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

            # ✅ wandb
            _wandb_log({"pretrain/phase1_loaded_ckpt": 1})
        else:
            logger.info(f"\n{'='*40}\n>>> [Phase 1] Start Vanilla GAT Training (LCG OFF)\n{'='*40}")
            args.use_lcg = False 

            # ✅ wandb
            _wandb_log({"pretrain/phase": 1, "pretrain/use_lcg": 0})

            # 학습 실행
            _ = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=20)
            
            # 결과 백업
            shutil.copy(os.path.join(ckpt_dir, "best.pt"), ckpt_vanilla)
            logger.info(f"   -> Phase 1 Saved to {ckpt_vanilla}")

            # ✅ wandb
            _wandb_log({"pretrain/phase1_saved": 1, "ckpt/vanilla_path": ckpt_vanilla})

        # --- [Step 2] Bridge: LCG Init ---
        logger.info(f"\n{'='*40}\n>>> [Bridge] Initializing LCG from Pre-trained CLS\n{'='*40}")

        # ✅ wandb
        _wandb_log({"pretrain/bridge_init_lcg": 1, "pretrain/lcg_strategy": str(getattr(args, "lcg_strategy", ""))})

        init_lcg(
            args, model_full, all_loaders, device, save_dir = ckpt_dir,
            strategy=args.lcg_strategy, injection_scale=0.1
        )
        args.use_lcg = True
        model_full.eval()
        with torch.no_grad():
            for src_name, loader in all_loaders.items():
                batch = next(iter(loader))
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                _ = model_full.predict(batch, return_all=True)
                pi = model_full.graph_quantizer.last_pi  # [B, M]
                
                logger.info(f"[INIT CHECK] {src_name}: mean pi={pi.mean(0).cpu().numpy().round(4)}")
                logger.info(f"[INIT CHECK] {src_name}: argmax counts={torch.bincount(pi.argmax(1), minlength=pi.shape[1]).cpu().numpy()}")
                
        fix_seed(args.random_seed)
        pdb.set_trace()
        # --- [Step 3] Phase 2: Joint Training ---
        logger.info(f"\n{'='*40}\n>>> [Phase 2] Start Joint Training (Global ON)\n{'='*40}")
        args.use_lcg = True 

        # ✅ wandb
        _wandb_log({"pretrain/phase": 2, "pretrain/use_lcg": 1})

        # [수정] patience 50으로 증가 (Global 학습 충분히)
        full_metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=10)
        
        # 최종 저장
        shutil.copy(os.path.join(ckpt_dir, "best.pt"), ckpt_final)
        logger.info(f"   -> Phase 2 Saved to {ckpt_final}")
        loaded_pretrain = True

        # ✅ wandb
        _wandb_log({"pretrain/phase2_saved": 1, "ckpt/final_path": ckpt_final})

    # 3) (옵션) Pretrain 모델로 소스 리포트 재평가 (eval-only)
    # [수정] few_shot이 4이거나 0일 때 (비교를 위해) 수행하도록 복구
    #pdb.set_trace()
    if loaded_pretrain and (args.few_shot == 4 or args.few_shot == 0):
        logger.info("[Full] Using loaded pretrain for source metrics report (eval only).")
        _bak = args.train_epochs
        args.train_epochs = 0
        args.use_lcg = True

        # ✅ wandb
        _wandb_log({"pretrain/eval_only_report": 1, "pretrain/eval_only_few_shot": int(args.few_shot)})

        full_metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=0)
        args.train_epochs = _bak
        fix_seed(args.random_seed)

        # ✅ wandb: source report 핵심만 summary로 (있을 때만)
        try:
            if full_metrics is not None:
                _wandb_summary_set({
                    "source_report/test_auc_full": float(full_metrics.get("test_auc_full", 0.0)),
                    "source_report/test_acc_full": float(full_metrics.get("test_acc_full", 0.0)),
                })
        except Exception:
            pass

    # 4) Target 적응 준비 공통 로직
    args.use_target_head = True
    args.use_lcg = True 
    model_few.args.use_lcg = True
    model_few.load_state_dict(model_full.state_dict(), strict=False)
    fix_seed(args.random_seed)

    # Target Data Load
    logger.info(f"[Target] target = {args.target_data}")
    r_t = prepare_embedding_dataloaders(args, args.target_data)
    train_loader_t, val_loader_t, test_loader_t = r_t['loaders']
    num_classes_t = r_t['num_classes']
    args.num_classes = num_classes_t
    args.output_dim  = num_classes_t if num_classes_t > 2 else 1
    
    is_binary_t = (args.num_classes == 2)
    crit_t = nn.BCEWithLogitsLoss() if is_binary_t else nn.CrossEntropyLoss()

    # ✅ wandb: target meta
    _wandb_log({
        "target/num_classes": int(args.num_classes),
        "target/is_binary": int(is_binary_t),
        "target/few_shot": int(args.few_shot),
        "target/support_resamples": int(getattr(args, "support_resamples", 1)),
        "target/train_epochs": int(getattr(args, "train_epochs", 0)),
    })

    # =========================================================
    # [분기 1] Zero-shot 평가 (학습 X, 평가 O, 종료)
    # =========================================================
    #pdb.set_trace()
    if args.few_shot == 0:
        logger.info("\n>>> [Zero-shot] Evaluating pretrained model directly on target test set...")
        
        evaluate_func = binary_evaluate if is_binary_t else multi_evaluate
        model_full.eval()
        # Validation set에서 threshold 결정
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

        (test_loss_zero, test_auc_zero, test_auprc_zero, test_precision_zero, test_recall_zero, 
         test_f1_zero, test_acc_zero, all_y_true_zero, all_y_pred_zero
        ) = final_test_evaluate(
            model_full, test_loader_t, crit_t, device, is_binary_t, 
            threshold=best_threshold_zero, mode='Full', args=args
        )
        
        logger.info(f"[Zero-shot] Test Results: "
           f"AUC={test_auc_zero:.4f} AUPRC={test_auprc_zero:.4f} ACC={test_acc_zero:.4f} "  # ✅ [AUPRC 추가]
           f"Prec={test_precision_zero:.4f} Rec={test_recall_zero:.4f} F1={test_f1_zero:.4f}")


        # ✅ wandb: zero-shot 결과
        _wandb_log({
            "zero_shot/test_loss": float(test_loss_zero),
            "zero_shot/test_auc": float(test_auc_zero),
            "zero_shot/test_auprc":float(test_auprc_zero),
            "zero_shot/test_acc": float(test_acc_zero),
            "zero_shot/test_precision": float(test_precision_zero),
            "zero_shot/test_recall": float(test_recall_zero),
            "zero_shot/test_f1": float(test_f1_zero),
            "zero_shot/threshold": float(best_threshold_zero) if best_threshold_zero is not None else None,
        })
        _wandb_summary_set({
            "final/zero_shot_test_auc": float(test_auc_zero),
            "final/zero_shot_test_auprc":float(test_auprc_zero),
            "final/zero_shot_test_acc": float(test_acc_zero),
        })
        
        # 결과 래핑
        if full_metrics is not None:
            full_test_auprc = compute_auprc(full_metrics["all_y_true_full"], full_metrics["all_y_pred_full"])  # ✅ [AUPRC 추가]

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
                test_accs=[full_metrics['test_acc_full']],
                test_auprcs = [full_test_auprc],
                best_ours_auprc = full_test_auprc
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
            train_accs=[], val_accs=[], test_accs=[test_acc_zero], 
            test_auprcs = [test_auprc_zero],
            best_ours_auprc = test_auprc_zero 
        )
        
        results = prepare_results_(full_ours_results, zero_shot_results)
        
        # 저장 및 종료
        logger.info("Saving Zero-shot results...")
        import copy
        args_for_save = copy.deepcopy(args)
        if isinstance(args_for_save.source_data, (list, tuple)):
            args_for_save.source_data = "+".join(map(str, args_for_save.source_data))
        else:
            args_for_save.source_data = str(args_for_save.source_data)

        save_results_(args_for_save, results)
        logger.info("Results saved")

        # ✅ wandb 종료
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

        return 

    # =========================================================
    # [분기 2] Few-shot 학습 (Target Adaptation)
    # =========================================================
    
    # 1. Freeze 설정 (GAT Unfreeze 전략)
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
    ep_test_metrics = [] 
    y_true_last, y_pred_last = None, None

    import numpy as _np

    for r in range(R):
        current_seed = args.random_seed + (r + 1)
        fix_seed(current_seed) 
        
        model_few.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()}, strict=False)
        model_few.set_freeze_target() 

        gat_params_few = []
        global_params_few = []
        
        for name, p in model_few.named_parameters():
            if not p.requires_grad: continue
            if 'basis' in name: 
                gat_params_few.append(p)
            else:
                global_params_few.append(p)
        
        gat_lr_few = args.source_lr_few * 0.1 
        global_lr_few = args.source_lr_few
        
        logger.info(f"[Few-shot][Ep {r+1}] GAT LR: {gat_lr_few} | Global LR: {global_lr_few}")

        # ✅ wandb: episode/seed/lr 기록
        _wandb_log({
            "few/episode": int(r + 1),
            "few/episode_seed": int(current_seed),
            "few/gat_lr": float(gat_lr_few),
            "few/global_lr": float(global_lr_few),
        })

        optimizer_few = optim.Adam(
            [
                {'params': gat_params_few,    'lr': gat_lr_few},
                {'params': global_params_few, 'lr': global_lr_few}
            ],
            weight_decay=3e-5
        )
        
        warmup_epochs_few = max(1, int(args.warmup_ratio * args.train_epochs))
        scheduler_few = make_warmup_cosine_epochs(
            optimizer_few,
            total_epochs=args.train_epochs,
            warmup_epochs=warmup_epochs_few,
            min_lr_mult=args.min_lr_mult
        )
        logger.info(f"[Few-shot][Ep {r+1}] LR schedule: warmup_epochs={warmup_epochs_few}, final_mult={args.min_lr_mult}")

        if args.few_shot > 0:
            #val_shot = int(math.ceil(args.few_shot * 0.25))
            val_shot = max(5, int(math.ceil(args.few_shot * 0.25)))
            import copy
            args_val = copy.deepcopy(args)
            args_val.few_shot = val_shot 
            train_loader_epi = get_few_shot_embedding_samples(train_loader_t, args)
            val_loader_epi = get_few_shot_embedding_samples(val_loader_t, args_val)
        else:
            train_loader_epi = train_loader_t 

        (train_losses_few, val_losses_few,
         train_aucs_few,   val_aucs_few,
         train_precisions_few, val_precisions_few,
         train_recalls_few,    val_recalls_few,
         train_f1s_few,        val_f1s_few,
         train_accs_few,       val_accs_few,
         best_epoch_few, best_val_auc_few, best_threshold_few
        ) = train_and_validate(args, model_few, train_loader_epi, val_loader_epi, crit_t,
                               optimizer_few, device, args.train_epochs, is_binary_t, patience=50,
                               mode="Few", scheduler=scheduler_few, warmup_epochs=warmup_epochs_few)

        (test_loss_few, test_auc_few, test_auprc_few, test_precision_few, test_recall_few, test_f1_few,
         test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
            model_few, test_loader_t, crit_t, device, is_binary_t, threshold=best_threshold_few,
            mode='Few', args = args
        )

        logger.info(f"[Few-shot][Ep {r+1}/{R}] AUC={test_auc_few:.4f} AUPRC={test_auprc_few:.4f} ACC={test_acc_few:.4f} "  # ✅ [AUPRC 추가]
            f"Prec={test_precision_few:.4f} Rec={test_recall_few:.4f} F1={test_f1_few:.4f}")

        # ✅ wandb: episode test 결과
        _wandb_log({
            "few_ep/test_loss": float(test_loss_few),
            "few_ep/test_auc": float(test_auc_few),
            "few_ep/test_auprc": float(test_auprc_few),  
            "few_ep/test_acc": float(test_acc_few),
            "few_ep/test_precision": float(test_precision_few),
            "few_ep/test_recall": float(test_recall_few),
            "few_ep/test_f1": float(test_f1_few),
            "few_ep/best_epoch": int(best_epoch_few),
            "few_ep/best_val_auc": float(best_val_auc_few),
            "few_ep/best_threshold": float(best_threshold_few) if best_threshold_few is not None else None,
            "few_ep/episode": int(r + 1),
        })

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
        ep_test_metrics.append((test_loss_few, test_auc_few, test_auprc_few, test_precision_few, test_recall_few, test_f1_few, test_acc_few))

        y_true_last, y_pred_last = all_y_true_few, all_y_pred_few 

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

    ep_arr = _np.asarray(ep_test_metrics, dtype=_np.float32)
    mean_test_loss, mean_test_auc, mean_test_auprc, mean_test_prec, mean_test_rec, mean_test_f1, mean_test_acc = ep_arr.mean(axis=0).tolist()
    Rf = float(max(len(ep_best_epochs), 1))
    best_epoch_few_mean = int(round(sum(ep_best_epochs) / Rf))

    # ✅ wandb: few-shot 최종 평균
    _wandb_log({
        "few_mean/test_loss": float(mean_test_loss),
        "few_mean/test_auc": float(mean_test_auc),
        "few_mean/test_auprc": float(mean_test_auprc),
        "few_mean/test_acc": float(mean_test_acc),
        "few_mean/test_precision": float(mean_test_prec),
        "few_mean/test_recall": float(mean_test_rec),
        "few_mean/test_f1": float(mean_test_f1),
        "few_mean/best_epoch": int(best_epoch_few_mean),
    })
    _wandb_summary_set({
        "final/few_shot_test_auc_mean": float(mean_test_auc),
        "final/few_shot_test_auprc_mean": float(mean_test_auprc),
        "final/few_shot_test_acc_mean": float(mean_test_acc),
        "final/few_shot_best_epoch_mean": int(best_epoch_few_mean),
    })

    # 결과 래핑
    if full_metrics is not None:
        full_test_auprc = compute_auprc(full_metrics["all_y_true_full"], full_metrics["all_y_pred_full"])  # ✅ [AUPRC 추가]
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
            test_accs=[full_metrics['test_acc_full']], 
            test_auprcs=[full_test_auprc],
            best_ours_auprc=full_test_auprc
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
        train_accs=train_accs_few_mean, val_accs=val_accs_few_mean, test_accs=[mean_test_acc],
        test_auprcs = [mean_test_auprc],
        best_ours_auprc=mean_test_auprc
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

    # ✅ wandb: 종료 전 최종 시간/종료
    _wandb_log({"exp/total_time_sec": float(time.time() - start_time)})
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
