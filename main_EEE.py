
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
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from utils.util import setup_logger, format_time, fix_seed, prepare_results_, save_results_, wrap_up_results_, make_warmup_cosine_epochs, make_warmup_cosine_steps, current_lr, build_epoch_scheduler
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from sklearn.model_selection import StratifiedKFold
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders, get_few_shot_embedding_samples_, generate_and_save_split_indices, prepare_exp_embedding_dataloaders
from models.TabularFLM_S_ import Model
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
                                 'forest_covertype_sampled','higgs_sampled','Cardiovascular_Disease_Dataset','Heart_disease_statlog','Medicaldataset', 'heart_failure_clinical_records','cardio_SAheart', 'Erbil_Cardiovascular_Health_Dataset',
                                 'mimic_mortality','eicu_mortality','hirid_mortality','support_mortality','zigong_mortality','sic_mortality'])
    parser.add_argument('--target_data', type=str, default='heart')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--source_lr_few', type=float, default=0.0001)
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
    parser.add_argument('--fgw_alpha', type = float, default = 0.3)
    parser.add_argument('--alpha' , type = float, default = 0.7)
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
    parser.add_argument('--lcg_strategy', type = str, default = 'hierarchical', choices = ['hierarchical', 'round_robin', 'sequential', 'robust_hierarchical'])
    parser.add_argument('--lcg_struct_type', type = str, default = 'static', choices = ['projection', 'static', ' residual'])
    parser.add_argument('--feat_distance', type = str, default = 'cosine', choices=['cosine','l2'])
    parser.add_argument('--orth_reg', type = float, default = 0.1)
    parser.add_argument('--div_reg', type = float, default = 1)
    parser.add_argument('--hp_reg', type=float, default=1.0)
    parser.add_argument('--hs_reg', type=float, default=0.1)
    parser.add_argument('--hs_warmup', type=int, default=30)
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
    # ── Exp A / Exp B (2026.04.02) ──
    parser.add_argument('--exp_mode', type=str, default='case1',
                        choices=['single_source', 'multi_source', 'case1', 'case2', 'case2_exclude',
                                 'exp_b_analysis', 'exp_b_retrain'],
                        help='Experiment mode for Exp A/B')
    parser.add_argument('--sampling_alpha', type=float, default=1.0,
                        help='Train set sampling ratio (0.0~1.0) for multi-source scaling')
    parser.add_argument('--exclude_sources', nargs='*', default=[],
                        help='Sources to exclude for Exp B retraining')
    parser.add_argument('--eval_source', type=str, default=None,
                        help='Single source to evaluate (single_source mode)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoint path for Exp B analysis (Case F model)')
    parser.add_argument('--freeze_ft', action='store_true',
                        help='Fine-tuning 시 set_freeze_target() 적용 (basis_layers + ghead2만 trainable)')
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args

WANDB_KEYS = [
    "alpha", "tau", "soft_tau", "vq_beta",
    "source_lr", "source_lr_few", "dropout_rate",
    "fgw_alpha", "few_shot", "random_seed", "entropy_reg"
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

    elif strategy == 'robust_hierarchical':
        logger.info(f">> [Robust Hierarchical] Trying 20 KMeans runs, selecting most diverse centroids")
        
        best_score = -1
        best_km = None
        
        for trial in range(20):
            km = KMeans(n_clusters=M, n_init=10, random_state=args.random_seed + trial).fit(data_pool)
            
            # centroid 간 cosine similarity로 diversity 측정
            centers = km.cluster_centers_
            norms = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
            gram = norms @ norms.T
            off_diag = gram[~np.eye(M, dtype=bool)]
            diversity = -off_diag.mean()
            
            # 클러스터 크기 균형
            counts = np.bincount(km.labels_, minlength=M)
            balance = counts.min() / (counts.max() + 1e-8)
            
            score = diversity + 0.3 * balance
            logger.info(f"   Trial {trial}: diversity={-diversity:.4f}, balance={balance:.3f}, score={score:.4f}, sizes={counts.tolist()}")
            
            if score > best_score:
                best_score = score
                best_km = km
        
        km_global = best_km
        counts_final = np.bincount(km_global.labels_, minlength=M)
        logger.info(f">> [Robust] Selected best trial: score={best_score:.4f}, sizes={counts_final.tolist()}")
        
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
        
    elif strategy == 'round_robin':
        kmeans = KMeans(n_clusters=M * K, n_init=10, random_state=args.random_seed).fit(data_pool)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        final_centroids = centers.view(K, M, D).transpose(0, 1).contiguous()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'hierarchical', 'robust_hierarchical', or 'round_robin'.")

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

    # # =========================================================================
    # # Save initial state for dead code reset
    # # =========================================================================
    # model.latent_graph.register_buffer(
    #     'init_node_embeddings', 
    #     model.latent_graph.node_embeddings.data.clone()
    # )
    # if model.latent_graph.struct_mode == 'static':
    #     model.latent_graph.register_buffer(
    #         'init_adj_param',
    #         model.latent_graph.adj_param.data.clone()
    #     )
    # logger.info(f">> ✅ Initial LCG state saved for dead code reset")

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



def load_one(args, name, sampling_alpha=1.0, use_exp=False):
    if use_exp:
        r = prepare_exp_embedding_dataloaders(args, name, alpha=sampling_alpha)
        train_loader, val_loader, test_loader = r['loaders']
        num_classes = r['num_classes']
    else:
        res = prepare_embedding_dataloaders(args, name, is_source=True)
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
    #checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}"
    checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}/{args.run_tag}"
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
def pretrain_and_eval_sources(args, model, device, sources, patience=20,
                              use_exp=False, sampling_alpha=1.0):
    import shutil
    logger_name = "my_experiment_logger"
    logger = logging.getLogger(logger_name)
    name_to_idx = {name: i for i, name in enumerate(sources)}
    trains, vals, tests, ncs = [], [], [], []
    for name in sources:
        tr, va, te, nc = load_one(args, name, sampling_alpha=sampling_alpha, use_exp=use_exp)
        trains.append(tr); vals.append(va); tests.append(te); ncs.append(nc)

    if len(set(ncs)) != 1:
        raise ValueError(f"num_classes mismatch across sources: {ncs}")
    args.num_classes = ncs[0]
    args.output_dim  = args.num_classes if args.num_classes > 2 else 1

    # 학습은 다중 소스를 섞어서
    tr_step = make_step(trains, mode='random', seed=args.random_seed)

    # 검증/개별-학습 평가용
    val_steps   = [MultiSourceStepLoader([vals[i]],   mode='round', seed=args.random_seed, src_idx=i) for i in range(len(vals))]
    train_steps = [MultiSourceStepLoader([trains[i]], mode='round', seed=args.random_seed, src_idx=i) for i in range(len(trains))]
    # Exp A/B: test loader가 존재하면 test_steps도 구성
    if use_exp and any(t is not None for t in tests):
        test_steps = [MultiSourceStepLoader([tests[i]], mode='round', seed=args.random_seed, src_idx=i)
                      for i in range(len(tests)) if tests[i] is not None]
    else:
        test_steps = None

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
    best_min_auc = -1.0
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
    #ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}/{args.run_tag}"
    
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
                model.graph_quantizer.phase2_start_epoch = model.current_epoch

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
        #current_min_auc = float(np.min(target_aucs))
        current_mean_auprc = float(np.mean(target_auprcs))
        #current_min_auprc = float(np.min(target_auprcs))

        # 평균이 기존 최고 평균보다 높으면 저장
        if current_mean_auc > best_mean_auc:
            best_mean_auc = current_mean_auc
            improved = True
        # if current_min_auc > best_min_auc:
        #     best_min_auc = current_min_auc 
        #     best_mean_auc = current_mean_auc 
        #     improved = True
        # === Logging ===
        mean_auc_l = float(np.mean(aucs_local))
        mean_auc_g = float(np.mean(aucs_global))
        
        # [main_EEE] fine-tune이면 현재 source 명시
        cur_src_tag = f"finetune={sources[0]}" if len(sources) == 1 else f"pretrain=ALL({len(sources)})"
        log_msg = (
            f"[Pre][Epoch {epoch+1}/{total_epochs}] [{cur_src_tag}]\n"
            f"   >>> Local (GAT): Mean AUC {mean_auc_l:.4f} | Per-Source: {['%.4f'%x for x in aucs_local]}\n"
            f"   >>> Global(LCG): Mean AUC {mean_auc_g:.4f} | Per-Source: {['%.4f'%x for x in aucs_global]}"
        )
        # log_msg = (
        #     f"[Pre][Epoch {epoch+1}/{total_epochs}]\n"
        #     f"   >>> Local (GAT): Mean AUC {mean_auc_l:.4f} | Min AUC {np.min(aucs_local):.4f} | Per-Source: {['%.4f'%x for x in aucs_local]}\n"
        #     f"   >>> Global(LCG): Mean AUC {mean_auc_g:.4f} | Min AUC {np.min(aucs_global):.4f} | Per-Source: {['%.4f'%x for x in aucs_global]}"
        # )
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
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'epoch': epoch,
            #     'val_auc_mean': best_mean_auc,
            #     'val_auc_min': best_min_auc,
            #     'val_aucs_per_source': target_aucs,
            #     'val_auprc_mean': current_mean_auprc,
            #     'val_auprc_min': current_min_auprc,
            #     'val_auprcs_per_source': target_auprcs,
            #     'args': args
            # }, ckpt_latest)
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
    # 최종 리포트 (train/val + test if use_exp)
    # -----------------------------
    per_train_loss, per_val_loss = [], []
    per_train_auc,  per_val_auc  = [], []
    per_train_precision, per_val_precision = [], []
    per_train_recall,    per_val_recall    = [], []
    per_train_f1,        per_val_f1        = [], []
    per_train_acc,       per_val_acc       = [], []
    per_train_auprc,     per_val_auprc     = [], []
    # Exp A/B: per-source test metrics
    per_test_auc, per_test_auprc, per_test_acc = [], [], []
    per_test_f1, per_test_precision, per_test_recall = [], [], []
    per_test_loss = []

    for i in range(len(sources)):
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

        per_train_loss.append(train_loss_i); per_val_loss.append(val_loss_i)
        per_train_auc.append(train_auc_i);   per_val_auc.append(val_auc_i)
        per_train_precision.append(train_precision_i); per_val_precision.append(val_precision_i)
        per_train_recall.append(train_recall_i);       per_val_recall.append(val_recall_i)
        per_train_f1.append(train_f1_i);               per_val_f1.append(val_f1_i)
        per_train_acc.append(train_acc_i);             per_val_acc.append(val_acc_i)
        per_train_auprc.append(train_auprc_i);         per_val_auprc.append(val_auprc_i)

        # Test (Exp A/B only)
        if test_steps is not None:
            (test_loss_i, y_true_te_i, y_pred_te_i), _ = eval_fn(model, test_steps[i], crit, device)
            if is_bin:
                te_auc_i   = roc_auc_score(y_true_te_i, y_pred_te_i)
                te_auprc_i = average_precision_score(y_true_te_i, y_pred_te_i)
                y_bin_te   = (y_pred_te_i > thr_i).astype(int)
                te_prec_i  = precision_score(y_true_te_i, y_bin_te, zero_division=0)
                te_rec_i   = recall_score(y_true_te_i, y_bin_te, zero_division=0)
                te_f1_i    = f1_score(y_true_te_i, y_bin_te, zero_division=0)
                te_acc_i   = accuracy_score(y_true_te_i, y_bin_te)
            else:
                n_cls = y_pred_te_i.shape[1]
                y_bin_te   = label_binarize(y_true_te_i, classes=range(n_cls))
                te_auc_i   = roc_auc_score(y_bin_te, y_pred_te_i, multi_class='ovr', average='macro')
                te_auprc_i = average_precision_score(y_bin_te, y_pred_te_i, average='macro')
                preds_te   = y_pred_te_i.argmax(axis=1)
                te_prec_i  = precision_score(y_true_te_i, preds_te, average='macro', zero_division=0)
                te_rec_i   = recall_score(y_true_te_i, preds_te, average='macro', zero_division=0)
                te_f1_i    = f1_score(y_true_te_i, preds_te, average='macro', zero_division=0)
                te_acc_i   = accuracy_score(y_true_te_i, preds_te)
            per_test_loss.append(test_loss_i)
            per_test_auc.append(te_auc_i);     per_test_auprc.append(te_auprc_i)
            per_test_acc.append(te_acc_i);     per_test_f1.append(te_f1_i)
            per_test_precision.append(te_prec_i); per_test_recall.append(te_rec_i)
            logger.info(f"   [Test] {sources[i]}: AUC={te_auc_i:.4f} AUPRC={te_auprc_i:.4f} ACC={te_acc_i:.4f}")

    # 평균 집계
    train_losses_full = [float(np.mean(per_train_loss))]
    val_losses_full   = [float(np.mean(per_val_loss))]
    train_aucs_full = [float(np.mean(per_train_auc))]
    val_aucs_full   = [float(np.mean(per_val_auc))]
    train_precisions_full = [float(np.mean(per_train_precision))]
    val_precisions_full   = [float(np.mean(per_val_precision))]
    train_recalls_full = [float(np.mean(per_train_recall))]
    val_recalls_full   = [float(np.mean(per_val_recall))]
    train_f1s_full = [float(np.mean(per_train_f1))]
    val_f1s_full   = [float(np.mean(per_val_f1))]
    train_accs_full = [float(np.mean(per_train_acc))]
    val_accs_full   = [float(np.mean(per_val_acc))]
    best_epoch_full = last_best_epoch
    train_auprcs_full = [float(np.mean(per_train_auprc))]
    val_auprcs_full   = [float(np.mean(per_val_auprc))]

    # ===========================
    # wandb: 최종 요약 로그
    # ===========================
    try:
        wandb_safe_log({
            "Pre/final_train_auc_mean": float(np.mean(per_train_auc)) if len(per_train_auc) else None,
            "Pre/final_val_auc_mean": float(np.mean(per_val_auc)) if len(per_val_auc) else None,
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
        train_aucs_full=train_aucs_full,
        val_aucs_full=val_aucs_full,
        train_precisions_full=train_precisions_full,
        val_precisions_full=val_precisions_full,
        train_recalls_full=train_recalls_full,
        val_recalls_full=val_recalls_full,
        train_f1s_full=train_f1s_full,
        val_f1s_full=val_f1s_full,
        train_accs_full=train_accs_full,
        val_accs_full=val_accs_full,
        best_epoch_full=best_epoch_full,
        train_auprcs_full=train_auprcs_full,
        val_auprcs_full=val_auprcs_full,
    )
    # Exp A/B: per-source test 결과 추가
    if test_steps is not None:
        full_pack['per_source_test'] = {
            'sources': list(sources),
            'auc':   per_test_auc,
            'auprc': per_test_auprc,
            'acc':   per_test_acc,
            'f1':    per_test_f1,
            'precision': per_test_precision,
            'recall': per_test_recall,
            'loss':  per_test_loss,
        }
        logger.info(f"[Test Summary] Per-source AUC: {['%.4f'%x for x in per_test_auc]}")
        logger.info(f"[Test Summary] Mean AUC: {np.mean(per_test_auc):.4f}, Mean AUPRC: {np.mean(per_test_auprc):.4f}")

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

'''
    2026.04.02
    ================================================================================
    main() — Exp A / Exp B 전용 파이프라인
    ================================================================================

    exp_mode 분기:
        single_source   — Exp A Step 2: 각 source별 독립 모델 학습 → 자기 test 평가
        multi_source    — Exp A Step 3/4: α scaling으로 multi-source 학습 → 모든 source test 평가
        exp_b_analysis  — Exp B Step 1/2: Case F 모델에서 LCG routing π 추출 → relevance 계산
        exp_b_retrain   — Exp B Step 3: 이질적 source 제외 후 retraining
'''


def main():
    start_time = time.time()
    args = get_args()
    if not getattr(args, "run_tag", ""):
        args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_init_and_override_args(args)
    wandb_update_config_minimal(args)
    import json, copy

    def _wandb_log(d, step=None):
        try:
            if wandb.run is not None:
                wandb.log(d, step=step)
        except Exception:
            pass

    def _wandb_summary_set(d):
        try:
            if wandb.run is not None:
                for k, v in d.items():
                    wandb.run.summary[k] = v
        except Exception:
            pass

    fix_seed(args.random_seed)

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

    logger.info(f"--- [main_E] Exp A/B Pipeline ---")
    logger.info(f"[RUN_TAG] {args.run_tag}  [EXP_MODE] {args.exp_mode}  [SEED] {args.random_seed}")
    logger.info(f"[sampling_alpha] {args.sampling_alpha}  [exclude_sources] {args.exclude_sources}")

    try:
        ncpu = os.cpu_count() or 1
        p.cpu_affinity(range(1, min(ncpu, 64)))
    except Exception as e:
        logger.warning(f"cpu_affinity not set: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Device: {device}")

    # ── 결과 저장 디렉토리 ──
    freeze_tag = '_freeze' if getattr(args, 'freeze_ft', False) else ''
    exp_results_dir = f"/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}/{args.exp_mode}{freeze_tag}"
    os.makedirs(exp_results_dir, exist_ok=True)

    # ── Step 0: Split index 생성 (최초 1회, 모든 모드에서 보장) ──
    src_list = args.source_data if isinstance(args.source_data, (list, tuple)) else [args.source_data]
    generate_and_save_split_indices(args, src_list)

    # ==================================================================
    # [Mode 1] single_source — Exp A Step 2
    # ==================================================================
    if args.exp_mode == 'single_source':
        target_src = args.eval_source
        if target_src is None:
            raise ValueError("--eval_source is required for single_source mode")
        alpha = args.sampling_alpha
        logger.info(f"\n{'='*60}\n>>> [Exp A] Single-Source Baseline: {target_src}, alpha={alpha}\n{'='*60}")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # Phase 1: Vanilla GAT (LCG OFF)
        args.use_lcg = False
        full_pack = pretrain_and_eval_sources(
            args, model, device, sources=[target_src], patience=20,
            use_exp=True, sampling_alpha=alpha
        )

        # LCG Init + Phase 2: Joint (LCG ON)
        tr, _, _, _ = load_one(args, target_src, sampling_alpha=alpha, use_exp=True)
        init_lcg(args, model, {target_src: tr}, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        full_pack = pretrain_and_eval_sources(
            args, model, device, sources=[target_src], patience=30,
            use_exp=True, sampling_alpha=alpha
        )

        # 결과 저장: exp_results_dir / {dataset} / {seed} / json
        single_save_dir = os.path.join(exp_results_dir, target_src, str(args.random_seed))
        os.makedirs(single_save_dir, exist_ok=True)
        result_file = os.path.join(
            single_save_dir,
            f"single_alpha{alpha:.2f}_{args.run_tag}.json"
        )
        save_data = {
            'exp_mode': 'single_source',
            'source': target_src,
            'sampling_alpha': alpha,
            'seed': args.random_seed,
            'per_source_test': full_pack.get('per_source_test', {}),
            'val_aucs_full': full_pack.get('val_aucs_full', []),
            'best_epoch': full_pack.get('best_epoch_full', -1),
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"[Single-Source] Results saved to {result_file}")

    # ==================================================================
    # [Mode 2] multi_source — Exp A Step 3/4
    # ==================================================================
    elif args.exp_mode == 'multi_source':
        alpha = args.sampling_alpha
        logger.info(f"\n{'='*60}\n>>> [Exp A] Multi-Source Scaling: alpha={alpha}\n{'='*60}")
        logger.info(f"Sources: {src_list}")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # Phase 1: Vanilla GAT (LCG OFF) — alpha applies to train only
        args.use_lcg = False
        full_pack = pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=20,
            use_exp=True, sampling_alpha=alpha
        )

        # LCG Init (alpha=1.0 for init — use full train to get good centroids)
        all_loaders_init = {}
        for s in src_list:
            tr, _, _, _ = load_one(args, s, sampling_alpha=alpha, use_exp=True)
            all_loaders_init[s] = tr
        init_lcg(args, model, all_loaders_init, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        # Phase 2: Joint Training (LCG ON)
        full_pack = pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=30,
            use_exp=True, sampling_alpha=alpha
        )

        # 결과 저장 (원래 flat 형식)
        result_file = os.path.join(
            exp_results_dir,
            f"multi_alpha{alpha:.2f}_seed{args.random_seed}_{args.run_tag}.json"
        )
        save_data = {
            'exp_mode': 'multi_source',
            'sources': src_list,
            'sampling_alpha': alpha,
            'seed': args.random_seed,
            'per_source_test': full_pack.get('per_source_test', {}),
            'val_aucs_full': full_pack.get('val_aucs_full', []),
            'best_epoch': full_pack.get('best_epoch_full', -1),
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"[Multi-Source] Results saved to {result_file}")

        # alpha=1.0이면 Case F 체크포인트도 저장
        if alpha >= 1.0:
            case_f_path = os.path.join(exp_results_dir, f"case_f_seed{args.random_seed}_{args.run_tag}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': args,
                'per_source_test': full_pack.get('per_source_test', {}),
            }, case_f_path)
            logger.info(f"[Case F] Checkpoint saved to {case_f_path}")

    # ==================================================================
    # [Mode 3] case1 — Pretrain ALL (alpha=1.0) → Fine-tune per-source (alpha varies)
    #   Pretrain은 전체 데이터로 한 번, fine-tune에서 alpha로 데이터 양 조절
    # ==================================================================
    elif args.exp_mode == 'case1':
        alpha = args.sampling_alpha
        logger.info(f"\n{'='*60}\n>>> [Case 1] Pretrain ALL → Fine-tune per-source (alpha={alpha})\n{'='*60}")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # ── Pretrain: 모든 source, alpha=1.0 ──
        args.use_lcg = False
        pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=20,
            use_exp=True, sampling_alpha=1.0
        )

        all_loaders_init = {}
        for s in src_list:
            tr, _, _, _ = load_one(args, s, sampling_alpha=1.0, use_exp=True)
            all_loaders_init[s] = tr
        init_lcg(args, model, all_loaders_init, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=30,
            use_exp=True, sampling_alpha=1.0
        )

        # pretrained state 저장
        pretrained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"[Case 1] Pretrained model saved (alpha=1.0, all sources)")

        # ── Fine-tune: 각 source별로, alpha 적용 ──
        ft_sources = []
        ft_auc, ft_auprc, ft_acc, ft_f1 = [], [], [], []
        ft_precision, ft_recall, ft_loss = [], [], []

        for src_name in src_list:
            pretrain_idx = src_list.index(src_name)
            logger.info(f"\n{'='*60}")
            logger.info(f"[Case 1 Fine-tune] dataset='{src_name}' (pretrain_src_idx={pretrain_idx})")
            logger.info(f"   alpha={alpha}, freeze={args.freeze_ft}, mode→Few (ghead→ghead2 copy)")
            logger.info(f"{'='*60}")
            model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
            # [main_EEE] fine-tune은 target adaptation처럼: ghead → ghead2 복사 후 mode='Few'
            model.ghead2.load_state_dict(model.ghead.state_dict())
            model.mode = 'Few'
            if args.freeze_ft:
                model.set_freeze_target()
            fix_seed(args.random_seed)

            ft_pack = pretrain_and_eval_sources(
                args, model, device, sources=[src_name], patience=20,
                use_exp=True, sampling_alpha=alpha
            )

            ps_test = ft_pack.get('per_source_test', {})
            ft_sources.append(src_name)
            for key, lst in [('auc', ft_auc), ('auprc', ft_auprc), ('acc', ft_acc),
                             ('f1', ft_f1), ('precision', ft_precision),
                             ('recall', ft_recall), ('loss', ft_loss)]:
                vals = ps_test.get(key, [])
                lst.append(vals[0] if vals else float('nan'))

        # 결과 저장
        result_file = os.path.join(
            exp_results_dir,
            f"case1{'_freeze' if args.freeze_ft else ''}_alpha{alpha:.2f}_seed{args.random_seed}_{args.run_tag}.json"
        )
        save_data = {
            'exp_mode': 'case1',
            'sources': src_list,
            'sampling_alpha': alpha,
            'seed': args.random_seed,
            'per_source_test': {
                'sources': ft_sources,
                'auc': ft_auc, 'auprc': ft_auprc, 'acc': ft_acc,
                'f1': ft_f1, 'precision': ft_precision,
                'recall': ft_recall, 'loss': ft_loss,
            },
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"[Case 1] Results saved to {result_file}")

    # ==================================================================
    # [Mode 4] case2 — Pretrain with alpha → Fine-tune per-source
    #   Pretrain에서 alpha로 데이터 양 조절, 그 위에 per-source fine-tuning 추가
    # ==================================================================
    elif args.exp_mode == 'case2':
        alpha = args.sampling_alpha
        logger.info(f"\n{'='*60}\n>>> [Case 2] Pretrain alpha={alpha} → Fine-tune per-source\n{'='*60}")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # ── Pretrain: 모든 source, alpha 적용 ──
        t_pretrain_start = time.time()
        args.use_lcg = False
        pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=20,
            use_exp=True, sampling_alpha=alpha
        )

        all_loaders_init = {}
        for s in src_list:
            tr, _, _, _ = load_one(args, s, sampling_alpha=alpha, use_exp=True)
            all_loaders_init[s] = tr
        init_lcg(args, model, all_loaders_init, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        pretrain_and_eval_sources(
            args, model, device, sources=src_list, patience=30,
            use_exp=True, sampling_alpha=alpha
        )

        pretrain_sec = time.time() - t_pretrain_start
        logger.info(f"[Case 2][TIME] Multi-source pretrain elapsed: {pretrain_sec:.1f}s ({pretrain_sec/60:.2f}min)")

        # pretrained state 저장
        pretrained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"[Case 2] Pretrained model saved (alpha={alpha}, all sources)")

        # ── Fine-tune: freeze=False → freeze=True 순차 실행 ──
        for do_freeze in [False, True]:
            freeze_tag = '_freeze' if do_freeze else ''
            ft_save_dir = f"/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}/case2{freeze_tag}"
            os.makedirs(ft_save_dir, exist_ok=True)

            logger.info(f"\n{'='*40}\n>>> [Case 2] Fine-tune (freeze={do_freeze})\n{'='*40}")

            ft_sources = []
            ft_auc, ft_auprc, ft_acc, ft_f1 = [], [], [], []
            ft_precision, ft_recall, ft_loss = [], [], []
            ft_sec = []
            t_ft_total_start = time.time()

            for src_name in src_list:
                pretrain_idx = src_list.index(src_name)
                logger.info(f"\n{'='*60}")
                logger.info(f"[Case 2 Fine-tune] dataset='{src_name}' (pretrain_src_idx={pretrain_idx})")
                logger.info(f"   alpha={alpha}, freeze={do_freeze}, mode→Few (ghead→ghead2 copy)")
                logger.info(f"{'='*60}")
                model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                # [main_EEE] fine-tune은 target adaptation처럼: ghead → ghead2 복사 후 mode='Few'
                model.ghead2.load_state_dict(model.ghead.state_dict())
                model.mode = 'Few'
                if do_freeze:
                    model.set_freeze_target()
                fix_seed(args.random_seed)

                t_src_start = time.time()
                ft_pack = pretrain_and_eval_sources(
                    args, model, device, sources=[src_name], patience=20,
                    use_exp=True, sampling_alpha=alpha
                )
                src_sec = time.time() - t_src_start
                ft_sec.append(src_sec)
                logger.info(f"[Case 2 Fine-tune][TIME] '{src_name}' freeze={do_freeze} elapsed: {src_sec:.1f}s ({src_sec/60:.2f}min)")

                ps_test = ft_pack.get('per_source_test', {})
                ft_sources.append(src_name)
                for key, lst in [('auc', ft_auc), ('auprc', ft_auprc), ('acc', ft_acc),
                                 ('f1', ft_f1), ('precision', ft_precision),
                                 ('recall', ft_recall), ('loss', ft_loss)]:
                    vals = ps_test.get(key, [])
                    lst.append(vals[0] if vals else float('nan'))

            ft_total_sec = time.time() - t_ft_total_start
            logger.info(f"[Case 2 Fine-tune][TIME] All sources total (freeze={do_freeze}): {ft_total_sec:.1f}s ({ft_total_sec/60:.2f}min)")

            # 결과 저장
            result_file = os.path.join(
                ft_save_dir,
                f"case2{freeze_tag}_alpha{alpha:.2f}_seed{args.random_seed}_{args.run_tag}.json"
            )
            save_data = {
                'exp_mode': 'case2',
                'freeze_ft': do_freeze,
                'sources': src_list,
                'sampling_alpha': alpha,
                'seed': args.random_seed,
                'pretrain_sec': pretrain_sec,
                'finetune_total_sec': ft_total_sec,
                'per_source_test': {
                    'sources': ft_sources,
                    'auc': ft_auc, 'auprc': ft_auprc, 'acc': ft_acc,
                    'f1': ft_f1, 'precision': ft_precision,
                    'recall': ft_recall, 'loss': ft_loss,
                    'finetune_sec': ft_sec,
                },
            }
            with open(result_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"[Case 2] Results saved to {result_file}")

            # wandb sweep metric
            mean_ft_auc = float(sum(ft_auc) / len(ft_auc)) if ft_auc else 0.0
            mean_ft_auprc = float(sum(ft_auprc) / len(ft_auprc)) if ft_auprc else 0.0
            try:
                per_src_log = {f"case2{freeze_tag}/{s}_auc": a for s, a in zip(ft_sources, ft_auc)}
                per_src_log.update({f"case2{freeze_tag}/{s}_auprc": p for s, p in zip(ft_sources, ft_auprc)})
                per_src_log[f"case2{freeze_tag}/mean_test_auc"] = mean_ft_auc
                per_src_log[f"case2{freeze_tag}/mean_test_auprc"] = mean_ft_auprc
                _wandb_log(per_src_log)

                per_src_summary = {f"final/case2{freeze_tag}_{s}_auc": a for s, a in zip(ft_sources, ft_auc)}
                per_src_summary.update({f"final/case2{freeze_tag}_{s}_auprc": p for s, p in zip(ft_sources, ft_auprc)})
                per_src_summary[f"final/case2{freeze_tag}_mean_auc"] = mean_ft_auc
                per_src_summary[f"final/case2{freeze_tag}_mean_auprc"] = mean_ft_auprc
                _wandb_summary_set(per_src_summary)
            except Exception:
                pass

        # sweep metric용: freeze=False 기준 mean AUC
        try:
            _wandb_log({"final/few_shot_test_auc_mean": mean_ft_auc})
            _wandb_summary_set({"final/few_shot_test_auc_mean": mean_ft_auc})
        except Exception:
            pass

    # ==================================================================
    # [Mode 4b] case2_exclude — Exp B Step 3: F(M-1) source exclusion
    #   특정 source 1개 제외 → 나머지 M-1개로 pretrain → M-1개 fine-tune (freeze 여부 둘 다)
    #   Alpha=1.0 고정
    # ==================================================================
    elif args.exp_mode == 'case2_exclude':
        if not args.exclude_sources:
            raise ValueError("--exclude_sources is required for case2_exclude (at least 1 source)")
        excluded = args.exclude_sources
        if isinstance(excluded, str):
            excluded = [excluded]
        remaining = [s for s in src_list if s not in excluded]
        if not remaining:
            raise ValueError("No sources remaining after exclusion!")

        excluded_tag = "+".join(excluded)
        logger.info(f"\n{'='*60}\n>>> [Case 2 Exclude] excluded={excluded_tag}\n"
                    f"    remaining={remaining}\n{'='*60}")

        alpha = 1.0  # 고정

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # ── Pretrain: remaining source, alpha=1.0 ──
        args.use_lcg = False
        pretrain_and_eval_sources(
            args, model, device, sources=remaining, patience=20,
            use_exp=True, sampling_alpha=alpha
        )

        all_loaders_init = {}
        for s in remaining:
            tr, _, _, _ = load_one(args, s, sampling_alpha=alpha, use_exp=True)
            all_loaders_init[s] = tr
        init_lcg(args, model, all_loaders_init, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        pretrain_and_eval_sources(
            args, model, device, sources=remaining, patience=30,
            use_exp=True, sampling_alpha=alpha
        )

        # pretrained state 저장
        pretrained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"[Case 2 Exclude] Pretrained model saved (excluded={excluded_tag})")

        # ── Fine-tune: remaining source 각각에 freeze=False → freeze=True 순차 ──
        for do_freeze in [False, True]:
            freeze_tag = '_freeze' if do_freeze else ''
            ft_save_dir = f"/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}/case2_exclude{freeze_tag}"
            os.makedirs(ft_save_dir, exist_ok=True)

            logger.info(f"\n{'='*40}\n>>> [Case 2 Exclude] Fine-tune (freeze={do_freeze})\n{'='*40}")

            ft_sources = []
            ft_auc, ft_auprc, ft_acc, ft_f1 = [], [], [], []
            ft_precision, ft_recall, ft_loss = [], [], []

            for src_name in remaining:
                logger.info(f"\n{'='*60}")
                logger.info(f"[Case 2 Exclude Fine-tune] dataset='{src_name}' "
                            f"(excluded={excluded_tag})")
                logger.info(f"   alpha={alpha}, freeze={do_freeze}, mode→Few (ghead→ghead2 copy)")
                logger.info(f"{'='*60}")
                model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                model.ghead2.load_state_dict(model.ghead.state_dict())
                model.mode = 'Few'
                if do_freeze:
                    model.set_freeze_target()
                fix_seed(args.random_seed)

                ft_pack = pretrain_and_eval_sources(
                    args, model, device, sources=[src_name], patience=20,
                    use_exp=True, sampling_alpha=alpha
                )

                ps_test = ft_pack.get('per_source_test', {})
                ft_sources.append(src_name)
                for key, lst in [('auc', ft_auc), ('auprc', ft_auprc), ('acc', ft_acc),
                                 ('f1', ft_f1), ('precision', ft_precision),
                                 ('recall', ft_recall), ('loss', ft_loss)]:
                    vals = ps_test.get(key, [])
                    lst.append(vals[0] if vals else float('nan'))

            # 결과 저장
            result_file = os.path.join(
                ft_save_dir,
                f"case2_exclude{freeze_tag}_excl-{excluded_tag}_seed{args.random_seed}_{args.run_tag}.json"
            )
            save_data = {
                'exp_mode': 'case2_exclude',
                'freeze_ft': do_freeze,
                'excluded_sources': excluded,
                'remaining_sources': remaining,
                'sampling_alpha': alpha,
                'seed': args.random_seed,
                'per_source_test': {
                    'sources': ft_sources,
                    'auc': ft_auc, 'auprc': ft_auprc, 'acc': ft_acc,
                    'f1': ft_f1, 'precision': ft_precision,
                    'recall': ft_recall, 'loss': ft_loss,
                },
            }
            with open(result_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"[Case 2 Exclude] Results saved to {result_file}")

            # wandb: mean test AUC 로그 (sweep metric용)
            mean_ft_auc = float(np.mean(ft_auc)) if ft_auc else 0.0
            mean_ft_auprc = float(np.mean(ft_auprc)) if ft_auprc else 0.0
            try:
                _wandb_log({
                    f"case2_exclude{freeze_tag}/mean_test_auc": mean_ft_auc,
                    f"case2_exclude{freeze_tag}/mean_test_auprc": mean_ft_auprc,
                    f"case2_exclude{freeze_tag}/excluded": excluded_tag,
                })
                _wandb_summary_set({
                    f"final/case2_exclude{freeze_tag}_mean_auc": mean_ft_auc,
                    f"final/case2_exclude{freeze_tag}_mean_auprc": mean_ft_auprc,
                })
            except Exception:
                pass

        # sweep metric용: freeze=False 기준 mean AUC를 'final/few_shot_test_auc_mean'에 기록
        try:
            _wandb_log({"final/few_shot_test_auc_mean": mean_ft_auc})
            _wandb_summary_set({"final/few_shot_test_auc_mean": mean_ft_auc})
        except Exception:
            pass

    # ==================================================================
    # [Mode 5] exp_b_analysis — Exp B Step 1/2: LCG routing analysis
    # ==================================================================
    elif args.exp_mode == 'exp_b_analysis':
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required for exp_b_analysis (Case F checkpoint)")

        logger.info(f"\n{'='*60}\n>>> [Exp B] LCG Routing Analysis\n{'='*60}")
        logger.info(f"Loading Case F model from: {args.ckpt_path}")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        args.use_lcg = True
        model.eval()

        # Source별 routing coefficient 수집
        pi_per_source = {}
        for src_name in src_list:
            r_src = prepare_exp_embedding_dataloaders(args, src_name, alpha=1.0)
            test_loader_src = r_src['loaders'][2]  # test loader
            all_pi = []
            with torch.no_grad():
                for batch in test_loader_src:
                    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                    _ = model.predict(batch, return_all=True)
                    pi = model.graph_quantizer.last_pi  # (B, M)
                    all_pi.append(pi.cpu())
            pi_cat = torch.cat(all_pi, dim=0)
            pi_mean = pi_cat.mean(dim=0).numpy()
            pi_per_source[src_name] = pi_mean
            logger.info(f"[Routing] {src_name}: mean pi = {pi_mean.round(4)}")

        # Target routing coefficient
        r_t = prepare_embedding_dataloaders(args, args.target_data, is_source=False)
        target_loader = r_t['loaders'][0]  # train pool (전체 target)
        all_pi_t = []
        with torch.no_grad():
            for batch in target_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                _ = model.predict(batch, return_all=True)
                pi = model.graph_quantizer.last_pi
                all_pi_t.append(pi.cpu())
        pi_target = torch.cat(all_pi_t, dim=0).mean(dim=0).numpy()
        logger.info(f"[Routing] Target ({args.target_data}): mean pi = {pi_target.round(4)}")

        # Relevance 계산: dot product
        relevance = {}
        for src_name, pi_src in pi_per_source.items():
            rel = float(np.dot(pi_target, pi_src))
            relevance[src_name] = rel
            logger.info(f"[Relevance] {src_name}: {rel:.4f}")

        # 결과 저장
        result_file = os.path.join(
            exp_results_dir,
            f"exp_b_routing_seed{args.random_seed}_{args.run_tag}.json"
        )
        save_data = {
            'exp_mode': 'exp_b_analysis',
            'sources': src_list,
            'target': args.target_data,
            'seed': args.random_seed,
            'pi_per_source': {k: v.tolist() for k, v in pi_per_source.items()},
            'pi_target': pi_target.tolist(),
            'relevance': relevance,
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"[Exp B Analysis] Results saved to {result_file}")

        # 정렬된 relevance 출력
        sorted_rel = sorted(relevance.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\n[Relevance Ranking]")
        for rank, (name, rel) in enumerate(sorted_rel, 1):
            logger.info(f"  {rank}. {name}: {rel:.4f}")

    # ==================================================================
    # [Mode 4] exp_b_retrain — Exp B Step 3: source exclusion retraining
    # ==================================================================
    elif args.exp_mode == 'exp_b_retrain':
        excluded = args.exclude_sources
        if not excluded:
            raise ValueError("--exclude_sources is required for exp_b_retrain")

        remaining = [s for s in src_list if s not in excluded]
        logger.info(f"\n{'='*60}\n>>> [Exp B] Retraining without: {excluded}\n{'='*60}")
        logger.info(f"Remaining sources: {remaining}")

        if not remaining:
            raise ValueError("No sources remaining after exclusion!")

        model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.dropout_rate, args.llm_model,
                      experiment_id, mode="Full").to(device)

        # Phase 1: Vanilla GAT (remaining sources only)
        args.use_lcg = False
        _ = pretrain_and_eval_sources(
            args, model, device, sources=remaining, patience=20,
            use_exp=True, sampling_alpha=1.0
        )

        # LCG Init
        all_loaders_init = {}
        for s in remaining:
            tr, _, _, _ = load_one(args, s, sampling_alpha=1.0, use_exp=True)
            all_loaders_init[s] = tr
        init_lcg(args, model, all_loaders_init, device, save_dir=exp_results_dir,
                 strategy=args.lcg_strategy, injection_scale=0.1)
        args.use_lcg = True
        # [main_EEE] Phase 2: mode='Full' 유지 (local_loss + ghead)
        fix_seed(args.random_seed)

        # Phase 2: Joint Training
        full_pack = pretrain_and_eval_sources(
            args, model, device, sources=remaining, patience=30,
            use_exp=True, sampling_alpha=1.0
        )

        # Target adaptation (few-shot) — main_SS.py 로직 재사용
        args.use_target_head = True
        model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                          args.dropout_rate, args.llm_model,
                          experiment_id, mode="Few").to(device)
        model_few.args.use_lcg = True
        model_few.load_state_dict(model.state_dict(), strict=False)
        fix_seed(args.random_seed)

        r_t = prepare_embedding_dataloaders(args, args.target_data, is_source=False)
        train_loader_t, _, test_loader_t = r_t['loaders']
        args.num_classes = r_t['num_classes']
        args.output_dim = args.num_classes if args.num_classes > 2 else 1
        is_binary_t = (args.num_classes == 2)
        crit_t = nn.BCEWithLogitsLoss() if is_binary_t else nn.CrossEntropyLoss()

        total_shot = args.few_shot * args.num_classes
        FEW_SHOT_EPOCHS = 100 if total_shot <= 16 else 240

        model_few.set_freeze_target()
        train_fn_few = binary_train if is_binary_t else multi_train
        eval_fn_few  = binary_evaluate if is_binary_t else multi_evaluate

        R = int(getattr(args, 'support_resamples', 1))
        base_state_cpu = {k: v.cpu() for k, v in model_few.state_dict().items()}
        ep_test_metrics = []

        for r in range(R):
            current_seed = args.random_seed + (r + 1)
            fix_seed(current_seed)
            model_few.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()}, strict=False)
            model_few.set_freeze_target()

            gat_params_few, lcg_params_few, head_params_few = [], [], []
            for name, p_param in model_few.named_parameters():
                if not p_param.requires_grad: continue
                if 'basis' in name:
                    gat_params_few.append(p_param)
                elif 'latent_graph' in name:
                    lcg_params_few.append(p_param)
                else:
                    head_params_few.append(p_param)

            optimizer_few = optim.Adam([
                {'params': gat_params_few,  'lr': args.source_lr_few},
                {'params': lcg_params_few,  'lr': args.source_lr_few},
                {'params': head_params_few, 'lr': args.source_lr_few}
            ], weight_decay=3e-5)

            def linear_lr_lambda(epoch, _tot=FEW_SHOT_EPOCHS):
                return max(0.0, 1.0 - epoch / float(_tot))
            scheduler_few = LambdaLR(optimizer_few, linear_lr_lambda)

            train_loader_epi = get_few_shot_embedding_samples_(train_loader_t, args, seed=current_seed)

            for epoch in range(FEW_SHOT_EPOCHS):
                model_few.train()
                train_fn_few(model_few, train_loader_epi, crit_t, optimizer_few, device)
                scheduler_few.step()

            # 최종 평가
            model_few.eval()
            res_train = eval_fn_few(model_few, train_loader_epi, crit_t, device)
            if isinstance(res_train, tuple) and len(res_train) == 2:
                res_g_tr, _ = res_train
                _, y_true_tr_few, y_pred_tr_few = res_g_tr
            else:
                _, y_true_tr_few, y_pred_tr_few = res_train
            thr_few = find_optimal_threshold(y_true_tr_few, y_pred_tr_few) if is_binary_t else None

            (_, tauc, tauprc, tprec, trec, tf1, tacc, _, _) = final_test_evaluate(
                model_few, test_loader_t, crit_t, device, is_binary_t,
                threshold=thr_few, mode='Few', args=args
            )
            ep_test_metrics.append((tauc, tauprc, tprec, trec, tf1, tacc))
            logger.info(f"[Retrain][r={r}] AUC={tauc:.4f} AUPRC={tauprc:.4f} ACC={tacc:.4f}")

        ep_arr = np.asarray(ep_test_metrics, dtype=np.float32)
        mean_metrics = ep_arr.mean(axis=0).tolist()
        std_metrics  = ep_arr.std(axis=0).tolist()

        logger.info(f"[Retrain Summary] AUC={mean_metrics[0]:.4f}+-{std_metrics[0]:.4f} "
                    f"AUPRC={mean_metrics[1]:.4f}+-{std_metrics[1]:.4f}")

        # 결과 저장
        excluded_tag = "+".join(excluded)
        result_file = os.path.join(
            exp_results_dir,
            f"retrain_excl-{excluded_tag}_seed{args.random_seed}_{args.run_tag}.json"
        )
        save_data = {
            'exp_mode': 'exp_b_retrain',
            'excluded': excluded,
            'remaining': remaining,
            'seed': args.random_seed,
            'few_shot': args.few_shot,
            'target': args.target_data,
            'per_source_test': full_pack.get('per_source_test', {}),
            'target_adaptation': {
                'mean_auc': mean_metrics[0], 'std_auc': std_metrics[0],
                'mean_auprc': mean_metrics[1], 'std_auprc': std_metrics[1],
                'mean_acc': mean_metrics[5], 'std_acc': std_metrics[5],
                'mean_f1': mean_metrics[4], 'std_f1': std_metrics[4],
            },
        }
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"[Exp B Retrain] Results saved to {result_file}")

    else:
        raise ValueError(f"Unknown exp_mode: {args.exp_mode}")

    logger.info(f"Total experiment time: {format_time(time.time() - start_time)}")
    _wandb_log({"exp/total_time_sec": float(time.time() - start_time)})
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()