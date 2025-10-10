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
from main_G import final_test_evaluate  # few-shot 학습/테스트 루틴 사용
import psutil
from utils.visualization import visualize_model_structure
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
    parser.add_argument('--num_shared_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--k_basis', type=int, default=8)
    parser.add_argument('--model', type=str, default='NORM_GNN')
    parser.add_argument('--source_data', nargs='+',
                        default=['heart_target_1', 'heart_target_2', 'heart_target_3', 'heart_target_4'],
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart',
                                 'heart_target_1','heart_target_2','heart_target_3','heart_target_4','myocardial',
                                 'cleveland','heart_statlog','hungarian','switzerland','breast','magic_telescope',
                                 'forest_covertype_sampled','higgs_sampled'])
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
    parser.add_argument('--model_type', type=str, default='TabularFLM',
                        choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3','GAT_edge_4','GAT_edge_5','TabularFLM'])
    parser.add_argument('--label', type=str, choices=['add','no'], default='add')
    parser.add_argument('--enc_type', type=str, choices=['ind','shared'], default='ind')
    parser.add_argument('--meta_type', type=str, choices=['meta_attn','meta_mlp'], default='meta_attn')
    parser.add_argument('--aggr_type', type=str, choices=['flatten','mean','attn'], default='attn')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--frozen', type=bool, default=False)
    parser.add_argument('--edge_type', default='mlp', choices=['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default='carte', choices=['carte','carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices=['gat_v1','att','gat_v2','gate'])
    parser.add_argument('--del_feat', nargs='+', default=[],
                        help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
    parser.add_argument('--viz_heatmap', action='store_true', help='Visualize heatmap')
    parser.add_argument('--viz_graph', action='store_true', help='Visualize graph')
    parser.add_argument('--use_target_head', type=bool, default=False)
    parser.add_argument('--sim_threshold', type=float, default=0.5, help='Subgraph attention similarity threshold')

    # coord_kmeans
    parser.add_argument('--coord_softmax_temp', type=float, default=0.5, help='Coordinator softmax temperature (lower = sharper).')
    parser.add_argument('--coord_reg_lambda', type=float, default=0.2, help='Weight of KL(coord) regularizer during Few-shot.')
    parser.add_argument('--coord_target_mode', type=str, default='soft', choices=['soft', 'hard'], help='Centroid target mode for coordinate regularization.')
    parser.add_argument('--coord_tau', type=float, default=0.3,help='Temperature for soft centroid mixing (soft target).')


    # kernel 전용
    parser.add_argument('--slot_kernel_rank', type=int, default=512)  # None이면 K로 세팅
    parser.add_argument('--slot_laplacian_lambda', type=float, default=0.0)
    parser.add_argument("--n_slots", type=int, default=8, help="Global slot space number M")
    parser.add_argument("--slot_dim", type=int, default=16, help="Global slot space latent dimension K")
    # 스칼라/온도/정규화 계수
    parser.add_argument('--slot_g_temp', type=float, default=1.0)
    parser.add_argument('--slot_g_sparse_l1', type=float, default=0.0)
        # BasisGAT
    # BasisGAT 스택 관련
    parser.add_argument('--num_basis_layers', type=int, default=2, help='Number of stacked BasisGAT layers.')
    parser.add_argument('--slot_orth_lambda',type=float, default = 0.1)
    parser.add_argument('--slot_usage_lambda',type=float , default = 0.1)
    parser.add_argument('--slot_g_mode', type=str, default='gw', choices=['markov','kernel','gw'])
    parser.add_argument('--g_frob_div_lambda', type=float, default=0.15)   # ✅ 추천: 0.01~0.02
    parser.add_argument('--gw_eps', type=float, default = 0.08)
    parser.add_argument("--gw_sigma", type = float, default = 1.2)
    parser.add_argument("--alpha_scale", type = float, default = 1.0)
    parser.add_argument("--gw_outer_iters", type=int, default=10)
    parser.add_argument("--gw_sinkhorn_iters", type=int, default=30)
    parser.add_argument("--gw_sinkhorn_eps", type=float, default=1e-6)
    # temperatures
    parser.add_argument('--slot_aff_temp', type=float, default=0.5)   # P의 softmax 온도(Attention)
    parser.add_argument('--slot_graph_temp', type=float, default=0.5) # Q의 softmax 온도

    # Sinkhorn 세부
    parser.add_argument('--slot_g_sinkhorn_iters', type=int, default=20)
    parser.add_argument('--slot_g_sinkhorn_eps', type=float, default=1e-6)

    # 관계 마스크 스코어러
    parser.add_argument('--relation_scorer_type', type=str, default='slot', choices=['pair_mlp', 'query'],help='How to build per-head Var-Var mask M: pairwise MLP or relation queries.')
    parser.add_argument('--rel_input_dim', type=int, default=512,help='Hidden size for relation scorer MLP. If -1, set to max(64, input_dim//2).')
    parser.add_argument('--rel_hidden_dim', type=int, default=256,help='Hidden size for relation scorer MLP. If -1, set to max(64, input_dim//2).')

    # 마스크를 로짓에 더할 때 세기(γ)
    parser.add_argument('--affinity_gate_gamma', type=float, default=2.0,help='Strength of pre-softmax logit bias from mask M.')

    # 재샘플링(한 seed 내에서 support set 여러 번 뽑아 평균)
    parser.add_argument('--support_resamples', type=int, default=5, help='How many support resamples per seed')
    parser.add_argument('--warmup_ratio', type=float, default=0.06,
                    help='Warmup steps/epochs ratio (0~1)')
    parser.add_argument('--min_lr_mult', type=float, default=0.10,
                        help='Final LR multiplier vs. base LR for cosine annealing (e.g., 0.1 means 10% of base)')
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args

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

def init_kmeans_centroids_from_sources(args, model, device):
    """
    소스들의 train_loader만 모아 coordinates를 수집하고,
    compute_coordinate_centroids_auto를 k=args.k_basis로 고정해 호출하여
    센트로이드를 얻는다.
    """
    src_names = args.source_data if isinstance(args.source_data, (list, tuple)) else [args.source_data]
    src_train_loaders = [load_one(args, s)[0] for s in src_names]  # [train_loader_i, ...]

    # k를 고정하려면 k_min=k_max=K
    K = max(2, int(args.k_basis))
    centroids, best_k, _scores = compute_coordinate_centroids_auto(
        model, src_train_loaders, device,
        k_min=K, k_max=K,         # 고정
        max_batches=0, max_points=0,
        silhouette_sample=50000,
        random_state=args.random_seed
    )
    logger.info(f"[KMeans] centroids shape={tuple(centroids.shape)}, best_k={best_k}")
    return centroids

def find_optimal_threshold(y_true, y_pred):
    # y_pred: sigmoid 결과(확률)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    idx = int(np.argmax(f1s))
    return thresholds[idx] if idx < len(thresholds) else thresholds[-1]


def train_and_validate(args, model, train_loader, val_loader,
                       criterion, optimizer, device, epochs,
                       is_binary, patience=10, mode="Full", scheduler=None, warmup_epochs = 0):
    """
    Train + Validation만 진행하고, Best Validation 성능을 기록한 모델 state를 반환.
    마지막에 Best Threshold도 함께 반환해서 별도의 Test 단계에서 사용.
    (스케줄러가 주어지면 epoch-wise로 step)
    """
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []

    train_func     = binary_train   if is_binary else multi_train
    evaluate_func  = binary_evaluate if is_binary else multi_evaluate

    best_val_auc   = 0.0
    best_epoch     = 0
    no_improve     = 0
    warmup_epochs  = int(warmup_epochs)
    best_threshold = 0.5
    best_model_state = None

    # 경로용 태그(리스트 방어)
    src_tag = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)

    model_sig = f"{args.model_type}_attn-{args.attn_type}_num_basis_{args.num_basis_layers}_num_shared_layers_{args.num_shared_layers}_num_basis_layers_{args.num_basis_layers}_scorer_{args.relation_scorer_type}_no_self_loop_{args.no_self_loop}"
    checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/{mode}/{model_sig}/{args.random_seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # -------- Train --------
        model.train()
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 스케줄러: epoch-wise step (per-step 원하면 train_func 내부에서 step 필요)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception as e:
                logger.warning(f"[Scheduler] step() failed at epoch {epoch+1}: {e}")

        # 현재 LR 로깅
        try:
            curr_lr = optimizer.param_groups[0]['lr']
        except Exception:
            curr_lr = None

        # -------- Evaluate (Train / Val) --------
        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        val_loss, y_true_val,   y_pred_val   = evaluate_func(model, val_loader,   criterion, device)
        val_losses.append(val_loss)

        if is_binary:
            # Binary
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            val_auc   = roc_auc_score(y_true_val,   y_pred_val)
            current_threshold = find_optimal_threshold(y_true_val, y_pred_val)

            y_pred_train_bin = (y_pred_train > current_threshold).astype(int)
            y_pred_val_bin   = (y_pred_val   > current_threshold).astype(int)

            train_precision = precision_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_precision   = precision_score(y_true_val,   y_pred_val_bin,   zero_division=0)
            train_recall    = recall_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_recall      = recall_score(y_true_val,   y_pred_val_bin,   zero_division=0)
            train_f1        = f1_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_f1          = f1_score(y_true_val,   y_pred_val_bin,   zero_division=0)
            train_acc       = accuracy_score(y_true_train, y_pred_train_bin)
            val_acc         = accuracy_score(y_true_val,   y_pred_val_bin)
        else:
            # Multi-class
            n_classes = y_pred_train.shape[1]
            y_true_train_bin = label_binarize(y_true_train, classes=range(n_classes))
            y_true_val_bin   = label_binarize(y_true_val,   classes=range(n_classes))

            train_auc = roc_auc_score(y_true_train_bin, y_pred_train, multi_class='ovr', average='macro')
            val_auc   = roc_auc_score(y_true_val_bin,   y_pred_val,   multi_class='ovr', average='macro')

            preds_train_argmax = y_pred_train.argmax(axis=1)
            preds_val_argmax   = y_pred_val.argmax(axis=1)

            train_precision = precision_score(y_true_train, preds_train_argmax, average='macro', zero_division=0)
            val_precision   = precision_score(y_true_val,   preds_val_argmax,   average='macro', zero_division=0)
            train_recall    = recall_score(y_true_train, preds_train_argmax, average='macro', zero_division=0)
            val_recall      = recall_score(y_true_val,   preds_val_argmax,   average='macro', zero_division=0)
            train_f1        = f1_score(y_true_train, preds_train_argmax, average='macro', zero_division=0)
            val_f1          = f1_score(y_true_val,   preds_val_argmax,   average='macro', zero_division=0)
            train_acc       = accuracy_score(y_true_train, preds_train_argmax)
            val_acc         = accuracy_score(y_true_val,   preds_val_argmax)
            current_threshold = None

        # 로그
        train_aucs.append(train_auc);    val_aucs.append(val_auc)
        train_precisions.append(train_precision); val_precisions.append(val_precision)
        train_recalls.append(train_recall);       val_recalls.append(val_recall)
        train_f1s.append(train_f1);              val_f1s.append(val_f1)
        train_accs.append(train_acc);            val_accs.append(val_acc)

        if curr_lr is not None:
            logger.info(f"[{mode}][Epoch {epoch+1}/{epochs}] "
                        f"LR:{curr_lr:.6f} | "
                        f"Train Loss:{train_loss:.4f} Val Loss:{val_loss:.4f} | "
                        f"Train AUC:{train_auc:.4f} Val AUC:{val_auc:.4f} | "
                        f"Train ACC:{train_acc:.4f} Val ACC:{val_acc:.4f}")
        else:
            logger.info(f"[{mode}][Epoch {epoch+1}/{epochs}] "
                        f"Train Loss:{train_loss:.4f} Val Loss:{val_loss:.4f} | "
                        f"Train AUC:{train_auc:.4f} Val AUC:{val_auc:.4f} | "
                        f"Train ACC:{train_acc:.4f} Val ACC:{val_acc:.4f}")

        # 베스트 갱신 & 체크포인트
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            no_improve   = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if current_threshold is not None:
                best_threshold = current_threshold

            ckpt_path = os.path.join(
                checkpoint_dir,
                f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}_S:{args.random_seed}_{experiment_id}.pt"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
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

    # 루프 종료 후 베스트로 복원
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
    import shutil  # ← 추가
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

    # 검증/테스트/개별-학습 평가용: 단일 로더를 래핑하고 src_idx "고정" 주입
    val_steps   = [MultiSourceStepLoader([vals[i]],   mode='round', seed=args.random_seed, src_idx=i) for i in range(len(vals))]
    test_steps  = [MultiSourceStepLoader([tests[i]],  mode='round', seed=args.random_seed, src_idx=i) for i in range(len(tests))]
    train_steps = [MultiSourceStepLoader([trains[i]], mode='round', seed=args.random_seed, src_idx=i) for i in range(len(trains))]

    is_bin = (args.num_classes == 2)
    crit   = nn.BCEWithLogitsLoss() if is_bin else nn.CrossEntropyLoss()
    opt    = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)

    # ✅ train_epochs==0이면 스케줄러/학습 루프 스킵 (eval-only)
    total_epochs = int(args.train_epochs)
    if total_epochs > 0:
        warmup_epochs = max(1, int(args.warmup_ratio * total_epochs))
        scheduler_ep  = make_warmup_cosine_epochs(
            opt,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_mult=args.min_lr_mult
        )
        logger.info(f"[Pretrain] LR schedule: warmup_epochs={warmup_epochs}, final_mult={args.min_lr_mult}")
    else:
        scheduler_ep = None
        logger.info("[Pretrain] Eval-only run (train_epochs=0). Skipping LR scheduler and training loop.")

    eval_fn  = binary_evaluate if is_bin else multi_evaluate
    train_fn = binary_train    if is_bin else multi_train

    best_per_source = [-1.0] * len(vals)
    no_improve = 0
    best_state = None
    last_best_epoch = -1

    # === 체크포인트 디렉토리 & 파일명 (고정 이름 + 히스토리) ===
    src_tag   = "+".join(args.source_data) if isinstance(args.source_data, (list, tuple)) else str(args.source_data)
    model_sig = f"{args.model_type}_attn-{args.attn_type}_num_basis_{args.num_basis_layers}_num_shared_layers_{args.num_shared_layers}_num_basis_layers_{args.num_basis_layers}_scorer_{args.relation_scorer_type}_no_self_loop_{args.no_self_loop}"
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_latest = os.path.join(ckpt_dir, "best.pt")  # ← 고정 이름(재사용용)
    ckpt_hist   = os.path.join(ckpt_dir, f"best_{experiment_id}.pt")  # ← 기록 보존용

    # === 학습 루프 (total_epochs==0이면 스킵됨) ===
    for epoch in range(total_epochs):
        _ = train_fn(model, tr_step, crit, opt, device)
        if scheduler_ep is not None:
            scheduler_ep.step()

        # val도 src_idx가 들어간 래핑 로더로 평가
        aucs = []
        for vl in val_steps:
            _, y_true, y_pred = eval_fn(model, vl, crit, device)
            if is_bin:
                aucs.append(roc_auc_score(y_true, y_pred))
            else:
                n_cls = y_pred.shape[1]
                y_bin = label_binarize(y_true, classes=range(n_cls))
                aucs.append(roc_auc_score(y_bin, y_pred, multi_class='ovr', average='macro'))

        improved = False
        for i, a in enumerate(aucs):
            if a > best_per_source[i]:
                best_per_source[i] = a
                improved = True

        mean_auc = float(np.mean(aucs))
        logger.info(f"[Pre][Epoch {epoch+1}/{total_epochs}] mean AUC: {mean_auc:.4f} per-source: {['%.4f'%x for x in aucs]}")

        if improved:
            best_state = model.state_dict()
            last_best_epoch = epoch
            no_improve = 0
            # 저장: 최신 고정 파일 + 히스토리 파일
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc_mean': mean_auc,
                'val_aucs_per_source': aucs,
                'args': args
            }, ckpt_latest)
            try:
                shutil.copyfile(ckpt_latest, ckpt_hist)
            except Exception as e:
                logger.warning(f"[Pretrain] history copy failed: {e}")
        else:
            if epoch + 1 > warmup_epochs: 
                no_improve += 1 
            else: 
                no_improve = 0 
            if no_improve >= patience : 
                logger.info(f"[Pre] Early stop at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------------
    # 소스별 threshold 산출(Val) → train/val/test 지표 계산
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
        # ---- val (threshold 산출) ----
        val_loss_i, y_true_val_i, y_pred_val_i = eval_fn(model, val_steps[i], crit, device)
        if is_bin:
            thr_i = find_optimal_threshold(y_true_val_i, y_pred_val_i)
        else:
            thr_i = None

        # ---- train 성능 ----
        train_loss_i, y_true_tr_i, y_pred_tr_i = eval_fn(model, train_steps[i], crit, device)
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

        # ---- val 성능 ----
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

        # ---- test 성능 ----
        test_loss_i, y_true_te_i, y_pred_te_i = eval_fn(model, test_steps[i], crit, device)
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

        # concat for all_y (Full 결과 저장용)
        all_y_true_full_list.append(y_true_te_i)
        all_y_pred_full_list.append(y_pred_te_i)

    # 평균 집계 (Full 파트)
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
    """
    우선순위:
    1) 고정 'best.pt'가 있으면 그걸 사용
    2) 없으면 'best_*.pt' 중 최신 파일을 사용
    3) 없으면 None
    """
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

    # cpu affinity(옵션): 안전 가드
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
    model_sig = f"{args.model_type}_attn-{args.attn_type}_num_basis_{args.num_basis_layers}_rel_id_{args.rel_input_dim}_rel_hd_{args.rel_hidden_dim}_scorer_{args.relation_scorer_type}_no_self_loop_{args.no_self_loop}"
    ckpt_dir  = f"/storage/personal/eungyeop/experiments/checkpoints__/{args.llm_model}/{src_tag}/Pre/{model_sig}/{args.random_seed}"

    loaded_pretrain = False
    full_metrics = None

    ckpt_path = find_pretrain_ckpt(ckpt_dir)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_full.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"[Pretrain] Loaded checkpoint: {ckpt_path}")
        loaded_pretrain = True
    else:
        logger.info(f"[Pretrain] No checkpoint at {ckpt_dir} → run pretraining.")
        _metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=10)
        # 4-shot 리포트는 아래에서 조건적으로 사용
        full_metrics = _metrics if args.few_shot == 4 else None

    # 3) (옵션) 4-shot일 때, 로드된 모델로 소스 리포트만 재평가(eval-only)
    if loaded_pretrain and args.few_shot == 4:
        logger.info("[Full] Using loaded pretrain for source metrics report (eval only).")
        _bak = args.train_epochs
        args.train_epochs = 0
        full_metrics = pretrain_and_eval_sources(args, model_full, device, args.source_data, patience=0)
        args.train_epochs = _bak

    # 4) few-shot 적응: 가중치 복사 → freeze 정책 적용
    args.use_target_head = True
    model_few.load_state_dict(model_full.state_dict(),strict=False)
    model_few.set_freeze_target()

    trainables = [n for n, p in model_few.named_parameters() if p.requires_grad]
    logger.info("Few-shot trainable params:\n" + "\n".join(trainables))

    # 5) KMeans 센트로이드 초기화
    centroids = init_kmeans_centroids_from_sources(args, model_few, device)
    model_few.set_kmeans_centroids(centroids)
    model_few.set_coord_temperature(args.coord_softmax_temp)

    # 6) Target dataloaders
    logger.info(f"[Few-shot] target = {args.target_data}")
    r_t = prepare_embedding_dataloaders(args, args.target_data)
    train_loader_t, val_loader_t, test_loader_t = r_t['loaders']
    num_classes_t = r_t['num_classes']
    args.num_classes = num_classes_t
    args.output_dim  = num_classes_t if num_classes_t > 2 else 1

    is_binary_t = (args.num_classes == 2)
    crit_t = nn.BCEWithLogitsLoss() if is_binary_t else nn.CrossEntropyLoss()

    # 7) Few-shot: support 재샘플링 R회 에피소드 평균
    R = int(getattr(args, 'support_resamples', 1))
    logger.info(f"[Few-shot] support resamples R = {R}")

    base_state_cpu = {k: v.cpu() for k, v in model_full.state_dict().items()}

    # === 에피소드 누적 버퍼 (가변 길이 안전 버전) ===
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
        # ---- 모델/옵티마이저 리셋 ----
        model_few.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()}, strict=False)
        model_few.set_freeze_target()

        optimizer_few = optim.Adam(
            (p for p in model_few.parameters() if p.requires_grad),
            lr=args.source_lr_few,
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
                               optimizer_few, device, args.train_epochs, is_binary_t, mode="Few", scheduler=scheduler_few, warmup_epochs=warmup_epochs_few)

        # ---- 테스트 ----
        (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
         test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
            model_few, test_loader_t, crit_t, device, is_binary_t, threshold=best_threshold_few
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