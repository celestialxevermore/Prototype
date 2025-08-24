import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datetime import datetime
import psutil

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import label_binarize

from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_results_, save_results_, wrap_up_results_
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_S import Model

# -------------------------------

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger()

p = psutil.Process()
try:
    p.cpu_affinity(range(1, 64))
except Exception:
    pass

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# -------------------------------
# Multi-source batch mixers
# -------------------------------

class _PseudoDataset:
    def __init__(self, length: int):
        self._len = int(length)
    def __len__(self):
        return self._len

class MultiSourceBatcher:
    """
    여러 source train 로더에서 step마다 랜덤으로 배치를 뽑아 하나의 '에폭용' 로더처럼 동작.
    - __len__: 한 에폭의 스텝 수(=배치 수) = steps_per_epoch
    - .dataset: 총 샘플 수 역할만 하는 가짜 dataset 제공
    """
    def __init__(self, loaders, steps_per_epoch):
        self.loaders = loaders
        self.steps_per_epoch = steps_per_epoch
        self._iters = [iter(ld) for ld in self.loaders]
        total_samples = 0
        for ld in self.loaders:
            n = getattr(ld, "dataset", None)
            total_samples += len(n) if n is not None else (len(ld) * getattr(ld, "batch_size", 1))
        self.dataset = _PseudoDataset(total_samples)

    def __iter__(self):
        steps = 0
        while steps < self.steps_per_epoch:
            li = random.randrange(len(self.loaders))
            try:
                batch = next(self._iters[li])
            except StopIteration:
                self._iters[li] = iter(self.loaders[li])
                batch = next(self._iters[li])
            yield batch
            steps += 1

    def __len__(self):
        return self.steps_per_epoch

class MultiSourceEvalLoader:
    """
    여러 source val/test 로더를 이어붙여 평가용으로 사용.
    - __len__: 총 배치 수 합
    - .dataset: 총 샘플 수 합(호환용)
    """
    def __init__(self, loaders):
        self.loaders = loaders
        total_samples = 0
        for ld in self.loaders:
            n = getattr(ld, "dataset", None)
            total_samples += len(n) if n is not None else (len(ld) * getattr(ld, "batch_size", 1))
        self.dataset = _PseudoDataset(total_samples)

    def __iter__(self):
        for ld in self.loaders:
            for batch in ld:
                yield batch

    def __len__(self):
        return sum(len(ld) for ld in self.loaders)

# -------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task (main_S)')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--k_basis', type=int, default=4)
    parser.add_argument('--model', type=str, default='NORM_GNN')

    # 멀티소스
    parser.add_argument('--source_data', nargs='+', default=['heart'],
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','heart_target_1','heart_target_2','heart_target_3','heart_target_4','myocardial','cleveland','heart_statlog','hungarian','switzerland','breast','magic_telescope','forest_covertype_sampled','higgs_sampled'])

    parser.add_argument('--target_data', type=str, default='hungarian')
    parser.add_argument('--few_shot', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=1e-4)
    parser.add_argument('--source_lr_few', type=float, default=1e-5)
    parser.add_argument('--llm_model', type=str, default='gpt2_mean',
                        choices=['gpt2_mean','gpt2_auto','sentence-bert','bio-bert','bio-clinical-bert','bio-llama','new','LLAMA_mean','LLAMA_auto'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--des', type=str)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'])
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")
    parser.add_argument('--model_type', type=str, default='TabularFLM',
                        choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3','GAT_edge_4','GAT_edge_5','TabularFLM'])
    parser.add_argument('--label', type=str, choices=['add','no'], default='add')
    parser.add_argument('--enc_type', type=str, choices=['ind','shared'], default='ind')
    parser.add_argument('--meta_type', type=str, choices=['meta_attn','meta_mlp'], default='meta_attn')
    parser.add_argument('--aggr_type', type=str, choices=['flatten','mean','attn'], default='attn')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--edge_type', default='mlp', choices=['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default='carte', choices=['carte','carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices=['gat_v1','att','gat_v2','gate'])
    parser.add_argument('--del_feat', nargs='+', default=[])
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true')
    parser.add_argument('--viz_heatmap', action='store_true')
    parser.add_argument('--viz_graph', action='store_true')


    # 멀티소스 스텝 수(한 에폭당)
    parser.add_argument('--steps_per_epoch', type=int, default=500)

    args = parser.parse_args()
    args.table_path = "/storage/personal/eungyeop/dataset/table/"
    return args

# -------------------------------

def _binary_log_loss(y_true, y_prob, eps=1e-7):
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    y = np.asarray(y_true).astype(np.float32)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _multiclass_log_loss(y_true, y_prob, eps=1e-7):
    P = np.asarray(y_prob)
    P = np.clip(P, eps, 1 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    y = np.asarray(y_true).astype(int)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))

def _describe_loader(loader):
    def _norm(x):
        if x is None: return None
        if isinstance(x, (list, tuple, set)):
            xs = [str(i) for i in x if i is not None]
            return ",".join(xs) if xs else None
        return str(x)
    cand = getattr(loader, "names", None)
    if cand:
        cand = _norm(cand)
        if cand: return cand
    cand = getattr(loader, "name", None) or getattr(getattr(loader, "dataset", None), "name", None)
    cand = _norm(cand)
    if cand: return cand
    loaders = getattr(loader, "loaders", None)
    if loaders:
        names = []
        for i, ld in enumerate(loaders):
            nm = getattr(ld, "name", None) or getattr(getattr(ld, "dataset", None), "name", None)
            nm = _norm(nm) or f"ds{i}"
            names.append(nm)
        if names:
            return ",".join(names)
    return "unknown"

def train_and_validate(args, model, train_loader, val_loader, criterion, optimizer, device, epochs, is_binary, patience=10, mode="Full"):
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []

    train_func    = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    best_val_auc   = 0.0
    no_improve     = 0
    best_epoch     = 0
    best_threshold = 0.5
    best_model_state = None

    train_src_str = _describe_loader(train_loader)
    val_src_str   = _describe_loader(val_loader)

    for epoch in range(epochs):
        _ = train_func(model, train_loader, criterion, optimizer, device)

        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        _, y_true_val,   y_pred_val   = evaluate_func(model, val_loader,   criterion, device)

        if is_binary:
            train_loss_disp = _binary_log_loss(y_true_train, y_pred_train)
            val_loss_disp   = _binary_log_loss(y_true_val,   y_pred_val)
        else:
            train_loss_disp = _multiclass_log_loss(y_true_train, y_pred_train)
            val_loss_disp   = _multiclass_log_loss(y_true_val,   y_pred_val)

        train_losses.append(train_loss_disp); val_losses.append(val_loss_disp)

        if is_binary:
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            val_auc   = roc_auc_score(y_true_val,   y_pred_val)

            precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_pred_val)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            idx = int(np.argmax(f1_scores))
            current_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

            y_pred_train_bin = (np.asarray(y_pred_train) > current_threshold).astype(int)
            y_pred_val_bin   = (np.asarray(y_pred_val)   > current_threshold).astype(int)

            train_precision = precision_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_precision   = precision_score(y_true_val,   y_pred_val_bin,   zero_division=0)
            train_recall    = recall_score(y_true_train,    y_pred_train_bin, zero_division=0)
            val_recall      = recall_score(y_true_val,      y_pred_val_bin,   zero_division=0)
            train_f1        = f1_score(y_true_train,        y_pred_train_bin, zero_division=0)
            val_f1          = f1_score(y_true_val,          y_pred_val_bin,   zero_division=0)
            train_acc       = accuracy_score(y_true_train,  y_pred_train_bin)
            val_acc         = accuracy_score(y_true_val,    y_pred_val_bin)
        else:
            n_classes        = np.asarray(y_pred_train).shape[1]
            y_true_train_bin = label_binarize(y_true_train, classes=range(n_classes))
            y_true_val_bin   = label_binarize(y_true_val,   classes=range(n_classes))
            train_auc        = roc_auc_score(y_true_train_bin, y_pred_train, multi_class='ovr', average='macro')
            val_auc          = roc_auc_score(y_true_val_bin,   y_pred_val,   multi_class='ovr', average='macro')

            preds_train_argmax = np.asarray(y_pred_train).argmax(axis=1)
            preds_val_argmax   = np.asarray(y_pred_val).argmax(axis=1)
            train_precision    = precision_score(y_true_train, preds_train_argmax, average='macro', zero_division=0)
            val_precision      = precision_score(y_true_val,   preds_val_argmax,   average='macro', zero_division=0)
            train_recall       = recall_score(y_true_train,    preds_train_argmax, average='macro', zero_division=0)
            val_recall         = recall_score(y_true_val,      preds_val_argmax,   average='macro', zero_division=0)
            train_f1           = f1_score(y_true_train,        preds_train_argmax, average='macro', zero_division=0)
            val_f1             = f1_score(y_true_val,          preds_val_argmax,   average='macro', zero_division=0)
            train_acc          = accuracy_score(y_true_train,  preds_train_argmax)
            val_acc            = accuracy_score(y_true_val,    preds_val_argmax)
            current_threshold  = None

        train_aucs.append(train_auc); val_aucs.append(val_auc)
        train_precisions.append(train_precision); val_precisions.append(val_precision)
        train_recalls.append(train_recall);       val_recalls.append(val_recall)
        train_f1s.append(train_f1);               val_f1s.append(val_f1)
        train_accs.append(train_acc);             val_accs.append(val_acc)

        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"[Train:{train_src_str} | Val:{val_src_str}] "
            f"Train Loss: {train_loss_disp:.4f}, Val Loss: {val_loss_disp:.4f}, "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
            f"Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            no_improve   = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if current_threshold is not None:
                best_threshold = current_threshold
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("No best_model_state saved; model not updated?")

    return (train_losses, val_losses,
            train_aucs, val_aucs,
            train_precisions, val_precisions,
            train_recalls, val_recalls,
            train_f1s, val_f1s,
            train_accs, val_accs,
            best_epoch, best_val_auc, best_threshold)

def final_test_evaluate(model, test_loader, criterion, device, is_binary, threshold=None):
    evaluate_func = binary_evaluate if is_binary else multi_evaluate
    test_loss, y_true_test, y_pred_test = evaluate_func(model, test_loader, criterion, device)

    if is_binary:
        test_auc = roc_auc_score(y_true_test, y_pred_test)
        if threshold is None:
            threshold = 0.5
        y_pred_test_bin = (y_pred_test > threshold).astype(int)
        test_precision = precision_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_f1 = f1_score(y_true_test, y_pred_test_bin, zero_division=0)
        test_acc = accuracy_score(y_true_test, y_pred_test_bin)
    else:
        n_classes = y_pred_test.shape[1]
        y_true_test_bin = label_binarize(y_true_test, classes=range(n_classes))
        test_auc = roc_auc_score(y_true_test_bin, y_pred_test, multi_class='ovr', average='macro')
        preds_argmax = y_pred_test.argmax(axis=1)
        test_precision = precision_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_recall = recall_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_f1 = f1_score(y_true_test, preds_argmax, average='macro', zero_division=0)
        test_acc = accuracy_score(y_true_test, preds_argmax)

    logger.info(f"[Test] Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return test_loss, test_auc, test_precision, test_recall, test_f1, test_acc, y_true_test, y_pred_test

def build_source_loaders(args, dataset_names):
    loaders_by_ds = {}
    num_classes_set = set()
    for ds in dataset_names:
        res = prepare_embedding_dataloaders(args, ds)
        tr, va, te = res['loaders']
        loaders_by_ds[ds] = {'train': tr, 'val': va, 'test': te, 'num_classes': res['num_classes']}
        num_classes_set.add(res['num_classes'])
        logger.info(f"[Source {ds}] num_classes={res['num_classes']} | train={len(tr)}, val={len(va)}, test={len(te)}")
    if len(num_classes_set) != 1:
        logger.warning(f"[Multi-source] Mixed num_classes across sources: {num_classes_set} (현재 동일 클래스 수 가정)")
    return loaders_by_ds, num_classes_set.pop()

# -------------------------------

def _log_fewshot_stats(dl, name):
    # embedding 형태: item은 dict이고 'y' 키 보유
    ds = getattr(dl, 'dataset', None)
    if isinstance(ds, list):
        labels = [int(x['y'].item()) for x in ds]
        cnt = dict(Counter(labels))
        logger.info(f"[{name}] size={len(labels)} class_counts={cnt}")
    else:
        try:
            labels = []
            for batch in dl:
                y = batch['y']
                labels.extend([int(t.item()) for t in y])
            cnt = dict(Counter(labels))
            logger.info(f"[{name}] size={len(labels)} class_counts={cnt}")
        except Exception:
            logger.info(f"[{name}] (stats skipped)")

# -------------------------------

def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    logger.info(f"Starting experiment with sources: {args.source_data} -> target: {args.target_data}")
    logger.info(f"Device: {device}")

    # --- 1) Multi-source pre-training (Full) ---
    src_loaders_by_ds, src_num_classes = build_source_loaders(args, args.source_data)
    args.num_classes = src_num_classes
    args.output_dim = src_num_classes if src_num_classes > 2 else 1
    is_binary = (src_num_classes == 2)

    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Full").to(device)

    # ✅ Full은 절대 동결 금지 (중요한 2줄)
    model_full.frozen = False
    model_full.apply_freeze_policy()

    train_mix = MultiSourceBatcher([src_loaders_by_ds[ds]['train'] for ds in args.source_data],
                                   steps_per_epoch=args.steps_per_epoch)
    val_mix   = MultiSourceEvalLoader([src_loaders_by_ds[ds]['val'] for ds in args.source_data])

    source_names = args.source_data if isinstance(args.source_data, (list, tuple)) else [args.source_data]
    if not getattr(train_mix, "names", None): train_mix.names = source_names
    if not getattr(val_mix,   "names", None): val_mix.names   = source_names

    optimizer_full = optim.Adam((p for p in model_full.parameters() if p.requires_grad),
                                lr=args.source_lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    logger.info("[Full-shot] Multi-source step-based training start...")
    (train_losses_full, val_losses_full,
     train_aucs_full, val_aucs_full,
     train_precisions_full, val_precisions_full,
     train_recalls_full, val_recalls_full,
     train_f1s_full, val_f1s_full,
     train_accs_full, val_accs_full,
     best_epoch_full, best_val_auc_full, best_threshold_full
    ) = train_and_validate(args, model_full, train_mix, val_mix, criterion, optimizer_full,
                           device, args.train_epochs, is_binary, mode="Full")

    # Full 테스트 집계(선택이지만 결과 저장에 포함)
    test_mix = MultiSourceEvalLoader([src_loaders_by_ds[ds]['test'] for ds in args.source_data])
    (test_loss_full, test_auc_full, test_precision_full, test_recall_full, test_f1_full,
     test_acc_full, all_y_true_full, all_y_pred_full) = final_test_evaluate(
        model_full, test_mix, criterion, device, is_binary, threshold=best_threshold_full
    )

    # --- 2) Target few-shot adaptation ---
    logger.info("Preparing target dataloaders...")
    tgt_res = prepare_embedding_dataloaders(args, args.target_data)
    tgt_train_full, tgt_val, tgt_test = tgt_res['loaders']
    tgt_num_classes = tgt_res['num_classes']
    if tgt_num_classes != args.num_classes:
        logger.warning(f"[Target] num_classes ({tgt_num_classes}) != source num_classes ({args.num_classes}). 동일 클래스 수 가정.")

    logger.info(f"Preparing target few-shot samples (K={args.few_shot})...")
    tgt_train_few = get_few_shot_embedding_samples(tgt_train_full, args)
    _log_fewshot_stats(tgt_train_few, f"Target-Train Few (K={args.few_shot})")

    model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Few").to(device)
    model_few.load_state_dict(model_full.state_dict(), strict=False)

    # ✅ Few는 args.frozen에만 따름 (중요한 2줄)
    model_few.frozen = args.frozen
    model_few.apply_freeze_policy()

    optimizer_few = optim.Adam((p for p in model_few.parameters() if p.requires_grad),
                               lr=args.source_lr_few, weight_decay=1e-5)

    logger.info("[Few-shot(Target)] Start adaptation...")
    (train_losses_few, val_losses_few,
     train_aucs_few, val_aucs_few,
     train_precisions_few, val_precisions_few,
     train_recalls_few, val_recalls_few,
     train_f1s_few, val_f1s_few,
     train_accs_few, val_accs_few,
     best_epoch_few, best_val_auc_few, best_threshold_few
    ) = train_and_validate(args, model_few, tgt_train_few, tgt_val, criterion, optimizer_few,
                           device, args.train_epochs, is_binary, mode="Few")

    logger.info("[Few-shot(Target)] Final Testing with best threshold from Validation")
    (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
     test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
        model_few, tgt_test, criterion, device, is_binary, threshold=best_threshold_few
    )

    # 결과 묶기
    full_ours_results = wrap_up_results_(
        train_losses_full, val_losses_full, [test_loss_full],
        train_aucs_full,   val_aucs_full,   [test_auc_full],
        train_precisions_full, val_precisions_full, [test_precision_full],
        train_recalls_full,    val_recalls_full,    [test_recall_full],
        train_f1s_full,        val_f1s_full,        [test_f1_full],
        [all_y_true_full], [all_y_pred_full],
        best_epoch_full, test_auc_full, test_acc_full,
        test_precision_full, test_recall_full, test_f1_full,
        train_accs=train_accs_full,
        val_accs=val_accs_full,
        test_accs=[test_acc_full]
    )

    few_ours_results = wrap_up_results_(
        train_losses_few, val_losses_few, [test_loss_few],
        train_aucs_few,   val_aucs_few,   [test_auc_few],
        train_precisions_few, val_precisions_few, [test_precision_few],
        train_recalls_few,    val_recalls_few,    [test_recall_few],
        train_f1s_few,        val_f1s_few,        [test_f1_few],
        [all_y_true_few], [all_y_pred_few],
        best_epoch_few, test_auc_few, test_acc_few,
        test_precision_few, test_recall_few, test_f1_few,
        train_accs=train_accs_few,
        val_accs=val_accs_few,
        test_accs=[test_acc_few]
    )

    results = prepare_results_(full_ours_results, few_ours_results)

    logger.info("Saving results...")
    import copy
    args_for_save = copy.copy(args)
    if isinstance(args_for_save.source_data, (list, tuple)):
        args_for_save.source_data = "_".join(map(str, args_for_save.source_data))
    save_results_(args_for_save, results)
    logger.info("Results saved")

    end_time = time.time()
    logger.info(f"Total experiment time: {format_time(end_time - start_time)}")

# -------------------------------

if __name__ == "__main__":
    main()
