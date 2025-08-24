import os
import time
import random
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, accuracy_score
)
from sklearn.preprocessing import label_binarize

from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_results_, save_results_, wrap_up_results_
from utils.train_test import (
    binary_train, binary_evaluate, multi_train, multi_evaluate,
    _binary_log_loss, _multiclass_log_loss
)
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_S import Model  # 모델 파일명은 기존 사용대로

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
# Multi-source helpers
# -------------------------------

class _PseudoDataset:
    """시각화/호환 목적(길이만 필요)으로 쓰는 더미 dataset"""
    def __init__(self, length: int):
        self._len = int(length)
    def __len__(self):
        return self._len

class MultiSourceConcatLoader:
    """
    여러 source 로더들을 '이어붙여' 평가에 사용 (validation/test 집계용).
    __len__은 총 배치 수 합, .dataset은 총 샘플 수 역할만 하는 더미.
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
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task (epoch-wise multi-source)')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--k_basis', type=int, default=4)
    parser.add_argument('--model', type=str, default='NORM_GNN')

    # 멀티소스(여러 소스를 공백으로 나열)
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
    parser.add_argument('--patience_local', type=int, default=10,
                        help='Early-stop patience counted per-source dataset (local).')

    parser.add_argument('--shared_connectivity', choices=['cls2var', 'full', 'baseline'], default='baseline', help="Shared encoder의 연결 정책: ""'cls2var'=CLS가 변수들만 봄, Var↔Var/Var→CLS 차단(권장); " "'full'=모두 허용; 'identity'=자기 자신만(사실상 mixing 없음).")
    parser.add_argument('--basis_connectivity', choices=['full', 'baseline', 'cls2var'], default='baseline', help="Basis 층 연결 정책: 'full'(기본), 'var2var'(변수끼리만), 'cls2var'(CLS가 변수만 보는 형태).")
    parser.add_argument('--basis_topk',type=int,default=0, help=">0이면 basis attention에서 query별 상위 k개 key만 사용(희소화). 0이면 비활성." )
    args = parser.parse_args()
    args.table_path = "/storage/personal/eungyeop/dataset/table/"
    return args

# -------------------------------

def _log_fewshot_stats(dataloader, name):
    dataset_obj = getattr(dataloader, 'dataset', None)
    if isinstance(dataset_obj, list):
        labels = [int(x['y'].item()) for x in dataset_obj]
        cnt = dict(Counter(labels))
        logger.info(f"[{name}] size={len(labels)} class_counts={cnt}")
    else:
        try:
            labels = []
            for batch in dataloader:
                y = batch['y']
                labels.extend([int(t.item()) for t in y])
            cnt = dict(Counter(labels))
            logger.info(f"[{name}] size={len(labels)} class_counts={cnt}")
        except Exception:
            logger.info(f"[{name}] (stats skipped)")

# -------------------------------
# Single-source train/validate (원래 관습 유지)
# -------------------------------

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

    # 이름을 loader에 미리 넣어준다(로그 보기 편하게)
    if not hasattr(train_loader, "name"):
        setattr(train_loader, "name", "train")
    if not hasattr(val_loader, "name"):
        setattr(val_loader, "name", "val")

    for epoch in range(epochs):
        # train one epoch
        _ = train_func(model, train_loader, criterion, optimizer, device)

        # evaluate train & val
        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        _, y_true_val,   y_pred_val   = evaluate_func(model, val_loader,   criterion, device)

        # display loss(표준화된 로그 로스)
        if is_binary:
            train_loss_disp = _binary_log_loss(y_true_train, y_pred_train)
            val_loss_disp   = _binary_log_loss(y_true_val,   y_pred_val)
        else:
            train_loss_disp = _multiclass_log_loss(y_true_train, y_pred_train)
            val_loss_disp   = _multiclass_log_loss(y_true_val,   y_pred_val)

        train_losses.append(train_loss_disp); val_losses.append(val_loss_disp)

        # metrics
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
            f"Train Loss: {train_loss_disp:.4f}, Val Loss: {val_loss_disp:.4f}, "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
            f"Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}"
        )

        # best
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

# -------------------------------
# Epoch-wise multi-source pretraining
# -------------------------------

def pretrain_multisource_epochwise(args, model, loaders_by_ds, criterion, optimizer, device, is_binary):
    """
    CALINEAR 스타일: 매 에폭 하나의 소스 데이터셋만 선택해 1 에폭 학습.
    Early-stopping은 '데이터셋별(local) 기준'으로: 모든 소스가 patience_local만큼 개선 없을 때 종료.
    반환값은 기존 wrap_up_results_에 맞춘 곡선/요약들(선택된 ds의 per-epoch 값들).
    """
    # per-ds trackers
    source_names = list(loaders_by_ds.keys())
    best_val_auc_by_ds = {ds: -float('inf') for ds in source_names}
    no_improve_by_ds   = {ds: 0 for ds in source_names}

    # curves (선택된 ds의 지표를 에폭별로 하나씩 쌓음)
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []

    best_epoch = -1
    last_threshold = 0.5  # 바이너리일 때 마지막으로 계산한 threshold(Val PR기반)

    train_func    = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    for epoch in range(args.train_epochs):
        # 1) 에폭당 하나의 소스 선택(균등 랜덤)
        ds = random.choice(source_names)
        tr_loader = loaders_by_ds[ds]['train']
        va_loader = loaders_by_ds[ds]['val']

        # 2) 한 에폭 학습
        train_loss = train_func(model, tr_loader, criterion, optimizer, device)

        # 3) Train/Val 평가
        _, y_true_train, y_pred_train = evaluate_func(model, tr_loader, criterion, device)
        val_loss, y_true_val,   y_pred_val   = evaluate_func(model, va_loader, criterion, device)

        # 4) 지표 계산
        if is_binary:
            # AUC
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            val_auc   = roc_auc_score(y_true_val,   y_pred_val)
            # Val-PR에서 F1 최대 threshold
            precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_pred_val)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            idx = int(np.argmax(f1_scores))
            current_thr = thresholds[idx] if idx < len(thresholds) else 0.5
            last_threshold = current_thr  # 마지막 threshold 업데이트

            # 이 threshold로 분류/정확도
            y_pred_train_bin = (np.asarray(y_pred_train) > current_thr).astype(int)
            y_pred_val_bin   = (np.asarray(y_pred_val)   > current_thr).astype(int)

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

        # 5) 로그 (명시적으로 ds 표시)
        logger.info(
            f"[Epoch {epoch+1}/{args.train_epochs}] [DS:{ds}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
            f"Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f} | "
            f"no_improve[{ds}]={no_improve_by_ds[ds]}/{args.patience_local}"
        )

        # 6) 곡선 누적(에폭별로 '그 에폭에 학습한 ds'의 값을 기록)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_aucs.append(float(train_auc));   val_aucs.append(float(val_auc))
        train_precisions.append(float(train_precision)); val_precisions.append(float(val_precision))
        train_recalls.append(float(train_recall));       val_recalls.append(float(val_recall))
        train_f1s.append(float(train_f1));               val_f1s.append(float(train_f1))
        train_accs.append(float(train_acc));             val_accs.append(float(val_acc))

        # 7) per-ds early-stop 카운트 갱신
        if val_auc > best_val_auc_by_ds[ds]:
            best_val_auc_by_ds[ds] = val_auc
            no_improve_by_ds[ds] = 0
            best_epoch = epoch
        else:
            no_improve_by_ds[ds] += 1

        # 8) 종료 조건: 모든 소스가 patience_local을 초과하면 멈춤
        if all(no_improve_by_ds[d] >= args.patience_local for d in source_names):
            logger.info(f"[Early Stop] All sources have no improvement for {args.patience_local} validations.")
            break

    # 하나의 global threshold는 정의하기 애매하므로 '마지막 에폭에서 계산된' threshold를 반환(이진만)
    return (train_losses, val_losses,
            train_aucs, val_aucs,
            train_precisions, val_precisions,
            train_recalls, val_recalls,
            train_f1s, val_f1s,
            train_accs, val_accs,
            best_epoch, max(val_aucs) if val_aucs else 0.0, last_threshold)

# -------------------------------

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

# -------------------------------

def build_source_loaders_epochwise(args, dataset_names):
    """
    각 소스 데이터셋별 (train/val/test) 로더를 만들어 dict로 반환.
    모든 소스의 클래스 수가 동일하다고 가정(다르면 경고).
    """
    loaders_by_ds = {}
    num_classes_set = set()
    for ds in dataset_names:
        res = prepare_embedding_dataloaders(args, ds)
        train_loader, val_loader, test_loader = res['loaders']

        # 보기 좋게 이름 부여
        setattr(train_loader, "name", f"{ds}-train")
        setattr(val_loader,   "name", f"{ds}-val")
        setattr(test_loader,  "name", f"{ds}-test")

        loaders_by_ds[ds] = {
            'train': train_loader,
            'val':   val_loader,
            'test':  test_loader,
            'num_classes': res['num_classes']
        }
        num_classes_set.add(res['num_classes'])
        logger.info(f"[Source {ds}] num_classes={res['num_classes']} | train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    if len(num_classes_set) != 1:
        logger.warning(f"[Multi-source] Mixed num_classes across sources: {num_classes_set} (현재 동일 클래스 수 가정)")

    return loaders_by_ds, num_classes_set.pop()

# -------------------------------

def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    logger.info(f"Starting experiment with sources: {args.source_data} -> target: {args.target_data}")
    logger.info(f"Device: {device}")

    # --- 1) Multi-source pre-training (Full, epoch-wise sampling) ---
    src_loaders_by_ds, src_num_classes = build_source_loaders_epochwise(args, args.source_data)
    args.num_classes = src_num_classes
    args.output_dim = src_num_classes if src_num_classes > 2 else 1
    is_binary = (src_num_classes == 2)

    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                       args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Full").to(device)

    # Full은 동결 없이 전체 학습(네 레거시 정책 그대로)
    optimizer_full = optim.Adam(model_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    logger.info("[Full-shot] Multi-source epoch-wise pretraining start (local-patience early-stop)...")

    (train_losses_full, val_losses_full,
     train_aucs_full, val_aucs_full,
     train_precisions_full, val_precisions_full,
     train_recalls_full, val_recalls_full,
     train_f1s_full, val_f1s_full,
     train_accs_full, val_accs_full,
     best_epoch_full, best_val_auc_full, best_threshold_full
    ) = pretrain_multisource_epochwise(
            args, model_full, src_loaders_by_ds, criterion, optimizer_full, device, is_binary
    )

    # (옵션) Full 테스트: 소스 전부를 이어붙여 '합산 테스트'로 집계 (이전 로직 유지)
    test_mix = MultiSourceConcatLoader([src_loaders_by_ds[ds]['test'] for ds in args.source_data])
    (test_loss_full, test_auc_full, test_precision_full, test_recall_full, test_f1_full,
     test_acc_full, all_y_true_full, all_y_pred_full) = final_test_evaluate(
        model_full, test_mix, criterion, device, is_binary, threshold=None  # 글로벌 threshold 없음 → None이면 0.5 기본
    )

    # --- 2) Target few-shot adaptation ---
    logger.info("Preparing target dataloaders...")
    tgt_res = prepare_embedding_dataloaders(args, args.target_data)
    target_train_full, target_val, target_test = tgt_res['loaders']
    setattr(target_train_full, "name", f"{args.target_data}-train")
    setattr(target_val,        "name", f"{args.target_data}-val")
    setattr(target_test,       "name", f"{args.target_data}-test")

    target_num_classes = tgt_res['num_classes']
    if target_num_classes != args.num_classes:
        logger.warning(f"[Target] num_classes ({target_num_classes}) != source num_classes ({args.num_classes}). 동일 클래스 수 가정.")

    logger.info(f"Preparing target few-shot samples (K={args.few_shot})...")
    target_train_few = get_few_shot_embedding_samples(target_train_full, args)
    _log_fewshot_stats(target_train_few, f"Target-Train Few (K={args.few_shot})")

    model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                      args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Few").to(device)
    # 소스 Full에서 학습된 파라미터로 초기화
    model_few.load_state_dict(model_full.state_dict(), strict=False)
    model_few.freeze()
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
    ) = train_and_validate(args, model_few, target_train_few, target_val, criterion, optimizer_few,
                           device, args.train_epochs, is_binary, mode="Few")

    logger.info("[Few-shot(Target)] Final Testing with best threshold from Validation")
    (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
     test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
        model_few, target_test, criterion, device, is_binary, threshold=best_threshold_few
    )

    # --- 결과 묶기(관습 유지: K=4일 때만 Full 저장/시각화) ---
    if args.few_shot == 4:
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
    else:
        full_ours_results = None

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