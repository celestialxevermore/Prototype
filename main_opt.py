import torch
#torch.use_deterministic_algorithms(False)
import os
import random,time
import argparse
import pandas as pd
import pdb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_results_, save_results_, wrap_up_results_
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from sklearn.model_selection import StratifiedKFold
from dataset.data_dataloaders import prepare_tabular_dataloaders,prepare_few_shot_dataloaders, get_few_shot_tabular_samples, get_few_shot_graph_samples
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM import Model
import psutil 
from torch_geometric.data import Batch
p = psutil.Process()

p.cpu_affinity(range(1, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--input_dim', type = int, default = 768)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--output_dim', type = int, default = 1)
    parser.add_argument('--num_layers', type = int, default = 1)
    parser.add_argument('--dropout_rate', type = float, default = 0.1)
    parser.add_argument('--threshold', type = float, default = 0.5)
    parser.add_argument('--heads', type = int, default = 4)
    parser.add_argument('--model', type = str, default = 'NORM_GNN')
    parser.add_argument('--source_dataset_name', type=str, default='heart', 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland'])
    #parser.add_argument('--source_dataset_names', nargs='+', type = str, default = ['cleveland', 'heart_statlog', 'heart'] , help = 'List of source dataaset name')
    parser.add_argument('--target_dataset_name', type = str, default = 'hungarian')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--source_lr_few', type=float, default=0.00001)
    parser.add_argument('--llm_model', type=str, default='gpt2')
    parser.add_argument('--meta_heads', type=int, default= 2)
    parser.add_argument('--meta_num_layers', type=int, default= 2)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--des', type=str, help='experimental memo')
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],help='List of baselines to use. Leave empty to use only our model.')
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")    
    parser.add_argument('--model_type', type=str, default='TabularFLM', choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3', 'GAT_edge_4', 'GAT_edge_5', 'TabularFLM'])
    parser.add_argument('--scaler_type', type=str, default='pow', choices=['pow'])
    parser.add_argument('--label', type = str, choices = ['add', 'no'], default = 'no')
    parser.add_argument('--enc_type', type = str, choices = ['ind', 'shared'], default = 'ind')
    parser.add_argument('--meta_type', type = str, choices = ['meta_attn', 'meta_mlp'], default = 'meta_attn')
    parser.add_argument('--aggr_type', type = str, choices = ['flatten', 'mean', 'attn'], default = 'flatten')
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()

    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, is_binary, patience=10):
    #pdb.set_trace()
    """
    Train + Validation만 진행하고, Best Validation 성능을 기록한 모델 state를 반환.
    마지막에 Best Threshold도 함께 반환해서 별도의 Test 단계에서 사용.
    """
    train_losses = []
    val_losses = []
    train_aucs, val_aucs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []

    # Binary / Multi 구분에 따라 함수 선택
    train_func = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate

    best_val_auc = 0.0
    no_improve = 0
    best_epoch = 0

    # Validation에서 찾은 best threshold (Binary 시에만)
    best_threshold = 0.5
    best_model_state = None

    for epoch in range(epochs):
        # 1) Training
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        #pdb.set_trace()
        # 2) Evaluate on Train / Validation
        #    - Train 평가는 단순 모니터링용
        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        val_loss, y_true_val, y_pred_val = evaluate_func(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if is_binary:
            # Binary Classification
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            val_auc = roc_auc_score(y_true_val, y_pred_val)
            current_threshold = find_optimal_threshold(y_true_val, y_pred_val)

            y_pred_train_bin = (y_pred_train > current_threshold).astype(int)
            y_pred_val_bin = (y_pred_val > current_threshold).astype(int)

            train_precision = precision_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_precision = precision_score(y_true_val, y_pred_val_bin, zero_division=0)
            train_recall = recall_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_recall = recall_score(y_true_val, y_pred_val_bin, zero_division=0)
            train_f1 = f1_score(y_true_train, y_pred_train_bin, zero_division=0)
            val_f1 = f1_score(y_true_val, y_pred_val_bin, zero_division=0)
            train_acc = accuracy_score(y_true_train, y_pred_train_bin)
            val_acc = accuracy_score(y_true_val, y_pred_val_bin)

        else:
            # Multi-class Classification
            n_classes = model.output_dim
            y_true_train_bin = label_binarize(y_true_train, classes=range(n_classes))
            y_true_val_bin = label_binarize(y_true_val, classes=range(n_classes))
            train_auc = roc_auc_score(y_true_train_bin, y_pred_train, multi_class='ovr', average='macro')
            val_auc = roc_auc_score(y_true_val_bin, y_pred_val, multi_class='ovr', average='macro')

            train_precision = precision_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
            val_precision = precision_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
            train_recall = recall_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
            val_recall = recall_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
            train_f1 = f1_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
            val_f1 = f1_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
            preds_train_argmax = y_pred_train.argmax(axis=1)
            preds_val_argmax   = y_pred_val.argmax(axis=1)
            train_acc = accuracy_score(y_true_train, preds_train_argmax)
            val_acc   = accuracy_score(y_true_val, preds_val_argmax)
            current_threshold = None

        # 로그 기록
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logger.info(f"[Epoch {epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                    f"Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

        # Early Stopping 로직
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            best_model_state = model.state_dict()
            if current_threshold is not None:
                best_threshold = current_threshold
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # 학습 종료 후, Best 모델로 복원
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
    """
    학습이 끝난 뒤, Test 로더에 대해 최종 성능을 측정.
    threshold가 있으면 Binary 분류 시 threshold 적용.
    """
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
        n_classes = model.output_dim
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

def find_pt(dataset_name, model_dir = "/home/eungyeop/LLM/tabular/ProtoLLM/pretrained_models"):
    model_path = os.path.join(model_dir,dataset_name)
    if os.path.exists(model_path):
        return model_path
    return None

import optuna
from optuna.trial import TrialState
import json
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour

def objective(trial, args):
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    # 하이퍼파라미터 탐색 범위 설정 - 문법만 수정
    args.source_lr_few = trial.suggest_categorical('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3, 1e-2])
    
    # 데이터 로더 준비
    results = prepare_embedding_dataloaders(args, args.source_dataset_name)
    train_loader_full_s, val_loader_full_s, test_loader_full_s = results['loaders']
    num_classes = results['num_classes']
    
    if args.few_shot > 0:
        train_loader_few_s = get_few_shot_embedding_samples(train_loader_full_s, args)
        val_loader_few_s = val_loader_full_s
        test_loader_few_s = test_loader_full_s

    is_binary = (num_classes == 2)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    
    # 모델 초기화
    model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim, 
                     args.num_layers, args.dropout_rate, args.llm_model).to(device)
    
    optimizer_few = optim.Adam(model_few.parameters(), 
                             lr=args.source_lr_few, 
                             weight_decay=weight_decay)

    # 학습 및 검증
    (_, _, _, val_aucs_few, _, _, _, _, _, _,
     _, _, _, _, best_threshold_few) = train_and_validate(
        model_few, train_loader_few_s, val_loader_few_s, criterion, 
        optimizer_few, device, args.train_epochs, is_binary
    )
    
    return max(val_aucs_few)

def main():
    start_time = time.time()
    args = get_args()
    
    # 현재 variant에 대한 디렉토리 생성
    variant_dir = f"A:{args.aggr_type}_L:{args.label}_E:{args.enc_type}_M:{args.meta_type}"
    results_path = os.path.join(
        f'experiments/source_to_source_{args.base_dir}',
        args.source_dataset_name,
        f'args_seed:{args.random_seed}',
        'TabularFLM',
        variant_dir
    )
    os.makedirs(results_path, exist_ok=True)
    
    # DB 파일 경로 설정
    db_path = os.path.join(results_path, "study.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    study_name = f"opt_{args.source_dataset_name}_{variant_dir}_seed{args.random_seed}"
    storage_name = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # 결과 저장
    best_params = {
        "learning_rate": study.best_trial.params["learning_rate"],
        "weight_decay": study.best_trial.params["weight_decay"],
        "best_value": study.best_trial.value
    }
    
    with open(os.path.join(results_path, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info("Best hyperparameters found:")
    logger.info(f"  Learning rate: {best_params['learning_rate']}")
    logger.info(f"  Weight decay: {best_params['weight_decay']}")
    logger.info(f"  Best validation AUC: {best_params['best_value']:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total optimization time: {format_time(total_time)}")

if __name__ == "__main__":
    main()