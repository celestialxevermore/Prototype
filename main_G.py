import torch
import os
import random, time
import argparse
import pandas as pd
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
from dataset.data_dataloaders import prepare_tabular_dataloaders,prepare_few_shot_dataloaders, get_few_shot_tabular_samples
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM import Model
import psutil
from utils.visualization import visualize_model_structure
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx               
import matplotlib.pyplot as plt
import numpy as np

# Optuna 추가
import optuna
from optuna.trial import TrialState

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task with Optuna')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--input_dim', type = int, default = 768)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--output_dim', type = int, default = 1)
    parser.add_argument('--num_layers', type = int, default = 3)
    parser.add_argument('--dropout_rate', type = float, default = 0.1)
    parser.add_argument('--n_heads', type = int, default = 4)
    parser.add_argument('--model', type = str, default = 'NORM_GNN')
    parser.add_argument('--source_data', type=str, default='heart', 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland','breast','magic_telescope','forest_covertype_sampled', 'higgs_sampled'])
    parser.add_argument('--target_data', type = str, default = 'hungarian')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--source_lr_few', type=float, default=0.00001)
    parser.add_argument('--llm_model', type=str, default = 'gpt2_mean', choices = ['gpt2_mean','gpt2_auto','sentence-bert','bio-bert','bio-clinical-bert','bio-llama', 'new', 'LLAMA_mean','LLAMA_auto'])
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--des', type=str, help='experimental memo')
    parser.add_argument('--base_dir', type=str, default='/storage/personal/eungyeop/experiments/optuna')
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],help='List of baselines to use. Leave empty to use only our model.')
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")    
    parser.add_argument('--model_type', type=str, default='TabularFLM', choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3', 'GAT_edge_4', 'GAT_edge_5', 'TabularFLM'])
    parser.add_argument('--label', type = str, choices = ['add', 'no'], default = 'add')
    parser.add_argument('--enc_type', type = str, choices = ['ind', 'shared'], default = 'ind')
    parser.add_argument('--meta_type', type = str, choices = ['meta_attn', 'meta_mlp'], default = 'meta_attn')
    parser.add_argument('--aggr_type', type = str, choices = ['flatten', 'mean', 'attn'], default = 'attn')
    parser.add_argument('--threshold', type = float, default = 0.5)
    parser.add_argument('--frozen', type = bool, default = False)
    parser.add_argument('--edge_type', default = 'mlp', choices= ['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v2', choices= ['gat_v1','att','gat_v2', 'gate'])
    parser.add_argument('--del_feat', nargs='+', default = [], help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
    # GMM 관련 인자 추가
    parser.add_argument('--use_gmm', action='store_true', help='Use GMM1 module')
    parser.add_argument('--use_gmm2', action='store_true', help='Use GMM2 module')
    parser.add_argument('--num_prototypes', type=int, default=32, help='Number of prototypes(phenotypes) in GMM')
    parser.add_argument('--gmm_stage_num', type=int, default=10, help='EM step iterations in GMM')
    parser.add_argument('--gmm_momentum', type=float, default=0.9, help='Momentum for prototype updates')
    parser.add_argument('--gmm_beta', type=float, default=1.0, help='Weight for reconstructed embedding')
    parser.add_argument('--gmm_lambda', type=float, default=2.0, help='Temperature parameter for responsibility')
    parser.add_argument('--gmm_eps', type=float, default=1e-6, help='Small value for numerical stability')
    parser.add_argument('--hard', action='store_true', help='Temperature parameter for Gumbel-Softmax')
    ## 시각화 관련 인자 추가
    parser.add_argument('--viz_heatmap', action='store_true', help='Visualize heatmap')
    parser.add_argument('--viz_graph', action='store_true', help='Visualize graph')

    # Optuna 관련 인자 추가
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--study_name', type=str, default='tabular_optimization', help='Optuna study name')
    
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

def train_and_validate(args, model, train_loader, val_loader, criterion, optimizer, device, epochs, is_binary, patience=10, mode="Full", trial=None):
    """
    Train + Validation만 진행하고, Best Validation 성능을 기록한 모델 state를 반환.
    Optuna pruning을 위해 trial 객체 추가.
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
    best_threshold = 0.5
    best_model_state = None

    for epoch in range(epochs):
        # 1) Training
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 2) Evaluate on Train / Validation
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
            n_classes = y_pred_train.shape[1]
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

        # Optuna intermediate reporting for pruning
        if trial is not None:
            trial.report(val_auc, epoch)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epoch % 10 == 0:
            logger.info(f"[{mode} Epoch {epoch+1}/{epochs}] "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                        f"Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            best_model_state = model.state_dict()
            if current_threshold is not None:
                best_threshold = current_threshold
            
            # 체크포인트 저장
            checkpoint_dir = f"/storage/personal/eungyeop/experiments/optuna/checkpoints/{args.llm_model}/{args.source_data}/{mode}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial.number if trial else 'test'}_{experiment_id}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
                'threshold': best_threshold,
                'args': args
            }, checkpoint_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # 학습 종료 후, Best 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_auc

def objective(trial, base_args):
    """Optuna objective function"""
    # 하이퍼파라미터 제안
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
    
    # Learning rates 추가
    source_lr = trial.suggest_float('source_lr', 1e-5, 1e-2, log=True)
    source_lr_few = trial.suggest_float('source_lr_few', 1e-6, 1e-3, log=True)
    
    # CLS 초기화 추가
    cls_init_type = trial.suggest_categorical('cls_init_type', ['normal', 'zeros', 'kaiming', 'xavier'])
    cls_init_std = trial.suggest_float('cls_init_std', 0.01, 0.05, step=0.01)  # normal용
    
    # 시드별 결과를 저장할 리스트
    seeds = [42, 44, 46, 48, 50]
    val_aucs = []
    
    for seed in seeds:
        # args 복사 및 하이퍼파라미터 적용
        args = argparse.Namespace(**vars(base_args))
        args.random_seed = seed
        args.dropout_rate = dropout_rate
        args.n_heads = n_heads
        args.num_layers = num_layers
        args.source_lr = source_lr          # ✅ Learning rate 적용
        args.source_lr_few = source_lr_few  # ✅ Few-shot learning rate 적용
        
        fix_seed(args.random_seed)
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        
        # 데이터 준비
        results = prepare_embedding_dataloaders(args, args.source_data)
        train_loader_full_s, val_loader_full_s, test_loader_full_s = results['loaders']
        num_classes = results['num_classes']
        
        args.num_classes = num_classes 
        args.output_dim = num_classes if num_classes > 2 else 1
        
        if args.few_shot > 0:
            train_loader_few_s = get_few_shot_embedding_samples(train_loader_full_s, args)
            val_loader_few_s = val_loader_full_s
            
        is_binary = (num_classes == 2)
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        
        # 모델 생성
        model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim, 
                         args.num_layers, args.dropout_rate, args.llm_model, 
                         experiment_id, mode="Few")
        
        # ✅ CLS 초기화 직접 적용 (argparser 수정 없음!)
        if cls_init_type == 'normal':
            nn.init.normal_(model_few.cls, mean=0, std=cls_init_std)
        elif cls_init_type == 'zeros':
            nn.init.zeros_(model_few.cls)
        elif cls_init_type == 'kaiming':
            nn.init.kaiming_uniform_(model_few.cls, a=math.sqrt(5))
        elif cls_init_type == 'xavier':
            nn.init.xavier_uniform_(model_few.cls)
        
        model_few = model_few.to(device)
        optimizer_few = optim.Adam(model_few.parameters(), lr=args.source_lr_few, weight_decay=1e-5)  # ✅ Few-shot LR 사용
        
        # 학습 및 평가
        try:
            val_auc = train_and_validate(args, model_few, train_loader_few_s, val_loader_few_s, 
                                       criterion, optimizer_few, device, args.train_epochs, 
                                       is_binary, mode="Few", trial=trial)
            val_aucs.append(val_auc)
            
        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number} pruned at seed {seed}")
            raise
        
        except Exception as e:
            logger.error(f"Error in trial {trial.number} at seed {seed}: {e}")
            val_aucs.append(0.0)
    
    # 시드별 평균 성능 반환
    mean_val_auc = np.mean(val_aucs)
    std_val_auc = np.std(val_aucs)
    
    logger.info(f"Trial {trial.number}: dropout={dropout_rate}, n_heads={n_heads}, "
                f"num_layers={num_layers}, source_lr={source_lr:.6f}, source_lr_few={source_lr_few:.6f}, "
                f"cls_init={cls_init_type} -> Mean AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    
    return mean_val_auc

def main():
    args = get_args()
    
    # Optuna 저장 디렉토리 생성
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Optuna 스터디 생성
    storage_url = f"sqlite:///{args.base_dir}/{args.study_name}.db"
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials")
    logger.info(f"Study will be saved to: {storage_url}")
    
    # 최적화 실행
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # 결과 출력
    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # 결과 저장
    results_path = os.path.join(args.base_dir, f"optimization_results_{experiment_id}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Optimization Results\n")
        f.write(f"===================\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.4f}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"\nTop 10 trials:\n")
        for i, trial in enumerate(study.trials_dataframe().nlargest(10, 'value').iterrows()):
            f.write(f"{i+1}. Trial {trial[1]['number']}: {trial[1]['value']:.4f} - {trial[1]['params_dropout_rate']}, {trial[1]['params_n_heads']}, {trial[1]['params_num_layers']}\n")
    
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"To view dashboard, run: optuna-dashboard {storage_url}")

if __name__ == "__main__":
    main()