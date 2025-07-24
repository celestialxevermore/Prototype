import torch
'''
    최대 성능을 위해 주석처리함. 
    2025.07.24. 주석처리 된 코드를 재활성화하면, 
    재현성이 확실히 보장됨.
    torch.use_deterministic_algorithms(False) : 재현성
    torch.use_deterministic_algorithms(True) : 성능
'''
torch.use_deterministic_algorithms(True)
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
from dataset.data_dataloaders import prepare_tabular_dataloaders,prepare_few_shot_dataloaders, get_few_shot_tabular_samples
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_A import Model
import psutil
from utils.visualization import visualize_model_structure
import copy 
from utils.ablation import extract_important_features, determine_del_feat_by_scenario, save_scenario_results_individually
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx               
import matplotlib.pyplot as plt
import numpy as np
experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()

p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],help='List of baselines to use. Leave empty to use only our model.')
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")    
    parser.add_argument('--model_type', type=str, default='TabularFLM', choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3', 'GAT_edge_4', 'GAT_edge_5', 'TabularFLM'])
    parser.add_argument('--label', type = str, choices = ['add', 'no'], default = 'add')
    parser.add_argument('--enc_type', type = str, choices = ['ind', 'shared'], default = 'ind')
    parser.add_argument('--meta_type', type = str, choices = ['meta_attn', 'meta_mlp'], default = 'meta_attn')
    parser.add_argument('--aggr_type', type = str, choices = ['flatten', 'mean', 'attn'], default = 'attn')
    parser.add_argument('--threshold', type = float, default = 0.5)
    parser.add_argument('--frozen', type = bool, default = False)
    #parser.add_argument('--use_edge_attr', action='store_true')
    parser.add_argument('--edge_type', default = 'mlp', choices= ['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices= ['gat_v1','att','gat_v2', 'gate'])
    parser.add_argument('--del_feat', nargs='+', default = [], help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
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
    args = parser.parse_args()

    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

def ad(adjacency):
    
    # 범위별 분포 분석
    adj_flat = adjacency.flatten()
    min_val = adj_flat.min().item()
    max_val = adj_flat.max().item()
    num_bins = 10
    step = (max_val - min_val) / num_bins
    
    for i in range(num_bins):
        lower = min_val + i * step
        upper = min_val + (i + 1) * step
        count = ((adj_flat >= lower) & (adj_flat < upper)).sum().item()
        print(f"범위 [{lower:.4f}, {upper:.4f}): {int(count)} 개")

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

def train_and_validate(args, model, train_loader, val_loader, criterion, optimizer, device, epochs, is_binary, patience=10, mode="Full"):
    
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
        # if args.viz_graph or args.viz_heatmap:
        #     if epoch % 10 == 0 or epoch == epochs - 1:
        #         visualize_model_structure(model, train_loader, device, args, mode, experiment_id, epoch, max_samples=5)
                
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

        logger.info(f"[Epoch {epoch+1}/{epochs}] "
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
            
            # 🔥 개선: validation AUC가 갱신될 때만 저장
            checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{args.source_data}/{mode}/{args.random_seed}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            # 항상 같은 파일명으로 덮어쓰기
            # 삭제된 feature가 많으면 "..." 처리
            if len(str(args.del_feat)) > 50:
                del_feat_str = "..."
            else:
                del_feat_str = str(args.del_feat)
            checkpoint_path = os.path.join(checkpoint_dir, f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}_D:{del_feat_str}_S:{args.random_seed}_{experiment_id}.pt")
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

def find_pt(dataset_name, model_dir = "/home/eungyeop/LLM/tabular/ProtoLLM/pretrained_models"):
    model_path = os.path.join(model_dir,dataset_name)
    if os.path.exists(model_path):
        return model_path
    return None

def main():
    start_time = time.time()
    args = get_args()
    
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    logger.info(f"Starting ablation study experiment with dataset: {args.source_data}")
    logger.info(f"Device: {device}")

    logger.info("Preparing Tabular datasets...")
    
    results = prepare_embedding_dataloaders(args, args.source_data)
    train_loader_full_s, val_loader_full_s, test_loader_full_s = results['loaders']
    num_classes = results['num_classes']
    
    args.num_classes = num_classes 
    args.output_dim = num_classes if num_classes > 2 else 1
    logger.info(f"Dataset: {args.source_data}, Classes: {num_classes}, Output dim: {args.output_dim}")
    
    if args.few_shot > 0:
        logger.info(f"Preparing few-shot samples (K={args.few_shot})...")
        train_loader_few_s = get_few_shot_embedding_samples(train_loader_full_s, args)
        val_loader_few_s = val_loader_full_s
        test_loader_few_s = test_loader_full_s
    logger.info(f"Datasets prepared, source dataset names : {args.source_data}")

    is_binary = (num_classes == 2)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    
    # =================================================================
    # Phase 1: Baseline (전체 변수 사용) & Attention Analysis
    # =================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Baseline analysis with all features")
    logger.info("=" * 60)
    
    # 전체 변수로 학습
    args_phase1 = copy.deepcopy(args)
    args_phase1.del_feat = []  # 전체 변수 사용
    
    logger.info(f"Phase 1 del_feat: {args_phase1.del_feat}")
    logger.info(f"Phase 1 should use all features")
    
    # Baseline Full-shot 실험
    logger.info("[Baseline - Full-shot] Start Training...")
    model_analysis_full = Model(args_phase1, args.input_dim, args.hidden_dim, args.output_dim, 
                               args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Baseline_Full")
    model_analysis_full = model_analysis_full.to(device)
    optimizer_analysis_full = optim.Adam(model_analysis_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
    
    # Baseline Full-shot 학습 & 평가
    (train_losses_baseline_full, val_losses_baseline_full, train_aucs_baseline_full, val_aucs_baseline_full,
     train_precisions_baseline_full, val_precisions_baseline_full, train_recalls_baseline_full, val_recalls_baseline_full,
     train_f1s_baseline_full, val_f1s_baseline_full, train_accs_baseline_full, val_accs_baseline_full,
     best_epoch_baseline_full, best_val_auc_baseline_full, best_threshold_baseline_full) = train_and_validate(
        args_phase1, model_analysis_full, train_loader_full_s, val_loader_full_s, 
        criterion, optimizer_analysis_full, device, args.train_epochs, is_binary, mode="Baseline_Full")
    
    # Baseline Full-shot 최종 테스트
    (test_loss_baseline_full, test_auc_baseline_full, test_precision_baseline_full, test_recall_baseline_full, 
     test_f1_baseline_full, test_acc_baseline_full, all_y_true_baseline_full, all_y_pred_baseline_full) = final_test_evaluate(
        model_analysis_full, test_loader_full_s, criterion, device, is_binary, threshold=best_threshold_baseline_full)
    
    # Baseline Few-shot 실험
    logger.info("[Baseline - Few-shot] Start Training...")
    model_analysis_few = Model(args_phase1, args.input_dim, args.hidden_dim, args.output_dim,
                              args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Baseline_Few")
    model_analysis_few = model_analysis_few.to(device)
    optimizer_analysis_few = optim.Adam(model_analysis_few.parameters(), lr=args.source_lr_few, weight_decay=1e-5)
    
    # Baseline Few-shot 학습 & 평가
    (train_losses_baseline_few, val_losses_baseline_few, train_aucs_baseline_few, val_aucs_baseline_few,
     train_precisions_baseline_few, val_precisions_baseline_few, train_recalls_baseline_few, val_recalls_baseline_few,
     train_f1s_baseline_few, val_f1s_baseline_few, train_accs_baseline_few, val_accs_baseline_few,
     best_epoch_baseline_few, best_val_auc_baseline_few, best_threshold_baseline_few) = train_and_validate(
        args_phase1, model_analysis_few, train_loader_few_s, val_loader_few_s, 
        criterion, optimizer_analysis_few, device, args.train_epochs, is_binary, mode="Baseline_Few")
    
    # Baseline Few-shot 최종 테스트
    (test_loss_baseline_few, test_auc_baseline_few, test_precision_baseline_few, test_recall_baseline_few,
     test_f1_baseline_few, test_acc_baseline_few, all_y_true_baseline_few, all_y_pred_baseline_few) = final_test_evaluate(
        model_analysis_few, test_loader_few_s, criterion, device, is_binary, threshold=best_threshold_baseline_few)
    
    # Attention 분석하여 중요한 변수 추출 (Full-shot 모델 사용)
    logger.info("Extracting attention weights and identifying important features...")
    important_features, all_features, unimportant_features = extract_important_features(model_analysis_full, train_loader_full_s, device)
    
    logger.info(f"Important features (Top-3): {important_features}")
    logger.info(f"All features ({len(all_features)} total): {all_features}")
    
    # Baseline 결과 정리
    baseline_full_results = wrap_up_results_(
        train_losses_baseline_full, val_losses_baseline_full, [],
        train_aucs_baseline_full, val_aucs_baseline_full, [test_auc_baseline_full],
        train_precisions_baseline_full, val_precisions_baseline_full, [test_precision_baseline_full],
        train_recalls_baseline_full, val_recalls_baseline_full, [test_recall_baseline_full],
        train_f1s_baseline_full, val_f1s_baseline_full, [test_f1_baseline_full],
        [all_y_true_baseline_full], [all_y_pred_baseline_full],
        best_epoch_baseline_full, test_auc_baseline_full, test_acc_baseline_full, 
        test_precision_baseline_full, test_recall_baseline_full, test_f1_baseline_full,
        train_accs_baseline_full, val_accs_baseline_full, [test_acc_baseline_full]
    )
    
    baseline_few_results = wrap_up_results_(
        train_losses_baseline_few, val_losses_baseline_few, [],
        train_aucs_baseline_few, val_aucs_baseline_few, [test_auc_baseline_few],
        train_precisions_baseline_few, val_precisions_baseline_few, [test_precision_baseline_few],
        train_recalls_baseline_few, val_recalls_baseline_few, [test_recall_baseline_few],
        train_f1s_baseline_few, val_f1s_baseline_few, [test_f1_baseline_few],
        [all_y_true_baseline_few], [all_y_pred_baseline_few],
        best_epoch_baseline_few, test_auc_baseline_few, test_acc_baseline_few,
        test_precision_baseline_few, test_recall_baseline_few, test_f1_baseline_few,
        train_accs_baseline_few, val_accs_baseline_few, [test_acc_baseline_few]
    )
    
    # =================================================================
    # Phase 2: Scenario-based Ablation Studies
    # =================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Scenario-based ablation studies")
    logger.info("=" * 60)
    
    scenarios = {
        0: "Baseline (No feature removal)",
        1: "Remove top-3 important features",
        2: "Remove top-3 important + 1 most unimportant features",  # 수정
        3: "Remove bottom-3 unimportant features", 
        4: "Keep only top-3 important features",
        5: "Keep only bottom-3 unimportant features"  # 새로 추가
    }
    
    all_scenario_results = {}
    
    # Baseline 결과를 시나리오 0으로 저장
    all_scenario_results[0] = {
        'description': "Baseline (No feature removal)",
        'removed_features': [],
        'full_results': baseline_full_results,
        'few_results': baseline_few_results,
        'scenario_args': args_phase1
    }
    
    logger.info(f"Scenario 0: Baseline (No feature removal)")
    logger.info(f"Full-shot - Test AUC: {test_auc_baseline_full:.4f}, Test ACC: {test_acc_baseline_full:.4f}")
    logger.info(f"Few-shot - Test AUC: {test_auc_baseline_few:.4f}, Test ACC: {test_acc_baseline_few:.4f}")
    
    for scenario_id, description in scenarios.items():
        if scenario_id == 0:  # Baseline은 이미 처리함
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario {scenario_id}: {description}")
        logger.info(f"{'='*60}")
        
        # 시나리오별 del_feat 설정
        args_scenario = copy.deepcopy(args)
        args_scenario.del_feat = determine_del_feat_by_scenario(scenario_id, important_features, all_features,unimportant_features)
        
        logger.info(f"Removing features: {args_scenario.del_feat}")
        
        # Full-shot 실험
        logger.info(f"[Scenario {scenario_id} - Full-shot] Start Training...")
        model_full = Model(args_scenario, args.input_dim, args.hidden_dim, args.output_dim,
                          args.num_layers, args.dropout_rate, args.llm_model, experiment_id, 
                          mode=f"Scenario_{scenario_id}_Full")
        model_full = model_full.to(device)
        optimizer_full = optim.Adam(model_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
        
        # Full-shot 학습 & 평가
        (train_losses_full, val_losses_full, train_aucs_full, val_aucs_full, 
         train_precisions_full, val_precisions_full, train_recalls_full, val_recalls_full,
         train_f1s_full, val_f1s_full, train_accs_full, val_accs_full,
         best_epoch_full, best_val_auc_full, best_threshold_full) = train_and_validate(
            args_scenario, model_full, train_loader_full_s, val_loader_full_s, 
            criterion, optimizer_full, device, args.train_epochs, is_binary, 
            mode=f"Scenario_{scenario_id}_Full")
        
        # Full-shot 최종 테스트
        (test_loss_full, test_auc_full, test_precision_full, test_recall_full, test_f1_full, 
         test_acc_full, all_y_true_full, all_y_pred_full) = final_test_evaluate(
            model_full, test_loader_full_s, criterion, device, is_binary, threshold=best_threshold_full)
        
        # Few-shot 실험
        logger.info(f"[Scenario {scenario_id} - Few-shot] Start Training...")
        model_few = Model(args_scenario, args.input_dim, args.hidden_dim, args.output_dim,
                         args.num_layers, args.dropout_rate, args.llm_model, experiment_id, 
                         mode=f"Scenario_{scenario_id}_Few")
        model_few = model_few.to(device)
        optimizer_few = optim.Adam(model_few.parameters(), lr=args.source_lr_few, weight_decay=1e-5)
        
        # Few-shot 학습 & 평가
        (train_losses_few, val_losses_few, train_aucs_few, val_aucs_few,
         train_precisions_few, val_precisions_few, train_recalls_few, val_recalls_few,
         train_f1s_few, val_f1s_few, train_accs_few, val_accs_few,
         best_epoch_few, best_val_auc_few, best_threshold_few) = train_and_validate(
            args_scenario, model_few, train_loader_few_s, val_loader_few_s, 
            criterion, optimizer_few, device, args.train_epochs, is_binary,
            mode=f"Scenario_{scenario_id}_Few")
        
        # Few-shot 최종 테스트
        (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
         test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(
            model_few, test_loader_few_s, criterion, device, is_binary, threshold=best_threshold_few)
        
        # 결과 정리
        full_results = wrap_up_results_(
            train_losses_full, val_losses_full, [],
            train_aucs_full, val_aucs_full, [test_auc_full],
            train_precisions_full, val_precisions_full, [test_precision_full],
            train_recalls_full, val_recalls_full, [test_recall_full],
            train_f1s_full, val_f1s_full, [test_f1_full],
            [all_y_true_full], [all_y_pred_full],
            best_epoch_full, test_auc_full, test_acc_full, test_precision_full, test_recall_full, test_f1_full,
            train_accs_full, val_accs_full, [test_acc_full]
        )
        
        few_results = wrap_up_results_(
            train_losses_few, val_losses_few, [],
            train_aucs_few, val_aucs_few, [test_auc_few],
            train_precisions_few, val_precisions_few, [test_precision_few],
            train_recalls_few, val_recalls_few, [test_recall_few],
            train_f1s_few, val_f1s_few, [test_f1_few],
            [all_y_true_few], [all_y_pred_few],
            best_epoch_few, test_auc_few, test_acc_few, test_precision_few, test_recall_few, test_f1_few,
            train_accs_few, val_accs_few, [test_acc_few]
        )
        
        # 시나리오 결과 저장
        all_scenario_results[scenario_id] = {
            'description': description,
            'removed_features': args_scenario.del_feat.copy(),
            'full_results': full_results,
            'few_results': few_results,
            'scenario_args': args_scenario
        }
        
        logger.info(f"Scenario {scenario_id} completed.")
        logger.info(f"Full-shot - Test AUC: {test_auc_full:.4f}, Test ACC: {test_acc_full:.4f}")
        logger.info(f"Few-shot - Test AUC: {test_auc_few:.4f}, Test ACC: {test_acc_few:.4f}")
    
    # =================================================================
    # 결과 저장 (각 시나리오별로 개별 JSON 파일 생성)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Saving scenario results...")
    logger.info("=" * 60)
    
    save_scenario_results_individually(args, all_scenario_results, important_features, all_features, unimportant_features)
    
    # 전체 요약 로그
    logger.info("=" * 60)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Important features identified: {important_features}")
    
    for scenario_id, scenario_data in all_scenario_results.items():
        logger.info(f"\nScenario {scenario_id}: {scenario_data['description']}")
        logger.info(f"  Removed features: {scenario_data['removed_features']}")
        
        full_results = scenario_data['full_results']
        few_results = scenario_data['few_results']

        logger.info(f"  Full-shot: AUC={full_results['best_ours_auc']:.4f}, ACC={full_results['best_ours_acc']:.4f}")
        logger.info(f"  Few-shot:  AUC={few_results['best_ours_auc']:.4f}, ACC={few_results['best_ours_acc']:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"\nTotal experiment time: {format_time(total_time)}")
    logger.info("Ablation study completed successfully!")

if __name__ == "__main__":
    main()