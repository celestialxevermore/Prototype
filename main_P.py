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
from dataset.data_dataloaders import get_few_shot_embedding_samples, prepare_embedding_dataloaders
from models.TabularFLM_P import Model, prototype_learning  # 같은 파일에서 둘 다 import
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
    
    # 프로토타입 학습 관련 인자 추가
    parser.add_argument('--prototype_momentum', type=float, default=0.9, help='Momentum for prototype updates')
    parser.add_argument('--few_shot_alpha', type=float, default=0.3, help='Weight for classification loss in few-shot phase')
    parser.add_argument('--few_shot_beta', type=float, default=0.7, help='Weight for prototype regularization in few-shot phase')
    
    args = parser.parse_args()

    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

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
            checkpoint_path = os.path.join(checkpoint_dir, f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}_D:{args.del_feat}_S:{args.random_seed}_{experiment_id}.pt")
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


def save_prototypes(self, save_dir):
    """Save prototypes after Phase 1 with unique naming"""
    os.makedirs(save_dir, exist_ok=True)
    
    no_self_loop_str = "NoSelfLoop" if getattr(self.args, 'no_self_loop', False) else "WithSelfLoop"
    
    filename = (f"prototypes_{self.args.source_data}_"
               f"Attn:{self.args.attn_type}_"
               f"{no_self_loop_str}_"
               f"Seed:{self.args.random_seed}.pt")
    
    save_path = os.path.join(save_dir, filename)
    self.Prototype.save_prototypes(save_path)
    logger.info(f"Source Prototypes saved to {save_path}")
    return save_path


def get_prototype_path(args, prototype_save_dir):
    """프로토타입 파일 경로를 생성하는 공통 함수"""
    no_self_loop_str = "NoSelfLoop" if getattr(args, 'no_self_loop', False) else "WithSelfLoop"
    
    filename = (f"prototypes_{args.source_data}_"
               f"Attn:{args.attn_type}_"
               f"{no_self_loop_str}_"
               f"Seed:{args.random_seed}.pt")
    
    return os.path.join(prototype_save_dir, filename)

def main():
    start_time = time.time()
    args  = get_args()
    
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    logger.info(f"Starting TPN-based prototype learning experiment with dataset: {args.source_data}")
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

    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model,experiment_id, mode="Full")
    model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode = "Few")

    model_full = prototype_learning(model_full, args)
    model_few = prototype_learning(model_few, args)
    
    model_full = model_full.to(device)
    model_few = model_few.to(device)
    optimizer_full = optim.Adam(model_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
    optimizer_few = optim.Adam(model_few.parameters(), lr=args.source_lr_few, weight_decay=1e-5)  # few-shot에는 더 작은 lr 사용

    # Phase 1: Full-shot Training with Prototype Learning (4-shot일 때만)
    if args.few_shot == 4:
        logger.info(f"[Phase 1: Full-shot] Start Training with Prototype Learning...")

        (train_losses_full, val_losses_full,
        train_aucs_full, val_aucs_full,
        train_precisions_full, val_precisions_full,
        train_recalls_full, val_recalls_full,
        train_f1s_full, val_f1s_full,
        train_accs_full, val_accs_full,
        best_epoch_full, best_val_auc_full, best_threshold_full
        ) = train_and_validate(args, model_full, train_loader_full_s, val_loader_full_s, criterion, optimizer_full, 
                            device, args.train_epochs, is_binary, mode="Full")

        prototype_save_dir = f"/storage/personal/eungyeop/experiments/prototypes/{args.llm_model}/{args.source_data}/{args.random_seed}"
        prototype_path = model_full.save_prototypes(prototype_save_dir)
        logger.info(f"Phase 1 completed. Prototypes saved to: {prototype_path}")

        logger.info("[Phase 1: Full-shot] Final Testing with best threshold from Validation")
        (test_loss_full, test_auc_full, test_precision_full, test_recall_full, test_f1_full,
        test_acc_full, all_y_true_full, all_y_pred_full) = final_test_evaluate(model_full, test_loader_full_s, criterion, device, is_binary, 
                                                                threshold=best_threshold_full)
    else:
        # 4-shot이 아닌 경우: 4-shot에서 생성된 prototype 파일을 사용
        logger.info(f"Using prototype file created by 4-shot experiment...")
        full_ours_results = None  # 4-shot이 아니므로 full training 결과 없음

    # 모든 few-shot 실험(4, 8, 16, 32, 64)은 동일한 prototype 파일 경로 사용
    prototype_save_dir = f"/storage/personal/eungyeop/experiments/prototypes/{args.llm_model}/{args.source_data}/{args.random_seed}"
    prototype_path = get_prototype_path(args, prototype_save_dir)
    
    # Phase 2: Few-shot Training with Prototype Regularization
    logger.info(f"[Phase 2: Few-shot {args.few_shot}] Start Training with Prototype Regularization...")
    
    # 프로토타입 파일 존재 확인
    if not os.path.exists(prototype_path):
        logger.error(f"Prototype file not found: {prototype_path}")
        logger.error(f"You need to run 4-shot experiment first to generate prototype files!")
        logger.error(f"Please run the following command first:")
        no_self_loop_flag = "--no_self_loop" if getattr(args, 'no_self_loop', False) else ""
        logger.error(f"python main_P.py --few_shot 4 --random_seed {args.random_seed} --source_data {args.source_data} --attn_type {args.attn_type} {no_self_loop_flag}")
        
        logger.info("Available prototype files:")
        if os.path.exists(prototype_save_dir):
            for file in os.listdir(prototype_save_dir):
                if file.endswith('.pt'):
                    logger.info(f"  - {file}")
        else:
            logger.error(f"Prototype directory does not exist: {prototype_save_dir}")
        raise FileNotFoundError(f"Required prototype file not found. Run 4-shot experiment first!")
    
    logger.info(f"Loading prototype from: {prototype_path}")
    # 4-shot에서 학습된 프로토타입을 Phase 2 모델에 로드
    model_few.load_source_prototypes(prototype_path)
    
    (train_losses_few, val_losses_few,
    train_aucs_few, val_aucs_few,
    train_precisions_few, val_precisions_few,
    train_recalls_few, val_recalls_few,
    train_f1s_few, val_f1s_few,
    train_accs_few, val_accs_few,
    best_epoch_few, best_val_auc_few, best_threshold_few
    ) = train_and_validate(args, model_few, train_loader_few_s, val_loader_few_s, criterion, optimizer_few, 
                        device, args.train_epochs, is_binary, mode="Few")

    logger.info("[Phase 2: Few-shot] Final Testing with best threshold from Validation")
    (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
    test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(model_few, test_loader_few_s, criterion, device, is_binary, 
                                                        threshold=best_threshold_few)

    # 기존과 동일한 결과 처리
    if args.few_shot == 4:
        full_ours_results = wrap_up_results_(
            train_losses=train_losses_full, 
            val_losses=val_losses_full,
            test_losses=[],  # 필요하면 test_loss 리스트 넣기
            train_aucs=train_aucs_full,
            val_aucs=val_aucs_full,
            test_aucs=[test_auc_full], 
            train_precisions=train_precisions_full,
            val_precisions=val_precisions_full,
            test_precisions=[test_precision_full],
            train_recalls=train_recalls_full,
            val_recalls=val_recalls_full,
            test_recalls=[test_recall_full],
            train_f1s=train_f1s_full,
            val_f1s=val_f1s_full,
            test_f1s=[test_f1_full],
            all_y_true=[all_y_true_full],
            all_y_pred=[all_y_pred_full],
            best_epoch=best_epoch_full,
            best_ours_auc=test_auc_full,
            best_ours_acc=test_acc_full,
            best_ours_precision=test_precision_full,
            best_ours_recall=test_recall_full,
            best_ours_f1=test_f1_full,
            train_accs=train_accs_full,
            val_accs=val_accs_full,
            test_accs=[test_acc_full]
            )
    else: 
        full_ours_results = None

    few_ours_results = wrap_up_results_(  # wrap_up_results에서 wrap_up_results_로 변경
    train_losses_few, val_losses_few, [],
    train_aucs_few, val_aucs_few, [test_auc_few],
    train_precisions_few, val_precisions_few, [test_precision_few],
    train_recalls_few, val_recalls_few, [test_recall_few],
    train_f1s_few, val_f1s_few, [test_f1_few],
    [all_y_true_few], [all_y_pred_few],
    best_epoch_few, test_auc_few, test_acc_few,
    test_precision_few, test_recall_few, test_f1_few,
    train_accs=train_accs_few,
    val_accs=val_accs_few,
    test_accs=[test_acc_few]
)

    results = prepare_results_(full_ours_results, few_ours_results)

    # 결과 저장
    logger.info("Saving results...")
    save_results_(args, results)
    logger.info("Results saved")
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")

if __name__ == "__main__":
    main()