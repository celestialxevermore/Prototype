import torch
#torch.use_deterministic_algorithms(False)
import os
import random,time
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_results, save_results, wrap_up_results
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from utils.metrics import get_best_performance
from dataset.data_dataloaders import prepare_graph_dataloaders, prepare_tabular_dataloaders
from models.Model import Model
from models.XGBoost import xgboost_benchmark
from models.LogReg import logistic_regression_benchmark
import psutil 
p = psutil.Process()

p.cpu_affinity(range(50, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--input_dim', type = int, default = 768)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--num_layers', type = int, default = 4)
    parser.add_argument('--dropout_rate', type = float, default = 0.3)
    parser.add_argument('--threshold', type = float, default = 0.5)
    parser.add_argument('--heads', type = int, default = 8)
    parser.add_argument('--model', type = str, default = 'NORM_GNN')
    parser.add_argument('--source_dataset_name', type=str, default='cleveland', 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland'])
    #parser.add_argument('--source_dataset_names', nargs='+', type = str, default = ['cleveland', 'heart_statlog', 'heart'] , help = 'List of source dataaset name')
    parser.add_argument('--target_dataset_name', type = str, default = 'hungarian')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--dataset_seed', type=int, default=4)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--llm_model', type=str, default='gpt2')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--des', type=str, help='experimental memo')
    parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],help='List of baselines to use. Leave empty to use only our model.')
    parser.add_argument('--graph_path', type=str, default="/storage/personal/eungyeop/dataset/graph")
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")    
    parser.add_argument('--model_type', type=str, default='NORM_GNN', choices=['NORM_GNN','GAT','GAT2','GAT3','GAT_edge','GAT_edge_2','GAT_edge_3'])
    parser.add_argument('--graph_type', type=str, default='star', 
                       choices=['star', 'full_one', 'full_mean'],
                       help='star: star graph, full_one: leaf-to-leaf with ones, full_mean: leaf-to-leaf with mean embeddings')
    parser.add_argument('--FD', type=str, default='N',
                       choices=['N', 'D', 'ND'],
                       help='N: Name embeddings, D: Description embeddings, ND: Name and Description embeddings')
    parser.add_argument('--label', action='store_true', help='Use Label Decoded Dataset')
    args = parser.parse_args()
    
    # 그래프 경로 설정
    args.graph_path = "/storage/personal/eungyeop/dataset/graph"
    
    # graph_type과 FD에 따른 하위 경로 설정
    graph_subdir = f"{args.graph_type}_{args.FD}"
    if args.label:
        graph_subdir += "_label"
    
    args.graph_path = os.path.join(args.graph_path, graph_subdir)
    
    return args

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epochs, is_binary):
   train_losses, val_losses, test_losses = [], [], []
   train_aucs, val_aucs, test_aucs = [], [], []
   train_precisions, val_precisions, test_precisions = [], [], []
   train_recalls, val_recalls, test_recalls = [], [], []
   train_f1s, val_f1s, test_f1s = [], [], []

   train_func = binary_train if is_binary else multi_train
   evaluate_func = binary_evaluate if is_binary else multi_evaluate
   all_y_true, all_y_pred = [], []
   
   best_val_auc = 0
   patience = 10  # early stopping patience
   no_improve = 0
   
   for epoch in range(epochs):
       # Training
       train_loss = train_func(model, train_loader, criterion, optimizer, device)
       train_losses.append(train_loss)
       
       # Evaluation on all sets
       _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
       val_loss, y_true_val, y_pred_val = evaluate_func(model, val_loader, criterion, device)
       test_loss, y_true_test, y_pred_test = evaluate_func(model, test_loader, criterion, device)
       
       val_losses.append(val_loss)
       test_losses.append(test_loss)
       all_y_true.append(y_true_test)
       all_y_pred.append(y_pred_test)
       
       if is_binary:
           # Binary classification metrics
           train_auc = roc_auc_score(y_true_train, y_pred_train)
           val_auc = roc_auc_score(y_true_val, y_pred_val)
           test_auc = roc_auc_score(y_true_test, y_pred_test)
           
           # Find optimal threshold using validation set
           optimal_threshold = find_optimal_threshold(y_true_val, y_pred_val)
           logger.info(f"Optimal threshold at epoch {epoch+1}: {optimal_threshold:.4f}")
           
           # Apply optimal threshold
           y_pred_train_bin = (y_pred_train > optimal_threshold).astype(int)
           y_pred_val_bin = (y_pred_val > optimal_threshold).astype(int)
           y_pred_test_bin = (y_pred_test > optimal_threshold).astype(int)
           
           # Calculate metrics with optimal threshold
           train_precision = precision_score(y_true_train, y_pred_train_bin, zero_division=0)
           val_precision = precision_score(y_true_val, y_pred_val_bin, zero_division=0)
           test_precision = precision_score(y_true_test, y_pred_test_bin, zero_division=0)
           
           train_recall = recall_score(y_true_train, y_pred_train_bin, zero_division=0)
           val_recall = recall_score(y_true_val, y_pred_val_bin, zero_division=0)
           test_recall = recall_score(y_true_test, y_pred_test_bin, zero_division=0)
           
           train_f1 = f1_score(y_true_train, y_pred_train_bin, zero_division=0)
           val_f1 = f1_score(y_true_val, y_pred_val_bin, zero_division=0)
           test_f1 = f1_score(y_true_test, y_pred_test_bin, zero_division=0)
           
       else:
           # Multi-class metrics
           y_true_train_bin = label_binarize(y_true_train, classes=range(model.output_dim))
           y_true_val_bin = label_binarize(y_true_val, classes=range(model.output_dim))
           y_true_test_bin = label_binarize(y_true_test, classes=range(model.output_dim))
           
           train_auc = roc_auc_score(y_true_train_bin, y_pred_train, multi_class='ovr', average='macro')
           val_auc = roc_auc_score(y_true_val_bin, y_pred_val, multi_class='ovr', average='macro')
           test_auc = roc_auc_score(y_true_test_bin, y_pred_test, multi_class='ovr', average='macro')
           
           train_precision = precision_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
           val_precision = precision_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
           test_precision = precision_score(y_true_test, y_pred_test.argmax(axis=1), average='macro', zero_division=0)
           
           train_recall = recall_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
           val_recall = recall_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
           test_recall = recall_score(y_true_test, y_pred_test.argmax(axis=1), average='macro', zero_division=0)
           
           train_f1 = f1_score(y_true_train, y_pred_train.argmax(axis=1), average='macro', zero_division=0)
           val_f1 = f1_score(y_true_val, y_pred_val.argmax(axis=1), average='macro', zero_division=0)
           test_f1 = f1_score(y_true_test, y_pred_test.argmax(axis=1), average='macro', zero_division=0)
       
       # Save metrics
       train_aucs.append(train_auc)
       val_aucs.append(val_auc)
       test_aucs.append(test_auc)
       
       train_precisions.append(train_precision)
       val_precisions.append(val_precision)
       test_precisions.append(test_precision)
       
       train_recalls.append(train_recall)
       val_recalls.append(val_recall)
       test_recalls.append(test_recall)
       
       train_f1s.append(train_f1)
       val_f1s.append(val_f1)
       test_f1s.append(test_f1)
       
       # Early stopping check
       if val_auc > best_val_auc:
           best_val_auc = val_auc
           best_model_state = model.state_dict()
           no_improve = 0
       else:
           no_improve += 1
           
       if no_improve >= patience:
           print(f"Early stopping at epoch {epoch}")
           model.load_state_dict(best_model_state)  # 최적의 모델로 복원
           break
           
       logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
       logger.info(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
   
   return (train_losses, val_losses, test_losses,
           train_aucs, val_aucs, test_aucs,
           train_precisions, val_precisions, test_precisions,
           train_recalls, val_recalls, test_recalls,
           train_f1s, val_f1s, test_f1s,
           all_y_true, all_y_pred)

def find_pt(dataset_name, model_dir = "/home/eungyeop/LLM/tabular/ProtoLLM/pretrained_models"):
    model_path = os.path.join(model_dir,dataset_name)
    if os.path.exists(model_path):
        return model_path
    return None


def main():
    start_time = time.time()
    args  = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    logger.info(f"Starting experiment with dataset: {args.source_dataset_name}")
    logger.info(f"Device: {device}")

    logger.info("Preparing Graph datasets...")
    train_loader_full_s, val_loader_full_s, test_loader_s, num_classes = prepare_graph_dataloaders(args, args.source_dataset_name, few_shot=False)
    train_loader_few_s, val_loader_few_s, test_loader_few_s, _ = prepare_graph_dataloaders(args, args.source_dataset_name, few_shot=True)
    
    # baseline이 지정된 경우에만 baseline용 데이터로더 준비
    if args.baseline:
        logger.info("Preparing Tabular datasets...")
        (X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full), _ = prepare_tabular_dataloaders(args, args.source_dataset_name, args.dataset_seed)
        (X_train_few, X_val_few, X_test_few, y_train_few, y_val_few, y_test_few), _ = prepare_tabular_dataloaders(args, args.source_dataset_name, args.dataset_seed, few_shot=True)


    logger.info(f"Datasets prepared, source dataset names : {args.source_dataset_name}")
    logger.info(f"Datasets prepared, target dataset name : {args.target_dataset_name}")


    is_binary = (num_classes == 2)
    model_full = Model(args, args.input_dim, args.hidden_dim, num_classes).to(device)
    model_few = Model(args, args.input_dim, args.hidden_dim, num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer_full = optim.Adam(model_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
    optimizer_few = optim.Adam(model_few.parameters(), lr=args.source_lr, weight_decay=1e-5)


    logger.info(f"Start Training..")

    (train_losses_full, val_losses_full, test_losses_full, train_aucs_full, val_aucs_full, test_aucs_full, train_precisions_full, val_precisions_full, test_precisions_full, train_recalls_full, val_recalls_full, test_recalls_full, train_f1s_full, val_f1s_full, test_f1s_full, all_y_true_full, all_y_pred_full) = train_and_evaluate(model_full, train_loader_full_s, val_loader_full_s, test_loader_s, criterion, optimizer_full, device, args.train_epochs, is_binary)
    best_epoch_full, best_ours_auc_full, best_ours_acc_full, best_ours_precision_full, best_ours_recall_full, best_ours_f1_full = get_best_performance(test_aucs_full, test_precisions_full, test_recalls_full, test_f1s_full, all_y_true_full, all_y_pred_full, is_binary)
    (train_losses_few, val_losses_few, test_losses_few, train_aucs_few, val_aucs_few, test_aucs_few, train_precisions_few, val_precisions_few, test_precisions_few, train_recalls_few, val_recalls_few, test_recalls_few, train_f1s_few, val_f1s_few, test_f1s_few, all_y_true_few, all_y_pred_few) = train_and_evaluate(model_few, train_loader_few_s, val_loader_few_s, test_loader_few_s, criterion, optimizer_few, device, args.train_epochs, is_binary)
    best_epoch_few, best_ours_auc_few, best_ours_acc_few, best_ours_precision_few, best_ours_recall_few, best_ours_f1_few = get_best_performance(test_aucs_few, test_precisions_few, test_recalls_few, test_f1s_few, all_y_true_few, all_y_pred_few, is_binary)

    full_ours_results = wrap_up_results(train_losses_full, val_losses_full, test_losses_full, train_aucs_full, val_aucs_full, test_aucs_full, train_precisions_full, val_precisions_full, test_precisions_full, train_recalls_full, val_recalls_full, test_recalls_full, train_f1s_full, val_f1s_full, test_f1s_full, all_y_true_full, all_y_pred_full, best_epoch_full, best_ours_auc_full, best_ours_acc_full, best_ours_precision_full, best_ours_recall_full, best_ours_f1_full)
    few_ours_results = wrap_up_results(train_losses_few, val_losses_few, test_losses_few, train_aucs_few, val_aucs_few, test_aucs_few, train_precisions_few, val_precisions_few, test_precisions_few, train_recalls_few, val_recalls_few, test_recalls_few, train_f1s_few, val_f1s_few, test_f1s_few, all_y_true_few, all_y_pred_few, best_epoch_few, best_ours_auc_few, best_ours_acc_few, best_ours_precision_few, best_ours_recall_few, best_ours_f1_few)


    baselines = args.baseline if isinstance(args.baseline, list) else [args.baseline]
    full_baseline_results = {}
    few_baseline_results = {}

    for baseline in baselines:
        if baseline == "Logistic_Regression":
            full_baseline_results[baseline] = logistic_regression_benchmark(args, X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full, is_binary=is_binary, max_iter=args.train_epochs)
            few_baseline_results[baseline] = logistic_regression_benchmark(args, X_train_few, X_val_few, X_test_few, y_train_few, y_val_few, y_test_few, is_binary=is_binary, max_iter=args.train_epochs)
        elif baseline == "XGBoost":
            full_baseline_results[baseline] = xgboost_benchmark(args, X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full, is_binary=is_binary)
            few_baseline_results[baseline] = xgboost_benchmark(args, X_train_few, X_val_few, X_test_few, y_train_few, y_val_few, y_test_few, is_binary=is_binary)
        else:
            logger.warning(f"Invalid baseline specified: {baseline}. Skipping.")

    if full_baseline_results:
        results = prepare_results(full_ours_results, few_ours_results, full_baseline_results=full_baseline_results, few_baseline_results=few_baseline_results, baselines=args.baseline)
    else:
        logger.info("No valid baseline specified, using only our model results.")
        results = prepare_results(full_ours_results, few_ours_results)

    logger.info("Saving results...")
    save_results(args, results)
    logger.info("Results saved")
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")

if __name__ == "__main__":
    main()