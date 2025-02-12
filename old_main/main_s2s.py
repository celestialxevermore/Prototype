import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from utils.metrics import calculate_binary_accuracy, calculate_multiclass_accuracy
from dataset.data import set_seed
from dataset.data_dataloaders import prepare_full_dataset, prepare_fewshot_dataset
from utils.util import save_results, fix_seed
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from models.GNN import NORM_GNN
from models.XGBoost import xgboost_benchmark
import psutil 
p = psutil.Process()
p.cpu_affinity(range(50, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
set_seed(42)



def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--fewshot_epochs', type=int, default=100, help='fewshot epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--dataset_name', type=str, default='adult', 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial'])
    parser.add_argument('--dataset_shot', type=int, default=16, help='the number of shot')
    parser.add_argument('--dataset_seed', type=int, default=4)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--fewshot_lr', type=float, default=0.0001)
    parser.add_argument('--llm_model', type=str, default='gpt2')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    return parser.parse_args()

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs, is_binary):
    train_losses, test_losses, train_aucs, test_aucs, train_accs, test_accs = [], [], [], [], [], []
    train_func = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate
    accuracy_func = calculate_binary_accuracy if is_binary else calculate_multiclass_accuracy
    all_y_true, all_y_pred = [], []
    
    for epoch in range(epochs):
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        test_loss, y_true_test, y_pred_test = evaluate_func(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        all_y_true.append(y_true_test)
        all_y_pred.append(y_pred_test)
        
        
        if is_binary:
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            test_auc = roc_auc_score(y_true_test, y_pred_test)
        else:
            y_true_train_bin = label_binarize(y_true_train, classes=range(model.output_dim))
            y_true_test_bin = label_binarize(y_true_test, classes=range(model.output_dim))
            train_auc = roc_auc_score(y_true_train_bin, y_pred_train, multi_class='ovr', average='macro')
            test_auc = roc_auc_score(y_true_test_bin, y_pred_test, multi_class='ovr', average='macro')
        
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    return train_losses, test_losses, train_aucs, test_aucs, all_y_true, all_y_pred

def main():
    args = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    (X_train_full, X_test, y_train_full, y_test), train_loader_full, test_loader, num_classes = prepare_full_dataset(args)
    
    is_binary = args.dataset_name not in ['car', 'communities']
    output_dim = 1 if is_binary else num_classes
    
    #Model Preparation
    model = NORM_GNN(input_dim=768, hidden_dim=128, output_dim=output_dim, num_layers=4, dropout_rate=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    # Full dataset training
    train_losses, test_losses, train_aucs, test_aucs, all_y_true, all_y_pred = train_and_evaluate(
        model, train_loader_full, test_loader, criterion, optimizer, device, args.train_epochs, is_binary
    )
    best_epoch = test_aucs.index(max(test_aucs))
    full_ours_auc = max(test_aucs)
    y_true = all_y_true[best_epoch]
    y_pred = all_y_pred[best_epoch]
    full_ours_acc = calculate_binary_accuracy(y_true, y_pred) if is_binary else calculate_multiclass_accuracy(y_true, np.argmax(y_pred, axis=1))
    
    full_xgb_auc, full_xgb_acc = xgboost_benchmark(X_train_full, X_test, y_train_full, y_test, is_binary=is_binary)
    
    print(f"Our model Best Test ROC AUC (Full-dataset): {full_ours_auc:.4f} at epoch {best_epoch + 1}")
    print(f"Our model Best Test Accuracy (Full-dataset): {full_ours_acc:.4f}")
    print(f"XGBoost ROC AUC (Full-dataset): {full_xgb_auc:.4f}")
    print(f"XGBoost Test Accuracy (Full-dataset): {full_xgb_acc:.4f}")
    
    torch.save(model.state_dict(), 'full_model.pth')

    # Few-shot learning
    args.fewshot = True
    (X_train_few, _, y_train_few, _), train_loader_few, _, _ = prepare_fewshot_dataset(args, X_train_full, y_train_full, X_test, y_test)
    
    model_few = NORM_GNN(input_dim=768, hidden_dim=128, output_dim=output_dim, num_layers=4, dropout_rate=0.3).to(device)
    model_few.load_state_dict(torch.load('full_model.pth'))
    optimizer_few = optim.Adam(model_few.parameters(), lr=args.fewshot_lr, weight_decay=1e-5)

    few_train_losses, few_test_losses, few_train_aucs, few_test_aucs, few_all_y_true, few_all_y_pred = train_and_evaluate(
        model_few, train_loader_few, test_loader, criterion, optimizer_few, device, args.fewshot_epochs, is_binary
    )
    
    best_few_epoch = few_test_aucs.index(max(few_test_aucs))
    few_ours_auc = max(few_test_aucs)
    y_true_few = few_all_y_true[best_few_epoch]
    y_pred_few = few_all_y_pred[best_few_epoch]
    few_ours_acc = calculate_binary_accuracy(y_true_few, y_pred_few) if is_binary else calculate_multiclass_accuracy(y_true_few, np.argmax(y_pred_few, axis=1))
    
    few_xgb_auc, few_xgb_acc = xgboost_benchmark(X_train_few, X_test, y_train_few, y_test, is_binary=is_binary)
    
    print(f"Our model Best Test ROC AUC (Few-shot): {few_ours_auc:.4f} at epoch {best_few_epoch + 1}")
    print(f"Our model Best Test Accuracy (Few-shot): {few_ours_acc:.4f}")
    print(f"XGBoost ROC AUC (Few-shot): {few_xgb_auc:.4f}")
    print(f"XGBoost Accuracy (Few-shot): {few_xgb_acc:.4f}")

    results = {
        "Ours": {
            "full-dataset auc": full_ours_auc,
            "full-dataset accuracy": full_ours_acc,
            "few-shot auc": few_ours_auc,
            "few-shot accuracy": few_ours_acc,
        },
        "XGBoost": {
            "full-dataset auc": full_xgb_auc,
            "full-dataset accuracy": full_xgb_acc,
            "few-shot auc": few_xgb_auc,
            "few-shot accuracy": few_xgb_acc,
        },
    }
    full_results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_aucs': train_aucs,
        'test_aucs': test_aucs,
    }
    few_shot_results = {
        'train_losses': few_train_losses,
        'test_losses': few_test_losses,
        'train_aucs': few_train_aucs,
        'test_aucs': few_test_aucs
    }

    
    save_results(args, model, results, full_results, few_shot_results)
    

if __name__ == "__main__":
    main()