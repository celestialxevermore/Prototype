import os
import random,time
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
from dataset.data_dataloaders import prepare_full_source_dataset, prepare_fewshot_source_dataset, prepare_multiple_dataset
from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_results_with_xgboost,prepare_results_with_logReg, save_source_to_source_nofewshot_results, wrap_up_results
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from utils.metrics import get_best_performance
from models.GNN import NORM_GNN
from models.XGBoost import xgboost_benchmark
from models.LogReg import logistic_regression_benchmark
import psutil 
p = psutil.Process()
p.cpu_affinity(range(50, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
set_seed(42)

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    
    # parser.add_argument('--source_dataset_name', type=str, default='adult', 
    #                     choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial'])
    parser.add_argument('--source_datasets', nargs='+', type = str, required = True,
                            help ='List of source dataset names')
    parser.add_argument('--target_dataset', type = str, required = True,
                            help='Target dataset name')
    
    
    parser.add_argument('--dataset_shot', type=int, default=16, help='the number of shot')
    parser.add_argument('--dataset_seed', type=int, default=4)
    parser.add_argument('--source_lr', type=float, default=0.0001)
    parser.add_argument('--llm_model', type=str, default='gpt2')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    return parser.parse_args()


def train_and_evaluate(source_models, target_model, all_loaders, criterions, source_optimizers, target_optimizer, device, epochs, source_datasets, target_dataset)

    for epoch in range(epochs):

        # Training
        for source_model in source_models.values():
            source_model.train()
        target_model.train()

        total_loss = 0 
        for batch_idx, target_batch in enumerate(all_loaders[target_dataset]['train']):
            target_batch = target_batch.to(device)

            source_batches = {} 
            #for dataset in source_datasets:





def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    if torch.cuda.is_available() is False:
        a = input("THE MODEL CANNOT LOAD GPU!!!")

    logger.info(f"Starting experiment with source datasets: {args.source_datasets}")
    logger.info(f"Target dataset: {args.target_dataset}")
    logger.info(f"Device: {device}")

    logger.info("Preparing datasets...")
    source_datasets, target_dataset, all_loaders, num_classes, original_data, feature_names = prepare_multiple_dataset(args)
    logger.info("Datasets prepared")

    source_models = {} 
    source_optimizers = {} 

    for dataset in source_datasets:
        is_binary = dataset not in ['car', 'communities']
        output_dim = 1 if is_binary else num_classes[dataset]
        source_models[dataset] = NORM_GNN(input_dim = 768, hidden_dim = 128, output_dim = output_dim, num_layers = 4, dropout_rate = 0.3).to(device)
        source_optimizers[dataset] = optim.Adam(source_models[dataset].parameters(), lr = args.source_lr, weight_decay = 1e-5)
    
    is_target_binary = target_dataset not in ['car', 'communities']
    target_output_dim = 1 if is_target_binary else num_classes[target_dataset]
    target_model = NORM_GNN(input_dim = 768, hidden_dim = 128, output_dim = target_output_dim, num_layer = 4, dropout_rate = 0.3).to(device)
    target_optimizer = optim.Adam(target_model.parameters(), lr = args.source_lr, weight_decay = 1e-5)
