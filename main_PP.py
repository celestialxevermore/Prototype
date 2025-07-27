import torch
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
from models.TabularFLM_PP import Model, prototype_learning
import psutil
from utils.visualization import visualize_model_structure
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx               
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import warnings
from torch.utils.data import DataLoader

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='Meta Learning ProtoNet For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=100, help='meta train epochs')
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
    parser.add_argument('--target_data', type = str, default = 'diabetes', choices=['heart_target_1','heart_target_2','heart_target_3', 'heart_target_4'])
    parser.add_argument('--few_shot', type=int, default=4, help='K-shot (support samples per class)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--source_lr_few', type=float, default=0.001, help='Meta learning rate (same as source_lr)')
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
    parser.add_argument('--edge_type', default = 'mlp', choices= ['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices= ['gat_v1','att','gat_v2', 'gate'])
    parser.add_argument('--del_feat', nargs='+', default = [], help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
    
    # Meta Learning 관련 인자
    parser.add_argument('--episodes_per_epoch', type=int, default=50, help='Number of meta training episodes per epoch')
    parser.add_argument('--num_query_per_class', type=int, default=30, help='Number of query samples per class in each episode')
    parser.add_argument('--val_episodes', type=int, default=50, help='Number of validation episodes')
    parser.add_argument('--test_episodes', type=int, default=100, help='Number of test episodes')
    
    # ProtoNet 관련 인자
    parser.add_argument('--distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'], help='Distance metric for prototypes')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for softmax in prototypical loss')
    
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

def sample_support_query_from_loader(data_loader, args, episode_idx=None):
    """DataLoader에서 Support + Query 샘플링 (Binary Classification용)"""
    if episode_idx is not None:
        random.seed(args.random_seed + episode_idx)
        np.random.seed(args.random_seed + episode_idx)
    
    # DataLoader에서 모든 데이터 수집
    all_data = []
    all_labels = []
    
    for batch in data_loader:
        # batch는 (batch_data, batch_labels) 튜플 형태
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            batch_data, batch_labels = batch
        else:
            # batch가 단일 dict인 경우 (y가 dict 안에 포함)
            batch_data = batch
            batch_labels = batch['y']
        
        batch_size = batch_labels.size(0)
        for i in range(batch_size):
            # 각 샘플을 개별적으로 저장
            sample_data = {}
            for key, value in batch_data.items():
                if key != 'y':  # y는 label이므로 제외
                    sample_data[key] = value[i:i+1]  # Keep batch dimension
            all_data.append(sample_data)
            all_labels.append(batch_labels[i].item())
    
    # 클래스별로 분류
    class_0_data = [all_data[i] for i, label in enumerate(all_labels) if label == 0]
    class_1_data = [all_data[i] for i, label in enumerate(all_labels) if label == 1]
    
    # Support + Query 샘플링
    support_data = []
    query_data = []
    
    # Class 0
    total_needed_0 = args.few_shot + args.num_query_per_class
    if len(class_0_data) < total_needed_0:
        warnings.warn(f"Class 0 has fewer samples ({len(class_0_data)}) than required ({total_needed_0})")
        selected_0 = random.choices(class_0_data, k=total_needed_0)
    else:
        selected_0 = random.sample(class_0_data, k=total_needed_0)
    
    support_data.extend(selected_0[:args.few_shot])
    query_data.extend(selected_0[args.few_shot:])
    
    # Class 1  
    total_needed_1 = args.few_shot + args.num_query_per_class
    if len(class_1_data) < total_needed_1:
        warnings.warn(f"Class 1 has fewer samples ({len(class_1_data)}) than required ({total_needed_1})")
        selected_1 = random.choices(class_1_data, k=total_needed_1)
    else:
        selected_1 = random.sample(class_1_data, k=total_needed_1)
    
    support_data.extend(selected_1[:args.few_shot])
    query_data.extend(selected_1[args.few_shot:])
    
    # Labels 생성
    support_labels = torch.tensor([0] * args.few_shot + [1] * args.few_shot, dtype=torch.long)
    query_labels = torch.tensor([0] * args.num_query_per_class + [1] * args.num_query_per_class, dtype=torch.long)
    
    return support_data, support_labels, query_data, query_labels

def compute_prototypical_loss(model, support_data, support_labels, query_data, query_labels, device, temperature=1.0):
    """ProtoNet Loss 계산"""
    
    # Support embeddings 계산
    support_embeddings_list = []
    for sample in support_data:
        embedding = model.base_model.compute_cls_embedding(sample)
        support_embeddings_list.append(embedding)
    
    support_embeddings = torch.cat(support_embeddings_list, dim=0)  # [2*K, embedding_dim]
    
    # Query embeddings 계산
    query_embeddings_list = []
    for sample in query_data:
        embedding = model.base_model.compute_cls_embedding(sample)
        query_embeddings_list.append(embedding)
    
    query_embeddings = torch.cat(query_embeddings_list, dim=0)  # [2*Q, embedding_dim]
    
    # Prototypes 계산 (각 클래스별 평균)
    class_0_mask = (support_labels == 0)
    class_1_mask = (support_labels == 1)
    
    prototype_0 = support_embeddings[class_0_mask].mean(dim=0)  # [embedding_dim]
    prototype_1 = support_embeddings[class_1_mask].mean(dim=0)  # [embedding_dim]
    
    prototypes = torch.stack([prototype_0, prototype_1])  # [2, embedding_dim]
    
    # Query와 prototype 간 거리 계산
    distances = torch.cdist(query_embeddings, prototypes)  # [2*Q, 2]
    
    # Prototypical loss (cross-entropy with negative distances)
    log_probs = torch.log_softmax(-distances / temperature, dim=1)
    query_labels = query_labels.to(device)
    loss = torch.nn.functional.nll_loss(log_probs, query_labels)
    
    # Accuracy 계산
    predictions = distances.argmin(dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss, accuracy.item()

def meta_train_epoch(model, train_loader, optimizer, args, device, epoch):
    """Meta Training 1 Epoch"""
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    logger.info(f"[Epoch {epoch+1}] Starting Meta Training...")
    
    for episode_idx in range(args.episodes_per_epoch):
        # Episode 샘플링
        support_data, support_labels, query_data, query_labels = sample_support_query_from_loader(
            train_loader, args, episode_idx=epoch * args.episodes_per_epoch + episode_idx
        )
        
        # Prototypical Loss 계산
        loss, accuracy = compute_prototypical_loss(
            model, support_data, support_labels, query_data, query_labels, device, args.temperature
        )
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
        
        if episode_idx % 10 == 0:
            logger.debug(f"Episode {episode_idx}/{args.episodes_per_epoch}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
    
    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accuracies)
    
    return avg_loss, avg_acc

def meta_validate(model, val_loader, args, device, epoch):
    """Meta Validation"""
    model.eval()
    val_losses = []
    val_accuracies = []
    
    logger.info(f"[Epoch {epoch+1}] Starting Meta Validation...")
    
    with torch.no_grad():
        for episode_idx in range(args.val_episodes):
            # Validation Episode 샘플링
            support_data, support_labels, query_data, query_labels = sample_support_query_from_loader(
                val_loader, args, episode_idx=episode_idx + 10000  # Validation용 다른 seed 범위
            )
            
            # Prototypical Loss 계산
            loss, accuracy = compute_prototypical_loss(
                model, support_data, support_labels, query_data, query_labels, device, args.temperature
            )
            
            val_losses.append(loss.item())
            val_accuracies.append(accuracy)
    
    avg_val_loss = np.mean(val_losses)
    avg_val_acc = np.mean(val_accuracies)
    val_acc_std = np.std(val_accuracies)
    
    return avg_val_loss, avg_val_acc, val_acc_std

def meta_test_single_episode(model, test_support_data, test_support_labels, test_query_data, test_query_labels, device):
    """단일 Test Episode 평가"""
    model.eval()
    
    with torch.no_grad():
        # Support embeddings
        support_embeddings_list = []
        for sample in test_support_data:
            embedding = model.base_model.compute_cls_embedding(sample)
            support_embeddings_list.append(embedding)
        
        support_embeddings = torch.cat(support_embeddings_list, dim=0)
        
        # Query embeddings 
        query_embeddings_list = []
        for sample in test_query_data:
            embedding = model.base_model.compute_cls_embedding(sample)
            query_embeddings_list.append(embedding)
        
        query_embeddings = torch.cat(query_embeddings_list, dim=0)
    
    # Prototypes 계산
    class_0_mask = (test_support_labels == 0)
    class_1_mask = (test_support_labels == 1)
    
    prototype_0 = support_embeddings[class_0_mask].mean(dim=0)
    prototype_1 = support_embeddings[class_1_mask].mean(dim=0)
    
    prototypes = torch.stack([prototype_0, prototype_1])
    
    # Prediction
    distances = torch.cdist(query_embeddings, prototypes)
    predictions = distances.argmin(dim=1)
    probs = torch.softmax(-distances, dim=1)
    
    test_query_labels = test_query_labels.to(device)
    
    # Metrics 계산
    accuracy = (predictions == test_query_labels).float().mean().item()
    
    # Binary classification metrics
    y_true = test_query_labels.cpu().numpy()
    y_pred_probs = probs[:, 1].cpu().numpy()  # Positive class probability
    y_pred_binary = predictions.cpu().numpy()
    
    if len(np.unique(y_true)) > 1:  # AUC는 양쪽 클래스가 모두 있을 때만 계산
        auc = roc_auc_score(y_true, y_pred_probs)
    else:
        auc = 0.5  # Default value when only one class present
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    return accuracy, auc, precision, recall, f1

def meta_test_source(model, test_loader, args, device):
    """Meta Testing on Source dataset - 여러 episodes 평가"""
    logger.info(f"[Source Meta Test] Multiple episodes evaluation...")
    
    test_accuracies = []
    test_aucs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    
    for test_episode in range(args.test_episodes):
        # 매번 새로운 Support + Query 샘플링
        support_data, support_labels, query_data, query_labels = sample_support_query_from_loader(
            test_loader, args, episode_idx=test_episode + 30000  # Source test용 seed 범위
        )
        
        # 단일 Episode 평가
        accuracy, auc, precision, recall, f1 = meta_test_single_episode(
            model, support_data, support_labels, query_data, query_labels, device
        )
        
        test_accuracies.append(accuracy)
        test_aucs.append(auc)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1s.append(f1)
    
    # 평균 계산
    avg_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    avg_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs)
    avg_precision = np.mean(test_precisions)
    avg_recall = np.mean(test_recalls)
    avg_f1 = np.mean(test_f1s)
    
    logger.info(f"Source Meta Test Results ({args.test_episodes} episodes):")
    logger.info(f"  Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"  Precision: {avg_precision:.4f}")
    logger.info(f"  Recall: {avg_recall:.4f}")
    logger.info(f"  F1: {avg_f1:.4f}")
    
    return avg_acc, avg_auc, avg_precision, avg_recall, avg_f1

def meta_test_target(model, test_loader, args, device):
    """Meta Testing on Target dataset - 단일 episode 평가"""
    logger.info(f"[Target Meta Test] Single {args.few_shot}-shot evaluation...")
    
    # 단일 Support + Query Set 샘플링 (한 번만!)
    fix_seed(args.random_seed)  # 재현성을 위해 고정 seed
    support_data, support_labels, query_data, query_labels = sample_support_query_from_loader(
        test_loader, args, episode_idx=99999  # Target test용 고정 seed
    )
    
    logger.info(f"Support Set: {args.few_shot} samples per class (total: {len(support_data)})")
    logger.info(f"Query Set: {args.num_query_per_class} samples per class (total: {len(query_data)})")
    
    # 단일 Episode 평가 (한 번만!)
    accuracy, auc, precision, recall, f1 = meta_test_single_episode(
        model, support_data, support_labels, query_data, query_labels, device
    )
    
    logger.info(f"Target Meta Test Results (Single Episode):")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    
    return accuracy, auc, precision, recall, f1

def main():
    start_time = time.time()
    args = get_args()
    
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    logger.info(f"Starting Meta Learning ProtoNet experiment")
    logger.info(f"Source dataset (Meta Training): {args.source_data}")
    logger.info(f"Target dataset (Meta Testing): {args.target_data}")
    logger.info(f"Device: {device}")
    logger.info(f"Meta Learning Setup: {args.few_shot}-shot, {args.episodes_per_epoch} episodes/epoch")
    
    # Source 데이터 준비 (Meta Training용)
    logger.info(f"Preparing Source dataset: {args.source_data}...")
    source_results = prepare_embedding_dataloaders(args, args.source_data)
    source_train_loader, source_val_loader, _ = source_results['loaders']
    source_num_classes = source_results['num_classes']
    
    # Target 데이터 준비 (Meta Testing용)
    logger.info(f"Preparing Target dataset: {args.target_data}...")
    target_results = prepare_embedding_dataloaders(args, args.target_data)
    _, _, target_test_loader = target_results['loaders']
    target_num_classes = target_results['num_classes']
    
    # 클래스 수 확인
    if source_num_classes != 2 or target_num_classes != 2:
        raise ValueError("Both source and target datasets must be binary classification!")
    
    args.num_classes = 2
    args.output_dim = 1
    logger.info(f"Source: {args.source_data} (Classes: {source_num_classes})")
    logger.info(f"Target: {args.target_data} (Classes: {target_num_classes})")
    
    # 모델 초기화
    model = Model(args, args.input_dim, args.hidden_dim, args.output_dim, 
                  args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Meta")
    
    # ProtoNet wrapper 적용
    model = prototype_learning(model, args)
    model = model.to(device)
    
    # Optimizer (args에서 learning rate 가져오기)
    optimizer = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)
    
    # Phase 1: Meta Training (args에서 epochs 가져오기)
    logger.info(f"[Phase 1: Meta Training] Starting {args.train_epochs} epochs...")
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    no_improve = 0
    
    for epoch in range(args.train_epochs):
        # Meta Training (Source dataset)
        train_loss, train_acc = meta_train_epoch(model, source_train_loader, optimizer, args, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Meta Validation (Source dataset)
        val_loss, val_acc, val_acc_std = meta_validate(model, source_val_loader, args, device, epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"[Epoch {epoch+1}/{args.train_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}±{val_acc_std:.4f}")
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve = 0
            
            # Checkpoint 저장
            checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/{args.source_data}_to_{args.target_data}/Meta/{args.random_seed}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"meta_model_{args.few_shot}shot_{experiment_id}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'args': args
            }, checkpoint_path)
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Best model 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Phase 1 완료 후 Meta Test 평가 (Source dataset - 여러 episodes)
    logger.info("[Phase 1: Meta Training] Final Testing on Source dataset...")
    phase1_test_acc, phase1_test_auc, phase1_test_precision, phase1_test_recall, phase1_test_f1 = meta_test_source(model, source_val_loader, args, device)
    
    # Phase 2: Meta Testing on Target dataset (단일 episode)
    logger.info(f"[Phase 2: Meta Testing] Testing on Target dataset: {args.target_data}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = meta_test_target(model, target_test_loader, args, device)
    
    # 결과 정리
    logger.info("=== Final Results ===")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    
    # 결과 저장을 위한 데이터 생성
    # Meta training 결과를 full_ours_results로 저장
    full_ours_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_aucs': train_accs,  # Meta learning에서는 accuracy 사용
        'val_aucs': val_accs,
        'train_precisions': train_accs,  # 더미 데이터
        'val_precisions': val_accs,
        'train_recalls': train_accs,
        'val_recalls': val_accs,
        'train_f1s': train_accs,
        'val_f1s': val_accs,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_aucs': [phase1_test_auc],
        'test_accs': [phase1_test_acc],
        'test_precisions': [phase1_test_precision],
        'test_recalls': [phase1_test_recall],
        'test_f1s': [phase1_test_f1],
        'all_y_true': [[]],  # 더미 데이터
        'all_y_pred': [[]],
        'best_epoch': 0,
        'best_ours_auc': phase1_test_auc,
        'best_ours_acc': phase1_test_acc,
        'best_ours_precision': phase1_test_precision,
        'best_ours_recall': phase1_test_recall,
        'best_ours_f1': phase1_test_f1
    }
    
    # Meta testing 결과를 few_ours_results로 저장
    few_ours_results = {
        'test_aucs': [test_auc],
        'test_accs': [test_acc],
        'test_precisions': [test_precision],
        'test_recalls': [test_recall], 
        'test_f1s': [test_f1],
        'train_losses': [],  # Meta test에는 training 없음
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'train_precisions': [],
        'val_precisions': [],
        'train_recalls': [],
        'val_recalls': [],
        'train_f1s': [],
        'val_f1s': [],
        'train_accs': [],
        'val_accs': [],
        'all_y_true': [[]],  # 더미 데이터
        'all_y_pred': [[]],
        'best_epoch': 0,
        'best_ours_auc': test_auc,
        'best_ours_acc': test_acc,
        'best_ours_precision': test_precision,
        'best_ours_recall': test_recall,
        'best_ours_f1': test_f1
    }
    
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