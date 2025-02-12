import os
import random,time
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from datetime import datetime
from utils.metrics import calculate_binary_accuracy, calculate_multiclass_accuracy
from dataset.data import set_seed
from dataset.data_dataloaders import CombinedDataLoader, prepare_graph_dataloaders
from utils.util import setup_logger, format_time, prepare_results, fix_seed, save_source_to_target_results, save_pretrained_model, load_pretrained_model, visualize_dataset_distributions
from utils.train_test import binary_train, binary_evaluate, multi_train, multi_evaluate
from utils.metrics import get_best_performance
from models.GNN import TOY
import psutil 
import wandb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    
    # Experimental Setup
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument('--random_seed', type=int, default=2024, help='random seed')
    exp_group.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    exp_group.add_argument('--wandb', action='store_true', help='use wandb')
    exp_group.add_argument('--cpu_start', type=int, default=0)
    exp_group.add_argument('--use_cpu_num', type=int, default=8)
    
    # Dataset
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--source_dataset_names', nargs='+', type=str, default=['blood'],
                        )
    data_group.add_argument('--target_dataset_name', type=str, default='blood', 
                        )
    data_group.add_argument('--dataset_shot', type=int, default=16, help='the number of shot')
    data_group.add_argument('--dataset_seed', type=int, default=4)
    data_group.add_argument('--batch_size', type=int, default=64, help='batch size')
    
    # Model
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, default='NORM_GNN')
    model_group.add_argument('--llm_model', type=str, default='gpt2')
    model_group.add_argument('--input_dim', type=int, default=768)
    model_group.add_argument('--hidden_dim', type=int, default=128)
    model_group.add_argument('--num_layers', type=int, default=4)
    model_group.add_argument('--dropout_rate', type=float, default=0.3)
    model_group.add_argument('--use_pretrained', action='store_true', help='Use pretrained model for source dataset if available')
    
    # Training
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    train_group.add_argument('--fewshot_epochs', type=int, default=100, help='fewshot epochs')
    train_group.add_argument('--source_lr', type=float, default=0.0001)
    train_group.add_argument('--fewshot_lr', type=float, default=0.0001)
    parser.add_argument('--few_shot_list', nargs='*', type=str, default=[], help='List of few shot number')
    parser.add_argument('--use_shared', action='store_true', help='Use shared MLP for all datasets')
    # Path
    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument('--save_dir', type=str, default='experiments', help='log directory')
    path_group.add_argument('--graph_path', type=str, default="/home/eungyeop/LLM/tabular/ProtoLLM/dataset/data/graph")
    path_group.add_argument('--table_path', type=str, default="/home/eungyeop/LLM/tabular/ProtoLLM/dataset/data/table")
    
    return parser.parse_args()

def train_and_evaluate(args, model, train_loader, test_loader, criterion, optimizer, device, epochs, is_binary):
    train_losses, test_losses, train_aucs, test_aucs = [], [], [], []
    train_func = binary_train if is_binary else multi_train
    evaluate_func = binary_evaluate if is_binary else multi_evaluate
    all_y_true, all_y_pred = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # max_loader_length 계산
        loader_lengths = train_loader.get_loader_lengths(phase='train')
        max_loader_length = max(loader_lengths.values())
        
        # Training phase
        for i in range(max_loader_length):
            batch = train_loader.get_batch(phase='train')
            optimizer.zero_grad()
            loss = model(batch)  # 여러 데이터셋의 batch를 한번에 처리
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss/(i+1)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            if isinstance(train_loader, CombinedDataLoader):
                _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
                if len(y_true_train) == 0 or len(y_pred_train) == 0:
                    print("Warning: Empty predictions or labels in training evaluation")
                    continue
                test_loss, y_true_test, y_pred_test = evaluate_func(model, test_loader, criterion, device)
            else:
                _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
                if len(y_true_train) == 0 or len(y_pred_train) == 0:
                    print("Warning: Empty predictions or labels in training evaluation")
                    continue
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
        
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_auc": train_auc,
                "test_auc": test_auc
            })
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    return train_losses, test_losses, train_aucs, test_aucs, all_y_true, all_y_pred


def evaluate_model(args, model, combined_loader, device, num_classes, phase='test'):

    model.eval()
    total_loss = 0
    labels = []
    probs = []
    with torch.no_grad():

        loader_lengths = combined_loader.get_loader_lengths(phase=phase)
        max_loader_length = max(loader_lengths.values())
        
        pbar = tqdm(range(max_loader_length), desc=f"{phase}")
        for i in pbar:
            batch = combined_loader.get_batch(phase = phase)
            for _, data in batch.items():
                labels.extend(data.y.cpu().numpy())
            loss, prob = model(batch)
            for _, source_prob in prob.items():
                probs.append(source_prob.detach().cpu().numpy())
            total_loss += loss.item()
    total_loss = total_loss/(i+1)
    probs_tensor = np.concatenate(probs, axis=0)
    overall_accuracy, overall_auc, overall_auprc, overall_f1, overall_recall, overall_precision = compute_overall_accuracy(probs_tensor, labels, num_classes, threshold=args.threshold)

    return total_loss, overall_accuracy, overall_auc, overall_auprc, overall_f1, overall_recall, overall_precision  

def train_model(model, optimizer, combined_loader, args, device):
    # best_acc = 0
    best_loss = float('inf')
    best_epoch = 0
    # train_losses, test_losses, train_aucs, test_aucs= [], [], [], []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0 
        labels = []
        probs = []

        loader_lengths = combined_loader.get_loader_lengths(phase='train')
        max_loader_length = max(loader_lengths.values())
        
        pbar = tqdm(range(max_loader_length), desc="Train")
        for i in pbar:
            batch = combined_loader.get_batch(phase = 'train')
            optimizer.zero_grad()
            for _, data in batch.items():
                labels.extend(data.y.cpu().numpy())
            loss, prob = model(batch) # dataset_predictions : {dataset_name: prediction}
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            for _, source_prob in prob.items():
                probs.append(source_prob.detach().cpu().numpy())
        # train_losses.append(epoch_loss)
        epoch_loss = epoch_loss/(i+1)
        probs_tensor = np.concatenate(probs, axis=0)

        # pdb.set_trace()
        print(f"Evaluating after Eopch {epoch+1}...")
        train_acc, train_auc, train_auprc, train_f1, train_recall, train_precision = compute_overall_accuracy(probs_tensor, labels, args.num_classes, threshold=args.threshold)
        valid_loss, valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = evaluate_model(args, model, combined_loader, device, args.num_classes)

        print(f"Epoch {epoch+1}/{args.epochs} :",
                f"\nTrain Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train AUPRC: {train_auprc:.4f}, Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}",
                f"\nValid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid AUPRC: {valid_auprc:.4f}, Valid F1: {valid_f1:.4f}, Valid Recall: {valid_recall:.4f}, Valid Precision: {valid_precision:.4f}    ")
        logging.info(f"Epoch {epoch+1}/{args.epochs} - \n"
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train AUPRC: {train_auprc:.4f}, Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}  \n"
                    f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid AUPRC: {valid_auprc:.4f}, Valid F1: {valid_f1:.4f}, Valid Recall: {valid_recall:.4f}, Valid Precision: {valid_precision:.4f}")
        if args.wandb:
            wandb.log({
                "Train Loss": epoch_loss,
                "Train Acc": train_acc,
                "Train AUC": train_auc,
                "Train AUPRC": train_auprc,
                "Train F1": train_f1,
                "Train Recall": train_recall,
                "Train Precision": train_precision,
                "Valid Loss": valid_loss,
                "Valid Acc": valid_acc,
                "Valid AUC": valid_auc,
                "Valid AUPRC": valid_auprc,
                "Valid F1": valid_f1,
                "Valid Recall": valid_recall,
                "Valid Precision": valid_precision,
                }, step = epoch)
            

        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     best_epoch = epoch
        #     torch.save(model, os.path.join(args.save_dir, f'{args.model}_{args.random_seed}.pt'))
        #     logging.info(f"Best epoch: {best_epoch}, Best acc: {best_acc}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model, os.path.join(args.save_dir, f'{args.model}_{args.random_seed}.pt'))
            logging.info(f"Best epoch: {best_epoch}, Best loss: {best_loss}")        



def main():
    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args = get_args()
    p = psutil.Process()
    p.cpu_affinity(range(args.cpu_start, args.cpu_start+args.use_cpu_num))
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    logger.info(f"Starting experiment with source datasets: {args.source_dataset_names}, target dataset: {args.target_dataset_name}")
    logger.info(f"Device: {device}")

    args.save_dir = os.path.join(args.save_dir, '_'.join(args.source_dataset_names))
    if args.wandb:
        WANDB_AUTH_KEY = 'f563277dff1229ebc314e702651cfd94ff3fc6a6'
        wandb.login(key=WANDB_AUTH_KEY)
        wandb.init(project="ProtoLLM",
                name=f"{'_'.join(args.source_dataset_names)}", 
                notes=f"{current_time}",
                )
        wandb.config.update(args)

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir, 'training.log'), level=logging.INFO)
    logging.info(f"Starting experiment with source datasets: {args.source_dataset_names}")

    logging.info("Preparing datasets...")
    
    # Source datasets
    train_loader_full_s = CombinedDataLoader(args, args.source_dataset_names, phase='train')
    test_loader_s = CombinedDataLoader(args, args.source_dataset_names, phase='test')
    
    # Check if all source datasets have the same number of classes
    num_classes_list = [train_loader_full_s.num_classes[name] for name in args.source_dataset_names]
    if not all(nc == num_classes_list[0] for nc in num_classes_list):
        raise ValueError(f"All source datasets must have the same number of classes. Found: {dict(zip(args.source_dataset_names, num_classes_list))}")
    num_classes_s = num_classes_list[0]
    

    # Target dataset
    train_loader_full_t, test_loader_t, num_classes_t = prepare_graph_dataloaders(args, args.target_dataset_name)
    assert num_classes_s == num_classes_t
    logger.info(f"Datasets prepared. Number of classes: {num_classes_s}")

    is_binary = all(name not in ['car', 'communities'] for name in args.source_dataset_names)
    output_dim = 1 if is_binary else num_classes_s
    
    logger.info("Preparing model...")
    model = TOY(args, device, train_loader_full_s.num_classes)
    
    evaluate_func = binary_evaluate if is_binary else multi_evaluate 
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    if args.use_pretrained and load_pretrained_model(model, '_'.join(args.source_dataset_names)):
        logger.info(f"Using pretrained model for {args.source_dataset_names}")
        
        _, y_true, y_pred = evaluate_func(model, test_loader_s, criterion, device) 
        full_ours_auc = roc_auc_score(y_true, y_pred) if is_binary else roc_auc_score(label_binarize(y_true, classes=range(output_dim)), y_pred, multi_class='ovr', average='macro')
        full_ours_acc = calculate_binary_accuracy(y_true, y_pred) if is_binary else calculate_multiclass_accuracy(y_true, np.argmax(y_pred, axis=1))
        train_losses = test_losses = train_aucs = test_aucs = None
        best_epoch = 0
    else:
        logger.info(f"Training new model for {args.source_dataset_names}...")
        optimizer = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)
        train_losses, test_losses, train_aucs, test_aucs, all_y_true, all_y_pred = train_and_evaluate(
            args, model, train_loader_full_s, test_loader_s, criterion, optimizer, device, args.train_epochs, is_binary
        )

        best_epoch, full_ours_auc, full_ours_acc = get_best_performance(
            test_aucs, all_y_true, all_y_pred, is_binary
        )
        save_pretrained_model(args, model, '_'.join(args.source_dataset_names))
        logger.info(f"Model trained and saved as pretrained")

    logger.info(f"Source model Best Test ROC AUC (Full-dataset): {full_ours_auc:.4f}" + (f" at epoch {best_epoch + 1}" if best_epoch > 0 else ""))
    logger.info(f"Source model Best Test Accuracy (Full-dataset): {full_ours_acc:.4f}")
    few_shot_results_dict = {}
    # Few-shot learning
    if args.few_shot_list:  # few_shot_list가 비어있지 않을 때만 실행
        
        for shot in args.few_shot_list:
            logger.info(f"\nStarting few-shot learning with {shot} shots...")
            
            train_loader_few_t = CombinedDataLoader(args, [args.target_dataset_name], phase='train')
            test_loader_few_t = CombinedDataLoader(args, [args.target_dataset_name], phase='test')
            
            logger.info(f"Train dataset size: {len(train_loader_few_t.train_loaders[args.target_dataset_name].dataset)}")
            logger.info(f"Test dataset size: {len(test_loader_few_t.test_loaders[args.target_dataset_name].dataset)}")

        model_few = TOY(args, device, train_loader_few_t.num_classes)
        model_few.load_state_dict(model.state_dict())
        optimizer_few = optim.Adam(model_few.parameters(), lr=args.fewshot_lr, weight_decay=1e-5)

        few_train_losses, few_test_losses, few_train_aucs, few_test_aucs, few_all_y_true, few_all_y_pred = train_and_evaluate(
            args, model_few, train_loader_few_t, test_loader_few_t, criterion, 
            optimizer_few, device, args.fewshot_epochs, is_binary
        )
        
        best_few_epoch, few_ours_auc, few_ours_acc = get_best_performance(
            few_test_aucs, few_all_y_true, few_all_y_pred, is_binary
        )

        few_shot_results_dict[f"shot_{shot}"] = {
            "train_losses": few_train_losses,
            "test_losses": few_test_losses,
            "train_aucs": few_train_aucs,
            "test_aucs": few_test_aucs,
            "best_epoch": best_few_epoch,
            "best_auc": few_ours_auc,
            "best_acc": few_ours_acc
        }
        
        logger.info(f"Shot {shot}: Best Test ROC AUC: {few_ours_auc:.4f} at epoch {best_few_epoch + 1}")
        logger.info(f"Shot {shot}: Best Test Accuracy: {few_ours_acc:.4f}")

        logger.info("\nFew-shot learning results:")
        for shot, results in few_shot_results_dict.items():
            logger.info(f"{shot}-shot: Best test AUC = {max(results['test_aucs']):.4f}")
    else:
        logger.info("\nSkipping few-shot learning (few_shot_list is empty)")

    logger.info("Saving results...")

    results = prepare_results(
        args, 
        full_ours_auc, 
        full_ours_acc,
        few_shot_results_dict
    )

    save_source_to_target_results(args, model, results)

    # few-shot loaders 초기화
    if args.few_shot_list:
        train_loader_few_t = CombinedDataLoader(args, [args.target_dataset_name], phase='train')
        test_loader_few_t = CombinedDataLoader(args, [args.target_dataset_name], phase='test')
    else:
        # few-shot을 실행하지 않을 때는 None으로 설정
        train_loader_few_t = None
        test_loader_few_t = None

    # visualization (few-shot loaders가 있을 때만)
    if train_loader_few_t is not None and test_loader_few_t is not None:
        visualize_dataset_distributions(
            train_loader_full_s, 
            test_loader_s, 
            train_loader_few_t, 
            test_loader_few_t, 
            args.source_dataset_name, 
            args.target_dataset_name
        )

    if args.wandb:
        wandb.finish()

    logger.info("Results saved")
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")

if __name__ == "__main__":
    main()