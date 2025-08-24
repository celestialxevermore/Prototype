import os 
import json
import torch
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np 
import pandas as pd 
from collections import Counter
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.manifold import TSNE
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
from utils.visualization import visualize_results
def fix_seed(seed):
    
    random.seed(seed)

    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    '''
        최대 성능을 위해 주석처리함. 
        2025.07.24. 주석처리 된 코드를 재활성화하면, 
        재현성이 확실히 보장됨.
    '''
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = True      # False → True
    torch.backends.cudnn.deterministic = False # True → False
    torch.use_deterministic_algorithms(False)  # True → False

def save_pretrained_model(args, model, dataset_name):
    save_dir = os.path.join('pretrained_models',dataset_name)
    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, 'pretrained_model.pth')
    torch.save(model.state_dict(),save_path)
    print(f"Pretrained model saved to {save_path}")

def load_pretrained_model(model, dataset_name):
    load_path = os.path.join('pretrained_models',dataset_name, 'pretrained_model.pth')
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"Pretrained model loaded from {load_path}")
        return True 
    else:
        print(f"No pretrained model found for {dataset_name}")
        return False 

def wrap_up_results(train_losses, val_losses, test_losses, 
                  train_aucs, val_aucs, test_aucs, 
                  train_precisions, val_precisions, test_precisions, 
                  train_recalls, val_recalls, test_recalls, 
                  train_f1s, val_f1s, test_f1s, 
                  all_y_true, all_y_pred, 
                  best_epoch, best_ours_auc, best_ours_acc, 
                  best_ours_precision, best_ours_recall, best_ours_f1) -> dict:
   """
   학습, 검증 및 평가 결과를 정리하는 함수.

   Args:
       train_losses, val_losses, test_losses: 학습, 검증 및 테스트 손실
       train_aucs, val_aucs, test_aucs: 학습, 검증 및 테스트 AUC
       train_precisions, val_precisions, test_precisions: 학습, 검증 및 테스트 Precision
       train_recalls, val_recalls, test_recalls: 학습, 검증 및 테스트 Recall
       train_f1s, val_f1s, test_f1s: 학습, 검증 및 테스트 F1-score
       all_y_true, all_y_pred: 모든 Epoch의 True/Predicted 값
       best_epoch: 최고 성능을 보인 Epoch
       best_ours_auc: 최고 AUC
       best_ours_acc: 최고 Accuracy
       best_ours_precision: 최고 Precision
       best_ours_recall: 최고 Recall
       best_ours_f1: 최고 F1-score

   Returns:
       dict: 결과를 포함한 딕셔너리
   """
   results = {
       'train_losses': train_losses,
       'val_losses': val_losses, 
       'test_losses': test_losses, 
       'train_aucs': train_aucs,
       'val_aucs': val_aucs,
       'test_aucs': test_aucs,
       'train_precisions': train_precisions,
       'val_precisions': val_precisions,
       'test_precisions': test_precisions,
       'train_recalls': train_recalls,
       'val_recalls': val_recalls,
       'test_recalls': test_recalls,
       'train_f1s': train_f1s,
       'val_f1s': val_f1s,
       'test_f1s': test_f1s,
       'all_y_true': all_y_true, 
       'all_y_pred': all_y_pred,
       'best_epoch': best_epoch,
       'best_ours_auc': best_ours_auc,
       'best_ours_acc': best_ours_acc,
       'best_ours_precision': best_ours_precision,
       'best_ours_recall': best_ours_recall,
       'best_ours_f1': best_ours_f1
   }
   return results



def wrap_up_results_(train_losses, val_losses, test_losses, 
                    train_aucs, val_aucs, test_aucs, 
                    train_precisions, val_precisions, test_precisions, 
                    train_recalls, val_recalls, test_recalls, 
                    train_f1s, val_f1s, test_f1s, 
                    all_y_true, all_y_pred, 
                    best_epoch, best_ours_auc, best_ours_acc, 
                    best_ours_precision, best_ours_recall, best_ours_f1,
                    train_accs, val_accs, test_accs,
                    train_auprcs=None, val_auprcs=None, test_auprcs=None,
                    best_ours_auprc=None) -> dict:
    """
    학습, 검증 및 평가 결과를 정리하는 함수.

    Args:
        train_losses, val_losses, test_losses: 학습, 검증 및 테스트 손실
        train_aucs, val_aucs, test_aucs: 학습, 검증 및 테스트 AUC
        train_precisions, val_precisions, test_precisions: 학습, 검증 및 테스트 Precision
        train_recalls, val_recalls, test_recalls: 학습, 검증 및 테스트 Recall
        train_f1s, val_f1s, test_f1s: 학습, 검증 및 테스트 F1-score
        all_y_true, all_y_pred: 모든 Epoch의 True/Predicted 값
        best_epoch: 최고 성능을 보인 Epoch
        best_ours_auc: 최고 AUC
        best_ours_acc: 최고 Accuracy
        best_ours_precision: 최고 Precision
        best_ours_recall: 최고 Recall
        best_ours_f1: 최고 F1-score
        train_accs, val_accs, test_accs: 학습, 검증 및 테스트 Accuracy

    Returns:
        dict: 결과를 포함한 딕셔너리
    """
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'test_losses': test_losses, 
        'train_aucs': train_aucs,
        'val_aucs': val_aucs,
        'test_aucs': test_aucs,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'test_precisions': test_precisions,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls,
        'test_recalls': test_recalls,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'test_f1s': test_f1s,
        'train_auprcs': train_auprcs or [],
        'val_auprcs': val_auprcs or [],
        'test_auprcs': test_auprcs or [],
        'all_y_true': all_y_true, 
        'all_y_pred': all_y_pred,
        'best_epoch': best_epoch,
        'best_ours_auc': best_ours_auc,
        'best_ours_acc': best_ours_acc,
        'best_ours_precision': best_ours_precision,
        'best_ours_recall': best_ours_recall,
        'best_ours_f1': best_ours_f1,
        'best_ours_auprc': best_ours_auprc,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_accs': test_accs
    }
    return results



def prepare_results_(full_ours_results, few_ours_results):
    if full_ours_results is None:
        results = {
                    'Best_results': {
            "Ours": "The results of source are already given. Check the initial results.",
            "Ours_few": {
                "Ours_best_few_auc": few_ours_results['best_ours_auc'],
                "Ours_best_few_acc": few_ours_results['best_ours_acc'],
                "Ours_best_few_precision": few_ours_results.get('best_ours_precision'),
                "Ours_best_few_recall": few_ours_results.get('best_ours_recall'),
                "Ours_best_few_f1": few_ours_results.get('best_ours_f1'),
                "Ours_best_few_auprc": few_ours_results.get('best_ours_auprc'),
            }
        },
            "Full_results": {
                "Ours": "The results of source are already given. Check the initial results.",
                "Ours_few": {
                    "Ours_train_few_auc": few_ours_results['train_aucs'],
                    "Ours_val_few_auc": few_ours_results['val_aucs'],
                    "Ours_train_few_losses": few_ours_results['train_losses'],
                    "Ours_val_few_losses": few_ours_results['val_losses'],
                    "Ours_train_few_precisions": few_ours_results.get('train_precisions'),
                    "Ours_val_few_precisions": few_ours_results.get('val_precisions'),
                    "Ours_train_few_recalls": few_ours_results.get('train_recalls'),
                    "Ours_val_few_recalls": few_ours_results.get('val_recalls'),
                    "Ours_train_few_f1s": few_ours_results.get('train_f1s'),
                    "Ours_val_few_f1s": few_ours_results.get('val_f1s'),
                }
            }
        }
    else:
        results = {
            'Best_results': {
                "Ours": {
                    "Ours_best_full_auc": full_ours_results['best_ours_auc'],
                    "Ours_best_full_acc": full_ours_results['best_ours_acc'],
                    "Ours_best_full_precision": full_ours_results.get('best_ours_precision'),
                    "Ours_best_full_recall": full_ours_results.get('best_ours_recall'),
                    "Ours_best_full_f1": full_ours_results.get('best_ours_f1'),
                },
                "Ours_few": {
                    "Ours_best_few_auc": few_ours_results['best_ours_auc'],
                    "Ours_best_few_acc": few_ours_results['best_ours_acc'],
                    "Ours_best_few_precision": few_ours_results.get('best_ours_precision'),
                    "Ours_best_few_recall": few_ours_results.get('best_ours_recall'),
                    "Ours_best_few_f1": few_ours_results.get('best_ours_f1'),
                    "Ours_best_few_auprc": few_ours_results.get('best_ours_auprc'),
                }
            },
            "Full_results": {
                "Ours": {
                    "Ours_train_full_auc": full_ours_results['train_aucs'],
                    "Ours_val_full_auc": full_ours_results['val_aucs'],
                    "Ours_train_full_losses": full_ours_results['train_losses'],
                    "Ours_val_full_losses": full_ours_results['val_losses'],
                    "Ours_train_full_precisions": full_ours_results.get('train_precisions'),
                    "Ours_val_full_precisions": full_ours_results.get('val_precisions'),
                    "Ours_train_full_recalls": full_ours_results.get('train_recalls'),
                    "Ours_val_full_recalls": full_ours_results.get('val_recalls'),
                    "Ours_train_full_f1s": full_ours_results.get('train_f1s'),
                    "Ours_val_full_f1s": full_ours_results.get('val_f1s'),
                },
                "Ours_few": {
                    "Ours_train_few_auc": few_ours_results['train_aucs'],
                    "Ours_val_few_auc": few_ours_results['val_aucs'],
                    "Ours_train_few_losses": few_ours_results['train_losses'],
                    "Ours_val_few_losses": few_ours_results['val_losses'],
                    "Ours_train_few_precisions": few_ours_results.get('train_precisions'),
                    "Ours_val_few_precisions": few_ours_results.get('val_precisions'),
                    "Ours_train_few_recalls": few_ours_results.get('train_recalls'),
                    "Ours_val_few_recalls": few_ours_results.get('val_recalls'),
                    "Ours_train_few_f1s": few_ours_results.get('train_f1s'),
                    "Ours_val_few_f1s": few_ours_results.get('val_f1s'),
                }
            }
        }
    return results



def prepare_results_s(args, full_ours_results):
    if args.is_pretrained is False:
        results = {
            'Best_results': {
                "Ours": {
                    "Ours_best_full_auc": full_ours_results['best_ours_auc'],
                    "Ours_best_full_acc": full_ours_results['best_ours_acc'],
                    "Ours_best_full_precision": full_ours_results.get('best_ours_precision'),
                    "Ours_best_full_recall": full_ours_results.get('best_ours_recall'),
                    "Ours_best_full_f1": full_ours_results.get('best_ours_f1'),
                },
            },
            "Full_results": {
                "Ours": {
                    "Ours_train_full_auc": full_ours_results['train_aucs'],
                    "Ours_test_full_auc": full_ours_results['test_aucs'],
                    "Ours_train_full_losses": full_ours_results['train_losses'],
                    "Ours_test_full_losses": full_ours_results['test_losses'],
                    "Ours_train_full_precisions": full_ours_results.get('train_precisions'),
                    "Ours_test_full_precisions": full_ours_results.get('test_precisions'),
                    "Ours_train_full_recalls": full_ours_results.get('train_recalls'),
                    "Ours_test_full_recalls": full_ours_results.get('test_recalls'),
                    "Ours_train_full_f1s": full_ours_results.get('train_f1s'),
                    "Ours_test_full_f1s": full_ours_results.get('test_f1s'),
                },
            }
        }
        return results
    else:
        return None



def prepare_results_t(full_ours_results, few_ours_results):
    results = {
        'Best_results': {
            "Ours": {
                "Ours_best_full_auc": full_ours_results['best_ours_auc'],
                "Ours_best_full_acc": full_ours_results['best_ours_acc'],
                "Ours_best_full_precision": full_ours_results.get('best_ours_precision'),
                "Ours_best_full_recall": full_ours_results.get('best_ours_recall'),
                "Ours_best_full_f1": full_ours_results.get('best_ours_f1'),
            },
            "Ours_few": {
                "Ours_best_few_auc": few_ours_results['best_ours_auc'],
                "Ours_best_few_acc": few_ours_results['best_ours_acc'],
                "Ours_best_few_precision": few_ours_results.get('best_ours_precision'),
                "Ours_best_few_recall": few_ours_results.get('best_ours_recall'),
                "Ours_best_few_f1": few_ours_results.get('best_ours_f1'),
            }
        },
        "Full_results": {
            "Ours": {
                "Ours_train_full_auc": full_ours_results['train_aucs'],
                "Ours_test_full_auc": full_ours_results['test_aucs'],
                "Ours_train_full_losses": full_ours_results['train_losses'],
                "Ours_test_full_losses": full_ours_results['test_losses'],
                "Ours_train_full_precisions": full_ours_results.get('train_precisions'),
                "Ours_test_full_precisions": full_ours_results.get('test_precisions'),
                "Ours_train_full_recalls": full_ours_results.get('train_recalls'),
                "Ours_test_full_recalls": full_ours_results.get('test_recalls'),
                "Ours_train_full_f1s": full_ours_results.get('train_f1s'),
                "Ours_test_full_f1s": full_ours_results.get('test_f1s'),
            },
            "Ours_few": {
                "Ours_train_few_auc": few_ours_results['train_aucs'],
                "Ours_test_few_auc": few_ours_results['test_aucs'],
                "Ours_train_few_losses": few_ours_results['train_losses'],
                "Ours_test_few_losses": few_ours_results['test_losses'],
                "Ours_train_few_precisions": few_ours_results.get('train_precisions'),
                "Ours_test_few_precisions": few_ours_results.get('test_precisions'),
                "Ours_train_few_recalls": few_ours_results.get('train_recalls'),
                "Ours_test_few_recalls": few_ours_results.get('test_recalls'),
                "Ours_train_few_f1s": few_ours_results.get('train_f1s'),
            }
        }
    }
    return results
def prepare_results_ss(full_ours_results, few_ours_results):
    results = {
        'Best_results': {
            "Ours": {
                "Ours_best_full_auc": full_ours_results['best_ours_auc'],
                "Ours_best_full_acc": full_ours_results['best_ours_acc'],
                "Ours_best_full_precision": full_ours_results.get('best_ours_precision'),
                "Ours_best_full_recall": full_ours_results.get('best_ours_recall'),
                "Ours_best_full_f1": full_ours_results.get('best_ours_f1'),
            },
            "Ours_few": {
                "Ours_best_few_auc": few_ours_results['best_ours_auc'],
                "Ours_best_few_acc": few_ours_results['best_ours_acc'],
                "Ours_best_few_precision": few_ours_results.get('best_ours_precision'),
                "Ours_best_few_recall": few_ours_results.get('best_ours_recall'),
                "Ours_best_few_f1": few_ours_results.get('best_ours_f1'),
            }
        },
        "Full_results": {
            "Ours": {
                "Ours_train_full_auc": full_ours_results['train_aucs'],
                "Ours_test_full_auc": full_ours_results['test_aucs'],
                "Ours_train_full_losses": full_ours_results['train_losses'],
                "Ours_test_full_losses": full_ours_results['test_losses'],
                "Ours_train_full_precisions": full_ours_results.get('train_precisions'),
                "Ours_test_full_precisions": full_ours_results.get('test_precisions'),
                "Ours_train_full_recalls": full_ours_results.get('train_recalls'),
                "Ours_test_full_recalls": full_ours_results.get('test_recalls'),
                "Ours_train_full_f1s": full_ours_results.get('train_f1s'),
                "Ours_test_full_f1s": full_ours_results.get('test_f1s'),
            },
            "Ours_few": {
                "Ours_train_few_auc": few_ours_results['train_aucs'],
                "Ours_test_few_auc": few_ours_results['test_aucs'],
                "Ours_train_few_losses": few_ours_results['train_losses'],
                "Ours_test_few_losses": few_ours_results['test_losses'],
                "Ours_train_few_precisions": few_ours_results.get('train_precisions'),
                "Ours_test_few_precisions": few_ours_results.get('test_precisions'),
                "Ours_train_few_recalls": few_ours_results.get('train_recalls'),
                "Ours_test_few_recalls": few_ours_results.get('test_recalls'),
                "Ours_train_few_f1s": few_ours_results.get('train_f1s'),
            }
        }
    }
    return results

def save_results_(args, results):
    exp_dir = os.path.join(
        f'/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}',
        args.source_data,f"args_seed:{args.random_seed}",
        #args.model_type, f"A:{args.aggr_type}_L:{args.label}_E:{args.enc_type}_M:{args.meta_type}"
        args.model_type, f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    # 데이터셋 파일 경로 구성
    dataset_file_path = os.path.join(
        args.table_path,  # 이미 완전한 경로가 구성되어 있음
        f"{args.source_data}.pkl"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"f{args.few_shot}_b{args.batch_size}_l{args.num_layers}_h{args.n_heads}_{timestamp}.json"
    filepath = os.path.join(exp_dir, filename)

    data = {
        "Experimental Memo": args.des,
        "dataset": args.source_data,
        "dataset_file_path": dataset_file_path,
        "timestamp": timestamp,
        "hyperparameters": {
            "seed": args.random_seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "full dataset learning_rate": args.source_lr,
            "few dataset learning_rate": args.source_lr_few,
            "llm_models": args.llm_model,
            "dropout_rate": args.dropout_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.n_heads,
            "few_shot": args.few_shot,
            "threshold": args.threshold,
        },
        "model_type": args.model_type,
        "label": args.label,
        "embed_type": args.embed_type,
        "edge_type": args.edge_type,
        "attn_type" : args.attn_type,
        "del_feature" : args.del_feat,
        "no_self_loop" : args.no_self_loop,
        "del_exp": getattr(args, 'del_exp', 'unknown'),
        "results": results['Best_results'], 
        "source_data" : args.source_data,
        "target_data" : args.target_data,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filepath}")
    visualize_results(args, results, exp_dir)


def save_results_A(args, results):
    """
    Ablation study용 결과 저장 함수 (시나리오 정보 포함)
    
    Args:
        args: 실험 설정
        results: 결과 딕셔너리 (scenario_info 포함)
    """
    exp_dir = os.path.join(
        f'/storage/personal/eungyeop/experiments/experiments/source_to_source_{args.base_dir}',
        args.source_data,f"args_seed:{args.random_seed}",
        args.model_type, f"Embed:{args.embed_type}_Edge:{args.edge_type}_A:{args.attn_type}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    
    # 데이터셋 파일 경로 구성
    dataset_file_path = os.path.join(
        args.table_path,  # 이미 완전한 경로가 구성되어 있음
        f"{args.source_data}.pkl"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 🔥 파일명에 시나리오 정보 추가
    scenario_id = results.get('scenario_info', {}).get('scenario_id', 'unknown')
    filename = f"f{args.few_shot}_b{args.batch_size}_l{args.num_layers}_h{args.n_heads}_scenario{scenario_id}_{timestamp}.json"
    filepath = os.path.join(exp_dir, filename)

    data = {
        "Experimental Memo": args.des,
        "dataset": args.source_data,
        "dataset_file_path": dataset_file_path,
        "timestamp": timestamp,
        "hyperparameters": {
            "seed": args.random_seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "full dataset learning_rate": args.source_lr,
            "few dataset learning_rate": args.source_lr_few,
            "llm_models": args.llm_model,
            "dropout_rate": args.dropout_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.n_heads,
            "few_shot": args.few_shot,
            "threshold": args.threshold,
        },
        "model_type": args.model_type,
        "label": args.label,
        "embed_type": args.embed_type,
        "edge_type": args.edge_type,
        "attn_type" : args.attn_type,
        "del_feature" : args.del_feat,
        # 🔥 시나리오 정보 추가 (results에서 가져옴)
        "scenario_info": results.get('scenario_info', {}),
        "results": results['Best_results']
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Ablation study results saved to {filepath}")




def save_results_st(args, results_s, results_t):
    exp_dir = os.path.join(
        'experiments/source_to_target_NEW',
        f"{args.source_data}_to_{args.target_dataset_name}",f"args_seed:{args.random_seed}",
        args.model_type,
        f"{args.graph_type}_{args.FD}_{args.center_type}"
    )

    os.makedirs(exp_dir, exist_ok=True)

    source_dataset_path = os.path.join(
        args.graph_path,
        f"{args.source_data}.pkl"
    )
    target_dataset_path = os.path.join(
        args.graph_path,
        f"{args.target_dataset_name}.pkl"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"f{args.few_shot}_b{args.batch_size}_l{args.num_layers}_h{args.heads}_{timestamp}.json"
    filepath = os.path.join(exp_dir, filename)

    data = {
        "Experimental Memo": args.des,
        "source_dataset": args.source_data,
        "target_dataset": args.target_dataset_name,
        "source_dataset_path": source_dataset_path,
        "target_dataset_path": target_dataset_path,
        "timestamp": timestamp,
        "hyperparameters": {
            "seed": args.random_seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "source_learning_rate": args.source_lr,
            "target_learning_rate": args.target_lr,
            "llm_models": args.llm_model,
            "dropout_rate": args.dropout_rate,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "few_shot": args.few_shot,
            "threshold": args.threshold
        },
        "model_type": args.model_type,
        "graph_type": args.graph_type,
        "center_type": args.center_type,
        "results": {
            "source_results": results_s['Best_results'] if args.is_pretrained is False else "The results of source are already given. Check the initial results.",
            "target_results": results_t['Best_results']
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filepath}")
    
    #visualize_results(args, results_s, os.path.join(exp_dir, 'source'))
    #visualize_results(args, results_t, os.path.join(exp_dir, 'target'))
def prepare_ml_results(args, full_baseline_results, few_baseline_results):
    """
    ML 모델들의 결과를 정리하는 함수
    """
    results = {
        'Best_results': {
            "full": {},
            "few": {}
        },

    }
    
    # 선택된 베이스라인 모델에 대해서만 처리
    for baseline in args.baseline:  # args.baseline이 하나의 모델만 포함해도 동작
        model_prefix = {
            'lr': 'lr',
            'xgb': 'xgb',
            'mlp': 'mlp',
            'cat': 'cat',
            'rf': 'rf'
        }[baseline]
        
        # Best results for full dataset
        results['Best_results']['full'][baseline] = {
            f"{baseline}_best_full_auc": full_baseline_results[baseline][f'test_{model_prefix}_auc'],
            f"{baseline}_best_full_acc": full_baseline_results[baseline][f'test_{model_prefix}_acc'],
            f"{baseline}_best_full_precision": full_baseline_results[baseline][f'test_{model_prefix}_precision'],
            f"{baseline}_best_full_recall": full_baseline_results[baseline][f'test_{model_prefix}_recall'],
            f"{baseline}_best_full_f1": full_baseline_results[baseline][f'test_{model_prefix}_f1']
        }
        
        # Best results for few-shot
        results['Best_results']['few'][baseline] = {
            f"{baseline}_best_few_auc": few_baseline_results[baseline][f'test_{model_prefix}_auc'],
            f"{baseline}_best_few_acc": few_baseline_results[baseline][f'test_{model_prefix}_acc'],
            f"{baseline}_best_few_precision": few_baseline_results[baseline][f'test_{model_prefix}_precision'],
            f"{baseline}_best_few_recall": few_baseline_results[baseline][f'test_{model_prefix}_recall'],
            f"{baseline}_best_few_f1": few_baseline_results[baseline][f'test_{model_prefix}_f1']
        }
    
    return results

def save_ml_results(args, results):
    """
    ML 모델들의 결과를 저장하는 함수
    """
    # 선택된 베이스라인 모델의 이름을 경로에 포함
    baseline_name = '_'.join(args.baseline)
    
    exp_dir = os.path.join(
        f'/storage/personal/eungyeop/experiments/experiments/ml_baselines_{args.base_dir}',
        args.source_data,
        f"args_seed:{args.random_seed}",
        baseline_name  # 모델 이름으로 서브디렉토리 생성
    )
    os.makedirs(exp_dir, exist_ok=True)

    dataset_file_path = os.path.join(
        args.table_path,
        f"{args.source_data}.csv"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"f{args.few_shot}_b{args.batch_size}_{timestamp}.json"
    filepath = os.path.join(exp_dir, filename)

    data = {
        "Experimental Memo": args.des,
        "dataset": args.source_data,
        "dataset_file_path": dataset_file_path,
        "timestamp": timestamp,
        "del_feature" : args.del_feat,
        "hyperparameters": {
            "seed": args.random_seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "dropout_rate": args.dropout_rate,
            "few_shot": args.few_shot,
            "threshold": args.threshold
        },
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filepath}")

def extract_center_nodes_and_labels(graphs):
    center_nodes = []
    labels = []
    for graph in graphs:
        center_nodes.append(graph.x[0, :768].numpy())
        labels.append(graph.y.item())
    return np.array(center_nodes), np.array(labels)
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
def visualize_dataset_distributions(train_loader_s, test_loader_s, train_loader_t, test_loader_t, source_name, target_name):
    # CombinedDataLoader에서 데이터셋 추출
    if isinstance(train_loader_s, CombinedDataLoader):
        # 첫 번째 데이터셋의 데이터만 사용
        dataset_name = list(train_loader_s.train_loaders.keys())[0]
        source_graphs = (list(train_loader_s.train_loaders[dataset_name].dataset) + 
                        list(test_loader_s.test_loaders[dataset_name].dataset))
    else:
        source_graphs = list(train_loader_s.dataset) + list(test_loader_s.dataset)
        
    if isinstance(train_loader_t, CombinedDataLoader):
        dataset_name = list(train_loader_t.train_loaders.keys())[0]
        target_graphs = (list(train_loader_t.train_loaders[dataset_name].dataset) + 
                        list(test_loader_t.test_loaders[dataset_name].dataset))
    else:
        target_graphs = list(train_loader_t.dataset) + list(test_loader_t.dataset)
    
    # Extract data
    source_center_nodes, source_labels = extract_center_nodes_and_labels(source_graphs)
    
    target_train_graphs = list(train_loader_t.dataset)
    target_train_nodes, target_train_labels = extract_center_nodes_and_labels(target_train_graphs)
    
    target_test_graphs = list(test_loader_t.dataset)
    target_test_nodes, target_test_labels = extract_center_nodes_and_labels(target_test_graphs)

    # Combine all data for normalization and dimensionality reduction
    all_nodes = np.vstack((source_center_nodes, target_train_nodes, target_test_nodes))
    
    # Normalize data
    scaler = StandardScaler()
    all_nodes_normalized = scaler.fit_transform(all_nodes)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    all_nodes_tsne = tsne.fit_transform(all_nodes_normalized)

    # Split back into source and target
    source_tsne = all_nodes_tsne[:len(source_center_nodes)]
    target_train_tsne = all_nodes_tsne[len(source_center_nodes):len(source_center_nodes)+len(target_train_nodes)]
    target_test_tsne = all_nodes_tsne[len(source_center_nodes)+len(target_train_nodes):]

    # Create DataFrame
    df = pd.DataFrame({
        'x': np.concatenate([source_tsne[:, 0], target_train_tsne[:, 0], target_test_tsne[:, 0]]),
        'y': np.concatenate([source_tsne[:, 1], target_train_tsne[:, 1], target_test_tsne[:, 1]]),
        'label': np.concatenate([source_labels, target_train_labels, target_test_labels]),
        'dataset': ['Source']*len(source_tsne) + ['Target (Train)']*len(target_train_tsne) + ['Target (Test)']*len(target_test_tsne)
    })

    # Plotting
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    
    # Use custom color palettes
    dataset_palette = {'Source': '#1f77b4', 'Target (Train)': '#ff7f0e', 'Target (Test)': '#2ca02c'}
    label_palette = {0: '#ff9999', 1: '#66b3ff'}  
    
    for dataset in df['dataset'].unique():
        for label in df['label'].unique():
            data = df[(df['dataset'] == dataset) & (df['label'] == label)]
            sns.scatterplot(data=data, x='x', y='y', 
                            hue='dataset',  
                            style='dataset', 
                            palette=dataset_palette,  
                            markers={'Source': 'o', 'Target (Train)': 's', 'Target (Test)': '^'},
                            size='dataset', sizes={'Source': 40, 'Target (Train)': 60, 'Target (Test)': 80}, 
                            alpha=0.6, edgecolor='none', ax=ax)
            
            confidence_ellipse(data['x'], data['y'], ax, n_std=2.0, 
                               edgecolor=dataset_palette[dataset], 
                               facecolor=label_palette[label],
                               alpha=0.1, 
                               linestyle='--', linewidth=2)
    
    plt.title(f"Distribution of {source_name} (Source) and {target_name} (Target Few-shot) datasets", fontsize=16)
    plt.xlabel("First t-SNE Component", fontsize=12)
    plt.ylabel("Second t-SNE Component", fontsize=12)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    dataset_legend = plt.legend(by_label.values(), by_label.keys(), title="Dataset", 
               loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.gca().add_artist(dataset_legend)
    
    label_handles = [plt.Rectangle((0,0),1,1, color=label_palette[label]) for label in label_palette]
    label_legend = plt.legend(label_handles, ['NO', 'YES'], title="Class", 
                              loc='center left', bbox_to_anchor=(1, 0.2), fontsize=10)

    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join('experiments', 'source_to_target', source_name, target_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{source_name}_{target_name}_distribution_comparison_tsne__.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Distribution comparison saved to: {save_path}")
def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce='mean')

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce='sum')

    return hyperedge
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
    elif minutes > 0:
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        return f"{seconds:.2f}s"
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss 
def check_class_imbalance(y_train, y_test, test_loader=None):
    y_all = np.concatenate([y_train, y_test])
    class_distribution = Counter(y_all)

    print("Total dataset Class distribution:")
    for class_label, count in class_distribution.items():
        print(f"Class {class_label}: {count} ({count/len(y_all)*100:.2f}%)")

    train_class_distribution = Counter(y_train)
    print("\Train dataset Class distribution::")
    for class_label, count in train_class_distribution.items():
        print(f"Class {class_label}: {count} ({count/len(y_train)*100:.2f}%)")

    test_class_distribution = Counter(y_test)
    print("\nTest dataset Class distribution:")
    for class_label, count in test_class_distribution.items():
        print(f"Class {class_label}: {count} ({count/len(y_test)*100:.2f}%)")

    if test_loader is not None:
        print("\nTest Loader:")
        for batch in test_loader:
            print(f"Batch y shape: {batch.y.shape}")
            print(f"Batch y sample: {batch.y[:5]}")
            break  # 첫 번째 배치만 출력

    print("\nClass imbalance Analysis Done.")