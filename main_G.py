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
from models.TabularFLM_G import Model
import psutil
from utils.visualization import visualize_model_structure
from torch_geometric.data import Batch
from datetime import datetime
import networkx as nx               
import matplotlib.pyplot as plt
import numpy as np
experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()

p.cpu_affinity(range(1, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=2095, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--input_dim', type = int, default = 768)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--output_dim', type = int, default = 1)
    parser.add_argument('--num_layers', type = int, default = 3)
    parser.add_argument('--dropout_rate', type = float, default = 0.1)
    parser.add_argument('--n_heads', type = int, default = 4)
    parser.add_argument('--model', type = str, default = 'NORM_GNN')
    parser.add_argument('--source_dataset_name', type=str, default='heart', 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland'])
    parser.add_argument('--target_dataset_name', type = str, default = 'hungarian')
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
    parser.add_argument('--use_edge_attr', action='store_true', default = False)
    # GMM 관련 인자 추가
    parser.add_argument('--use_gmm', action='store_true', help='Use GMM1 module')
    parser.add_argument('--use_gmm2', action='store_true', help='Use GMM2 module')
    parser.add_argument('--num_prototypes', type=int, default=32, help='Number of prototypes(phenotypes) in GMM')
    parser.add_argument('--gmm_stage_num', type=int, default=10, help='EM step iterations in GMM')
    parser.add_argument('--gmm_momentum', type=float, default=0.9, help='Momentum for prototype updates')
    parser.add_argument('--gmm_beta', type=float, default=1.0, help='Weight for reconstructed embedding')
    parser.add_argument('--gmm_lambda', type=float, default=2.0, help='Temperature parameter for responsibility')
    parser.add_argument('--gmm_eps', type=float, default=1e-6, help='Small value for numerical stability')

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
    
    # T2G 스타일의 2단계 학습을 위한 변수 추가
    frozen_switch = True  # 그래프 구조 고정 전환 여부
    
    best_threshold = 0.5
    best_model_state = None
    
    # viz_dir = os.path.join(f"visualizations/{args.llm_model}/cosine_similarity/{args.source_dataset_name}/{mode}/{experiment_id}")
    # os.makedirs(viz_dir, exist_ok=True)
    
    # graph_viz_dir = os.path.join(f"visualizations/{args.llm_model}/graph_structure/{args.source_dataset_name}/{mode}/{experiment_id}")
    # os.makedirs(graph_viz_dir, exist_ok=True)

    # max_samples = 20
    # sample_dirs = []
    # for i in range(max_samples):  # 최대 20개 샘플 디렉토리 미리 생성
    #     sample_dir = os.path.join(graph_viz_dir, f'sample_{i}')
    #     os.makedirs(sample_dir, exist_ok=True)
    #     sample_dirs.append(sample_dir)
        
    #     # 각 샘플 내에 레이어별 서브폴더 생성
    #     for layer_idx in range(len(model.layers)):
    #         layer_dir = os.path.join(sample_dir, f'layer_{layer_idx}')
    #         os.makedirs(layer_dir, exist_ok=True)

    for epoch in range(epochs):
        # 1) Training
        train_loss = train_func(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        # 2) Evaluate on Train / Validation
        _, y_true_train, y_pred_train = evaluate_func(model, train_loader, criterion, device)
        val_loss, y_true_val, y_pred_val = evaluate_func(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
           visualize_model_structure(model, val_loader, device, args, mode, experiment_id, epoch, max_samples=10)
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     with torch.no_grad():
        #         model.eval()
                
                
        #         sample_count = 0
                
        #         for batch_idx, batch in enumerate(val_loader):
        #             batch_on_device = {
        #                 k: v.to(device) if isinstance(v, torch.Tensor) else v
        #                 for k, v in batch.items()
        #             }
        #             prediction = model.predict(batch_on_device)
                    
        #             # 배치 크기 확인model.layers[0].global_topology_proj_head
        #             batch_size = model.layers[0].attn_weights.shape[0]
                    
        #             for sample_idx in range(batch_size):
        #                 # 3) Feature names 정리 (모든 레이어에서 공통으로 사용)
        #                 feature_names = []
        #                 if 'cat_desc_texts' in batch_on_device:
        #                     for feature in batch_on_device['cat_desc_texts']:
        #                         if isinstance(feature, tuple):
        #                             clean_name = str(feature[0])
        #                         else:
        #                             try:
        #                                 clean_name = feature.split("'")[1] if "'" in feature else feature
        #                                 clean_name = clean_name.split(',')[0]
        #                             except:
        #                                 clean_name = str(feature)
        #                         feature_names.append(clean_name)

        #                 if 'num_desc_texts' in batch_on_device:
        #                     for feature in batch_on_device['num_desc_texts']:
        #                         if isinstance(feature, tuple):
        #                             clean_name = str(feature[0])
        #                         else:
        #                             try:
        #                                 clean_name = feature.split("'")[1] if "'" in feature else feature
        #                                 clean_name = clean_name.split(',')[0]
        #                             except:
        #                                 clean_name = str(feature)
        #                         feature_names.append(clean_name)
                                
        #                 # 중복 제거 (순서 유지)
        #                 seen = set()
        #                 unique_features = []
        #                 for feat in feature_names:
        #                     if feat not in seen:
        #                         seen.add(feat)
        #                         unique_features.append(feat)
        #                 feature_names = unique_features
                        

        #                 '''
        #                     1. heatmap
        #                 '''
        #                 for layer_idx in range(len(model.layers)):
        #                     fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                            
        #                     # 1. global_sim 히트맵
        #                     global_sim_np = model.layers[layer_idx].global_sim[sample_idx].cpu().numpy()
        #                     im1 = axes[0].imshow(global_sim_np, cmap='viridis', interpolation='nearest')
        #                     axes[0].set_title('Global Similarity (Cosine)', fontsize=14)
        #                     fig.colorbar(im1, ax=axes[0])
                            
        #                     for i in range(len(feature_names)):
        #                         for j in range(len(feature_names)):
        #                             axes[0].text(j, i, f"{global_sim_np[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)

        #                     # 2. sample_sim 히트맵
        #                     global_topology_A = model.layers[layer_idx].global_topology_A[sample_idx].cpu().numpy()
        #                     im2 = axes[1].imshow(global_topology_A, cmap='plasma', interpolation='nearest')
        #                     axes[1].set_title('global_topology (Cosine + Sigmoid)', fontsize=14)
        #                     fig.colorbar(im2, ax=axes[1])
                            
        #                     for i in range(len(feature_names)):
        #                         for j in range(len(feature_names)):
        #                             axes[1].text(j, i, f"{global_topology_A[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)
                            
        #                     # 3. adjacency 히트맵
        #                     adjacency_np = model.layers[layer_idx].adjacency[sample_idx].cpu().numpy()
        #                     im3 = axes[2].imshow(adjacency_np, cmap='cividis', interpolation='nearest')
        #                     axes[2].set_title('Final Adjacency (Softmax)', fontsize=14)
        #                     fig.colorbar(im3, ax=axes[2])
        #                     #pdb.set_trace()
        #                     for i in range(len(feature_names)):
        #                         for j in range(len(feature_names)):
        #                             axes[2].text(j, i, f"{adjacency_np[i,j]:.2f}", ha="center", va="center", color="white", fontsize=5)
                            
        #                     # 모든 축에 feature_names 적용
        #                     for ax in axes:
        #                         ax.set_xticks(np.arange(len(feature_names)))
        #                         ax.set_yticks(np.arange(len(feature_names)))
        #                         ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
        #                         ax.set_yticklabels(feature_names, fontsize=8)
        #                         ax.grid(False)
                            
        #                     # 전체 타이틀
        #                     fig.suptitle(f'Graph Construction Process - Epoch {epoch} - Sample {sample_count}', fontsize=16)
        #                     plt.tight_layout()
                            
        #                     # cosine_similarity 폴더에 각 샘플별로 저장
        #                     # viz_dir에서 graph_structure 부분을 cosine_similarity로 변경
        #                     cosine_dir = viz_dir.replace('graph_structure', 'cosine_similarity')
        #                     sample_cosine_dir = os.path.join(cosine_dir, f'sample_{sample_count}' ,f'layer_{layer_idx}')
        #                     os.makedirs(sample_cosine_dir, exist_ok=True)
        #                     #layer_dir = os.path.join(sample_dirs[sample_count], f'layer_{layer_idx}')
        #                     sim_viz_path = os.path.join(sample_cosine_dir, f'epoch_{epoch}.png')
        #                     fig.savefig(sim_viz_path, dpi=300, bbox_inches='tight')
        #                     plt.close(fig)
                            
        #                     logger.info(f"Epoch {epoch} - 샘플 {sample_count} 히트맵 저장: {sim_viz_path}")

        #                 '''
        #                     2. graph structure
        #                 '''

        #                 # 각 레이어별로 시각화 수행
        #                 for layer_idx in range(len(model.layers)):
        #                     # 1) Attention 가중치(헤드 평균)
        #                     attn_weights = model.layers[layer_idx].attn_weights[sample_idx]  # [n_heads, seq, seq]
        #                     attn_weights_mean = attn_weights.mean(dim=0).cpu()

        #                     # 원본 adjacency 사용 (히트맵과 일치하는 값)
        #                     adjacency = model.layers[layer_idx].adjacency[sample_idx].cpu()
        #                     new_seq = attn_weights_mean.shape[0]
        #                     graph_matrix = torch.zeros((new_seq, new_seq), device=attn_weights_mean.device)

        #                     graph_matrix[1:, 1:] = adjacency  # 변수 간 연결은 원본 adjacency 사용
        #                     graph_matrix[0, 1:] = 1.0  # CLS->변수 연결
        #                     graph_matrix[1:, 0] = 0.0  # 변수->CLS 연결
                            
        #                     mask = (graph_matrix ==0)
        #                     final_graph_matrix = (attn_weights_mean * graph_matrix).numpy()
        #                     final_graph_matrix[mask.numpy()] = 0.0 
        #                     n_nodes = final_graph_matrix.shape[0]
                            
        #                     # 2) Edge 리스트(모든 i->j) 수집 (Barplot용)
        #                     cls_edges_info = []  # CLS에서 나가는 엣지
        #                     var_edges_info = []  # 나머지 엣지
                            
        #                     for i in range(n_nodes):
        #                         for j in range(n_nodes):
        #                             if i != j:
        #                                 w = final_graph_matrix[i, j]
        #                                 if i == 0:
        #                                     cls_edges_info.append((f"{i}->{j}", w))
        #                                 else:
        #                                     var_edges_info.append((f"{i}->{j}", w))
                            
        #                     # topK 적용
        #                     top_k = min(10, len(var_edges_info))
        #                     var_edges_info.sort(key=lambda x: x[1], reverse=True)
        #                     var_edges_info = var_edges_info[:top_k]
                            
        #                     # 전체 합치기
        #                     edges_info = cls_edges_info + var_edges_info
        #                     edges_info.sort(key=lambda x: x[1], reverse=True)
        #                     edge_labels = [x[0] for x in edges_info]
        #                     edge_weights = [x[1] for x in edges_info]
                            
        #                     # CLS 엣지와 일반 엣지 구분을 위한 색상 리스트
        #                     bar_colors = []
        #                     for label in edge_labels:
        #                         if label.startswith("0->"):
        #                             bar_colors.append("crimson")  # CLS 엣지는 빨간색
        #                         else:
        #                             bar_colors.append("cornflowerblue")  # 일반 엣지는 파란색
                            
        #                     # 노드 이름 매핑
        #                     node_name_map = {0: "CLS"}
        #                     for i in range(1, n_nodes):
        #                         idx_feat = i - 1
        #                         if idx_feat < len(feature_names):
        #                             node_name_map[i] = feature_names[idx_feat]
        #                         else:
        #                             node_name_map[i] = f"feature_{i}"
                                    
        #                     # x축 라벨에 사용할 이름 변환
        #                     display_edge_labels = []
        #                     for label in edge_labels:
        #                         i, j = map(int, label.split('->'))
        #                         display_edge_labels.append(f"{node_name_map[i]}->{node_name_map[j]}")
                            
        #                     # Figure & 2 Subplots 생성
        #                     fig, axes = plt.subplots(2,2, figsize=(24,20))
        #                     ax_bar = axes[0,0]
        #                     # -----(A) Left Subplot: Barplot)-----
        #                     bars = ax_bar.bar(range(len(edge_weights)), edge_weights, color=bar_colors)
                            
        #                     # 각 바 위에 attention score 값 표시
        #                     for i, (weight, label) in enumerate(zip(edge_weights, edge_labels)):
        #                         ax_bar.text(i, weight + 0.01, f"{weight:.3f}", 
        #                                    ha='center', va='bottom', rotation=45, 
        #                                    fontsize=7, color='black')
                            
        #                     ax_bar.set_title(f'Top Edge Weights - Layer {layer_idx}', fontsize=12)
        #                     ax_bar.set_xlabel('Edge (i->j)')
        #                     ax_bar.set_ylabel('Attention Weight')
        #                     # x축 라벨 (너무 많으면 회전)
        #                     ax_bar.set_xticks(range(len(edge_labels)))
        #                     ax_bar.set_xticklabels(display_edge_labels, rotation=90, fontsize=8)
                            
        #                     # -----(B) Right Subplot: Network Graph)-----
        #                     ax_graph = axes[0,1]
        #                     G = nx.DiGraph()
        #                     node_labels = {}

        #                     for i in range(n_nodes):
        #                         if i == 0:
        #                             node_name = "CLS"
        #                             node_color = "red"
        #                         else:
        #                             idx_feat = i - 1
        #                             if idx_feat < len(feature_names):
        #                                 node_name = feature_names[idx_feat]
        #                                 node_color = "blue"
        #                             else:
        #                                 node_name = f"feature_{i}"
        #                                 node_color = "blue"

        #                         G.add_node(i, name=node_name, color=node_color)
        #                         node_labels[i] = node_name

        #                     # CLS->Var / Var->Var 구분해서 그리기
        #                     min_edge_weight = 0.00
        #                     for i in range(n_nodes):
        #                         for j in range(n_nodes):
        #                             if i == j:
        #                                 continue

        #                             w = final_graph_matrix[i, j]
        #                             if w > min_edge_weight:
        #                                 if i == 0 and j != 0:
        #                                     # CLS->Var
        #                                     G.add_edge(i, j, weight=w, cls_to_var=True)
        #                                 elif j == 0:
        #                                     # Var->CLS는 표시 안 함
        #                                     continue
        #                                 else:
        #                                     # Var->Var
        #                                     G.add_edge(i, j, weight=w, cls_to_var=False)

        #                     pos = {}
        #                     pos[0] = np.array([0, 0])
        #                     non_center_nodes = n_nodes - 1
        #                     radius = 1.0
        #                     for i_ in range(1, n_nodes):
        #                         angle_ = 2 * np.pi * (i_ - 1) / non_center_nodes
        #                         pos[i_] = np.array([radius * np.cos(angle_), radius * np.sin(angle_)])

        #                     # 배경 그리드
        #                     for r_ in [0.25, 0.5, 0.75, 1.0]:
        #                         circle = plt.Circle((0, 0), r_, fill=False, color='lightgray', linestyle='--', alpha=0.5)
        #                         ax_graph.add_patch(circle)
        #                     for i_ in range(1, n_nodes):
        #                         angle__ = 2 * np.pi * (i_ - 1) / non_center_nodes
        #                         x_ = 1.1 * np.cos(angle__)
        #                         y_ = 1.1 * np.sin(angle__)
        #                         ax_graph.plot([0, x_], [0, y_], color='lightgray', linestyle='--', alpha=0.5)

        #                     node_colors = [d["color"] for _, d in G.nodes(data=True)]
        #                     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax_graph, edgecolors='gray')

        #                     cls_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('cls_to_var')]
        #                     var_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('cls_to_var')]

        #                     cls_weights = [G[u][v]['weight'] for (u, v) in cls_edges]
        #                     var_weights = [G[u][v]['weight'] for (u, v) in var_edges]

        #                     # CLS->Var: 빨강 굵은선
        #                     if cls_edges:
        #                         nx.draw_networkx_edges(
        #                             G, pos,
        #                             edgelist=cls_edges,
        #                             width=[2 + w * 5 for w in cls_weights],
        #                             alpha=0.7,
        #                             edge_color='crimson',
        #                             connectionstyle='arc3,rad=0.1',  
        #                             arrowstyle='-|>',  # 화살표 스타일 변경
        #                             arrowsize=15,      # 화살표 크기 키우기 (기본값보다 크게)
        #                             node_size=800,
        #                             ax=ax_graph
        #                         )

        #                     # Var->Var: 파랑 점선
        #                     if var_edges:
        #                         nx.draw_networkx_edges(
        #                             G, pos,
        #                             edgelist=var_edges,
        #                             width=[1 + w * 2 for w in var_weights],
        #                             edge_color='blue',
        #                             style='dashed',
        #                             arrowstyle='-|>',
                                    
        #                             arrowsize=30,
        #                             alpha=0.5,
        #                             ax=ax_graph,
        #                             arrows=True
        #                         )

        #                     label_options = {
        #                         "font_size": 9,
        #                         "font_color": "black",
        #                         "bbox": dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        #                     }
        #                     nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_graph, **label_options)

        #                     ax_graph.set_title(f'Graph Structure - Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=12)
        #                     ax_graph.axis('off')
        #                     ax_graph.set_aspect('equal')
        #                     ax_graph.set_xlim([-1.2, 1.2])
        #                     ax_graph.set_ylim([-1.2, 1.2])

        #                     # 3. 확장된 graph matrix heatmap
        #                     ax_graph_matrix = axes[1,0]
        #                     graph_matrix_np = graph_matrix.cpu().numpy() 
        #                     im_graph = ax_graph_matrix.imshow(graph_matrix_np, cmap="Blues", interpolation='nearest')
        #                     ax_graph_matrix.set_title("Graph Matrix (with CLS)", fontsize= 14)
        #                     fig.colorbar(im_graph, ax=ax_graph_matrix)

        #                     all_node_names = ["CLS"] + feature_names 
        #                     ax_graph_matrix.set_xticks(np.arange(len(all_node_names)))
        #                     ax_graph_matrix.set_yticks(np.arange(len(all_node_names)))
        #                     ax_graph_matrix.set_xticklabels(all_node_names, rotation=90, fontsize=8)
        #                     ax_graph_matrix.set_yticklabels(all_node_names, fontsize=8)

        #                     for i in range(len(all_node_names)):
        #                         for j in range(len(all_node_names)):
        #                             ax_graph_matrix.text(j,i, f"{graph_matrix_np[i,j]:.2f}", ha="center", va="center", color="black" if graph_matrix_np[i,j] < 0.5 else "white", fontsize=8)

        #                     ax_final = axes[1,1]
        #                     vmax = final_graph_matrix.max()
        #                     vmin = 0.0  # 0부터 시작하도록 설정

        #                     # 다른 컬러맵 사용 및 범위 조정
        #                     im_final = ax_final.imshow(final_graph_matrix, 
        #                                             cmap='YlOrRd',  # 'YlOrRd', 'hot', 'OrRd' 등 시도해볼 수 있음
        #                                             interpolation='nearest',
        #                                             vmin=vmin, 
        #                                             vmax=vmax)
        #                     ax_final.set_title("Final Graph Matrix (Attention * Graph_matrix)", fontsize=14)
        #                     fig.colorbar(im_final, ax=ax_final)
                            
        #                     ax_final.set_xticks(np.arange(len(all_node_names)))
        #                     ax_final.set_yticks(np.arange(len(all_node_names)))
        #                     ax_final.set_xticklabels(all_node_names, rotation=90, fontsize=8)
        #                     ax_final.set_yticklabels(all_node_names, fontsize=8)
        #                     # 각 셀에 값 표시
        #                     for i in range(len(all_node_names)):
        #                         for j in range(len(all_node_names)):
        #                             # 상대적인 값에 따라 텍스트 색상 결정 (0에 가까울수록 검정, 최대값에 가까울수록 흰색)
        #                             relative_value = final_graph_matrix[i,j] / vmax if vmax > 0 else 0
        #                             text_color = "black" if relative_value < 0.7 else "white"
                                    
        #                             # 값이 0일 경우 빈 문자열 표시할 수도 있음
        #                             value_text = f"{final_graph_matrix[i,j]:.3f}" #if final_graph_matrix[i,j] > 0.001 else ""
                                    
        #                             ax_final.text(j, i, value_text, 
        #                                         ha="center", va="center", 
        #                                         color=text_color, 
        #                                         fontsize=7)
                            
        #                     # 전체 제목 설정
        #                     fig.suptitle(f'Layer {layer_idx} - Epoch {epoch} - Sample {sample_count}', fontsize=18)
        #                     fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # suptitle을 위한 여백 확보
                            
        #                     # 레이어별 폴더에 저장
        #                     layer_dir = os.path.join(sample_dirs[sample_count], f'layer_{layer_idx}')
        #                     graph_path = os.path.join(layer_dir, f'epoch_{epoch}_complete.png')
        #                     fig.savefig(graph_path, dpi=300, bbox_inches='tight')
        #                     plt.close(fig)
                            
        #                     logger.info(f"샘플 {sample_count} - 레이어 {layer_idx} - 에포크 {epoch} 종합 시각화 저장: {graph_path}")

        #                 sample_count += 1
        #                 if sample_count >= max_samples:
        #                     break

        #             if sample_count >= max_samples:
        #                 break
                
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
        
        # 2단계 학습 전환 로직 (T2G와 동일)
        if no_improve >= patience:
            if frozen_switch and hasattr(model, 'froze_topology'):
                # 첫 번째 단계 종료 후 그래프 구조 고정
                model.froze_topology()
                logger.info(f"[Epoch {epoch+1}] Freezing graph topology and continuing training")
                frozen_switch = False
                no_improve = 0  # 카운터 초기화
            else:
                # 두 번째 단계도 개선이 없으면 완전히 종료
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

def main():
    start_time = time.time()
    args  = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    logger.info(f"Starting experiment with dataset: {args.source_dataset_name}")
    logger.info(f"Device: {device}")

    logger.info("Preparing Tabular datasets...")

    results = prepare_embedding_dataloaders(args, args.source_dataset_name)
    train_loader_full_s, val_loader_full_s, test_loader_full_s = results['loaders']
    num_classes = results['num_classes']
    
    if args.few_shot > 0:
        logger.info(f"Preparing few-shot samples (K={args.few_shot})...")
        train_loader_few_s = get_few_shot_embedding_samples(train_loader_full_s, args)
        val_loader_few_s = val_loader_full_s
        test_loader_few_s = test_loader_full_s
    logger.info(f"Datasets prepared, source dataset names : {args.source_dataset_name}")

    is_binary = (num_classes == 2)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    
    model_full = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model)
    model_few = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model)
    model_full = model_full.to(device)
    model_few = model_few.to(device)
    optimizer_full = optim.Adam(model_full.parameters(), lr=args.source_lr, weight_decay=1e-5)
    optimizer_few = optim.Adam(model_few.parameters(), lr=args.source_lr, weight_decay=1e-5)


    if args.few_shot == 4:
        logger.info(f"[Source-Only: Full] Start Training..")

        (train_losses_full, val_losses_full,
        train_aucs_full, val_aucs_full,
        train_precisions_full, val_precisions_full,
        train_recalls_full, val_recalls_full,
        train_f1s_full, val_f1s_full,
        train_accs_full, val_accs_full,
        best_epoch_full, best_val_auc_full, best_threshold_full
        ) = train_and_validate(args, model_full, train_loader_full_s, val_loader_full_s, criterion, optimizer_full, 
                            device, args.train_epochs, is_binary, mode="Full")

        logger.info("[Full-shot] Final Testing with best threshold from Validation")
        (test_loss_full, test_auc_full, test_precision_full, test_recall_full, test_f1_full,
        test_acc_full, all_y_true_full, all_y_pred_full) = final_test_evaluate(model_full, test_loader_full_s, criterion, device, is_binary, 
                                                                threshold=best_threshold_full)

    # 4-2) 최종 Test - Few
    logger.info("[Few-shot] Start Training...")
    (train_losses_few, val_losses_few,
    train_aucs_few, val_aucs_few,
    train_precisions_few, val_precisions_few,
    train_recalls_few, val_recalls_few,
    train_f1s_few, val_f1s_few,
    train_accs_few, val_accs_few,
    best_epoch_few, best_val_auc_few, best_threshold_few
    ) = train_and_validate(args, model_few, train_loader_few_s, val_loader_few_s, criterion, optimizer_few, 
                        device, args.train_epochs, is_binary, mode="Few")

    logger.info("[Few-shot] Final Testing with best threshold from Validation")
    (test_loss_few, test_auc_few, test_precision_few, test_recall_few, test_f1_few,
    test_acc_few, all_y_true_few, all_y_pred_few) = final_test_evaluate(model_few, test_loader_few_s, criterion, device, is_binary, 
                                                        threshold=best_threshold_few)



    
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