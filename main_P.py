# import torch
# import os
# import random,time
# import argparse
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from math import ceil
# import torch.nn.utils as nn_utils
# from utils.util import setup_logger, format_time, fix_seed
# from models.TabularFLM_P import Model, prototype_learning
# import psutil
# from datetime import datetime
# from torch.utils.data import DataLoader
# import higher  # MAML 구현을 위한 라이브러리
# from transformers import get_linear_schedule_with_warmup  # Learning rate scheduler
# from dataset.data_dataloaders import create_episode_batch, sample_episode_from_dataset, load_dataset_once, create_episode_batch_from_memory
# from sklearn.metrics import roc_auc_score

# experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# p = psutil.Process()
# p.cpu_affinity(range(1, 64))
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# logger = setup_logger()

# def get_args():
#     parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
#     parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
#     parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
#     parser.add_argument('--input_dim', type = int, default = 768)
#     parser.add_argument('--hidden_dim', type = int, default = 128)
#     parser.add_argument('--output_dim', type = int, default = 1)
#     parser.add_argument('--num_layers', type = int, default = 3)
#     parser.add_argument('--dropout_rate', type = float, default = 0.1)
#     parser.add_argument('--n_heads', type = int, default = 4)
#     parser.add_argument('--k_basis', type = int, default = 4, help='Number of basis functions/expert heads for CoordinatorMLP')
#     parser.add_argument('--gate_temperature', type=float, default=1.0, help='Temperature for gating softmax (lower = sharper)')
#     parser.add_argument('--model', type = str, default = 'NORM_GNN')
#     parser.add_argument('--source_data', nargs='+', default=['adult', 'heart', 'blood'], 
#                         choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland','breast','magic_telescope','forest_covertype_sampled', 'higgs_sampled'],
#                         help='List of source datasets for meta-learning')
#     parser.add_argument('--target_data', type = str, default = 'diabetes')
#     parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--source_lr', type=float, default=0.0005)  # Outer loop learning rate (5e-4)
#     parser.add_argument('--source_lr_few', type=float, default=0.0005)  # 안정적인 learning rate
#     parser.add_argument('--target_lr', type=float, default=0.0001)  # Target learning rate
#     parser.add_argument('--inner_lr', type=float, default=0.001)    # Inner loop learning rate (1e-3)
#     parser.add_argument('--llm_model', type=str, default = 'gpt2_mean', choices = ['gpt2_mean','gpt2_auto','sentence-bert','bio-bert','bio-clinical-bert','bio-llama', 'new', 'LLAMA_mean','LLAMA_auto'])
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--des', type=str, help='experimental memo')
#     parser.add_argument('--base_dir', type=str, required=True)
#     parser.add_argument('--baseline', nargs='*', default=[], choices=['Logistic_Regression', 'XGBoost'],help='List of baselines to use. Leave empty to use only our model.')
#     parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")    
#     parser.add_argument('--model_type', type=str, default='TabularFLM', choices=['NORM_GNN','GAT_edge','GAT_edge_2','GAT_edge_3', 'GAT_edge_4', 'GAT_edge_5', 'TabularFLM'])
#     parser.add_argument('--label', type = str, choices = ['add', 'no'], default = 'add')
#     parser.add_argument('--enc_type', type = str, choices = ['ind', 'shared'], default = 'ind')
#     parser.add_argument('--meta_type', type = str, choices = ['meta_attn', 'meta_mlp'], default = 'meta_attn')
#     parser.add_argument('--aggr_type', type = str, choices = ['flatten', 'mean', 'attn'], default = 'attn')
#     parser.add_argument('--threshold', type = float, default = 0.5)
#     parser.add_argument('--pos_weight', type=float, default=None, help='Positive weight for BCEWithLogitsLoss (N_neg/N_pos)')
#     parser.add_argument('--use_optimal_threshold', action='store_true', help='Use optimal threshold from support set for accuracy calculation')
#     parser.add_argument('--frozen', type = bool, default = False)
#     #parser.add_argument('--use_edge_attr', action='store_true')
#     parser.add_argument('--edge_type', default = 'mlp', choices= ['mlp','normal','no_use'])
#     parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
#     parser.add_argument('--attn_type', default='gat_v1', choices= ['gat_v1','att','gat_v2', 'gate'])
#     parser.add_argument('--del_feat', nargs='+', default = [], help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
#     parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
#     parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
#     ## 시각화 관련 인자 추가
#     parser.add_argument('--viz_heatmap', action='store_true', help='Visualize heatmap')
#     parser.add_argument('--viz_graph', action='store_true', help='Visualize graph')
    
#     # 프로토타입 학습 관련 인자 추가
#     parser.add_argument('--prototype_momentum', type=float, default=0.9, help='Momentum for prototype updates')
#     parser.add_argument('--few_shot_alpha', type=float, default=0.3, help='Weight for classification loss in few-shot phase')
#     parser.add_argument('--few_shot_beta', type=float, default=0.7, help='Weight for prototype regularization in few-shot phase')
    
#     # 메타러닝 관련 인자 추가
#     parser.add_argument('--query_size', type=int, default=40, help='Number of query samples per class')
#     parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes for meta-training')
#     parser.add_argument('--inner_steps', type=int, default=1, help='Number of inner loop adaptation steps')  # 과적응 방지
#     parser.add_argument('--meta_batch_size', type=int, default=8, help='Number of episodes per meta-batch')  # 변동성 완화
#     parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Ratio of total training steps for learning rate warmup')
    

    
#     args = parser.parse_args()

#     args.table_path = f"/storage/personal/eungyeop/dataset/table/"
#     return args 

# def decide_flip_by_support_auc(fmodel, support_loader, device):
#     """support에서 AUC로 flip 결정 (글로벌 방향 결정용)"""
#     probs, ys = [], []
#     with torch.no_grad():
#         for b in support_loader:
#             y = b['y'].to(device).view(-1).float()
#             z = fmodel.predict(b).view(-1)      # logits
#             p = torch.sigmoid(z)
#             probs.append(p.cpu().numpy())
#             ys.append(y.cpu().numpy())
#     p = np.concatenate(probs)
#     y = np.concatenate(ys)
#     # 표본 8개라 AUC=0.5 부근일 수 있음 → 그대로 사용, 집계는 여러 에피소드
#     auc = roc_auc_score(y, p)
#     return (auc < 0.5), auc



# def find_optimal_threshold_on_support(fmodel, support_loader, device, use_flip):
#     """support에서 F1/Youden 지표로 최적 임계값 찾기"""
#     probs = []
#     ys = []
#     with torch.no_grad():
#         for b in support_loader:
#             y = b['y'].to(device).view(-1).float()
#             z = fmodel.predict(b).view(-1)
#             p = torch.sigmoid(-z) if use_flip else torch.sigmoid(z)
#             probs.append(p.cpu().numpy())
#             ys.append(y.cpu().numpy())
#     probs = np.concatenate(probs)
#     ys = np.concatenate(ys)

#     # Youden's J 최대화
#     ts = np.linspace(0.05, 0.95, 37)
#     best_t, best_j = 0.5, -1
#     for t in ts:
#         pred = (probs >= t).astype(int)
#         # Youden J = TPR + TNR - 1
#         tp = ((pred==1)&(ys==1)).sum()
#         tn = ((pred==0)&(ys==0)).sum()
#         fp = ((pred==1)&(ys==0)).sum()
#         fn = ((pred==0)&(ys==1)).sum()
#         sens = tp / max(tp+fn, 1)  # TPR
#         spec = tn / max(tn+fp, 1)  # TNR
#         j = sens + spec - 1
#         if j > best_j:
#             best_j, best_t = j, t
#     return best_t

# def meta_train_episode(model, support_loader, query_loader, criterion, device, inner_steps, inner_lr, meta_optimizer):
#     """higher 라이브러리를 사용한 MAML 스타일의 메타러닝 Episode 수행"""
    
#     # --- support set에서 양/음 비율 로깅만 유지 (pos_weight는 끄기) ---
#     if model.num_classes == 2:
#         pos_cnt, neg_cnt = 0, 0
#         for b in support_loader:
#             yb = b['y'].view(-1)
#             pos_cnt += (yb == 1).sum().item()
#             neg_cnt += (yb == 0).sum().item()
        
#         # support 통계 로깅 (균형 확인)
#         logger.info(f"support pos/neg: {pos_cnt}/{neg_cnt}")
        
#         # 라벨 검증 로깅 (데이터 타입/스케일 확인)
#         sample_y = next(iter(support_loader))['y']
#         logger.info(
#             f"Label check - dtype: {sample_y.dtype}, shape: {sample_y.shape}, "
#             f"unique: {torch.unique(sample_y).tolist()}, "
#             f"min: {sample_y.min().item()}, max: {sample_y.max().item()}"
#         )
    
#     for param in model.parameters():
#         param.requires_grad = True
    
#     # Inner loop를 위한 optimizer를 정의 (실제 파라미터는 업데이트하지 않음)
#     inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr)

#     # higher의 컨텍스트 안에서 Inner Loop를 수행
#     with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
#         # fmodel: 임시 모델 (계산 그래프 추적 가능)
#         # diffopt: 임시 옵티마이저
        
#         # Inner Loop: Support set으로 'fmodel'을 학습시켜 빠른 적응
#         fmodel.train()
#         for step in range(inner_steps):
#             for batch in support_loader:
#                 # fmodel을 사용하여 손실 계산 (항상 로짓 기준, pos_weight=None)
#                 raw_target = batch['y'].to(device)
#                 if model.num_classes == 2:
#                     logits = fmodel.predict(batch).view(-1)  # 로짓이어야 함
#                     target = raw_target.view(-1).float()
#                     support_loss = F.binary_cross_entropy_with_logits(logits, target)  # pos_weight=None
#                 else:
#                     target_for_loss = raw_target.squeeze().long()
#                     logits = fmodel.predict(batch)
#                     support_loss = F.cross_entropy(logits, target_for_loss)
                
#                 # diffopt.step()은 fmodel의 파라미터를 임시로 업데이트
#                 diffopt.step(support_loss)
        
#         # Outer Loop: Query set으로 meta-loss 계산
#         fmodel.eval()
#         query_loss = 0.0
#         y_true_list = []
#         y_pred_list = []
        
#         for batch in query_loader:
#             # 최종적으로 적응된 fmodel을 사용하여 쿼리셋에 대한 손실 계산
#             raw_target = batch['y'].to(device)
            
#             # 손실 계산을 functional로 (항상 로짓 기준)
#             if model.num_classes == 2:
#                 logits = fmodel.predict(batch).view(-1)
#                 target = raw_target.view(-1).float()
#                 loss = F.binary_cross_entropy_with_logits(logits, target)  # pos_weight=None
#             else:
#                 target_for_loss = raw_target.squeeze().long()
#                 pred = fmodel.predict(batch)
#                 loss = F.cross_entropy(pred, target_for_loss)
            
#             query_loss += loss
            
#             # 예측 결과 저장 (항상 same 방향)
#             with torch.no_grad():
#                 # 지표용 라벨은 1차원으로
#                 y_true_list.extend(raw_target.view(-1).cpu().numpy().tolist())
#                 # 확률 저장 (항상 same 방향)
#                 if model.num_classes == 2:
#                     prob = torch.sigmoid(logits)  # << 확률 저장(항상 same 방향)
#                     y_pred_list.extend(prob.cpu().numpy())
#                 else:
#                     prob = torch.softmax(pred, dim=1)
#                     y_pred_list.extend(prob.detach().cpu().numpy().tolist())
        
#         query_loss /= len(query_loader)
        
#         # >>> [핵심] 컨텍스트 안에서 backward()와 grad 누적 처리 <<<
#         # (meta_optimizer.zero_grad() 는 제거!)
#         query_loss.backward()
        
#         # 이름 기준 매칭으로 안전성 향상
#         f_named = dict(fmodel.named_parameters())
#         for name, p in model.named_parameters():
#             fp = f_named.get(name, None)
#             if fp is None or fp.grad is None:
#                 continue
#             if p.grad is None:
#                 p.grad = fp.grad.detach().clone()
#             else:
#                 p.grad.add_(fp.grad.detach())   # 누적!
        
#         return query_loss.detach(), y_true_list, y_pred_list

# def meta_learning_training(args, model, device, criterion):
#     """메타러닝 메인 학습 루프 (Inner Loop Freeze, Outer Loop Train)"""
    
#     # 메타러닝용 데이터셋 목록 (args.source_data 직접 사용)
#     source_datasets = args.source_data.copy()
#     logger.info(f"Meta-learning with {len(source_datasets)} source datasets: {source_datasets}")
    
#     # 모든 데이터셋을 한 번만 로드 (메모리에 저장)
#     logger.info("Loading all source datasets into memory...")
#     all_datasets = {}
#     for dataset_name in source_datasets:
#         all_datasets[dataset_name] = load_dataset_once(dataset_name, args)
#         logger.info(f"Loaded {dataset_name}: {len(all_datasets[dataset_name]['embeddings'])} samples")
    
#     # ---- AdamW: weight decay는 bias/LayerNorm 제외 ----
#     no_decay_keys = ['bias', 'LayerNorm.weight', 'layer_norm', 'ln', 'norm']
#     decay_params, nodecay_params = [], []
#     for n, p in model.named_parameters():
#         if any(k in n for k in no_decay_keys):
#             nodecay_params.append(p)
#         else:
#             decay_params.append(p)

#     meta_optimizer = optim.AdamW(
#         [
#             {'params': decay_params,   'weight_decay': 0.01},
#             {'params': nodecay_params, 'weight_decay': 0.0},
#         ],
#         lr=args.source_lr
#     )
    
#     # ---- 스케줄러 step 수: '메타 업데이트 횟수' 기준으로 ----
#     num_meta_updates = ceil(args.num_episodes / args.meta_batch_size)
#     num_warmup_steps = int(num_meta_updates * args.warmup_ratio)

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer=meta_optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_meta_updates
#     )
    
#     train_losses = []
#     meta_optimizer.zero_grad()   # 바깥에서 1번!
#     accum = 0
    
#     for episode in range(args.num_episodes):
#         # Episode 배치 생성 (메모리에서 샘플링)
#         episode_batch = create_episode_batch_from_memory(
#             all_datasets, source_datasets, args.few_shot, args.query_size, args.batch_size
#         )
        
#         # 메타러닝 Episode 수행. query_loss는 계산 그래프를 포함한 Tensor.
#         # Inner loop LR을 독립 하이퍼로 설정
#         query_loss, y_true, y_pred = meta_train_episode(
#             model,
#             episode_batch['support_loader'],
#             episode_batch['query_loader'],
#             criterion,
#             device,
#             args.inner_steps,
#             args.inner_lr,  # 독립적인 Inner loop LR
#             meta_optimizer  # meta_optimizer 전달
#         )
        
#         train_losses.append(query_loss.item())
#         accum += 1
        
#         # 간소화된 로깅 (main_G.py 형식 참고)
#         logger.info(f"[Episode {episode+1}/{args.num_episodes}] "
#                    f"Dataset: {episode_batch['dataset_name']}, "
#                    f"Meta Loss: {query_loss.item():.4f}")
        
#         # meta-batch에 도달하면 평균/step
#         if accum % args.meta_batch_size == 0:
#             # 스케일을 맞추려면 grad를 나눠주거나, 위에서 loss를 /meta_batch_size 해서 backward 하세요.
#             for p in model.parameters():
#                 if p.grad is not None:
#                     p.grad.div_(args.meta_batch_size)
#             nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             meta_optimizer.step()
#             scheduler.step()
#             meta_optimizer.zero_grad()
        
#         # 로깅
#         if (episode + 1) % 10 == 0:
#             avg_loss = np.mean(train_losses[-10:])
#             logger.info(f"[Episode {episode+1}] Average Loss: {avg_loss:.4f}")
    
#     # 남은 잔여 그래디언트 처리
#     if accum % args.meta_batch_size != 0:
#         for p in model.parameters():
#             if p.grad is not None:
#                 p.grad.div_(accum % args.meta_batch_size)
#         nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         meta_optimizer.step()
#         scheduler.step()
#         meta_optimizer.zero_grad()
    
#     return train_losses



# def create_target_episode(args, target_dataset, few_shot, query_size=15):
#     """Target dataset에서 few_shot 기반 episode 생성"""
    
#     # Target dataset에서 episode 샘플링 (few_shot 사용)
#     support_data, query_data = sample_episode_from_dataset(
#         target_dataset, few_shot, query_size, args
#     )
    
#     # DataLoader 생성
#     support_loader = DataLoader(
#         support_data,
#         batch_size=args.batch_size,
#         shuffle=True
#     )
    
#     query_loader = DataLoader(
#         query_data,
#         batch_size=args.batch_size,
#         shuffle=False
#     )
    
#     # 동적으로 n_way 결정 (실제 데이터에서 클래스 수 확인)
#     n_way = len(set([data['y'].item() for data in support_data]))
    
#     return {
#         'dataset_name': target_dataset,
#         'support_loader': support_loader,
#         'query_loader': query_loader,
#         'n_way': n_way,
#         'k_shot': few_shot,
#         'support_size': len(support_data),
#         'query_size': len(query_data)
#     }

# def evaluate_on_target_dataset(args, model, device, criterion, num_episodes=50):
#     """Target dataset에서 few_shot 기반 평가"""
    
#     # 기본값 사용
#     num_episodes = 50  # 고정값
#     adaptation_steps = args.inner_steps  # 기존 inner_steps 사용
    
#     logger.info(f"Evaluating on target dataset: {args.target_data}")
#     logger.info(f"Using few_shot: {args.few_shot}")
#     logger.info(f"Number of target episodes: {num_episodes}")
#     logger.info(f"Target adaptation steps: {adaptation_steps}")
    
#     # 글로벌 flip 결정을 위한 변수들
#     warmup_eps = 20   # 처음 20개 에피소드로 방향 결정
#     flip_votes = []   # 1: flip, 0: no flip
#     global_flip = None
    
#     # Target dataset에서 여러 episode 생성하여 평가
#     target_episodes = []
#     for episode in range(num_episodes):
#         episode_data = create_target_episode(
#             args, args.target_data, args.few_shot, args.query_size
#         )
#         target_episodes.append(episode_data)
    
#     # 각 episode에서 성능 측정
#     episode_results = []
    
#     # Model.predict가 로짓을 반환하는지 확인 (초기에 한 번)
#     if model.num_classes == 2:
#         sample_episode = target_episodes[0]
#         sample_batch = next(iter(sample_episode['support_loader']))
#         with torch.no_grad():
#             z = model.predict(sample_batch)
#             logger.info(f"[Target] Model.predict check - shape: {z.shape}, "
#                        f"min: {z.min().item():.3f}, max: {z.max().item():.3f}")
    
#     for episode_idx, episode_data in enumerate(target_episodes):
#         logger.info(f"[Target Episode {episode_idx+1}/{num_episodes}] "
#                    f"Dataset: {episode_data['dataset_name']}, "
#                    f"N-way: {episode_data['n_way']}, K-shot: {episode_data['k_shot']}")
        
#         # --- support set에서 양/음 비율로 pos_weight 계산 (binary일 때만) ---
#         use_pos_weight = (model.num_classes == 2)
#         if use_pos_weight:
#             pos_cnt, neg_cnt = 0, 0
#             for b in episode_data['support_loader']:
#                 yb = b['y'].view(-1)
#                 pos_cnt += (yb == 1).sum().item()
#                 neg_cnt += (yb == 0).sum().item()
#             # 안전장치 (라플라스 스무딩 + 클램프)
#             pos_cnt = max(pos_cnt, 0)
#             neg_cnt = max(neg_cnt, 0)
#             pw = (neg_cnt + 1) / (pos_cnt + 1)
#             pw = float(np.clip(pw, 0.5, 2.0))  # 보수적 범위 (AUC 안정성 우선)
#             pos_weight_tensor = torch.tensor([pw], device=device, dtype=torch.float)
            
#             # support 통계 로깅 (불균형/임계값 감지)
#             logger.info(f"[Target] support pos/neg: {pos_cnt}/{neg_cnt}, pos_weight≈{pw:.3f}")
            
#             # 라벨 검증 로깅 (데이터 타입/스케일 확인)
#             sample_y = next(iter(episode_data['support_loader']))['y']
#             logger.info(f"[Target] Label check - dtype: {sample_y.dtype}, shape: {sample_y.shape}, "
#                        f"unique: {torch.unique(sample_y).tolist()}, "
#                        f"min: {sample_y.min().item()}, max: {sample_y.max().item()}")
#         else:
#             pos_weight_tensor = None
        
#         # 훈련 때와 동일한 MAML 스타일 적응을 수행
#         model.train() # Adaptation을 위해 train 모드로 설정
        
#         # Inner loop optimizer 정의 (훈련과 동일한 LR 사용)
#         inner_optimizer = optim.Adam(model.parameters(), lr=args.inner_lr)
        
#         with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
#             # 2) inner adaptation (항상 로짓 기준, pos_weight=None)
#             fmodel.train()
#             for step in range(adaptation_steps):
#                 for batch in episode_data['support_loader']:
#                     raw = batch['y'].to(device).view(-1).float()
#                     logits = fmodel.predict(batch).view(-1)
#                     if model.num_classes == 2:
#                         loss = F.binary_cross_entropy_with_logits(logits, raw)  # pos_weight=None
#                     else:
#                         tgt = raw.long()
#                         loss = F.cross_entropy(fmodel.predict(batch), tgt)
                    
#                     diffopt.step(loss)
                    
#                     # 메모리 정리
#                     del loss
#                     torch.cuda.empty_cache()
            
#             # 1) fmodel inner adaptation 끝난 직후: flip 결정
#             if model.num_classes == 2:
#                 flip_ep, support_auc = decide_flip_by_support_auc(fmodel, episode_data['support_loader'], device)
                
#                 if (episode_idx + 1) <= warmup_eps:
#                     flip_votes.append(1 if flip_ep else 0)
#                     use_flip = flip_ep  # 워밍업 구간은 에피소드 결정 사용
#                 else:
#                     if global_flip is None:
#                         # 워밍업 종료 시점에 글로벌 방향 고정
#                         global_flip = (np.median(flip_votes) > 0.5)
#                         logger.info(f"[Target] Global flip locked: {global_flip} "
#                                     f"(warmup votes={sum(flip_votes)}/{len(flip_votes)})")
#                     use_flip = global_flip
                
#                 logger.info(f"[Target Ep {episode_idx+1}] flip={use_flip}, support_auc={support_auc:.3f}")
#             else:
#                 use_flip = False
            
#             # 3) (옵션) 최적 임계값
#             if args.use_optimal_threshold and model.num_classes == 2:
#                 thr = find_optimal_threshold_on_support(fmodel, episode_data['support_loader'], device, use_flip)
#                 logger.info(f"[Target Ep {episode_idx+1}] Optimal threshold: {thr:.3f}")
#             else:
#                 thr = 0.5
            
#             # Query set으로 '적응된' 모델의 성능 평가
#             fmodel.eval()
#             query_loss = 0.0
#             y_true_list = []
#             y_pred_list = []
            
#             with torch.no_grad():
#                 for batch in episode_data['query_loader']:
#                     # 적응된 fmodel로 예측 수행 (predict 메소드 직접 호출)
#                     pred = fmodel.predict(batch)                      # logits
#                     raw_target = batch['y'].to(device)               # [B]
                    
#                     # 손실 계산을 functional로 (항상 로짓 기준)
#                     if model.num_classes == 2:
#                         pred_flat = pred.view(-1)
#                         target_flat = raw_target.view(-1).float()
#                         loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)  # pos_weight=None
#                     else:
#                         target_for_loss = raw_target.squeeze().long()
#                         loss = F.cross_entropy(pred, target_for_loss)
                    
#                     query_loss += loss.item()
                    
#                     # 확률 산출은 여기서만 방향 적용
#                     if model.num_classes == 2:
#                         proba = torch.sigmoid(-pred).squeeze(-1) if use_flip else torch.sigmoid(pred).squeeze(-1)
#                     else:
#                         proba = torch.softmax(pred, dim=1)
                    
#                     # 확률 저장
#                     y_pred_list.extend(proba.cpu().numpy().tolist())
                    
#                     # 지표용 라벨은 1차원으로
#                     y_true_list.extend(raw_target.view(-1).cpu().numpy().tolist())
            
#             # 평균 loss 계산
#             query_loss /= len(episode_data['query_loader'])
            
#             # 정확도 계산
#             y_true = np.array(y_true_list)
#             y_pred = np.asarray(y_pred_list)
            
#             if model.num_classes == 2:
#                 # 최적 임계값 사용 (이미 위에서 계산됨)
#                 y_pred_classes = (y_pred >= thr).astype(int)
#             else:
#                 # Multi-class
#                 y_pred_classes = y_pred.argmax(axis=1)
            
#             accuracy = np.mean(y_true == y_pred_classes)
            
#             # 디버깅: 각 에피소드의 y_true와 y_pred 길이 확인
#             logger.info(f"[Target Episode {episode_idx+1}] y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
            
#             # numpy array로 변환하여 저장
#             y_true_np = np.array(y_true_list)
#             y_pred_np = np.array(y_pred_list)
            
#             episode_results.append({
#                 'episode': episode_idx + 1,
#                 'loss': query_loss,
#                 'accuracy': accuracy,
#                 'y_true': y_true_np,
#                 'y_pred': y_pred_np,
#                 'threshold': thr if model.num_classes == 2 else None  # 임계값 저장
#             })
            
#             # 디버깅: Target에서의 예측 분포 확인
#             pred_distribution = np.bincount(y_pred_classes, minlength=2)
#             logger.info(f"[Target Episode {episode_idx+1}] Loss: {query_loss:.4f}, "
#                        f"Accuracy: {accuracy:.4f}, Pred Distribution: {pred_distribution}")
            
#             # AUC 디버그 로깅 (해석 명확화)
#             if model.num_classes == 2:
#                 try:
#                     from sklearn.metrics import roc_auc_score
#                     # y_pred는 이미 '양성 확률' (flip 적용됨)
#                     auc = roc_auc_score(y_true, y_pred)          # y_pred는 '양성 확률'
#                     auc_alt = roc_auc_score(y_true, 1 - y_pred)  # 대조용
#                     logger.info(f"[Ep {episode_idx+1}] flip={use_flip}, AUC={auc:.3f}, AUC_alt={auc_alt:.3f}")
                    
#                     # 확률 히스토그램 (폭주 감지)
#                     probs = np.asarray(y_pred_list)
#                     logger.info(f"probs mean={probs.mean():.3f}, std={probs.std():.3f}, "
#                                 f"p<0.01={(probs<0.01).mean():.2%}, p>0.99={(probs>0.99).mean():.2%}")
#                 except Exception as e:
#                     logger.info(f"AUC error: {e}")
    
#     # 저장 전에 per-episode AUC 분포 로그
#     if model.num_classes == 2:
#         per_ep_aucs = []
#         for ep in episode_results:
#             yt = ep['y_true'].ravel()
#             yp = ep['y_pred'].ravel()
#             try:
#                 from sklearn.metrics import roc_auc_score
#                 per_ep_aucs.append(roc_auc_score(yt, yp))
#             except:
#                 pass
#         if per_ep_aucs:
#             logger.info(f"[Target] Per-episode AUC median={np.median(per_ep_aucs):.3f} "
#                         f"mean={np.mean(per_ep_aucs):.3f}")
    
#     # 전체 결과 요약
#     avg_loss = np.mean([result['loss'] for result in episode_results])
#     avg_accuracy = np.mean([result['accuracy'] for result in episode_results])
    
#     logger.info(f"Target Dataset Evaluation Summary:")
#     logger.info(f"Average Loss: {avg_loss:.4f}")
#     logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    
#     return {
#         'episode_results': episode_results,
#         'avg_loss': avg_loss,
#         'avg_accuracy': avg_accuracy,
#         'num_episodes': num_episodes
#     }

# def save_meta_learning_results(args, train_losses, target_performance):
#     """메타러닝 결과를 기존 형식과 동일하게 저장"""
    
#     from utils.util import wrap_up_results_, prepare_results_, save_results_
    
#     # Target evaluation 결과를 기존 형식에 맞게 변환
#     target_episode_results = target_performance['episode_results']
    
#     # 모든 episode의 y_true, y_pred 수집
#     all_y_true_target = []
#     all_y_pred_target = []
#     target_losses = []
#     target_accs = []
#     target_auprcs = []
    
#     for episode_result in target_episode_results:
#         # numpy array를 list로 변환하여 extend
#         if isinstance(episode_result['y_true'], np.ndarray):
#             all_y_true_target.extend(episode_result['y_true'].tolist())
#         else:
#             all_y_true_target.extend(episode_result['y_true'])
            
#         if isinstance(episode_result['y_pred'], np.ndarray):
#             all_y_pred_target.extend(episode_result['y_pred'].tolist())
#         else:
#             all_y_pred_target.extend(episode_result['y_pred'])
            
#         target_losses.append(episode_result['loss'])
#         target_accs.append(episode_result['accuracy'])
    
#     # Target에서의 최고 성능 찾기
#     best_target_idx = np.argmax(target_accs)
#     best_target_acc = target_accs[best_target_idx]
#     best_target_loss = target_losses[best_target_idx]
    
#     # AUC, Precision, Recall, F1 계산 (binary classification)
#     y_true_array = np.array(all_y_true_target)
#     y_pred_array = np.array(all_y_pred_target)
    
#     # 디버깅: 예측 배열 형태 확인
#     logger.info(f"y_true_array shape: {y_true_array.shape}")
#     logger.info(f"y_pred_array shape: {y_pred_array.shape}")
#     logger.info(f"y_true_array sample: {y_true_array[:5]}")
#     logger.info(f"y_pred_array sample: {y_pred_array[:5]}")
    
#     # 길이 불일치 문제 해결: 더 짧은 길이에 맞춤
#     min_length = min(len(y_true_array), len(y_pred_array))
#     logger.info(f"Adjusting to minimum length: {min_length}")
    
#     y_true_array = y_true_array[:min_length]
#     y_pred_array = y_pred_array[:min_length]
    
#     logger.info(f"After adjustment - y_true_array shape: {y_true_array.shape}, y_pred_array shape: {y_pred_array.shape}")
    
#     if len(y_pred_array.shape) > 1 and y_pred_array.shape[1] > 1:
#         y_pred_proba = y_pred_array[:, 1]  # positive class probability
#         y_pred_classes = y_pred_array.argmax(axis=1)
#     else:
#         # 1차원이거나 2차원이지만 1개 열만 있는 경우
#         if len(y_pred_array.shape) == 1:
#             y_pred_proba = y_pred_array
#         else:
#             y_pred_proba = y_pred_array[:, 0]  # 첫 번째 열 사용
        
#         # 에피소드별 임계값을 사용하여 클래스 결정 (정합성 보장)
#         if len(target_episode_results) > 0 and target_episode_results[0].get('threshold') is not None:
#             # 모든 에피소드의 임계값이 동일한지 확인
#             thresholds = [ep.get('threshold', 0.5) for ep in target_episode_results]
#             if len(set(thresholds)) == 1:  # 모든 임계값이 동일
#                 threshold = thresholds[0]
#                 logger.info(f"Using consistent threshold: {threshold:.3f}")
#             else:
#                 # 임계값이 다르면 평균 사용
#                 threshold = np.mean(thresholds)
#                 logger.info(f"Using average threshold: {threshold:.3f} (varied across episodes)")
#             y_pred_classes = (y_pred_proba > threshold).astype(int)
#         else:
#             # 기본 임계값 사용
#             y_pred_classes = (y_pred_proba > 0.5).astype(int)
    
#     # Metrics 계산
#     from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
    
#     try:
#         target_auc = roc_auc_score(y_true_array, y_pred_proba)
#     except:
#         target_auc = 0.5  # fallback
    
#     try:
#         target_auprc = average_precision_score(y_true_array, y_pred_proba)
#     except:
#         target_auprc = 0.5  # fallback
    
#     target_precision = precision_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
#     target_recall = recall_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
#     target_f1 = f1_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
    
#     # 기존 wrap_up_results_ 형식에 맞춰 target results 생성
#     target_results = wrap_up_results_(
#         train_losses=[],  # 메타러닝에서는 train_losses가 없음
#         val_losses=[],    # 메타러닝에서는 val_losses가 없음
#         test_losses=target_losses,
#         train_aucs=[],    # 메타러닝에서는 train_aucs가 없음
#         val_aucs=[],      # 메타러닝에서는 val_aucs가 없음
#         test_aucs=[target_auc],
#         train_precisions=[],  # 메타러닝에서는 train_precisions가 없음
#         val_precisions=[],    # 메타러닝에서는 val_precisions가 없음
#         test_precisions=[target_precision],
#         train_recalls=[],     # 메타러닝에서는 train_recalls가 없음
#         val_recalls=[],       # 메타러닝에서는 val_recalls가 없음
#         test_recalls=[target_recall],
#         train_f1s=[],         # 메타러닝에서는 train_f1s가 없음
#         val_f1s=[],           # 메타러닝에서는 val_f1s가 없음
#         test_f1s=[target_f1],
#         all_y_true=[all_y_true_target],
#         all_y_pred=[all_y_pred_target],
#         best_epoch=best_target_idx + 1,
#         best_ours_auc=target_auc,
#         best_ours_acc=best_target_acc,
#         best_ours_precision=target_precision,
#         best_ours_recall=target_recall,
#         best_ours_f1=target_f1,
#         train_accs=[],        # 메타러닝에서는 train_accs가 없음
#         val_accs=[],          # 메타러닝에서는 val_accs가 없음
#         test_accs=[best_target_acc],
#         train_auprcs=[],      # 메타러닝에서는 train_auprcs가 없음
#         val_auprcs=[],        # 메타러닝에서는 val_auprcs가 없음
#         test_auprcs=[target_auprc],
#         best_ours_auprc=target_auprc
#     )
    
#     # prepare_results_ 형식에 맞춰 results 구성
#     results = {
#         'Best_results': {
#             "Ours": "Meta-learning experiment - no full training phase",
#             "Ours_few": {
#                 "Ours_best_few_auc": target_auc,
#                 "Ours_best_few_acc": best_target_acc,
#                 "Ours_best_few_precision": target_precision,
#                 "Ours_best_few_recall": target_recall,
#                 "Ours_best_few_f1": target_f1,
#                 "Ours_best_few_auprc": target_auprc,
#             }
#         },
#         "Full_results": {
#             "Ours": "Meta-learning experiment - no full training phase",
#             "Ours_few": {
#                 "Ours_train_few_auc": [],  # 메타러닝에서는 없음
#                 "Ours_val_few_auc": [],    # 메타러닝에서는 없음
#                 "Ours_train_few_losses": train_losses,  # 메타러닝 loss
#                 "Ours_val_few_losses": target_losses,   # target evaluation loss
#                 "Ours_train_few_precisions": [],
#                 "Ours_val_few_precisions": [target_precision],
#                 "Ours_train_few_recalls": [],
#                 "Ours_val_few_recalls": [target_recall],
#                 "Ours_train_few_f1s": [],
#                 "Ours_val_few_f1s": [target_f1],
#             }
#         }
#     }
    
#     # 기존 save_results_ 함수 사용
#     # 하지만 메타러닝용으로 경로 수정 필요
#     exp_dir = os.path.join(
#         f'/storage/personal/eungyeop/experiments/experiments/meta_learning_{args.base_dir}',
#         args.target_data, f"args_seed:{args.random_seed}",
#         args.model_type, f"Meta_F{args.few_shot}"
#     )
#     os.makedirs(exp_dir, exist_ok=True)
    
#     # 데이터셋 파일 경로 구성
#     dataset_file_path = os.path.join(
#         args.table_path,
#         f"{args.target_data}.pkl"
#     )
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"meta_f{args.few_shot}_b{args.batch_size}_{timestamp}.json"
#     filepath = os.path.join(exp_dir, filename)
    
#     # 기존 형식과 동일한 데이터 구조
#     data = {
#         "Experimental Memo": getattr(args, 'des', 'Meta-learning experiment'),
#         "dataset": args.target_data,
#         "dataset_file_path": dataset_file_path,
#         "timestamp": timestamp,
#         "experiment_type": "Meta-Learning",
#         "hyperparameters": {
#             "seed": args.random_seed,
#             "batch_size": args.batch_size,
#             "few_shot": args.few_shot,
#             "query_size": args.query_size,
#             "num_episodes": args.num_episodes,
#             "source_lr": args.source_lr,
#             "target_lr": args.target_lr,
#             "inner_steps": args.inner_steps,
#             "llm_models": args.llm_model,
#             "dropout_rate": args.dropout_rate,
#             "hidden_dim": args.hidden_dim,
#             "num_layers": args.num_layers,
#             "num_heads": args.n_heads,
#             "k_basis": args.k_basis,
#         },
#         "model_type": args.model_type,
#         "embed_type": args.embed_type,
#         "edge_type": args.edge_type,
#         "attn_type": args.attn_type,
#         "del_feature": args.del_feat,
#         "no_self_loop": args.no_self_loop,
#         "source_datasets": args.source_data,
#         "results": results['Best_results']
#     }
    
#     # JSON 파일로 저장
#     import json
#     with open(filepath, 'w') as f:
#         json.dump(data, f, indent=4, default=str)
    
#     logger.info(f"Results saved to: {filepath}")
#     return filepath

# def main():
#     start_time = time.time()
#     args = get_args()
    
#     fix_seed(args.random_seed)
#     device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
#     logger.info(f"Starting Meta-Learning experiment with few_shot={args.few_shot}")
#     logger.info(f"Device: {device}")
    
#     # 메타러닝용 데이터셋 목록 (args.source_data 직접 사용)
#     source_datasets = args.source_data.copy()
#     logger.info(f"Meta-learning with {len(source_datasets)} source datasets: {source_datasets}")
    
#     # 메타러닝에서는 각 episode마다 클래스 수가 다를 수 있으므로 동적으로 설정
#     # 기본값으로 few_shot 사용 (실제로는 episode에서 자동 결정)
#     # 첫 번째 source dataset으로 기본 클래스 수 설정 (실제로는 episode에서 동적 결정)
#     first_source = source_datasets[0]
#     from dataset.data_dataloaders import sample_episode_from_dataset
#     temp_support, _ = sample_episode_from_dataset(first_source, args.few_shot, args.query_size, args)
#     temp_classes = len(set([data['y'].item() for data in temp_support]))
    
#     args.num_classes = temp_classes
#     args.output_dim = temp_classes if temp_classes > 2 else 1
#     logger.info(f"Initial setup - Classes: {temp_classes}, Output dim: {args.output_dim}")
    
#     # 모델 생성 (메타러닝 모드)
#     model = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Meta")
#     model = prototype_learning(model, args)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss() if temp_classes > 2 else nn.BCEWithLogitsLoss()
    
#     # 메타러닝 학습 수행
#     logger.info("Starting meta-learning training...")
    
#     # 디버깅: 간단한 테스트
#     logger.info("Testing episode creation...")
#     try:
#         test_batch = create_episode_batch(args, source_datasets, args.few_shot, args.query_size)
#         logger.info(f"Test episode created successfully: {test_batch['dataset_name']}")
#     except Exception as e:
#         logger.error(f"Episode creation failed: {e}")
#         return
    
#     logger.info("Starting meta-learning training...")
#     train_losses = meta_learning_training(args, model, device, criterion)
    
#     # Target dataset 평가
#     logger.info("Starting target dataset evaluation...")
#     target_performance = evaluate_on_target_dataset(args, model, device, criterion)
    
#     # 최종 결과 저장
#     logger.info("Meta-learning completed!")
    
#     # 체크포인트 저장
#     checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/meta_learning/{args.random_seed}"
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_path = os.path.join(checkpoint_dir, f"Meta_F{args.few_shot}_Seed{args.random_seed}_{experiment_id}.pt")
    
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'args': args,
#         'train_losses': train_losses,
#         'final_episode': args.num_episodes,
#         'target_performance': target_performance
#     }, checkpoint_path)
    
#     logger.info(f"Model checkpoint saved to: {checkpoint_path}")
    
#     # 결과 저장
#     results_path = save_meta_learning_results(args, train_losses, target_performance)
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     logger.info(f"Total experiment time: {format_time(total_time)}")
    
#     # 간단한 결과 요약
#     if train_losses:
#         final_loss = train_losses[-1]
#         avg_loss = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else np.mean(train_losses)
#         logger.info(f"Final Episode Loss: {final_loss:.4f}")
#         logger.info(f"Average Loss (last 10 episodes): {avg_loss:.4f}")
#         logger.info(f"Total Episodes: {len(train_losses)}")
    
#     logger.info(f"Target Dataset: {args.target_data}")
#     logger.info(f"Target Few-shot: {args.few_shot}")
#     logger.info(f"Target Average Loss: {target_performance['avg_loss']:.4f}")
#     logger.info(f"Target Average Accuracy: {target_performance['avg_accuracy']:.4f}")
#     logger.info(f"Results saved to: {results_path}")

# if __name__ == "__main__":
#     main()
import torch
import os
import random,time
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
import torch.nn.utils as nn_utils
from utils.util import setup_logger, format_time, fix_seed
from models.TabularFLM_P import Model, prototype_learning
import psutil
from datetime import datetime
from torch.utils.data import DataLoader
import higher  # MAML 구현을 위한 라이브러리
from transformers import get_linear_schedule_with_warmup  # Learning rate scheduler
from dataset.data_dataloaders import create_episode_batch, sample_episode_from_dataset, load_dataset_once, create_episode_batch_from_memory
from sklearn.metrics import roc_auc_score
from contextlib import nullcontext  # AMP fallback

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

p = psutil.Process()
p.cpu_affinity(range(1, 64))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

# ===== AMP helper (bfloat16) =====
def amp_autocast():
    # Ampere 이상이면 bfloat16이 안전. 아니면 nullcontext 반환
    try:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    except Exception:
        return nullcontext()

# ===== 타깃 inner-loop에서만 가볍게 튜닝할 파라미터 지정 (ANIL 스타일) =====
def select_target_inner_params(model):
    """
    Target few-shot inner loop에서 업데이트할 파라미터만 골라냄.
    - coordinator.*
    - basis_layer.* (BasisGAT)
    - basis_layer_norm.*
    - expert_predictors.*
    나머지는 requires_grad=False로 동결.
    """
    ALLOWED_PREFIXES = (
        "coordinator.",
        "basis_layer.",
        "basis_layer_norm.",
        "expert_predictors."
    )
    inner_params, frozen, tuned = [], [], []
    for name, p in model.named_parameters():
        allow = any(name.startswith(pref) for pref in ALLOWED_PREFIXES)
        p.requires_grad = allow
        if allow:
            inner_params.append(p)
            tuned.append(name)
        else:
            frozen.append(name)
    try:
        logger.info(f"[Target inner] Tunable params: {len(tuned)}; Frozen: {len(frozen)}")
    except Exception:
        pass
    return inner_params

def restore_requires_grad_all(model):
    for p in model.parameters():
        p.requires_grad = True

def get_args():
    parser = argparse.ArgumentParser(description='ProtoLLM For Tabular Task')
    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='(meta 트레이닝 에피소드용) batch_size')
    # ▶ 추가: support/query 배치 크기 분리 (target eval/inner에서 사용)
    parser.add_argument('--support_batch_size', type=int, default=8)
    parser.add_argument('--query_batch_size', type=int, default=32)

    parser.add_argument('--input_dim', type = int, default = 768)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--output_dim', type = int, default = 1)
    parser.add_argument('--num_layers', type = int, default = 3)
    parser.add_argument('--dropout_rate', type = float, default = 0.1)
    parser.add_argument('--n_heads', type = int, default = 4)
    parser.add_argument('--k_basis', type = int, default = 4, help='Number of basis functions/expert heads for CoordinatorMLP')
    parser.add_argument('--gate_temperature', type=float, default=1.0, help='Temperature for gating softmax (lower = sharper)')
    parser.add_argument('--model', type = str, default = 'NORM_GNN')
    parser.add_argument('--source_data', nargs='+', default=['adult', 'heart', 'blood'], 
                        choices=['adult','bank','blood','car','communities','credit-g','diabetes','heart','myocardial','cleveland', 'heart_statlog','hungarian','switzerland','breast','magic_telescope','forest_covertype_sampled', 'higgs_sampled'],
                        help='List of source datasets for meta-learning')
    parser.add_argument('--target_data', type = str, default = 'diabetes')
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source_lr', type=float, default=0.0005)  # Outer loop learning rate (5e-4)
    parser.add_argument('--source_lr_few', type=float, default=0.0005)  # 안정적인 learning rate
    parser.add_argument('--target_lr', type=float, default=0.0001)  # Target learning rate
    parser.add_argument('--inner_lr', type=float, default=0.001)    # Inner loop learning rate (1e-3)
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
    parser.add_argument('--pos_weight', type=float, default=None, help='Positive weight for BCEWithLogitsLoss (N_neg/N_pos)')
    parser.add_argument('--use_optimal_threshold', action='store_true', help='Use optimal threshold from support set for accuracy calculation')
    parser.add_argument('--frozen', type = bool, default = False)
    #parser.add_argument('--use_edge_attr', action='store_true')
    parser.add_argument('--edge_type', default = 'mlp', choices= ['mlp','normal','no_use'])
    parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
    parser.add_argument('--attn_type', default='gat_v1', choices= ['gat_v1','att','gat_v2', 'gate'])
    parser.add_argument('--del_feat', nargs='+', default = [], help='Features to remove from the model. Usage: --del_feat feature1 feature2 feature3')
    parser.add_argument('--del_exp', default="You did not entered the exp type", choices=['exp1','exp2','exp3','exp4','exp5'])
    parser.add_argument('--no_self_loop', action='store_true', help="activate the self loop of the Graph attention network")
    ## 시각화 관련 인자 추가
    parser.add_argument('--viz_heatmap', action='store_true', help='Visualize heatmap')
    parser.add_argument('--viz_graph', action='store_true', help='Visualize graph')
    
    # 프로토타입 학습 관련 인자 추가
    parser.add_argument('--prototype_momentum', type=float, default=0.9, help='Momentum for prototype updates')
    parser.add_argument('--few_shot_alpha', type=float, default=0.3, help='Weight for classification loss in few-shot phase')
    parser.add_argument('--few_shot_beta', type=float, default=0.7, help='Weight for prototype regularization in few-shot phase')
    
    # 메타러닝 관련 인자 추가
    parser.add_argument('--query_size', type=int, default=40, help='Number of query samples per class')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes for meta-training')
    parser.add_argument('--inner_steps', type=int, default=1, help='Number of inner loop adaptation steps')  # 과적응 방지
    parser.add_argument('--meta_batch_size', type=int, default=8, help='Number of episodes per meta-batch')  # 변동성 완화
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Ratio of total training steps for learning rate warmup')
    
    args = parser.parse_args()
    args.table_path = f"/storage/personal/eungyeop/dataset/table/"
    return args 

def decide_flip_by_support_auc(fmodel, support_loader, device):
    """support에서 AUC로 flip 결정 (글로벌 방향 결정용)"""
    probs, ys = [], []
    with torch.no_grad():
        for b in support_loader:
            y = b['y'].to(device).view(-1).float()
            # AMP 캐스팅 (평가용이지만 안전하게 동일 로직 유지)
            with amp_autocast():
                z = fmodel.predict(b).view(-1)      # logits
                p = torch.sigmoid(z)
            probs.append(p.cpu().numpy())
            ys.append(y.cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(ys)
    auc = roc_auc_score(y, p)
    return (auc < 0.5), auc

def find_optimal_threshold_on_support(fmodel, support_loader, device, use_flip):
    """support에서 F1/Youden 지표로 최적 임계값 찾기"""
    probs = []
    ys = []
    with torch.no_grad():
        for b in support_loader:
            y = b['y'].to(device).view(-1).float()
            with amp_autocast():
                z = fmodel.predict(b).view(-1)
                p = torch.sigmoid(-z) if use_flip else torch.sigmoid(z)
            probs.append(p.cpu().numpy())
            ys.append(y.cpu().numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)

    ts = np.linspace(0.05, 0.95, 37)
    best_t, best_j = 0.5, -1
    for t in ts:
        pred = (probs >= t).astype(int)
        tp = ((pred==1)&(ys==1)).sum()
        tn = ((pred==0)&(ys==0)).sum()
        fp = ((pred==1)&(ys==0)).sum()
        fn = ((pred==0)&(ys==1)).sum()
        sens = tp / max(tp+fn, 1)
        spec = tn / max(tn+fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t

def meta_train_episode(model, support_loader, query_loader, criterion, device, inner_steps, inner_lr, meta_optimizer):
    """higher 라이브러리를 사용한 MAML 스타일의 메타러닝 Episode 수행"""
    if model.num_classes == 2:
        pos_cnt, neg_cnt = 0, 0
        for b in support_loader:
            yb = b['y'].view(-1)
            pos_cnt += (yb == 1).sum().item()
            neg_cnt += (yb == 0).sum().item()
        logger.info(f"support pos/neg: {pos_cnt}/{neg_cnt}")
        sample_y = next(iter(support_loader))['y']
        logger.info(
            f"Label check - dtype: {sample_y.dtype}, shape: {sample_y.shape}, "
            f"unique: {torch.unique(sample_y).tolist()}, "
            f"min: {sample_y.min().item()}, max: {sample_y.max().item()}"
        )
    for param in model.parameters():
        param.requires_grad = True
    
    # NOTE: 메타훈련 inner는 기존 그대로(전체 파라미터) 유지. 필요시 아래 두 줄로 교체하면 더 가벼워짐.
    # inner_params = select_target_inner_params(model)
    # inner_optimizer = optim.SGD(inner_params, lr=inner_lr, momentum=0.9)
    inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr)

    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
        # Inner Loop: Support
        fmodel.train()
        for step in range(inner_steps):
            for batch in support_loader:
                raw_target = batch['y'].to(device)
                with amp_autocast():
                    if model.num_classes == 2:
                        logits = fmodel.predict(batch).view(-1)  # logits
                        target = raw_target.view(-1).float()
                        support_loss = F.binary_cross_entropy_with_logits(logits, target)
                    else:
                        target_for_loss = raw_target.squeeze().long()
                        logits = fmodel.predict(batch)
                        support_loss = F.cross_entropy(logits, target_for_loss)
                diffopt.step(support_loss)
        
        # Outer Loop: Query
        fmodel.eval()
        query_loss = 0.0
        y_true_list = []
        y_pred_list = []
        
        for batch in query_loader:
            raw_target = batch['y'].to(device)
            with amp_autocast():
                if model.num_classes == 2:
                    logits = fmodel.predict(batch).view(-1)
                    target = raw_target.view(-1).float()
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                else:
                    target_for_loss = raw_target.squeeze().long()
                    pred = fmodel.predict(batch)
                    loss = F.cross_entropy(pred, target_for_loss)
            query_loss += loss

            with torch.no_grad():
                y_true_list.extend(raw_target.view(-1).cpu().numpy().tolist())
                if model.num_classes == 2:
                    prob = torch.sigmoid(logits)
                    y_pred_list.extend(prob.cpu().numpy())
                else:
                    prob = torch.softmax(pred, dim=1)
                    y_pred_list.extend(prob.detach().cpu().numpy().tolist())
        
        query_loss /= len(query_loader)
        query_loss.backward()
        
        f_named = dict(fmodel.named_parameters())
        for name, p in model.named_parameters():
            fp = f_named.get(name, None)
            if fp is None or fp.grad is None:
                continue
            if p.grad is None:
                p.grad = fp.grad.detach().clone()
            else:
                p.grad.add_(fp.grad.detach())
        
        return query_loss.detach(), y_true_list, y_pred_list

def meta_learning_training(args, model, device, criterion):
    """메타러닝 메인 학습 루프 (Inner Loop Freeze, Outer Loop Train)"""
    source_datasets = args.source_data.copy()
    logger.info(f"Meta-learning with {len(source_datasets)} source datasets: {source_datasets}")
    logger.info("Loading all source datasets into memory...")
    all_datasets = {}
    for dataset_name in source_datasets:
        all_datasets[dataset_name] = load_dataset_once(dataset_name, args)
        logger.info(f"Loaded {dataset_name}: {len(all_datasets[dataset_name]['embeddings'])} samples")
    
    no_decay_keys = ['bias', 'LayerNorm.weight', 'layer_norm', 'ln', 'norm']
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in no_decay_keys):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    meta_optimizer = optim.AdamW(
        [
            {'params': decay_params,   'weight_decay': 0.01},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ],
        lr=args.source_lr
    )
    
    num_meta_updates = ceil(args.num_episodes / args.meta_batch_size)
    num_warmup_steps = int(num_meta_updates * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=meta_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_meta_updates
    )
    
    train_losses = []
    meta_optimizer.zero_grad()
    accum = 0
    
    for episode in range(args.num_episodes):
        episode_batch = create_episode_batch_from_memory(
            all_datasets, source_datasets, args.few_shot, args.query_size, args.batch_size
        )
        query_loss, y_true, y_pred = meta_train_episode(
            model,
            episode_batch['support_loader'],
            episode_batch['query_loader'],
            criterion,
            device,
            args.inner_steps,
            args.inner_lr,
            meta_optimizer
        )
        
        train_losses.append(query_loss.item())
        accum += 1
        logger.info(f"[Episode {episode+1}/{args.num_episodes}] "
                   f"Dataset: {episode_batch['dataset_name']}, "
                   f"Meta Loss: {query_loss.item():.4f}")
        
        if accum % args.meta_batch_size == 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(args.meta_batch_size)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            meta_optimizer.step()
            scheduler.step()
            meta_optimizer.zero_grad()
        
        if (episode + 1) % 10 == 0:
            avg_loss = np.mean(train_losses[-10:])
            logger.info(f"[Episode {episode+1}] Average Loss: {avg_loss:.4f}")
    
    if accum % args.meta_batch_size != 0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(accum % args.meta_batch_size)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        meta_optimizer.step()
        scheduler.step()
        meta_optimizer.zero_grad()
    
    return train_losses

def create_target_episode(args, target_dataset, few_shot, query_size=15):
    """Target dataset에서 few_shot 기반 episode 생성"""
    support_data, query_data = sample_episode_from_dataset(
        target_dataset, few_shot, query_size, args
    )
    # ▶ target 적응은 support를 작게, query는 크게
    support_loader = DataLoader(
        support_data,
        batch_size=args.support_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    query_loader = DataLoader(
        query_data,
        batch_size=args.query_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    n_way = len(set([data['y'].item() for data in support_data]))
    return {
        'dataset_name': target_dataset,
        'support_loader': support_loader,
        'query_loader': query_loader,
        'n_way': n_way,
        'k_shot': few_shot,
        'support_size': len(support_data),
        'query_size': len(query_data)
    }

def evaluate_on_target_dataset(args, model, device, criterion, num_episodes=50):
    """Target dataset에서 few_shot 기반 평가 (ANIL inner + AMP + 마이크로배치)"""
    num_episodes = 50
    adaptation_steps = args.inner_steps
    
    logger.info(f"Evaluating on target dataset: {args.target_data}")
    logger.info(f"Using few_shot: {args.few_shot}")
    logger.info(f"Number of target episodes: {num_episodes}")
    logger.info(f"Target adaptation steps: {adaptation_steps}")
    
    warmup_eps = 20
    flip_votes = []
    global_flip = None
    
    target_episodes = []
    for episode in range(num_episodes):
        episode_data = create_target_episode(
            args, args.target_data, args.few_shot, args.query_size
        )
        target_episodes.append(episode_data)
    
    episode_results = []
    
    if model.num_classes == 2:
        sample_episode = target_episodes[0]
        sample_batch = next(iter(sample_episode['support_loader']))
        with torch.no_grad(), amp_autocast():
            z = model.predict(sample_batch)
            logger.info(f"[Target] Model.predict check - shape: {z.shape}, "
                       f"min: {z.min().item():.3f}, max: {z.max().item():.3f}")
    
    for episode_idx, episode_data in enumerate(target_episodes):
        logger.info(f"[Target Episode {episode_idx+1}/{num_episodes}] "
                   f"Dataset: {episode_data['dataset_name']}, "
                   f"N-way: {episode_data['n_way']}, K-shot: {episode_data['k_shot']}")
        
        use_pos_weight = (model.num_classes == 2)
        if use_pos_weight:
            pos_cnt, neg_cnt = 0, 0
            for b in episode_data['support_loader']:
                yb = b['y'].view(-1)
                pos_cnt += (yb == 1).sum().item()
                neg_cnt += (yb == 0).sum().item()
            pos_cnt = max(pos_cnt, 0); neg_cnt = max(neg_cnt, 0)
            pw = (neg_cnt + 1) / (pos_cnt + 1)
            pw = float(np.clip(pw, 0.5, 2.0))
            pos_weight_tensor = torch.tensor([pw], device=device, dtype=torch.float)
            logger.info(f"[Target] support pos/neg: {pos_cnt}/{neg_cnt}, pos_weight≈{pw:.3f}")
            sample_y = next(iter(episode_data['support_loader']))['y']
            logger.info(f"[Target] Label check - dtype: {sample_y.dtype}, shape: {sample_y.shape}, "
                       f"unique: {torch.unique(sample_y).tolist()}, "
                       f"min: {sample_y.min().item()}, max: {sample_y.max().item()}")
        else:
            pos_weight_tensor = None
        
        model.train()  # adaptation 모드
        # ▶ 타깃 적응에서는 작은 모듈만 튜닝
        inner_params = select_target_inner_params(model)
        inner_optimizer = optim.SGD(inner_params, lr=args.inner_lr, momentum=0.9)
        
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
            # Inner adaptation on support
            fmodel.train()
            for step in range(adaptation_steps):
                for batch in episode_data['support_loader']:
                    raw = batch['y'].to(device).view(-1).float()
                    with amp_autocast():
                        logits = fmodel.predict(batch).view(-1)
                        if model.num_classes == 2:
                            loss = F.binary_cross_entropy_with_logits(logits, raw)
                        else:
                            tgt = raw.long()
                            loss = F.cross_entropy(fmodel.predict(batch), tgt)
                    diffopt.step(loss)
                    del loss
                    torch.cuda.empty_cache()
            
            # flip 결정
            if model.num_classes == 2:
                flip_ep, support_auc = decide_flip_by_support_auc(fmodel, episode_data['support_loader'], device)
                if (episode_idx + 1) <= warmup_eps:
                    flip_votes.append(1 if flip_ep else 0)
                    use_flip = flip_ep
                else:
                    if global_flip is None:
                        global_flip = (np.median(flip_votes) > 0.5)
                        logger.info(f"[Target] Global flip locked: {global_flip} "
                                    f"(warmup votes={sum(flip_votes)}/{len(flip_votes)})")
                    use_flip = global_flip
                logger.info(f"[Target Ep {episode_idx+1}] flip={use_flip}, support_auc={support_auc:.3f}")
            else:
                use_flip = False
            
            # 최적 임계값
            if args.use_optimal_threshold and model.num_classes == 2:
                thr = find_optimal_threshold_on_support(fmodel, episode_data['support_loader'], device, use_flip)
                logger.info(f"[Target Ep {episode_idx+1}] Optimal threshold: {thr:.3f}")
            else:
                thr = 0.5
            
            # Query 평가
            fmodel.eval()
            query_loss = 0.0
            y_true_list, y_pred_list = [], []
            with torch.no_grad():
                for batch in episode_data['query_loader']:
                    with amp_autocast():
                        pred = fmodel.predict(batch)   # logits
                        raw_target = batch['y'].to(device)
                        if model.num_classes == 2:
                            pred_flat = pred.view(-1)
                            target_flat = raw_target.view(-1).float()
                            loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)
                        else:
                            target_for_loss = raw_target.squeeze().long()
                            loss = F.cross_entropy(pred, target_for_loss)
                    query_loss += loss.item()
                    if model.num_classes == 2:
                        proba = torch.sigmoid(-pred).squeeze(-1) if use_flip else torch.sigmoid(pred).squeeze(-1)
                    else:
                        proba = torch.softmax(pred, dim=1)
                    y_pred_list.extend(proba.cpu().numpy().tolist())
                    y_true_list.extend(raw_target.view(-1).cpu().numpy().tolist())
            query_loss /= len(episode_data['query_loader'])
            
            y_true = np.array(y_true_list)
            y_pred = np.asarray(y_pred_list)
            if model.num_classes == 2:
                y_pred_classes = (y_pred >= thr).astype(int)
            else:
                y_pred_classes = y_pred.argmax(axis=1)
            accuracy = np.mean(y_true == y_pred_classes)
            
            logger.info(f"[Target Episode {episode_idx+1}] y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
            y_true_np = np.array(y_true_list)
            y_pred_np = np.array(y_pred_list)
            episode_results.append({
                'episode': episode_idx + 1,
                'loss': query_loss,
                'accuracy': accuracy,
                'y_true': y_true_np,
                'y_pred': y_pred_np,
                'threshold': thr if model.num_classes == 2 else None
            })
            pred_distribution = np.bincount(y_pred_classes, minlength=2)
            logger.info(f"[Target Episode {episode_idx+1}] Loss: {query_loss:.4f}, "
                       f"Accuracy: {accuracy:.4f}, Pred Distribution: {pred_distribution}")
            if model.num_classes == 2:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    auc_alt = roc_auc_score(y_true, 1 - y_pred)
                    logger.info(f"[Ep {episode_idx+1}] flip={use_flip}, AUC={auc:.3f}, AUC_alt={auc_alt:.3f}")
                    probs = np.asarray(y_pred_list)
                    logger.info(f"probs mean={probs.mean():.3f}, std={probs.std():.3f}, "
                                f"p<0.01={(probs<0.01).mean():.2%}, p>0.99={(probs>0.99).mean():.2%}")
                except Exception as e:
                    logger.info(f"AUC error: {e}")
        # ▶ 컨텍스트 종료 후 require_grad 복구
        restore_requires_grad_all(model)
    
    if model.num_classes == 2:
        per_ep_aucs = []
        for ep in episode_results:
            yt = ep['y_true'].ravel()
            yp = ep['y_pred'].ravel()
            try:
                per_ep_aucs.append(roc_auc_score(yt, yp))
            except:
                pass
        if per_ep_aucs:
            logger.info(f"[Target] Per-episode AUC median={np.median(per_ep_aucs):.3f} "
                        f"mean={np.mean(per_ep_aucs):.3f}")
    
    avg_loss = np.mean([result['loss'] for result in episode_results])
    avg_accuracy = np.mean([result['accuracy'] for result in episode_results])
    logger.info(f"Target Dataset Evaluation Summary:")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    
    return {
        'episode_results': episode_results,
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'num_episodes': num_episodes
    }

def save_meta_learning_results(args, train_losses, target_performance):
    from utils.util import wrap_up_results_, prepare_results_, save_results_
    target_episode_results = target_performance['episode_results']
    all_y_true_target, all_y_pred_target = [], []
    target_losses, target_accs = [], []
    for episode_result in target_episode_results:
        if isinstance(episode_result['y_true'], np.ndarray):
            all_y_true_target.extend(episode_result['y_true'].tolist())
        else:
            all_y_true_target.extend(episode_result['y_true'])
        if isinstance(episode_result['y_pred'], np.ndarray):
            all_y_pred_target.extend(episode_result['y_pred'].tolist())
        else:
            all_y_pred_target.extend(episode_result['y_pred'])
        target_losses.append(episode_result['loss'])
        target_accs.append(episode_result['accuracy'])
    best_target_idx = np.argmax(target_accs)
    best_target_acc = target_accs[best_target_idx]
    best_target_loss = target_losses[best_target_idx]
    y_true_array = np.array(all_y_true_target)
    y_pred_array = np.array(all_y_pred_target)
    logger.info(f"y_true_array shape: {y_true_array.shape}")
    logger.info(f"y_pred_array shape: {y_pred_array.shape}")
    logger.info(f"y_true_array sample: {y_true_array[:5]}")
    logger.info(f"y_pred_array sample: {y_pred_array[:5]}")
    min_length = min(len(y_true_array), len(y_pred_array))
    logger.info(f"Adjusting to minimum length: {min_length}")
    y_true_array = y_true_array[:min_length]
    y_pred_array = y_pred_array[:min_length]
    logger.info(f"After adjustment - y_true_array shape: {y_true_array.shape}, y_pred_array shape: {y_pred_array.shape}")
    if len(y_pred_array.shape) > 1 and y_pred_array.shape[1] > 1:
        y_pred_proba = y_pred_array[:, 1]
        y_pred_classes = y_pred_array.argmax(axis=1)
    else:
        if len(y_pred_array.shape) == 1:
            y_pred_proba = y_pred_array
        else:
            y_pred_proba = y_pred_array[:, 0]
        if len(target_episode_results) > 0 and target_episode_results[0].get('threshold') is not None:
            thresholds = [ep.get('threshold', 0.5) for ep in target_episode_results]
            if len(set(thresholds)) == 1:
                threshold = thresholds[0]
                logger.info(f"Using consistent threshold: {threshold:.3f}")
            else:
                threshold = np.mean(thresholds)
                logger.info(f"Using average threshold: {threshold:.3f} (varied across episodes)")
            y_pred_classes = (y_pred_proba > threshold).astype(int)
        else:
            y_pred_classes = (y_pred_proba > 0.5).astype(int)
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
    try:
        target_auc = roc_auc_score(y_true_array, y_pred_proba)
    except:
        target_auc = 0.5
    try:
        target_auprc = average_precision_score(y_true_array, y_pred_proba)
    except:
        target_auprc = 0.5
    target_precision = precision_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
    target_recall = recall_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
    target_f1 = f1_score(y_true_array, y_pred_classes, average='binary', zero_division=0)
    target_results = wrap_up_results_(
        train_losses=[],
        val_losses=[],
        test_losses=target_losses,
        train_aucs=[],
        val_aucs=[],
        test_aucs=[target_auc],
        train_precisions=[],
        val_precisions=[],
        test_precisions=[target_precision],
        train_recalls=[],
        val_recalls=[],
        test_recalls=[target_recall],
        train_f1s=[],
        val_f1s=[],
        test_f1s=[target_f1],
        all_y_true=[all_y_true_target],
        all_y_pred=[all_y_pred_target],
        best_epoch=best_target_idx + 1,
        best_ours_auc=target_auc,
        best_ours_acc=best_target_acc,
        best_ours_precision=target_precision,
        best_ours_recall=target_recall,
        best_ours_f1=target_f1,
        train_accs=[],
        val_accs=[],
        test_accs=[best_target_acc],
        train_auprcs=[],
        val_auprcs=[],
        test_auprcs=[target_auprc],
        best_ours_auprc=target_auprc
    )
    results = {
        'Best_results': {
            "Ours": "Meta-learning experiment - no full training phase",
            "Ours_few": {
                "Ours_best_few_auc": target_auc,
                "Ours_best_few_acc": best_target_acc,
                "Ours_best_few_precision": target_precision,
                "Ours_best_few_recall": target_recall,
                "Ours_best_few_f1": target_f1,
                "Ours_best_few_auprc": target_auprc,
            }
        },
        "Full_results": {
            "Ours": "Meta-learning experiment - no full training phase",
            "Ours_few": {
                "Ours_train_few_auc": [],
                "Ours_val_few_auc": [],
                "Ours_train_few_losses": train_losses,
                "Ours_val_few_losses": target_losses,
                "Ours_train_few_precisions": [],
                "Ours_val_few_precisions": [target_precision],
                "Ours_train_few_recalls": [],
                "Ours_val_few_recalls": [target_recall],
                "Ours_train_few_f1s": [],
                "Ours_val_few_f1s": [target_f1],
            }
        }
    }
    exp_dir = os.path.join(
        f'/storage/personal/eungyeop/experiments/experiments/meta_learning_{args.base_dir}',
        args.target_data, f"args_seed:{args.random_seed}",
        args.model_type, f"Meta_F{args.few_shot}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    dataset_file_path = os.path.join(args.table_path, f"{args.target_data}.pkl")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"meta_f{args.few_shot}_b{args.batch_size}_{timestamp}.json"
    filepath = os.path.join(exp_dir, filename)
    import json
    data = {
        "Experimental Memo": getattr(args, 'des', 'Meta-learning experiment'),
        "dataset": args.target_data,
        "dataset_file_path": dataset_file_path,
        "timestamp": timestamp,
        "experiment_type": "Meta-Learning",
        "hyperparameters": {
            "seed": args.random_seed,
            "batch_size": args.batch_size,
            "few_shot": args.few_shot,
            "query_size": args.query_size,
            "num_episodes": args.num_episodes,
            "source_lr": args.source_lr,
            "target_lr": args.target_lr,
            "inner_steps": args.inner_steps,
            "llm_models": args.llm_model,
            "dropout_rate": args.dropout_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.n_heads,
            "k_basis": args.k_basis,
        },
        "model_type": args.model_type,
        "embed_type": args.embed_type,
        "edge_type": args.edge_type,
        "attn_type": args.attn_type,
        "del_feature": args.del_feat,
        "no_self_loop": args.no_self_loop,
        "source_datasets": args.source_data,
        "results": results['Best_results']
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    logger.info(f"Results saved to: {filepath}")
    return filepath

def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Starting Meta-Learning experiment with few_shot={args.few_shot}")
    logger.info(f"Device: {device}")
    source_datasets = args.source_data.copy()
    logger.info(f"Meta-learning with {len(source_datasets)} source datasets: {source_datasets}")
    first_source = source_datasets[0]
    from dataset.data_dataloaders import sample_episode_from_dataset
    temp_support, _ = sample_episode_from_dataset(first_source, args.few_shot, args.query_size, args)
    temp_classes = len(set([data['y'].item() for data in temp_support]))
    args.num_classes = temp_classes
    args.output_dim = temp_classes if temp_classes > 2 else 1
    logger.info(f"Initial setup - Classes: {temp_classes}, Output dim: {args.output_dim}")
    model = Model(args, args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate, args.llm_model, experiment_id, mode="Meta")
    model = prototype_learning(model, args)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if temp_classes > 2 else nn.BCEWithLogitsLoss()
    logger.info("Starting meta-learning training...")
    logger.info("Testing episode creation...")
    try:
        test_batch = create_episode_batch(args, source_datasets, args.few_shot, args.query_size)
        logger.info(f"Test episode created successfully: {test_batch['dataset_name']}")
    except Exception as e:
        logger.error(f"Episode creation failed: {e}")
        return
    logger.info("Starting meta-learning training...")
    train_losses = meta_learning_training(args, model, device, criterion)
    logger.info("Starting target dataset evaluation...")
    target_performance = evaluate_on_target_dataset(args, model, device, criterion)
    logger.info("Meta-learning completed!")
    checkpoint_dir = f"/storage/personal/eungyeop/experiments/checkpoints/{args.llm_model}/meta_learning/{args.random_seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"Meta_F{args.few_shot}_Seed{args.random_seed}_{experiment_id}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'train_losses': train_losses,
        'final_episode': args.num_episodes,
        'target_performance': target_performance
    }, checkpoint_path)
    logger.info(f"Model checkpoint saved to: {checkpoint_path}")
    results_path = save_meta_learning_results(args, train_losses, target_performance)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")
    if train_losses:
        final_loss = train_losses[-1]
        avg_loss = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else np.mean(train_losses)
        logger.info(f"Final Episode Loss: {final_loss:.4f}")
        logger.info(f"Average Loss (last 10 episodes): {avg_loss:.4f}")
        logger.info(f"Total Episodes: {len(train_losses)}")
    logger.info(f"Target Dataset: {args.target_data}")
    logger.info(f"Target Few-shot: {args.few_shot}")
    logger.info(f"Target Average Loss: {target_performance['avg_loss']:.4f}")
    logger.info(f"Target Average Accuracy: {target_performance['avg_accuracy']:.4f}")
    logger.info(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()