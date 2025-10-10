import json
import torch
import torch.nn as nn
import numpy as np 
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv,DynamicEdgeConv, EdgeConv,RGCNConv, TransformerConv, GINConv, global_mean_pool

from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from torch_geometric.nn import HypergraphConv, GCNConv, Set2Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

torch.use_deterministic_algorithms(False)
class NORM_GNN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3):
        super(NORM_GNN, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim 
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.args.use_shared:
            return x  # [batch_size, hidden_dim]
        
        x = self.fc2(x)  # [batch_size, output_dim]
        return x



class GAT(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=8):
        super(GAT, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(self.input_dim, self.hidden_dim // self.heads, heads=self.heads))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        # Output layer
        self.convs.append(GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, concat=False))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim // self.heads))

        self.fc1 = nn.Linear(self.hidden_dim // self.heads, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)

        if self.args.use_shared:
            return x  # [batch_size, hidden_dim]

        x = self.fc2(x)  # [batch_size, output_dim]
        return x



class Transformer_GNN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.3, heads=4):
        super(Transformer_GNN, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # TransformerConv for transformer-style attention
        self.convs.append(TransformerConv(input_dim, hidden_dim // heads, heads=heads, edge_dim=768))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=768))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#############################################

# class GMM(object):
#     def __init__(self, components = 16, input_dim = 128, stage_num = 10, momentum = 0.9, beta = 3):
#         """
#         GMM 클래스
#         :param k: Gaussian mixture의 component 개수
#         :param input_dim: 입력 embedding의 차원
#         :param stage_num: EM 알고리즘 반복 횟수
#         :param momentum: momentum update
#         :param beta: 최종 결과에 추가되는 스케일링 파라미터
#         """
#         self.components = components 
#         self.input_dim = input_dim
#         self.stage_num = stage_num 
#         self.momentum = momentum 
#         self.beta = beta 

#         #GMM Initializatoin 
#         self.membership = torch.ones(self.components) / self.components 
#         self.means = torch.randn(self.components, self.input_dim)
#         self.covariances = torch.eye(input_dim).reapeat(self.components,1,1)

#         # Learnable
#         self.membership = nn.Parameter(self.membership)
#         self.means = nn.Parameter(self.means)
#         self.covariances = nn.Parameter(self.covariances)
    
#     def __call__(self, embeddings, training = True):
#         """
#         EM 알고리즘 수행
#         :param embeddings: 입력 embedding (batch_size, input_dim)
#         :param if_train: 학습 모드 여부
#         :return: 업데이트된 embedding과 KL divergence loss
#         """
#         if training:
#             self.last_input_embeddings = embeddings.detach().clone() 

#         batch_size, input_dim = embeddings.size()
#         z = None # membership 
#         _embeddings = embeddings 

#         with torch.no_grad():
#             for step in range(self.stage_num):
#                 memberships = [] 
#                 for i in range(self.components):
#                     mvn = torch.distributions.MultivariateNormal(
#                         self.means[i],
#                         covariance_matrix = self.covariances[i] + 1e-6 * torch.eye(self.input_dim).to(embeddings.device),
#                     )
#                     # Log Likelihood
#                     log_probs = mvn.log_prob(_embeddings) + torch.log(self.prior[i] + 1e-6)
#                     memberships.append(log_probs)
#                 memberships = torch.stack(memberships, dim = 1)
#                 memberships = F.softmax(memberships, dim = 1)

#                 # (M-step)
#                 for i in range(self.components):
#                     weight = memberships[:,i].unsqueeze(-1)
#                     total_weight = weight.sum(dim = 0)

#                     # Mean Update 
#                     self.means[i] = weight (_embeddings).sum(dim = 0) / (total_weight + 1e-6)

#                     # Covariance Update
#                     diff = _embeddings - self.means[i]
#                     self.covariance[i] = (
#                         torch.bmm((weight * diff).unsqueeze(2), diff.unsqueeze(1)).sum(dim=0) / (total_weight + 1e-6)
#                     )
#                 z = memberships
#         _embeddings = torch.mm(z, self.means)

#         if training:
#             self.prior.data = self.momentum * self.prior.data + (1 - self.momentum) * z.mean(dim=0).data

#         return self.beta * _embeddings + embeddings

#     def compute_KLD(self, y_pred, memberships):
#         """
#             param_y_pred : P(Y|X_{source})
#             memberships : GMM responsibilities
#             return KLD
#         """
#         y_pred = F.softmax(y_pred, dim =1)
#         gmm_distribution = torch.sum(memberships, dim = 1, keepdim = True)
#         kl_loss = F.kl_div(y_pred.log(), gmm_distribution, reduction = 'batchmean')
#         return kl_loss

class TOY(nn.Module):
    def __init__(self, args, device, num_classes):
        super(TOY, self).__init__()
        self.device = device
        self.encoders = nn.ModuleDict()  
        self.output_heads = nn.ModuleDict()
        self.args = args

        for dataset_name in args.source_dataset_names:
            is_binary = num_classes[dataset_name] == 2
            output_dim = 1 if is_binary else num_classes[dataset_name]
            
            self.encoders[dataset_name] = NORM_GNN(
                args=args,
                input_dim=args.input_dim, 
                hidden_dim=args.hidden_dim, 
                output_dim=output_dim, 
                num_layers=args.num_layers, 
                dropout_rate=args.dropout_rate
            ).to(device)
        
        if args.use_shared:
            self.shared_mlp = nn.Sequential(
                nn.Linear(len(args.source_dataset_names) * args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(args.hidden_dim, args.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(args.hidden_dim // 2, output_dim)
            ).to(device)


    def forward(self, batch_data):
        if self.args.use_shared:
            graph_representations = []
            
            # 각 데이터셋의 node representations 수집
            for dataset_name, data in batch_data.items():
                data = data.to(self.device)
                graph_rep = self.encoders[dataset_name](data.x, data.edge_index, data.batch)
                graph_representations.append(graph_rep)
            
            # Concatenate all representations
            combined_rep = torch.cat(graph_representations, dim=1)
            
            # Shared MLP를 통과
            predictions = self.shared_mlp(combined_rep)
            
            # Loss 계산
            total_loss = 0
            for dataset_name, data in batch_data.items():
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(predictions.squeeze(), data.y.float())
                total_loss += loss
                
            return total_loss / len(batch_data)

        else:
            # 기존 방식
            total_loss = 0
            for dataset_name, data in batch_data.items():
                data = data.to(self.device)
                encoder = self.encoders[dataset_name]
                predictions = encoder(data.x, data.edge_index, data.batch)
                
                is_binary = predictions.shape[1] == 1 if len(predictions.shape) > 1 else True
                criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
                
                if is_binary:
                    loss = criterion(predictions.squeeze(), data.y.float())
                else:
                    loss = criterion(predictions, data.y)
                
                total_loss += loss
            
            return total_loss / len(batch_data)





class GMM(nn.Module):
    def __init__(self, components=16, input_dim=128, stage_num=10, momentum=0.9, beta=3):
        """
        GMM 클래스
        :param components: Gaussian mixture의 component 개수
        :param input_dim: 입력 embedding의 차원
        :param stage_num: EM 알고리즘 반복 횟수
        :param momentum: prior 업데이트에 사용할 momentum
        :param beta: 최종 결과에 추가되는 스케일링 파라미터
        """
        super(GMM, self).__init__()
        self.components = components
        self.input_dim = input_dim
        self.stage_num = stage_num
        self.momentum = momentum
        self.beta = beta

        # GMM Initialization
        self.prior = nn.Parameter(torch.ones(self.components) / self.components, requires_grad=False)
        self.means = nn.Parameter(torch.randn(self.components, self.input_dim))
        self.covariances = nn.Parameter(
            torch.stack([torch.eye(self.input_dim) for _ in range(self.components)])
        )

        # Visualization data
        self.last_input_embeddings = None
        self.visualization_data = []

    def __call__(self, embeddings, training=True):
        """
        EM 알고리즘 수행
        :param embeddings: 입력 embedding (batch_size, input_dim)
        :param training: 학습 모드 여부
        :return: 업데이트된 embedding과 KL divergence loss
        """
        if training:
            self.last_input_embeddings = embeddings.detach().clone()

        batch_size, input_dim = embeddings.size()
        _embeddings = embeddings
        z = None  # Membership (Responsibility)

        with torch.no_grad():
            for step in range(self.stage_num):
                memberships = []
                for i in range(self.components):
                    mvn = MultivariateNormal(
                        self.means[i],
                        covariance_matrix=self.covariances[i] + 1e-6 * torch.eye(self.input_dim).to(embeddings.device),
                    )
                    log_probs = mvn.log_prob(_embeddings) + torch.log(self.prior[i] + 1e-6)
                    memberships.append(log_probs)
                memberships = torch.stack(memberships, dim=1)
                memberships = F.softmax(memberships, dim=1)  # Normalize responsibilities

                # (M-step)
                for i in range(self.components):
                    weight = memberships[:, i].unsqueeze(-1)
                    total_weight = weight.sum(dim=0)

                    # Mean Update
                    self.means.data[i] = (weight * _embeddings).sum(dim=0) / (total_weight + 1e-6)

                    # Covariance Update
                    diff = _embeddings - self.means[i]
                    self.covariances.data[i] = (
                        torch.bmm((weight * diff).unsqueeze(2), diff.unsqueeze(1)).sum(dim=0) / (total_weight + 1e-6)
                    )

                z = memberships

        # Responsibility-weighted embeddings
        _embeddings = torch.mm(z, self.means)

        if training:
            self.prior.data = self.momentum * self.prior.data + (1 - self.momentum) * z.mean(dim=0).data
            self.visualization_data.append((z.detach().cpu(), self.means.detach().cpu()))

        return self.beta * _embeddings + embeddings, z

    def compute_kl_loss(self, source_embedding, updated_embedding, output_head):
        """
        KL Divergence Loss 계산
        :param source_embedding: GMM 적용 전 임베딩
        :param updated_embedding: GMM 적용 후 임베딩
        :param output_head: Task-specific head
        :return: KL Divergence Loss
        """
        y_pred_x = output_head(source_embedding)
        y_pred_z = output_head(updated_embedding)

        P_y_given_x = F.softmax(y_pred_x, dim=-1)
        P_y_given_z = F.softmax(y_pred_z, dim=-1)

        kl_loss = F.kl_div(P_y_given_z.log(), P_y_given_x, reduction="batchmean")
        return kl_loss


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.device = devices
        self.encoders = nn.ModuleDict()  
        self.output_heads = nn.ModuleDict()
        self.args = args
        self.kl_weight = args.kl_weight

        for dataset_name in args.source_datasets:
            is_binary = args.num_classes[dataset_name] == 2
            output_dim = 1 if is_binary else args.num_classes[dataset_name]
            

            self.encoders[dataset_name] = NORM_GNN(
                input_dim=args.input_dim, 
                hidden_dim=args.hidden_dim, 
                output_dim=output_dim, 
                num_layers=args.num_layers, 
                dropout_rate=args.dropout_rate
            ).to(device)

        self.output_heads[dataset_name] = nn.Linear(args.hidden_dim, output_dim).to(device)

        self.GMM = GMM(components = self.args.num_gaussian_components, input_dim = args.hidden_dim).to(device)

    def forward(self, batch):
        total_loss = 0
        all_embeddings = [] 
        all_labels = [] 
        all_dataset_names = [] 

        for dataset_name, data in batch.items():

            encoder = self.encoders[dataset_name]
            embedding = encoder(data.x)

            all_embeddings.append(embedding)
            all_labels.append(data.y)
            all_dataset_names.append(dataset_name)
            output_head = self.output_heads[dataset_name]


        source_embeddings= torch.cat(all_embeddings, dim =0) # X_1 + ... + X_source
        updated_embeddings, responsibilities = self.GMM(source_embeddings, training = self.training)

        split_sizes = [emb.size(0) for emb in all_embeddings]
        updated_embeddings_list = torch.split(updated_embeddings, split_sizes)
        responsibilities_list = torch.split(responsibilities, split_sizes)

        for i, dataset_name in enumerate(all_dataset_names):
            data_y = all_labels[i]
            source_embedding = all_embeddings[i] # X_{source}
            latent_embedding = updated_embeddings_list[i] # Z 
            responsibility = responsibilities_list[i]

            #latent_embedding은 이미 P(Y|Z)에서 Z를 잘 반영하는 embedding 이지만, 원래 source의 embedding을 통해 expresiveness를 높혔다. 
            latent_embedding = self.alpha * latent_embedding + (1 - self.alpha) * torch.mm(responsibility, source_embedding)
            membership_list = membership_list[i] 

            output_head = self.output_heads[dataset_name]
            # P(Y_{source} | X_source) GMM이 반영되지 않은, 원래의 embedding에서 얻은 prediction & distribution
            y_pred_x = output_head(source_embedding)
            P_y_given_x = nn.functional.softmax(y_pred_x, dim = -1)
            # P(Y_{source} | Z) GMM이 반영된 embedding에서 얻은 prediction & distribution
            y_pred_z = output_head(latent_embedding)
            P_y_given_z = nn.functional.softmax(y_pred_z, dim = -1)

            task_loss = self.compute_loss(y_pred_z, data_y, dataset_name)
            kl_loss = F.kl_div(P_y_given_z.log(), P_y_given_x, reduction = 'batchmean')
            collapse_loss = collapse_loss(responsibility, self.GMM.components, reg_weight = self.args.reg_weight)

            total_loss += task_loss + self.kl_weight * kl_loss
            total_loss += collapse_loss

        return total_loss
    
    def compute_loss(self, prediction, target, dataset_name):
        """
        Task-specific loss 계산 (e.g., binary or multi-class classification)
        """
        is_binary = self.output_heads[dataset_name].out_features == 1
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        return criterion(prediction, target)

    def collapse_loss(self, responsibility, components, reg_weight):
        responsibility = responsibility.mean(dim = 0)

        uniform = torch.full_like(responsibility, 1.0 / components)
        reg_loss = F.kl_div(responsibility.log(), uniform, reduction = 'batchmean')
        return reg_weight * reg_loss