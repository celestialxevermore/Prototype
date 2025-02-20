import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import PowerTransformer, StandardScaler
import json

class Model(nn.Module):
    def __init__(
            self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.args = args

        self.llm_model = llm_model
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.source_dataset_name = args.source_dataset_name
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.input_dim))
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        
        # Meta-embedding MLP
        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            #nn.Dropout(dropout_rate)
        ).to(self.device)

        self.cat_val_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        ).to(self.device)

        self.num_val_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        ).to(self.device)

        self.feature_attentions = nn.ModuleList([
            nn.MultiheadAttention(
            embed_dim = self.input_dim,
            num_heads = args.heads,
            batch_first = True,
            dropout = dropout_rate
            ) for _ in range(self.num_layers)])
        
        # MLP for final prediction
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # device 설정
        
        # 다른 모듈들도 GPU로
        self.name_desc_mlp = self.name_desc_mlp.to(self.device)
        self.cat_val_mlp = self.cat_val_mlp.to(self.device)
        self.num_val_mlp = self.num_val_mlp.to(self.device)
    
    def forward(self, batch, y):
        pred = self.predict(batch)
        target = y.to(self.device).view(-1,1).float()

        loss = self.criterion(pred, target)
        return loss
    
    def predict(self, batch):
        cat_name_embeddings = batch['cat_name_embeddings'].to(self.device).squeeze(-2)
        cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
        cat_value_embeddings = batch['cat_value_embeddings'].to(self.device).squeeze(-2)
        num_name_embeddings = batch['num_name_embeddings'].to(self.device).squeeze(-2)
        num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
        num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        #pdb.set_trace()

        cat_name_desc_embeddings = torch.cat([cat_name_embeddings, cat_desc_embeddings], dim=-1)
        cat_name_desc_embeddings = self.name_desc_mlp(cat_name_desc_embeddings)
        cat_value_embeddings = torch.cat([cat_value_embeddings, cat_name_desc_embeddings], dim=-1)
        cat_value_embeddings = self.cat_val_mlp(cat_value_embeddings)

        num_name_desc_embeddings = torch.cat([num_name_embeddings, num_desc_embeddings], dim=-1)
        num_name_desc_embeddings = self.name_desc_mlp(num_name_desc_embeddings)
        num_value_embeddings = torch.cat([num_prompt_embeddings, num_name_desc_embeddings], dim=-1)
        num_value_embeddings = self.num_val_mlp(num_value_embeddings)

        sample_embeddings = torch.cat([cat_value_embeddings, num_value_embeddings], dim=1)
        
        
        if self.args.label == 'add': 
            sample_embeddings = torch.cat([label_description_embeddings, sample_embeddings], dim=1)
        if self.args.mode =='sa':
            attention_output = sample_embeddings
            for attention_layer in self.feature_attentions:
                attention_output, _ = attention_layer(
                    query=label_description_embeddings,
                    key=attention_output,
                    value=attention_output
                )
            pred = attention_output[:,0]

        elif self.args.mode == 'mean':
            pred = sample_embeddings.mean(dim=1)
        pred = self.predictor(pred)
        return pred





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pdb

# class MetaLevelCrossAttention(nn.Module):
#     def __init__(self, args, input_dim, dropout_rate):
#         super(MetaLevelCrossAttention, self).__init__()
#         self.args = args
#         self.input_dim = input_dim
#         self.dropout_rate = dropout_rate
#         self.meta_num_layers = args.meta_num_layers

#         # label_emb => (Key, Value), feature_emb => (Query)
#         self.cross_attentions = nn.ModuleList([
#             nn.MultiheadAttention(
#                 embed_dim=self.input_dim,
#                 num_heads=self.args.meta_heads,
#                 batch_first=True,
#                 dropout=dropout_rate
#             ) for _ in range(args.meta_num_layers)
#         ])

#     def forward(self, label_emb, cat_emb, num_emb):
#         """
#         label_emb : shape [B, 1, D]
#           - (Key/Value) = 라벨 토큰
#         cat_emb   : shape [B, n_cat, D]
#         num_emb   : shape [B, n_num, D]
#           => (Query) = cat+num 토큰
          
#         반환:
#           meta_emb: [B, (n_cat + n_num), D]
#             - 각 feature가 label_emb와 cross-attn 한 결과
#           attn_weights: [B, (n_cat + n_num), 1]
#             - 각 feature(token)마다 label 토큰(1개)에 대한 attention 가중치
#         """
#         # 1) feature 임베딩(cat+num) concat => Query
#         #    shape = [B, (n_cat+n_num), D]
#         #feature_emb = torch.cat([cat_emb, num_emb], dim=1)
#         meta_emb = torch.cat([cat_emb, num_emb], dim=1)
        
#         # Multiple layers of attention
#         for i, attention_layer in enumerate(self.cross_attentions):
#             meta_emb, attn_weights = attention_layer(
#                 query=meta_emb,    # Previous layer's output
#                 key=label_emb,     # [B, 1, D]
#                 value=label_emb,   # [B, 1, D]
#             )
#             if i == len(self.cross_attentions) - 1:
#                 final_weights = attn_weights
        
#         return meta_emb, final_weights

# class FeatureLevelAttention(nn.Module):
#     def __init__(self, args, input_dim, dropout_rate):
#         super(FeatureLevelAttention, self).__init__()
#         self.args = args
#         self.input_dim = input_dim
#         self.dropout_rate = dropout_rate
        
#         # Multiple attention layers
#         self.feature_attentions = nn.ModuleList([
#             nn.MultiheadAttention(
#                 embed_dim=self.input_dim,
#                 num_heads=args.heads,
#                 batch_first=True,
#                 dropout=dropout_rate
#             ) for _ in range(args.num_layers)
#         ])

#     def forward(self, label_emb, sample_emb):
#         """
#         label_emb: shape [B, 1, D]
#           - Query = 라벨 토큰
#         sample_emb: shape [B, N, D]
#           - Key/Value = feature value 토큰들
          
#         반환:
#           attention_output: [B, 1, D]
#             - 모든 attention layer를 통과한 최종 출력
#         """
#         attention_output = sample_emb
#         for attention_layer in self.feature_attentions:
#             attention_output, _ = attention_layer(
#                 query=label_emb,
#                 key=attention_output,
#                 value=attention_output
#             )
        
#         return attention_output[:, 0]  # [B, D]

# class InstanceWiseEncoder(nn.Module):
#     def __init__(self, args, input_dim, hidden_dim):
#         super(InstanceWiseEncoder, self).__init__()
#         self.args = args
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim 

#         self.cat_val_mlp = nn.Sequential(
#             nn.Linear(2 * self.input_dim, self.input_dim),
#             nn.ReLU(),
#             nn.Linear(self.input_dim, self.input_dim)
#         )

#         self.num_val_mlp = nn.Sequential(
#             nn.Linear(2 * self.input_dim, self.input_dim),
#             nn.ReLU(),
#             nn.Linear(self.input_dim, self.input_dim)
#         )
#     def forward(self, cat_meta_emb, cat_weights, cat_value_emb, num_meta_emb, num_weights, num_value_emb):
#         cat_name_desc_embeddings = torch.cat([cat_meta_emb, cat_value_emb], dim = -1)
#         cat_name_desc_embeddings = self.cat_val_mlp(cat_name_desc_embeddings) + cat_weights * cat_value_emb

#         num_name_desc_embeddings = torch.cat([num_meta_emb, num_value_emb], dim = -1)
#         num_name_desc_embeddings = self.num_val_mlp(num_name_desc_embeddings) + num_weights * num_value_emb

#         return cat_name_desc_embeddings, num_name_desc_embeddings
    


# class Model(nn.Module):
#     def __init__(
#             self, args, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, llm_model):
#         super(Model, self).__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.args = args

#         self.llm_model = llm_model
#         self.input_dim = input_dim
#         self.num_layers = num_layers
#         self.source_dataset_name = args.source_dataset_name
#         self.cls_token = nn.Parameter(torch.randn(1, 1, args.input_dim))
#         num_layers = args.num_layers
#         dropout_rate = args.dropout_rate
#         llm_model = args.llm_model
#         self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        

#         self.meta_level_cross_attention = MetaLevelCrossAttention(args, input_dim, dropout_rate)
#         self.instance_wise_encoder = InstanceWiseEncoder(args, input_dim, hidden_dim)
#         self.feature_level_attention = FeatureLevelAttention(args, input_dim, dropout_rate)
#         # Meta-embedding MLP
#         self.name_desc_mlp = nn.Sequential(
#             nn.Linear(2 * self.input_dim, self.input_dim),
#             nn.LayerNorm(self.input_dim),
#             nn.ReLU(),
#             nn.Linear(self.input_dim, self.input_dim),
#         ).to(self.device)

#         self.cat_val_mlp = nn.Sequential(
#             nn.Linear(2 * self.input_dim, self.input_dim),
#             nn.ReLU(),
#             nn.Linear(self.input_dim, self.input_dim),
#         ).to(self.device)

#         self.num_val_mlp = nn.Sequential(
#             nn.Linear(2 * self.input_dim, self.input_dim),
#             nn.ReLU(),
#             nn.Linear(self.input_dim, self.input_dim),
#         ).to(self.device)

#         self.feature_attentions = nn.ModuleList([
#             nn.MultiheadAttention(
#             embed_dim = self.input_dim,
#             num_heads = args.heads,
#             batch_first = True,
#             dropout = dropout_rate
#             ) for _ in range(self.num_layers)])
        
#         # MLP for final prediction
#         self.predictor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(hidden_dim, 1)
#         ).to(self.device)

#         # device 설정
        
#         self.name_desc_mlp = self.name_desc_mlp.to(self.device)
#         self.cat_val_mlp = self.cat_val_mlp.to(self.device)
#         self.num_val_mlp = self.num_val_mlp.to(self.device)
    
#     def forward(self, batch, y):
#         pred = self.predict(batch)
#         target = y.to(self.device).view(-1,1).float()

#         loss = self.criterion(pred, target)
#         return loss
    
#     def predict(self, batch):
#         cat_name_embeddings = batch['cat_name_embeddings'].to(self.device).squeeze(-2)
#         cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
#         cat_value_embeddings = batch['cat_value_embeddings'].to(self.device).squeeze(-2)
#         num_name_embeddings = batch['num_name_embeddings'].to(self.device).squeeze(-2)
#         num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
#         num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)
#         label_description_embeddings = batch['label_description_embeddings'].to(self.device)

#         cat_name_desc_embeddings = torch.cat([cat_name_embeddings, cat_desc_embeddings], dim=-1)
#         cat_name_desc_embeddings = self.name_desc_mlp(cat_name_desc_embeddings)
        
#         num_name_desc_embeddings = torch.cat([num_name_embeddings, num_desc_embeddings], dim=-1)
#         num_name_desc_embeddings = self.name_desc_mlp(num_name_desc_embeddings)
        
#         meta_level_description_embeddings, attention_weights = self.meta_level_cross_attention(
#             label_description_embeddings, cat_name_desc_embeddings, num_name_desc_embeddings
#         )


#         cat_meta_emb = meta_level_description_embeddings[:, :cat_desc_embeddings.size(1), :]
#         num_meta_emb = meta_level_description_embeddings[:, cat_desc_embeddings.size(1):, :]
#         cat_weights = attention_weights[:, :cat_desc_embeddings.size(1), :]
#         num_weights = attention_weights[:, cat_desc_embeddings.size(1):, :]
        
#         cat_val_emb, num_val_emb = self.instance_wise_encoder(cat_meta_emb, cat_weights, cat_value_embeddings, num_meta_emb, num_weights, num_prompt_embeddings)
#         sample_embeddings = torch.cat([cat_val_emb, num_val_emb], dim = 1)
#         sample_embeddings = sample_embeddings.mean(dim=1)
#         #pred = self.feature_level_attention(label_description_embeddings, sample_embeddings)
#         #pdb.set_trace()
#         pred = self.predictor(sample_embeddings)
#         return pred
        
        

    
        

    