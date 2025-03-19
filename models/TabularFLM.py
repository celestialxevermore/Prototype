import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import PowerTransformer, StandardScaler
import json
import os
from torch import Tensor
import math
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, use_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = torch.sqrt(torch.tensor(embed_dim).float())
        self.use_residual = use_residual
        
        self.W_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias = False)
        
    def forward(self, x, attn_mask = None):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attention_weights = torch.bmm(Q, K.transpose(1, 2))
        attention_weights = attention_weights / self.scaling

        if attn_mask is not None:
            attention_weights = attention_weights + attn_mask 

        attention_weights = F.softmax(attention_weights, dim=-1)

        out = torch.bmm(attention_weights, V)
        if self.use_residual:
            out = out + x  # residual connection
            
        return out, attention_weights
    
class Meta(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_type, meta_type):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.scaling = torch.sqrt(torch.tensor(input_dim).float())
        self.enc_type = enc_type
        self.meta_type = meta_type
        if self.meta_type == 'meta_attn':
            self.W_q = nn.Linear(input_dim, input_dim)
            self.W_k = nn.Linear(input_dim, input_dim)
            self.W_v = nn.Linear(input_dim, input_dim)

        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        )

    def forward(self, name_embeddings, desc_embeddings, label_description_embeddings):
        if self.meta_type == 'meta_mlp':
            name_desc_embeddings = torch.cat([name_embeddings, desc_embeddings], dim = -1)
            name_desc_embeddings = self.name_desc_mlp(name_desc_embeddings)
            return name_desc_embeddings
        if self.meta_type == 'meta_attn':
            name_desc_embeddings = torch.cat([name_embeddings, desc_embeddings], dim = -1)
            name_desc_embeddings = self.name_desc_mlp(name_desc_embeddings)
            Q = self.W_q(label_description_embeddings)
            K = self.W_k(name_desc_embeddings)
            V = self.W_v(name_desc_embeddings)
            attention_weights = torch.bmm(Q, K.transpose(1, 2))
            attention_weights = attention_weights / self.scaling
            attention_weights = F.softmax(attention_weights, dim=-1)
            meta_attn_out = torch.bmm(attention_weights, V)
            return meta_attn_out, attention_weights
            

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
        self.meta_type = args.meta_type
        self.enc_type = args.enc_type
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()

        # Meta-embedding MLP
        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            #nn.Dropout(dropout_rate)
        ).to(self.device)

        if self.args.meta_type in ['meta_mlp', 'meta_attn']:
            self.meta_level = Meta(input_dim, hidden_dim, self.enc_type, self.meta_type)

        if self.args.enc_type == 'ind':
            self.cat_val_mlp = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
                ).to(self.device)

            self.num_val_mlp = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            ).to(self.device)
        elif self.args.enc_type == 'shared':
            self.shared_val_mlp = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim)
            ).to(self.device)


        ### self.args.aggr_type ###
        if (self.args.aggr_type == 'mean'):
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
        elif (self.args.aggr_type == 'attn'):
            self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
            nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))
            self.feature_attentions = nn.ModuleList([
                SimpleAttention(embed_dim=self.input_dim) 
                for _ in range(self.args.num_layers)
            ])  
            self.feature_attentions = self.feature_attentions.to(self.device)
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
    
    def build_prune_to_readout_mask(self, batch_size, num_tokens):
        """
            T2GFormer-like prune_to_readout:
                - block (i>0, col=0) so that row = i doesn't attent col=0
                - keep (row = 0, col=j>0) open so [CLS] can gather from features
        """
        mask = torch.zeros(num_tokens, num_tokens) # shape [F, F]
        mask[1:,0] = -10000
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask 



    def forward(self, batch, y):
        pred = self.predict(batch)
        target = y.to(self.device).view(-1,1).float()

        loss = self.criterion(pred, target)
        return loss
    
    def predict(self, batch):
        label_description_embeddings = batch['label_description_embeddings'].to(self.device)
        sample_embeddings = []
        if all(k in batch for k in ['cat_name_embeddings', 'cat_desc_embeddings', 'cat_value_embeddings']):
            cat_name_embeddings = batch['cat_name_embeddings'].to(self.device).squeeze(-2)
            cat_desc_embeddings = batch['cat_desc_embeddings'].to(self.device).squeeze(-2)
            cat_value_embeddings = batch['cat_value_embeddings'].to(self.device).squeeze(-2)

            if self.args.meta_type == 'meta_mlp':
                meta_cat_embeddings = self.meta_level(cat_name_embeddings, cat_desc_embeddings, label_description_embeddings)
                
                if self.args.enc_type == 'ind':
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    final_cat_value_embeddings = self.cat_val_mlp(concat_cat_value_embeddings)
                elif self.args.enc_type == 'shared':
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    final_cat_value_embeddings = self.shared_val_mlp(concat_cat_value_embeddings)

            elif self.args.meta_type == 'meta_attn':
                meta_cat_embeddings, meta_weights = self.meta_level(cat_name_embeddings, cat_desc_embeddings, label_description_embeddings)
                pdb.set_trace()
                if self.args.enc_type == 'ind':
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    final_cat_value_embeddings = self.cat_val_mlp(concat_cat_value_embeddings) + meta_weights.expand(-1,-1,self.input_dim) * cat_value_embeddings
                    
                elif self.args.enc_type == 'shared':
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    final_cat_value_embeddings = self.shared_val_mlp(concat_cat_value_embeddings) + meta_weights.expand(-1,-1,self.input_dim) * cat_value_embeddings
            sample_embeddings.append(final_cat_value_embeddings)
        #pdb.set_trace()
        # Numerical features 처리
        if all(k in batch for k in ['num_name_embeddings', 'num_desc_embeddings', 'num_prompt_embeddings']):
            num_name_embeddings = batch['num_name_embeddings'].to(self.device).squeeze(-2)
            num_desc_embeddings = batch['num_desc_embeddings'].to(self.device).squeeze(-2)
            num_prompt_embeddings = batch['num_prompt_embeddings'].to(self.device).squeeze(-2)

            if self.args.meta_type == 'meta_mlp':
                meta_num_embeddings = self.meta_level(num_name_embeddings, num_desc_embeddings, label_description_embeddings)
                concat_num_value_embeddings = torch.cat([num_prompt_embeddings, meta_num_embeddings], dim=-1)
                if self.args.enc_type == 'ind':
                    final_num_value_embeddings = self.num_val_mlp(concat_num_value_embeddings)
                elif self.args.enc_type == 'shared':
                    final_num_value_embeddings = self.shared_val_mlp(concat_num_value_embeddings)
            elif self.args.meta_type =='meta_attn':
                meta_num_embeddings, column_weights = self.meta_level(num_name_embeddings, num_desc_embeddings, label_description_embeddings)
                concat_num_value_embeddings = torch.cat([num_prompt_embeddings, meta_num_embeddings], dim=-1)
                
                if self.args.enc_type == 'ind':
                    num_value_embeddings = self.num_val_mlp(concat_num_value_embeddings) 
                    final_num_value_embeddings = num_value_embeddings+ column_weights.expand(-1,-1,self.input_dim) * num_value_embeddings
                elif self.args.enc_type == 'shared':
                    num_value_embeddings = self.shared_val_mlp(concat_num_value_embeddings)
                    final_num_value_embeddings = num_value_embeddings + column_weights.expand(-1,-1,self.input_dim) * num_value_embeddings
            sample_embeddings.append(final_num_value_embeddings)

        if len(sample_embeddings) > 0:
            sample_embeddings = torch.cat(sample_embeddings, dim=1)
        else:
            raise ValueError("Neither categorical nor numerical features found in batch")

        if self.args.label == 'add':
            sample_embeddings = torch.concat([label_description_embeddings, sample_embeddings], dim=1)

        
        if self.args.aggr_type == 'mean':
            sample_embeddings = sample_embeddings.mean(dim=1)
            pred = self.predictor(sample_embeddings)
        elif self.args.aggr_type == 'attn':
            attention_output = sample_embeddings
            B, F ,D = attention_output.shape
            
            cls_token = self.cls.expand(B, -1, -1)
            attention_output = torch.cat([cls_token, attention_output], dim=1)

            attn_mask = self.build_prune_to_readout_mask(B,F + 1).to(self.device)

            for attention_layer in self.feature_attentions:
                attention_output, _ = attention_layer(attention_output, attn_mask)
            pred = attention_output[:, 0, :]
            pred = self.predictor(pred)
        return pred

