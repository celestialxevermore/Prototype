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
import torch.nn.init as nn_init

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=4, use_residual=True, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.use_residual = use_residual
        self.dropout = dropout
        
        # Q, K, V 투영 레이어
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 출력 투영 레이어
        self.W_out = nn.Linear(embed_dim, embed_dim)
        
        # 드롭아웃
        self.attn_dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        nn_init.xavier_uniform_(self.W_q.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.W_k.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.W_v.weight, gain=1 / math.sqrt(2))
        nn_init.xavier_uniform_(self.W_out.weight)
        nn_init.zeros_(self.W_out.bias)
    
    def _split_heads(self, x):
        """텐서를 (batch_size, seq_len, n_heads, head_dim) 형태로 분할"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
    
    def _merge_heads(self, x):
        """헤드 결과를 다시 합침"""
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, n_heads, head_dim)
        return x.reshape(batch_size, seq_len, self.embed_dim)
    
    def forward(self, x, attn_mask=None):
        # 원본 입력 저장 (residual 연결용)
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 선형 투영
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 헤드 분할
        q = self._split_heads(q)  # (batch_size, n_heads, seq_len, head_dim)
        k = self._split_heads(k)  # (batch_size, n_heads, seq_len, head_dim)
        v = self._split_heads(v)  # (batch_size, n_heads, seq_len, head_dim)
        
        # 스케일링된 닷-프로덕트 어텐션
        scaling = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling  # (batch_size, n_heads, seq_len, seq_len)
        
        # 마스크 적용 (필요한 경우)
        if attn_mask is not None:
            # 마스크를 헤드 차원에 맞게 확장
            attn_mask = attn_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            scores = scores + attn_mask
        
        # 소프트맥스 및 드롭아웃 적용
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 값에 가중치 적용 및 헤드 합치기
        context = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        context = self._merge_heads(context)  # (batch_size, seq_len, embed_dim)
        
        # 최종 선형 투영
        output = self.W_out(context)
        
        # residual 연결 적용
        if self.use_residual:
            output = output + residual
        
        return output, attn_weights

class Meta(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, enc_type, meta_type):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.scaling = torch.sqrt(torch.tensor(input_dim).float())
        self.enc_type = enc_type
        self.meta_type = meta_type
        self.args = args
        if self.meta_type == 'meta_attn':
            self.W_q = nn.Linear(input_dim, input_dim)
            self.W_k = nn.Linear(input_dim, input_dim)
            self.W_v = nn.Linear(input_dim, input_dim)

        # name_desc_mlp 초기화
        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(self.args.meta_dropout_rate),
            nn.Linear(self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim)
        )
        
        # Sequential 내의 Linear 레이어 초기화
        for m in self.name_desc_mlp.modules():
            if isinstance(m, nn.Linear):
                nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # ReLU에 적합한 He 초기화
                if m.bias is not None:
                    nn_init.zeros_(m.bias)

    def forward(self, name_embeddings, desc_embeddings, label_description_embeddings):
        if self.meta_type == 'meta_mlp':
            name_desc_embeddings = torch.cat([name_embeddings, desc_embeddings], dim = -1)
            name_desc_embeddings = self.name_desc_mlp(name_desc_embeddings)
            return name_desc_embeddings
        if self.meta_type == 'meta_attn':
            name_desc_embeddings = torch.cat([name_embeddings, desc_embeddings], dim = -1)
            name_desc_embeddings = self.name_desc_mlp(name_desc_embeddings)
            Q = self.W_q(name_desc_embeddings)
            K = self.W_k(label_description_embeddings)
            V = self.W_v(label_description_embeddings)
            attention_weights = torch.matmul(Q, K.transpose(1, 2))
            attention_weights = attention_weights / self.scaling
            attention_weights = F.softmax(attention_weights, dim=-1)
            meta_attn_out = torch.matmul(attention_weights, V)
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
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.meta_type = args.meta_type
        self.enc_type = args.enc_type
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
        
        # MLP 초기화 함수 정의
        def init_mlp(module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn_init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # ReLU에 적합한 He 초기화
                    if m.bias is not None:
                        nn_init.zeros_(m.bias)
        
        # 다양한 MLP 초기화
        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        ).to(self.device)
        init_mlp(self.name_desc_mlp)
        
        if self.args.meta_type in ['meta_mlp', 'meta_attn']:
            self.meta_level = Meta(args, input_dim, hidden_dim, self.enc_type, self.meta_type)

        if self.args.enc_type == 'ind':
            self.cat_val_mlp = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
                ).to(self.device)
            init_mlp(self.cat_val_mlp)
            
            self.num_val_mlp = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            ).to(self.device)
            init_mlp(self.num_val_mlp)
            
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
            init_mlp(self.shared_val_mlp)

        
        if (self.args.aggr_type == 'attn'):
            self.cls = nn.Parameter(Tensor(1, 1, self.input_dim))
            nn.init.kaiming_uniform_(self.cls, a = math.sqrt(5))
            
            self.feature_attentions = nn.ModuleList([
                MultiheadAttention(
                    embed_dim=self.input_dim,
                    n_heads=args.n_heads if hasattr(args, 'n_heads') else 4,  # 기본값 4
                    use_residual=True,
                    dropout=dropout_rate
                ) 
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
            init_mlp(self.predictor)
    
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
                
                if self.args.enc_type == 'ind':
                    
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    
                    final_cat_value_embeddings = self.cat_val_mlp(concat_cat_value_embeddings) + meta_weights.expand(-1,-1,self.input_dim) * cat_value_embeddings
                    
                elif self.args.enc_type == 'shared':
                    concat_cat_value_embeddings = torch.cat([cat_value_embeddings, meta_cat_embeddings], dim=-1)
                    final_cat_value_embeddings = self.shared_val_mlp(concat_cat_value_embeddings) + meta_weights.expand(-1,-1,self.input_dim) * cat_value_embeddings
            sample_embeddings.append(final_cat_value_embeddings)

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

        if self.args.aggr_type == 'attn':
            attention_output = sample_embeddings
            B, F, D = attention_output.shape
            
            cls_token = self.cls.expand(B, -1, -1)
            attention_output = torch.cat([cls_token, attention_output], dim=1)

            attn_mask = self.build_prune_to_readout_mask(B, F + 1).to(self.device)

            for attention_layer in self.feature_attentions:
                attention_output, _ = attention_layer(attention_output, attn_mask)
            pred = attention_output[:, 0, :]
            pred = self.predictor(pred)
        return pred

