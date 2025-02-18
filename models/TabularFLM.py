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

        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()

        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        
        self.name_desc_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        ).to(self.device)
        self.cat_val_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        ).to(self.device)
        self.num_val_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        ).to(self.device)
        self.feature_attentions = nn.ModuleList([
            nn.MultiheadAttention(
            embed_dim = self.input_dim,
            num_heads = 2,
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

        pdb.set_trace()
        cat_name_desc_embeddings = torch.cat([cat_name_embeddings, cat_desc_embeddings], dim=-1)
        cat_name_desc_embeddings = self.name_desc_mlp(cat_name_desc_embeddings)
        cat_value_embeddings = torch.cat([cat_value_embeddings, cat_name_desc_embeddings], dim=-1)
        cat_value_embeddings = self.cat_val_mlp(cat_value_embeddings)

        num_name_desc_embeddings = torch.cat([num_name_embeddings, num_desc_embeddings], dim=-1)
        num_name_desc_embeddings = self.name_desc_mlp(num_name_desc_embeddings)
        num_value_embeddings = torch.cat([num_prompt_embeddings, num_name_desc_embeddings], dim=-1)
        num_value_embeddings = self.num_val_mlp(num_value_embeddings)

        sample_embeddings = torch.cat([cat_value_embeddings, num_value_embeddings], dim=1)
        
        attention_output = self.feature_attentions(
            query = sample_embeddings, key = sample_embeddings, value = sample_embeddings
        )
        
        pred = self.predictor(attention_output)
        return pred
        
        

    