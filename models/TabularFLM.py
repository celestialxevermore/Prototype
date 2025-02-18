import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import PowerTransformer, StandardScaler
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
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
        self.is_fitted_ = False
        self.scaler_type = args.scaler_type
        self.llm_model_name = args.llm_model
        self.source_dataset_name = args.source_dataset_name
        self._load_lm_model()
        self.llm_model = self.llm_model.to(self.device)
        #self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.label_description = self._get_label_desc()
        self.label_embedding = self._transform_label() 
        num_layers = args.num_layers
        dropout_rate = args.dropout_rate
        llm_model = args.llm_model
        self.criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()

        for param in self.llm_model.parameters():
            param.requires_grad = False
        
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
        
        # LLM 모델과 tokenizer를 GPU로
        self.llm_model = self.llm_model.to(self.device)
        
        # 다른 모듈들도 GPU로
        self.name_desc_mlp = self.name_desc_mlp.to(self.device)
        self.cat_val_mlp = self.cat_val_mlp.to(self.device)
        self.num_val_mlp = self.num_val_mlp.to(self.device)

    def fit(self, X: pd.DataFrame, y = None):
        self.y_ = y
        self.is_fitted_ = False
        self.col_names = X.columns.tolist()
        # Load language model
        if not hasattr(self, "lm_model_"):
            self._load_lm_model()
        
        cat_col_names = X.select_dtypes(include="object").columns 
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True)
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns 
        num_col_names = num_col_names.str.replace("\n", " ", regex=True)
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names # n_i  = \{n_1, n_2, ... n_d\}
        self.cat_name_to_description = {name: self._get_feature_description(name) for name in self.cat_col_names}
        self.num_name_to_description = {name: self._get_feature_description(name) for name in self.num_col_names}

        if self.scaler_type == 'pow':
            self.num_transformer = PowerTransformer().set_output(transform="pandas")
            X_num = X[self.num_col_names]
            self.num_transformer.fit(X_num)
        self.is_fitted_ = True
        return self
    
    def _get_label_desc(self) -> str:
        if self.source_dataset_name is None:
            raise ValueError("source_dataset_name is not set")
        label_desc_path = f"/storage/personal/eungyeop/dataset/feature_description/{self.source_dataset_name}/{self.source_dataset_name}-metadata.json"
    
        with open(label_desc_path, 'r') as f:
            metadata = json.load(f)
            # target_binary 키에 해당하는 description을 가져옴
            label_description = metadata.get('target_binary')
            if label_description is None:
                raise ValueError("target_binary description not found in metadata")
        return label_description
    def _transform_label(self):
        """
        레이블 설명을 임베딩으로 변환
        """
        label_input = self.tokenizer(self.label_description, return_tensors="pt", padding=True, truncation=True)
        label_input = {k: v.to(self.device) for k, v in label_input.items()}  # GPU로 이동
        
        with torch.no_grad():
            label_embedding = self.llm_model(**label_input).last_hidden_state.mean(dim=1)  # (1, embed_dim)
        return label_embedding
        
    def _get_feature_description(self, feature_name: str) -> str:
        if self.source_dataset_name is None:
            ValueError("source_dataset_name is not set")
        feature_desc_path = f"/storage/personal/eungyeop/dataset/feature_description/{self.source_dataset_name}/{self.source_dataset_name}-metadata.json"
        
        if not hasattr(self, 'feature_descriptions'):
            with open(feature_desc_path, 'r') as f:
                metadata = json.load(f)
                # target_binary를 제외한 feature descriptions만 저장
                self.feature_descriptions = {k: v for k, v in metadata.items() if k != 'target_binary'}
        return self.feature_descriptions.get(feature_name, feature_name)

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0] # N : sample number 
        
        self.col_names = X.columns.tolist()
        # Load language model
        if not hasattr(self, "lm_model_"):
            self._load_lm_model()
        #pdb.set_trace()
        cat_col_names = X.select_dtypes(include="object").columns 
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True)
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns 
        num_col_names = num_col_names.str.replace("\n", " ", regex=True)
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names # n_i  = \{n_1, n_2, ... n_d\}
        self.cat_name_to_description = {name: self._get_feature_description(name) for name in self.cat_col_names}
        self.num_name_to_description = {name: self._get_feature_description(name) for name in self.num_col_names}
    
        if self.scaler_type == 'pow':
            self.num_transformer = PowerTransformer().set_output(transform="pandas")
            X_num = X[self.num_col_names]
            self.num_transformer.fit(X_num)
        self.is_fitted_ = True

        label_embedding = self.label_embedding.expand(num_data, -1, -1)
        # Separate categorical and numerical features
        X_categorical = X_[self.cat_col_names] if len(self.cat_col_names) > 0 else pd.DataFrame()
        X_numerical_raw = X_[self.num_col_names].copy() if len(self.num_col_names) > 0 else pd.DataFrame() 
        X_numerical = X_[self.num_col_names].copy() if len(self.num_col_names) > 0 else pd.DataFrame()
        

        if len(self.num_col_names) > 0 and self.scaler_type == 'pow':
            if not self.is_fitted_:
                X_numerical = self.num_transformer.fit_transform(X_numerical)
            else:
                X_numerical = self.num_transformer.transform(X_numerical)
    
        
        sample_embeddings = self._get_embeddings(X_categorical, X_numerical, X_numerical_raw, self.cat_name_to_description, self.num_name_to_description)
        sample_embeddings = torch.cat([label_embedding, sample_embeddings], dim = 1)
        attention_output = self.feature_attentions(
            query = sample_embeddings, key = sample_embeddings, value = sample_embeddings
        )
    
        return attention_output 
    def _move_inputs_to_device(self, inputs_dict, device):
        for k, v in inputs_dict.items():
            inputs_dict[k] = v.to(device)
        return inputs_dict
    # cat_name_to_description : {feature_name : feature_description}
    # 카테고리컬 데이터와 description을 받아서 카테고리컬 임베딩과 설명 임베딩을 반환함.
    def _transform_cat(self, X_categorical, cat_name_to_description):
        """
        배치 단위로 범주형 데이터의 임베딩을 생성
        Args:
            X_categorical: pd.DataFrame - categorical features
            cat_name_to_description: dict - {feature_name: feature_description}
        Returns:
            name_embeddings: (n_cat_features, embed_dim)
            desc_embeddings: (n_cat_features, embed_dim)
            value_embeddings: (n_cat_features, batch_size, embed_dim)
        """
        name_embeddings = []
        desc_embeddings = []
        value_embeddings = []
        
        for feature_name in X_categorical.columns:
            # Name embedding
            name_input = self.tokenizer(feature_name, return_tensors="pt", padding=True, truncation=True).to(self.device)
            name_input = self._move_inputs_to_device(name_input, self.device)
            with torch.no_grad():
                name_emb = self.llm_model(**name_input).last_hidden_state.mean(dim=1)
                
            # Description embedding
            desc = cat_name_to_description[feature_name]
            desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(self.device)
            desc_input = self._move_inputs_to_device(desc_input, self.device)
            with torch.no_grad():
                desc_emb = self.llm_model(**desc_input).last_hidden_state.mean(dim=1)
                
            # Value embeddings
            values = X_categorical[feature_name].values
            value_inputs = self.tokenizer([str(v) for v in values], return_tensors="pt", padding=True, truncation=True).to(self.device)
            value_inputs = self._move_inputs_to_device(value_inputs, self.device)
            with torch.no_grad():
                value_emb = self.llm_model(**value_inputs).last_hidden_state.mean(dim=1)
                
            name_embeddings.append(name_emb)
            desc_embeddings.append(desc_emb)
            value_embeddings.append(value_emb)
        
        name_embeddings = torch.cat(name_embeddings, dim=0)
        desc_embeddings = torch.cat(desc_embeddings, dim=0)
        value_embeddings = torch.stack(value_embeddings)
        
        return name_embeddings, desc_embeddings, value_embeddings

    def _transform_num(self, x_num, x_num_raw, descriptions):
        """
        배치 단위로 수치형 데이터의 임베딩을 생성
        Args:
            x_num: torch.Tensor - PowerTransformer 통과한 값들 (batch_size, n_num_features)
            x_num_raw: torch.Tensor - 원본 numerical 값들 (batch_size, n_num_features)
            descriptions: dict - {feature_name: feature_description}
        Returns:
            name_embeddings: (n_num_features, embed_dim)
            desc_embeddings: (n_num_features, embed_dim)
            prompt_embeddings: (n_num_features, embed_dim)
        """
        name_embeddings = []
        desc_embeddings = []
        prompt_embeddings = []
        
        for i, col_name in enumerate(self.num_col_names):
            # Name embedding
            name_input = self.tokenizer(col_name, return_tensors="pt", padding=True, truncation=True).to(self.device)
            name_input = self._move_inputs_to_device(name_input, self.device)
            with torch.no_grad():
                name_emb = self.llm_model(**name_input).last_hidden_state.mean(dim=1)
            
            # Description embedding
            desc = descriptions.get(col_name, col_name)
            desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(self.device)
            desc_input = self._move_inputs_to_device(desc_input, self.device)
            with torch.no_grad():
                desc_emb = self.llm_model(**desc_input).last_hidden_state.mean(dim=1)
            
            # 이 column의 통계값으로 prompt embedding 생성
            col_values = x_num_raw[:, i]
            min_val = col_values.min().item()
            max_val = col_values.max().item()
            mean_val = col_values.mean().item()
            std_val = col_values.std().item()
            
            prompt = (
                f"<|start_prompt|>"
                f"Column Name: {col_name}. "
                f"Statistics: Min = {min_val:.2f}, Max = {max_val:.2f}, "
                f"Mean = {mean_val:.2f}, Std = {std_val:.2f}. "
                "<|end_prompt|>"
            )
            
            prompt_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            prompt_input = self._move_inputs_to_device(prompt_input, self.device)
            with torch.no_grad():
                prompt_emb = self.llm_model(**prompt_input).last_hidden_state.mean(dim=1)
            
            name_embeddings.append(name_emb)
            desc_embeddings.append(desc_emb)
            prompt_embeddings.append(prompt_emb)
        
        name_embeddings = torch.cat(name_embeddings, dim=0)
        desc_embeddings = torch.cat(desc_embeddings, dim=0)
        prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
        
        return name_embeddings, desc_embeddings, prompt_embeddings

    def _get_embeddings(self, 
            X_categorical, 
            X_numerical, 
            X_numerical_raw, 
            cat_name_to_description, 
            num_name_to_description):
        batch_size = X_categorical.shape[0]

        # 데이터 전처리
        X_categorical = X_categorical.fillna("")
        X_categorical = X_categorical.replace("\n", " ", regex=True)
        X_numerical = X_numerical.fillna(0)
        X_numerical_raw = X_numerical_raw.fillna(0)

        # Categorical 데이터 변환
        cat_name_embeddings, cat_desc_embeddings, cat_value_embeddings = self._transform_cat(
            X_categorical, cat_name_to_description
        )

        # Numerical 데이터를 tensor로 변환하고 GPU로 이동
        x_num = torch.tensor(X_numerical[self.num_col_names].values, dtype=torch.float32).to(self.device)
        x_num_raw = torch.tensor(X_numerical_raw[self.num_col_names].values, dtype=torch.float32).to(self.device)

        # Numerical 데이터 변환
        num_name_embeddings, num_desc_embeddings, num_prompt_embeddings = self._transform_num(
            x_num, x_num_raw, num_name_to_description
        )

        # 임베딩 결합 및 변환 (모두 GPU에서 수행)
        cat_name_desc_embeddings = torch.cat([cat_name_embeddings, cat_desc_embeddings], dim=-1)
        cat_name_desc_embeddings = self.name_desc_mlp(cat_name_desc_embeddings)
        cat_value_embeddings = torch.cat([cat_value_embeddings, cat_name_desc_embeddings], dim=-1)
        cat_value_embeddings = self.cat_val_mlp(cat_value_embeddings)

        num_name_desc_embeddings = torch.cat([num_name_embeddings, num_desc_embeddings], dim=-1)
        num_name_desc_embeddings = self.name_desc_mlp(num_name_desc_embeddings)
        num_value_embeddings = x_num.unsqueeze(-1) * num_prompt_embeddings
        num_value_embeddings = torch.cat([num_value_embeddings, num_name_desc_embeddings], dim=-1)
        num_value_embeddings = self.num_val_mlp(num_value_embeddings)

        sample_embeddings = torch.cat([cat_value_embeddings, num_value_embeddings], dim=1)
        return sample_embeddings
    
    def forward(self, X,y):
        pred = self.predict(X)
        target = y.view(-1,1).float()

        loss = self.criterion(pred, target)
        return loss
    def predict(self, X):
        #pdb.set_trace()
        output = self.transform(X)
        pred = self.predictor(output)
        return pred

    def _load_lm_model(self):
        if self.llm_model_name == "gpt2":
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = 12
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            #self.llm_model = self.llm_model.to(self.device)
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            except EnvironmentError:  # downloads tokenizer from HF if not already done
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            #self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.model = GPT2Model.from_pretrained("gpt2")
            for param in self.llm_model.parameters():
                param.requires_grad = False
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        
        

    