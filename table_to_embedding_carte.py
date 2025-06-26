from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
import numpy as np 
import pandas as pd 
import pdb
import torch 
import torch.nn as nn
from scipy.io import arff
import os
import random
import openai
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import networkx as nx
from typing import Union
from torch_geometric.data import Data

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline


class Table2EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, args, source_dataset_name):
        super(Table2EmbeddingTransformer, self).__init__()
        
        self.args = args
        self.input_dim = args.input_dim
        self.is_fitted_ = False
        self.scaler_type = args.scaler_type
        self.llm_model_name = args.llm_model
        self.source_dataset_name = source_dataset_name
        self._load_lm_model()
        self.label_description = self._get_label_desc()
        self.label_embedding = self._transform_label()
        self.n_components = args.input_dim
        n_jobs: int = 1
        
        # 컬럼 정보를 저장할 변수들 초기화
        self.cat_col_names = None
        self.num_col_names = None
        self.col_names = None
        self.cat_name_to_description = None
        self.num_name_to_description = None
        
    def _process_column_names(self, X):
        """컬럼명 처리 로직을 분리한 메서드"""
        cat_col_names = X.select_dtypes(include="object").columns 
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True)
        cat_col_names = list(cat_col_names)
        
        num_col_names = X.select_dtypes(exclude="object").columns 
        num_col_names = num_col_names.str.replace("\n", " ", regex=True)
        num_col_names = list(num_col_names)
        
        col_names = cat_col_names + num_col_names
        
        return cat_col_names, num_col_names, col_names
        
    def fit(self, X, y=None):
        """Fit function used for the Table2GraphTransformer

        Parameters
        ----------
        X : pandas DataFrame (n_samples, n_features)
            The input data used to transform to graphs.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.y_ = y
        self.is_fitted_ = False

        if not hasattr(self, "lm_model_"):
            self._load_lm_model()
        
        # 컬럼명 처리 - fit에서 한 번만 수행
        self.cat_col_names, self.num_col_names, self.col_names = self._process_column_names(X)
        
        # Description 매핑 생성 - fit에서 한 번만 수행
        self.cat_name_to_description = {name: self._get_feature_description(name) for name in self.cat_col_names}
        self.num_name_to_description = {name: self._get_feature_description(name) for name in self.num_col_names}

        # Numerical transformer 설정
        if self.scaler_type == 'pow':
            self.num_transformer = PowerTransformer().set_output(transform="pandas")
            if len(self.num_col_names) > 0:
                X_num = X[self.num_col_names]
                self.num_transformer.fit(X_num)

        return self

    def _load_lm_model(self):
        # GPT2 모델
        if (self.llm_model_name == "gpt2_mean") or (self.llm_model_name == "gpt2_auto"):
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = 12
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            
            for param in self.llm_model.parameters():
                param.requires_grad = False
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        # Sentence-BERT 모델
        elif self.llm_model_name == 'sentence-bert':
            try:
                from sentence_transformers import SentenceTransformer
                self.llm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cpu')
                self.tokenizer = None 
                print("Sentence-BERT 모델 로딩 완료")
            except ImportError:
                raise ImportError("sentence-transformers 패키지를 설치해주세요: pip install sentence-transformers")
        elif (self.llm_model_name == 'LLAMA_mean') or (self.llm_model_name == 'LLAMA_auto'):
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.output_attentions = True 
            self.llama_config.output_hidden_states = True 
                
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config)
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,local_files_only=False)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # BioBERT 모델
        elif self.llm_model_name == 'bio-bert':
            try:
                self.llm_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                print("BioBERT 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                
        # Bio_ClinicalBERT 모델
        elif self.llm_model_name == "bio-clinical-bert":
            try:
                self.llm_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                print("Bio_ClinicalBERT 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                
        # LLaMA 계열 모델
        elif self.llm_model_name == "bio-llama":
            try:
                self.llm_model = LlamaModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                print("Bio-Medical Llama 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")
                self.tokenizer = LlamaTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        else:
            raise ValueError(f"지원하지 않는 LLM 모델: {self.llm_model_name}")
            
        # 모든 모델 파라미터 고정
        if hasattr(self, 'llm_model') and self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False

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
        with torch.no_grad():
            # 1. Sentence-BERT 모델 처리 (tokenizer 없음)
            if self.llm_model_name == "sentence-bert":
                label_embedding = self.llm_model.encode(self.label_description, convert_to_tensor=True)
                label_embedding = label_embedding.unsqueeze(0)
            
            # 2. 다른 모델들 (BERT 계열, GPT2, LLaMA)
            else:
                label_input = self.tokenizer(self.label_description, return_tensors="pt", padding=True, truncation=True)
                outputs = self.llm_model(**label_input)
                
                # BERT 계열 모델은 [CLS] 토큰 사용
                if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                    label_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 (1, embed_dim)
                elif self.llm_model_name in ["gpt2_auto","LLAMA_auto"]:
                    last_token_idx = (label_input.attention_mask[0] == 1).sum() - 1 
                    label_embedding = outputs.last_hidden_state[0, last_token_idx, :].unsqueeze(0)
                elif self.llm_model_name in ["gpt2_mean", "LLAMA_mean"]:
                    label_embedding = outputs.last_hidden_state.mean(dim=1)  # (1, embed_dim)
                else:
                    label_embedding = outputs.last_hidden_state.mean(dim=1)

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
        """Apply Table2GraphTransformer to each row of the data

        Parameters
        ----------
        X : Pandas DataFrame. (n_samples, n_features)
            The input data used to transform to embeddings.
            Heterogeneous features are separated into categorical and numerical features.

        y : None
            Ignored.

        Returns
        -------
        Embedding Data : list of size (n_samples).
            The list of transformed embedding data.
        """
        # fit()에서 설정된 컬럼 정보 사용 (중복 제거)
        if self.cat_col_names is None or self.num_col_names is None:
            raise ValueError("Transformer must be fitted before transform. Call fit() first.")
            
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        y_ = None 
        if self.y_ is not None:
            y_ = np.array(self.y_)
            y_ = torch.tensor(y_).reshape((num_data, 1))

        # fit()에서 설정된 컬럼명 사용
        X_categorical = X_.select_dtypes(include="object").copy()
        X_categorical.columns = self.cat_col_names
        X_numerical = X_.select_dtypes(exclude="object").copy()
        X_numerical.columns = self.num_col_names

        # Numerical transformer 처리
        if len(self.num_col_names) != 0:
            X_numerical = self._get_num(X_numerical)
            
        if not self.is_fitted_:
            self.is_fitted_ = True 
        
        embedding_data = [
            self._get_embeddings(
                X_categorical,
                X_numerical,
                self.cat_name_to_description,  # fit()에서 생성된 것 사용
                self.num_name_to_description,  # fit()에서 생성된 것 사용
                y_[i] if y_ is not None else None,
                idx=i,
            )
            for i in range(num_data)
        ]

        return embedding_data
    
    def _get_num(self, X_numerical):
        """Transform numerical columns using power transformer"""
        return self.num_transformer.transform(X_numerical)   
        
    def _transform_cat(self, X_categorical, cat_name_to_description): 
        cat_name_embeddings = []
        cat_desc_embeddings = [] 
        cat_value_embeddings = []
        cat_desc_texts = [] 
        for feature_name in X_categorical.columns:
            
            descriptions_ = cat_name_to_description[feature_name]
            feature_values_ = X_categorical[feature_name].values[0]
            cat_desc_texts.append(feature_name)
            descriptions_ = f"{feature_name} : {descriptions_}"
            if self.llm_model_name == 'sentence-bert':
                with torch.no_grad():
                    # Name embedding
                    feature_name_emb = self.llm_model.encode(feature_name, convert_to_tensor=True)
                    # Description embedding
                    feature_desc_emb = self.llm_model.encode(descriptions_, convert_to_tensor=True)
                    # value embedding 
                    feature_value_emb = self.llm_model.encode(feature_values_, convert_to_tensor=True)
                    cat_name_embeddings.append(feature_name_emb)
                    cat_desc_embeddings.append(feature_desc_emb)
                    cat_value_embeddings.append(feature_value_emb)
            else:
                # Name embedding
                feature_name_input = self.tokenizer(feature_name, return_tensors="pt", padding=True, truncation=True)
                # Description embedding
                feature_desc_input = self.tokenizer(descriptions_, return_tensors="pt", padding=True, truncation=True)
                # Value embedding 
                feature_value_input = self.tokenizer(feature_values_, return_tensors="pt", padding=True, truncation=True)    
                with torch.no_grad():
                    feature_name_output = self.llm_model(**feature_name_input)
                    feature_desc_output = self.llm_model(**feature_desc_input)
                    feature_value_output = self.llm_model(**feature_value_input)
                    if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                        feature_name_emb = feature_name_output.last_hidden_state[:, 0, :].squeeze(0)
                        feature_desc_emb = feature_desc_output.last_hidden_state[:, 0, :].squeeze(0)
                        feature_value_emb = feature_value_output.last_hidden_state[:, 0, :].squeeze(0)
                    elif self.llm_model_name in ["gpt2_auto", "LLAMA_auto"]:
                        name_last_token_idx = (feature_name_input.attention_mask[0] == 1).sum() -1
                        desc_last_token_idx = (feature_desc_input.attention_mask[0] == 1).sum() - 1
                        value_last_token_idx = (feature_value_input.attention_mask[0] ==1).sum() - 1
                        feature_name_emb = feature_name_output.last_hidden_state[0, name_last_token_idx, :].squeeze(0)
                        feature_desc_emb = feature_desc_output.last_hidden_state[0, desc_last_token_idx, :].squeeze(0)
                        feature_value_emb = feature_value_output.last_hidden_state[0, value_last_token_idx, :].squeeze(0)
                    elif self.llm_model_name in ["gpt2_mean", "LLAMA_mean"]:
                        feature_name_emb = feature_name_output.last_hidden_state.mean(dim=1).squeeze(0)
                        feature_desc_emb = feature_desc_output.last_hidden_state.mean(dim=1).squeeze(0)
                        feature_value_emb = feature_value_output.last_hidden_state.mean(dim=1).squeeze(0)
                    else:
                        feature_name_emb = feature_name_output.last_hidden_state.mean(dim=1).squeeze(0)
                        feature_desc_emb = feature_desc_output.last_hidden_state.mean(dim=1).squeeze(0)
                        feature_value_emb = feature_value_output.last_hidden_state.mean(dim=1).squeeze(0)
                cat_name_embeddings.append(feature_name_emb)
                cat_desc_embeddings.append(feature_desc_emb)
                cat_value_embeddings.append(feature_value_emb)
        cat_name_embeddings = torch.stack(cat_name_embeddings, dim= 0)
        cat_desc_embeddings = torch.stack(cat_desc_embeddings, dim= 0)
        cat_value_embeddings = torch.stack(cat_value_embeddings, dim = 0)
        return cat_name_embeddings, cat_desc_embeddings, cat_value_embeddings, cat_desc_texts 

    def _transform_num(self, data_num, num_name_to_description):
        """
        Numerical features의 name embedding과 description embedding을 생성
        
        Args:
            data_num: pandas Series - 해당 행의 numerical 데이터
            num_name_to_description: dict - {feature_name: description}
        
        Returns:
            num_name_embeddings: torch.Tensor
            num_desc_embeddings: torch.Tensor  
            num_desc_texts: list
        """
        num_name_embeddings = []
        num_desc_embeddings = []
        num_desc_texts = []
        
        for feature_name, value in data_num.items():
            description = num_name_to_description[feature_name]
            num_desc_texts.append(feature_name)
            
            if self.llm_model_name == 'sentence-bert':
                with torch.no_grad():
                    # Feature name embedding
                    name_emb = self.llm_model.encode(feature_name, convert_to_tensor=True)
                    # Feature description embedding
                    desc_emb = self.llm_model.encode(description, convert_to_tensor=True)
                    
                    num_name_embeddings.append(name_emb)
                    num_desc_embeddings.append(desc_emb)
            else:
                # Feature name embedding
                name_input = self.tokenizer(feature_name, return_tensors="pt", padding=True, truncation=True)
                # Feature description embedding
                desc_input = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    name_output = self.llm_model(**name_input)
                    desc_output = self.llm_model(**desc_input)
                    
                    if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                        name_emb = name_output.last_hidden_state[:, 0, :].squeeze(0)
                        desc_emb = desc_output.last_hidden_state[:, 0, :].squeeze(0)
                    elif self.llm_model_name in ["gpt2_auto", "LLAMA_auto"]:
                        name_last_idx = (name_input.attention_mask[0] == 1).sum() - 1
                        desc_last_idx = (desc_input.attention_mask[0] == 1).sum() - 1
                        name_emb = name_output.last_hidden_state[0, name_last_idx, :].squeeze(0)
                        desc_emb = desc_output.last_hidden_state[0, desc_last_idx, :].squeeze(0)
                    elif self.llm_model_name in ["gpt2_mean", "LLAMA_mean"]:
                        name_emb = name_output.last_hidden_state.mean(dim=1).squeeze(0)
                        desc_emb = desc_output.last_hidden_state.mean(dim=1).squeeze(0)
                    else:
                        name_emb = name_output.last_hidden_state.mean(dim=1).squeeze(0)
                        desc_emb = desc_output.last_hidden_state.mean(dim=1).squeeze(0)
                
                num_name_embeddings.append(name_emb)
                num_desc_embeddings.append(desc_emb)
        
        num_name_embeddings = torch.stack(num_name_embeddings, dim=0)
        num_desc_embeddings = torch.stack(num_desc_embeddings, dim=0)

        return num_name_embeddings, num_desc_embeddings, num_desc_texts

    def _get_embeddings(self, X_categorical, X_numerical, cat_name_to_description, num_name_to_description, y, idx):
        """Transform to graph objects.

        Parameters
        ----------
        X_categorical : Pandas DataFrame of shape (n_samples, n_categorical_features)
            The input pandas DataFrame containing only the categorical features.
        X_numerical : Pandas DataFrame of shape (n_samples, n_numerical_features)
            The input pandas DataFrame containing only the numerical features.
        y : array-like of shape (n_samples,)
            The target variable to try to predict.
        idx: int
            The index of a particular data point used to transform into graphs

        Returns
        -------
        Embedding
        """
        if y is not None:
            y_ = y.clone() 
        else:
            y_ = torch.tensor([])
        s_idx = idx 
        data = {
            'label_description_embeddings' : self.label_embedding,
            'y':y_,
            's_idx' : s_idx
        }

        # Categorical features 
        if len(X_categorical.columns) > 0:
            data_cat = X_categorical.iloc[idx]
            data_cat = data_cat.dropna() 

            num_cat = len(data_cat)
            if num_cat != 0:
                data_cat = data_cat.str.replace("\n", " ", regex = True)
                cat_name_embeddings, cat_desc_embeddings, cat_value_embeddings, cat_desc_texts = self._transform_cat(
                    pd.DataFrame(data_cat).T,
                    cat_name_to_description
                )
                '''
                    args.carte
                    기본 형태는 CarTE 형식을 따른다.
                    node embedding :  feature value (M), 
                    edge_attribute : feature name (Gender)
                    Main에서의 혼선을 줄이기 위해, 부득이 cat_desc_embeddings에 cat_name_embeddings를 넣도록 한다.
                    따라서, 기본 형태에서는 반드시 --use_desc_attr를 써야 한다. 
                '''
                data.update({
                    'cat_name_value_embeddings' : cat_value_embeddings,
                    'cat_desc_embeddings' : cat_name_embeddings,
                    'cat_desc_texts' : cat_desc_texts 
                })
        if len(X_numerical.columns) > 0: 
            data_num = X_numerical.iloc[idx]
            data_num = data_num.dropna() 
            num_num = len(data_num)

            if num_num > 0:
                # 1. Name과 Description embedding 생성
                num_name_embeddings, num_desc_embeddings, num_desc_texts = self._transform_num(
                    data_num, num_name_to_description
                )
                # 2. PowerTransformed된 수치값들
                x_num_ = torch.tensor(np.array(data_num).astype("float32"))
                
                # 3. 수치값 × name embedding (CarTE 방식)
                # x_num_를 (n_features, 1)로 reshape하고 name_embeddings와 곱함
                num_prompt_embeddings = x_num_.view(-1, 1) * num_name_embeddings
                
                data.update({
                    'num_prompt_embeddings': num_prompt_embeddings,  # 수치값 × name embedding
                    'num_desc_embeddings': num_name_embeddings,     # description embedding (edge attribute용)
                    'num_desc_texts': num_desc_texts
                })
        return data